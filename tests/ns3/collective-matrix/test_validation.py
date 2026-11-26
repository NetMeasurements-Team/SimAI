"""Regression tests for the matrix validator, without a simulator dependency."""
import csv
import contextlib
import io
from unittest import mock
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('matrix', Path(__file__).with_name('run.py'))
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.case = matrix.cases_for('v2.30')[0]
        self.log = ('all passes finished at time: 123\n'
                    'All messages finished. Stopping simulation\n'
                    'all-reduce forward pass collective issued for layer: collective_layer\n'
                    'Total collectives issued for this layer: 1\n'
                    'Total streams injected: 1\nTotal streams finished: 1\n')
        (self.folder/'run.log').write_text(self.log)
        self.rows = [dict(collective='test', op='0', data_size=str(self.case['bytes']),
                          algorithm='1', protocol='0', channel_id='0', flow_id=str(rank),
                          src=str(rank), dest=str((rank+1) % 4), flow_size='1024',
                          chunk_id='0', chunk_count='1', conn_type='RING',
                          parent_flow_ids='', prev_flow_ids='') for rank in range(4)]
        self.write_csv()
        self.fct = ''.join(f'{(0x0b000001 + (r << 8)):08x} '
                           f'{(0x0b000001 + (((r+1) % 4) << 8)):08x} '
                           '100 100 1024 0 10 10\n' for r in range(4))
        (self.folder/'test.fct.txt').write_text(self.fct)

    def write_csv(self, version='v2.30'):
        fields = list(self.rows[0]) if self.rows else list(self.fieldnames)
        if version == 'v2.20':
            fields = [f for f in fields if f not in ['algorithm', 'protocol']]
        self.fieldnames = fields
        with (self.folder/'ncclFlowModel_detailed_flows.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.rows)

    def check(self, version='v2.30'):
        return matrix.check_outputs(self.folder, 'test', self.case, version)

    def test_complete_matching_flows(self):
        self.assertEqual(self.check()['completed_messages'], 4)

    def test_missing_message_despite_clean_exit_markers(self):
        (self.folder/'test.fct.txt').write_text('\n'.join(self.fct.splitlines()[:-1])+'\n')
        with self.assertRaisesRegex(ValueError, 'message mismatch'):
            self.check()

    def test_wrong_protocol_cannot_pass(self):
        self.rows[0]['protocol'] = '2'
        self.write_csv()
        with self.assertRaisesRegex(ValueError, 'fallback'):
            self.check()

    def test_wrong_algorithm_cannot_pass(self):
        self.rows[0]['algorithm'] = '6'
        self.write_csv()
        with self.assertRaisesRegex(ValueError, 'fallback'):
            self.check()

    def test_empty_schedule_cannot_pass(self):
        self.rows = []
        self.write_csv()
        with self.assertRaisesRegex(ValueError, 'empty generated'):
            self.check()

    def test_wrong_binary_version_cannot_pass(self):
        self.write_csv('v2.20')
        with self.assertRaisesRegex(ValueError, 'schema'):
            self.check()

    def test_legacy_schema(self):
        self.write_csv('v2.20')
        self.assertEqual(self.check('v2.20')['generated_flows'], 4)

    def test_missing_completion_marker(self):
        (self.folder/'run.log').write_text(self.log.replace('All messages finished.', ''))
        with self.assertRaisesRegex(ValueError, 'marker'):
            self.check()

    def test_duplicate_schedule_cannot_pass(self):
        self.rows.append(self.rows[0].copy())
        self.write_csv()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.check()

    def test_wrong_completed_size_cannot_pass(self):
        (self.folder/'test.fct.txt').write_text(self.fct.replace('1024', '2048'))
        with self.assertRaisesRegex(ValueError, 'message mismatch'):
            self.check()

    def test_reduce_scatter_pxn_labels_are_valid(self):
        self.case = next(c for c in matrix.cases_for('v2.30') if c['name'] == 'pxn-reducescatter-LL')
        (self.folder/'run.log').write_text(self.log.replace('all-reduce', 'reduce-scatter'))
        self.rows[0]['conn_type'] = 'PXN_INIT'
        self.rows[1]['conn_type'] = 'PXN'
        self.write_csv()
        self.assertEqual(self.check()['completed_messages'], 4)

    def test_reduce_scatter_pxn_requires_forwarding_labels(self):
        self.case = next(c for c in matrix.cases_for('v2.30') if c['name'] == 'pxn-reducescatter-LL')
        (self.folder/'run.log').write_text(self.log.replace('all-reduce', 'reduce-scatter'))
        with self.assertRaisesRegex(ValueError, 'forwarding labels are absent'):
            self.check()

    def test_pxn_labels_rejected_when_disabled(self):
        self.rows[0]['conn_type'] = 'PXN'
        self.write_csv()
        with self.assertRaisesRegex(ValueError, 'unexpected flow types'):
            self.check()

    def test_generated_inputs(self):
        for version in ['v2.20', 'v2.30']:
            cases = matrix.cases_for(version)
            self.assertEqual(len(cases), len({c['name'] for c in cases}))
            for case in cases:
                directory = self.folder/(version + '-' + case['name'])
                directory.mkdir()
                matrix.prepare(directory, case)
                lines = (directory/'topology.txt').read_text().splitlines()
                nodes, ppn, nvs, switches, links = map(int, lines[0].split()[:5])
                self.assertEqual(nodes, case['ranks'] + nvs + switches)
                self.assertEqual(len(lines[1].split()), nvs + switches)
                self.assertEqual(len(lines[2:]), links)
                self.assertEqual(case['ranks'] % ppn, 0)
                self.assertNotIn('@BUFFER_MB@', (directory/'network.conf').read_text())


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.runner = Path(__file__).with_name('run.py')

    def invoke(self, *args, env=None):
        return subprocess.run([sys.executable, str(self.runner), *args],
                              text=True, capture_output=True, timeout=15, env=env)

    def fake_binary(self, body):
        binary = self.folder/'fake-simulator'
        binary.write_text('#!/usr/bin/env python3\n' + body + '\n')
        binary.chmod(0o755)
        return binary

    def test_prepare_both_versions_without_binary(self):
        for version, count in [('2.20', 15), ('2.30', 55)]:
            output = self.folder/version
            result = self.invoke('--nccl-version', version, '--prepare-only',
                                 '--binary', str(self.folder/'missing'), '--output', str(output))
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(len(summary['cases']), count)
            self.assertEqual({r['status'] for r in summary['cases'].values()}, {'PREPARED'})

    @unittest.skipIf(os.name == 'nt', 'executable-script subprocess test requires Linux/WSL')
    def test_failed_processes_continue_and_environment_is_isolated(self):
        binary = self.fake_binary('import os, sys\nprint(os.getenv("AS_SEND_LAT", "unset"))\nsys.exit(7)')
        output = self.folder/'failed'
        env = {**os.environ, 'AS_SEND_LAT': '999999'}
        result = self.invoke('--nccl-version', '2.30', '--binary', str(binary),
                             '--output', str(output), '--case', 'local-allreduce-LL',
                             '--case', 'local-allgather-LL', env=env)
        self.assertEqual(result.returncode, 1, result.stderr)
        summary = json.loads((output/'summary.json').read_text())
        self.assertEqual(len(summary['cases']), 2)
        for case, record in summary['cases'].items():
            self.assertEqual(record['status'], 'FAIL')
            self.assertIn('7', record['reason'])
            self.assertEqual((output/case/'run.log').read_text().strip(), 'unset')

    @unittest.skipIf(os.name == 'nt', 'executable-script subprocess test requires Linux/WSL')
    def test_timeout_is_recorded(self):
        binary = self.fake_binary('import time\ntime.sleep(30)')
        output = self.folder/'timeout'
        result = self.invoke('--nccl-version', '2.20', '--binary', str(binary),
                             '--output', str(output), '--case', 'odd3-allreduce', '--timeout', '1')
        self.assertEqual(result.returncode, 1, result.stderr)
        record = json.loads((output/'summary.json').read_text())['cases']['odd3-allreduce']
        self.assertEqual(record['status'], 'FAIL')
        self.assertIn('timed out', record['reason'])

    def test_interrupt_is_saved(self):
        output = self.folder/'interrupted'
        argv = ['run.py', '--nccl-version', '2.20', '--binary', sys.executable,
                '--case', 'odd3-allreduce', '--output', str(output)]
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(
                matrix.subprocess, 'run', side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(matrix.main(), 130)
        record = json.loads((output/'summary.json').read_text())['cases']['odd3-allreduce']
        self.assertEqual(record['status'], 'INTERRUPTED')
        self.assertGreaterEqual(record['elapsed_seconds'], 0)

    def test_pxn_rerun_includes_reference(self):
        output = self.folder/'pxn'
        result = self.invoke('--nccl-version', '2.30', '--prepare-only',
                             '--case', 'pxn-reducescatter-LL', '--output', str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        cases = json.loads((output/'summary.json').read_text())['cases']
        self.assertEqual(set(cases), {'cross-reducescatter-LL', 'pxn-reducescatter-LL'})

if __name__ == '__main__':
    unittest.main()
