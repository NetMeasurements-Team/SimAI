"""Regression checks for sparse queue traces in the collective/PFC validator."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('pfc', Path(__file__).with_name('run.py'))
pfc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pfc)


class QueueTraceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        (self.folder / 'run.log').write_text(
            'all passes finished at time: 100\n'
            'All messages finished. Stopping simulation\n'
            'all-to-all forward pass collective issued for layer: collective_layer\n'
            'all-reduce weight grad collective issued for layer: collective_layer\n'
            'Total collectives issued for this layer: 2\n'
            'Total streams injected: 2\nTotal streams finished: 2\n')
        (self.folder / 'case.fct.txt').write_text(
            '0b000001 0b000101 100 100 1024 0 10 10\n'
            '0b000201 0b000301 100 100 1024 0 10 10\n')
        (self.folder / 'case.pfc.txt').write_text('')
        self.queue = self.folder / 'case.qlen.csv'
        self.queue.write_text('time,sw_id,port_id,q_id,qlen,port_len\n')

    def test_header_only_baseline(self):
        self.assertEqual(pfc.check_outputs(self.folder, 'case', 'absent')
                         ['peak_sampled_egress_port_bytes'], 0)

    def test_congestion_still_requires_queue_samples(self):
        (self.folder / 'case.pfc.txt').write_text('1 0 1 1 1\n2 0 1 1 0\n')
        with self.assertRaisesRegex(ValueError, 'no samples'):
            pfc.check_outputs(self.folder, 'case', 'present')
        with self.queue.open('a') as handle:
            handle.write('1, 4, 1, 1, 2048, 2048\n')
        self.assertEqual(pfc.check_outputs(self.folder, 'case', 'present')
                         ['peak_sampled_egress_port_bytes'], 2048)

    def test_empty_trace_still_fails(self):
        self.queue.write_text('')
        with self.assertRaisesRegex(ValueError, 'empty queue trace'):
            pfc.check_outputs(self.folder, 'case', 'absent')

    def test_missing_trace_still_fails(self):
        self.queue.unlink()
        with self.assertRaisesRegex(ValueError, 'missing queue trace'):
            pfc.check_outputs(self.folder, 'case', 'absent')

    def test_malformed_sample_still_fails(self):
        with self.queue.open('a') as handle:
            handle.write('1, 2\n')
        with self.assertRaisesRegex(ValueError, 'invalid queue trace'):
            pfc.check_outputs(self.folder, 'case', 'absent')


if __name__ == '__main__':
    unittest.main()
