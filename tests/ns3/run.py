#!/usr/bin/env python3
"""Prepare and run a fixture-based SimAI collective integration test (stdlib only)."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def check_outputs(folder, name, expectation):
    log = (folder / "run.log").read_text(errors="replace")
    for marker in ("all passes finished at time:", "All messages finished. Stopping simulation"):
        if marker not in log:
            raise ValueError(f"missing completion marker: {marker}")
    for marker in ("all-to-all forward pass collective issued for layer: collective_layer",
                   "all-reduce weight grad collective issued for layer: collective_layer"):
        if marker not in log:
            raise ValueError(f"fixture collective was not issued: {marker}")
    collectives = re.search(r"Total collectives issued for this layer:\s*(\d+)", log)
    if not collectives or int(collectives[1]) != 2:
        raise ValueError("expected two collectives for the fixture layer")
    if "dropping lossless packet" in log:
        raise ValueError("MMU packet loss detected")
    injected = re.search(r"Total streams injected:\s*(\d+)", log)
    finished = re.search(r"Total streams finished:\s*(\d+)", log)
    if not injected or not finished or int(injected[1]) == 0 or injected[1] != finished[1]:
        raise ValueError("rank-0 stream counts are missing, zero or incomplete")
    flows = (folder / f"{name}.fct.txt").read_text().splitlines()
    ranks = set()
    for line in flows:
        fields = line.split()
        if len(fields) != 8 or int(fields[4]) <= 0 or int(fields[6]) <= 0:
            raise ValueError("invalid completed-message FCT record")
        ranks.update((int(address, 16) >> 8) & 0xffff for address in fields[:2])
    if ranks != {0, 1, 2, 3}:
        raise ValueError(f"completed network messages do not cover all four ranks: {ranks}")
    counts = Counter()
    for line in (folder / f"{name}.pfc.txt").read_text().splitlines():
        values = list(map(int, line.split()))
        if len(values) != 5 or values[4] not in (0, 1):
            raise ValueError("invalid PFC trace record")
        counts[values[4]] += 1
    if expectation == "present" and (not counts[1] or not counts[0]):
        raise ValueError("workload completed but did not exercise both PFC pause and resume")
    if expectation == "absent" and sum(counts.values()):
        raise ValueError("unexpected PFC in baseline")
    peak = 0
    queue_trace = folder / f"{name}.qlen.csv"
    if not queue_trace.is_file():
        raise ValueError("missing queue trace: enable ENABLE_MONITOR 1 in network.conf "
                         "and rebuild the simulator with queue-monitor support")
    queue_rows = queue_trace.read_text().splitlines()[1:]
    if not queue_rows:
        raise ValueError("queue trace contains no samples; check MON_START and workload duration")
    for line in queue_rows:
        fields = line.split(',')
        if len(fields) != 6:
            raise ValueError("invalid queue trace record")
        peak = max(peak, int(fields[5]))
    return dict(completed_messages=len(flows), rank0_streams=int(injected[1]),
                xoff=counts[1], xon=counts[0], peak_sampled_egress_port_bytes=peak)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/ns3/collective-pfc')
    parser.add_argument('--binary', type=Path, default=ROOT / 'bin/SimAI_simulator')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', default='all', help='fixture case name, or all')
    parser.add_argument('--timeout', type=int, default=300, help='wall-clock seconds per case')
    parser.add_argument('--prepare-only', action='store_true', help='generate inputs without executing SimAI')
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    cases = json.loads((fixture / 'cases.json').read_text())
    if args.case != 'all' and args.case not in cases:
        parser.error(f'unknown case {args.case}; available: {list(cases)}')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.case != 'all':
        cases = {args.case: cases[args.case]}
    binary = args.binary.resolve()
    if not args.prepare_only and not (binary.is_file() and os.access(binary, os.X_OK)):
        parser.error(f'not executable: {binary}; build with scripts/build.sh -c ns3')
    output = (args.output or ROOT / ('simai-test-results-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
    if any(c.isspace() for c in str(output)):
        parser.error('output path must not contain whitespace (network config parser limitation)')
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    settings = {'AS_NVLS_ENABLE': '0', 'AS_PXN_ENABLE': '0', 'AS_SEND_LAT': '0'}
    env.update(settings)
    summary = {}
    for case, params in cases.items():
        folder = output / case
        folder.mkdir()
        name = 'test-' + output.name + '-' + case
        replacements = {'@BYTES@': str(params['bytes']), '@TRUNK@': params['trunk'],
                        '@BUFFER_MB@': str(params['buffer_mb'])}
        for source, target in [('workload.txt.in', 'workload.txt'), ('topology.txt.in', 'topology.txt'),
                               ('network.conf.in', 'network.conf'), ('system.txt', 'system.txt')]:
            text = (fixture / source).read_text()
            for token, value in replacements.items():
                text = text.replace(token, value)
            if re.search(r'@[A-Z_]+@', text):
                raise ValueError(f'unresolved template token in {source}')
            (folder / target).write_text(text)
        (folder / 'flow.txt').write_text('0\n')
        (folder / 'trace.txt').write_text('0\n')
        command = [str(binary), '-t', '1', '-w', str(folder/'workload.txt'),
                   '-n', str(folder/'topology.txt'), '-c', str(folder/'network.conf'),
                   '-s', str(folder/'system.txt'), '-r', name]
        metadata = dict(command=command, cwd=str(folder), environment_overrides=settings,
                        parameters=params, input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in folder.iterdir() if p.is_file()})
        (folder/'invocation.json').write_text(json.dumps(metadata, indent=2)+'\n')
        if args.prepare_only:
            print(f'PREPARED {case}: {folder}')
            continue
        print(f'Running {case}', flush=True)
        try:
            with (folder/'run.log').open('w') as log:
                subprocess.run(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=args.timeout, check=True)
            summary[case] = dict(status='PASS', **check_outputs(folder, name, params['pfc']))
            print(f'PASS {case}: {json.dumps(summary[case])}', flush=True)
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            summary[case] = dict(status='FAIL', reason=str(exc))
            (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            print(f'FAIL {case}: {exc}\nInspect {folder}', file=sys.stderr)
            return 1
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Results: {output}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
