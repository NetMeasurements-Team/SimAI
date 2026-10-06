#!/usr/bin/env python3
"""Completion/selection smoke matrix for the ns-3 frontend (Python stdlib only)."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / 'tests/ns3/collective-pfc'
OPS = {'ALLREDUCE': 'all-reduce', 'ALLGATHER': 'all-gather',
       'REDUCESCATTER': 'reduce-scatter', 'ALLTOALL': 'all-to-all'}
ALGOS = {'Ring': 1, 'NVLS': 4, 'PAT': 6}
PROTOS = {'LL': 0, 'LL128': 1, 'Simple': 2, 'UNDEF': -1}
GAPS = [
    'Tree, CollNet Direct/Chain and NVLS-tree: not selectable by the current mock selector.',
    'Broadcast: no workload issue path in this frontend.',
    'PAT LL/LL128: PAT always selects Simple; a forced protocol does not override it.',
    'NVLS LL/LL128: excluded from the supported matrix (NVLS is Simple-only in NCCL).',
    'v2.20: no PAT or protocol selection; flow labels supply weaker selection evidence than v2.30.',
    'Completion does not validate reduced tensor values, schedule dependency correctness, or timing accuracy.',
]
BASE_ENV = {'AS_NVLS_ENABLE': '0', 'AS_NVLS_MIN_BYTES': '2097152',
            'AS_PXN_ENABLE': '0', 'SIMAI_PAT_ENABLE': '0',
            'SIMAI_PAT_MIN_BYTES': '524288', 'SIMAI_PAT_MAX_BYTES': '1048576',
            'SIMAI_PROTO_AWARE': '1', 'SIMAI_DUMP_DETAILED_FLOWS': '1'}


def cases_for(version):
    cases = []
    def add(name, op, ranks=4, ppn=2, size=786432, algo='Ring', proto=None, **env):
        cases.append(dict(name=name, op=op, ranks=ranks, ppn=ppn, bytes=size,
                          algorithm=algo, protocol=proto or ('LL' if version == 'v2.30' else 'UNDEF'),
                          environment=env))
    protocols = ['LL', 'LL128', 'Simple'] if version == 'v2.30' else ['UNDEF']
    for proto in protocols:
        force = {'SIMAI_FORCE_PROTO': proto} if version == 'v2.30' else {}
        for location, ppn in [('local', 4), ('cross', 2)]:
            for op in OPS:
                add(f'{location}-{op.lower()}-{proto}', op, ppn=ppn, proto=proto, **force)
        for op in ['ALLTOALL', 'REDUCESCATTER']:
            add(f'pxn-{op.lower()}-{proto}', op, proto=proto, AS_PXN_ENABLE='1', **force)
    for op in OPS:
        add(f'odd3-{op.lower()}', op, ranks=3, ppn=1)
    add('nvls-allreduce', 'ALLREDUCE', ranks=8, ppn=8, size=4194304,
        algo='NVLS', proto='Simple' if version == 'v2.30' else 'UNDEF', AS_NVLS_ENABLE='1')
    if version == 'v2.30':
        for ranks in [2, 3, 4]:
            for op in ['ALLGATHER', 'REDUCESCATTER']:
                add(f'pat{ranks}-{op.lower()}', op, ranks=ranks, ppn=1,
                    algo='PAT', proto='Simple', SIMAI_PAT_ENABLE='1')
        for op in ['ALLGATHER', 'REDUCESCATTER']:
            add(f'pat-auto-{op.lower()}', op, ppn=1, size=524288,
                algo='PAT', proto='Simple', SIMAI_PAT_ENABLE='2')
            add(f'pat-ineligible-{op.lower()}', op, SIMAI_PAT_ENABLE='1')
        for op in ['ALLGATHER', 'REDUCESCATTER']:
            for suffix, size in [('below', 524284), ('above', 1048580)]:
                add(f'pat-auto-{suffix}-{op.lower()}', op, ppn=1, size=size, SIMAI_PAT_ENABLE='2')
        for force in ['LL', 'LL128']:
            add(f'pat-ignores-forced-{force}', 'ALLGATHER', ppn=1, algo='PAT',
                proto='Simple', SIMAI_PAT_ENABLE='1', SIMAI_FORCE_PROTO=force)
        for proto, size in [('LL', 4194304), ('LL128', 4194308), ('Simple', 16777220)]:
            add(f'auto-proto-{proto}', 'ALLGATHER', ranks=2, ppn=1, size=size, proto=proto)
        add('protocol-disabled', 'ALLREDUCE', proto='UNDEF', SIMAI_PROTO_AWARE='0')
    return cases


def topology(case):
    ranks, ppn = case['ranks'], case['ppn']
    nvls = case['algorithm'] == 'NVLS'
    nvs = ranks // ppn if nvls else 0
    tor = ranks + nvs
    links = [f'{rank} {tor} 100Gbps 0.001ms 0' for rank in range(ranks)]
    if nvls:
        links += [f'{rank} {ranks + rank // ppn} 400Gbps 0.000025ms 0' for rank in range(ranks)]
    return (f'{tor + 1} {ppn} {nvs} 1 {len(links)} {"H100" if nvls else "A100"}\n'
            + ' '.join(map(str, range(ranks, tor + 1))) + '\n' + '\n'.join(links) + '\n')


def prepare(folder, case):
    n = case['ranks']
    (folder/'topology.txt').write_text(topology(case))
    (folder/'workload.txt').write_text(
        f'HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: {n} ep: 1 pp: 1 '
        f'vpp: 1 ga: 1 all_gpus: {n} checkpoints: 0 checkpoint_initiates: 0\n1\n'
        f'collective_layer -1 1000 {case["op"]} {case["bytes"]} 1000 NONE 0 1000 NONE 0 1\n')
    (folder/'system.txt').write_text((FIXTURE/'system.txt').read_text())
    config = (FIXTURE/'network.conf.in').read_text().replace('@BUFFER_MB@', '32')
    config = config.replace('ENABLE_MONITOR 1', 'ENABLE_MONITOR 0')
    (folder/'network.conf').write_text(config)
    for name in ['flow.txt', 'trace.txt']:
        (folder/name).write_text('0\n')


def check_outputs(folder, run_name, case, version):
    log = (folder/'run.log').read_text(errors='replace')
    for marker in ['all passes finished at time:', 'All messages finished. Stopping simulation',
                   f'{OPS[case["op"]]} forward pass collective issued for layer: collective_layer']:
        if marker not in log:
            raise ValueError(f'missing completion/issue marker: {marker}')
    issued = re.findall(r'Total collectives issued for this layer:\s*(\d+)', log)
    injected = re.search(r'Total streams injected:\s*(\d+)', log)
    finished = re.search(r'Total streams finished:\s*(\d+)', log)
    if not issued or set(issued) != {'1'}:
        raise ValueError(f'expected exactly one collective per layer, got {issued}')
    if not injected or not finished or int(injected[1]) <= 0 or injected[1] != finished[1]:
        raise ValueError('rank-0 stream counts missing, zero or incomplete')
    if 'dropping lossless packet' in log:
        raise ValueError('MMU packet loss detected')
    with (folder/'ncclFlowModel_detailed_flows.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError('empty generated flow model: cannot count a no-op as PASS')
    has_selection = {'algorithm', 'protocol'} <= fields
    if has_selection != (version == 'v2.30'):
        raise ValueError('flow CSV schema does not match requested SimCCL version; rebuild first')
    if has_selection:
        selected = {(int(r['algorithm']), int(r['protocol'])) for r in rows}
        expected = {(ALGOS[case['algorithm']], PROTOS[case['protocol']])}
        if selected != expected:
            raise ValueError(f'algorithm/protocol fallback: expected {expected}, observed {selected}')
    labels = {r['conn_type'] for r in rows}
    required_label = {'NVLS': 'NVLS', 'PAT': 'PAT'}.get(case['algorithm'])
    if required_label and labels != {required_label}:
        raise ValueError(f'expected {required_label} flows, got {labels}')
    if case['algorithm'] == 'Ring':
        allowed = {'PTP', 'PTP_PXN_START', 'PTP_PXN_END'} if case['op'] == 'ALLTOALL' else {'RING'}
        if case['op'] == 'REDUCESCATTER' and case['environment'].get('AS_PXN_ENABLE') == '1':
            allowed |= {'PXN_INIT', 'PXN'}
            if not {'PXN_INIT', 'PXN'} <= labels:
                raise ValueError('PXN ReduceScatter requested but forwarding labels are absent')
        if not labels <= allowed:
            raise ValueError(f'unexpected flow types: {labels}')
    if case['op'] == 'ALLTOALL' and case['environment'].get('AS_PXN_ENABLE') == '1':
        if not {'PTP_PXN_START', 'PTP_PXN_END'} <= labels:
            raise ValueError('PXN requested but forwarding flows are absent')
    planned = Counter()
    identities = set()
    for row in rows:
        key = (row['collective'], row['channel_id'], row['flow_id'])
        if key in identities:
            raise ValueError(f'duplicate generated flow: {key}')
        identities.add(key)
        src, dst, size = (int(row[k]) for k in ['src', 'dest', 'flow_size'])
        if size <= 0 or src == dst or int(row['data_size']) != case['bytes']:
            raise ValueError('invalid flow size, self-flow, or workload size mismatch')
        planned[src, dst, size] += 1
    completed = Counter()
    for line in (folder/f'{run_name}.fct.txt').read_text().splitlines():
        fields = line.split()
        if len(fields) != 8 or int(fields[4]) <= 0 or int(fields[6]) <= 0:
            raise ValueError('invalid completed-message FCT record')
        src, dst = ((int(address, 16) >> 8) & 0xffff for address in fields[:2])
        completed[src, dst, int(fields[4])] += 1
    if completed != planned:
        missing, extra = planned - completed, completed - planned
        raise ValueError(f'generated/completed message mismatch: missing={sum(missing.values())}, '
                         f'extra={sum(extra.values())}; examples missing={list(missing.items())[:4]}, '
                         f'extra={list(extra.items())[:4]}')
    ranks = {rank for src, dst, _ in completed for rank in [src, dst] if rank < case['ranks']}
    if ranks != set(range(case['ranks'])):
        raise ValueError(f'completed messages do not cover all GPU ranks: {sorted(ranks)}')
    return dict(generated_flows=len(rows), completed_messages=sum(completed.values()),
                rank0_streams=int(injected[1]), flow_types=sorted(labels),
                selection_evidence='CSV algorithm/protocol' if has_selection else 'flow types (v2.20 has no selection columns)')


def sha256(path):
    with path.open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nccl-version', required=True, choices=['2.20', 'v2.20', '2.30', 'v2.30'])
    parser.add_argument('--binary', type=Path, default=ROOT/'bin/SimAI_simulator')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', action='append', help='exact name, repeatable; default: all')
    parser.add_argument('--timeout', type=int, default=120, help='wall seconds per case')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    version = 'v' + args.nccl_version.lstrip('v')
    cases = cases_for(version)
    if args.case:
        unknown = set(args.case) - {c['name'] for c in cases}
        if unknown:
            parser.error(f'unknown cases: {sorted(unknown)}; use --list')
        selected_names = set(args.case)
        for name in args.case:
            if name.startswith('pxn-reducescatter-'):
                selected_names.add(name.replace('pxn-', 'cross-', 1))
        cases = [c for c in cases if c['name'] in selected_names]
    if args.list:
        for case in cases:
            print(f'{case["name"]}: {case["ranks"]} ranks/{case["ppn"]} per server, '
                  f'{case["bytes"]} bytes, {case["algorithm"]}/{case["protocol"]}')
        print(f'{len(cases)} cases. Coverage gaps:')
        print('\n'.join(GAPS))
        return 0
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    binary = args.binary.resolve()
    if not args.prepare_only and not (binary.is_file() and os.access(binary, os.X_OK)):
        parser.error(f'not executable: {binary}')
    output = (args.output or ROOT/('simai-test-results-matrix-' + version + '-' +
                                   datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
    if any(c.isspace() for c in str(output)):
        parser.error('output path must not contain whitespace (network parser limitation)')
    output.mkdir(parents=True, exist_ok=False)
    # Prevent the shell environment from silently changing selection/latency. Record removals.
    removed = sorted(k for k in os.environ if k.startswith(('AS_', 'SIMAI_', 'NCCL_')))
    clean_env = {k: v for k, v in os.environ.items() if k not in removed}
    summary = dict(version=version, binary=str(binary),
                   binary_sha256=None if args.prepare_only else sha256(binary),
                   coverage_gaps=GAPS, cases={})
    def save():
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for case in cases:
        folder = output/case['name']
        folder.mkdir()
        prepare(folder, case)
        name = 'matrix-' + output.name + '-' + case['name']
        settings = {**BASE_ENV, **case['environment']}
        command = [str(binary), '-t', '1', '-w', str(folder/'workload.txt'),
                   '-n', str(folder/'topology.txt'), '-c', str(folder/'network.conf'),
                   '-s', str(folder/'system.txt'), '-r', name]
        metadata = dict(command=command, cwd=str(folder), case=case, version=version,
                        binary_sha256=summary['binary_sha256'], timeout=args.timeout,
                        environment_overrides=settings, environment_removed=removed,
                        input_sha256={p.name: sha256(p) for p in folder.iterdir() if p.is_file()})
        (folder/'invocation.json').write_text(json.dumps(metadata, indent=2)+'\n')
        result = dict(status='PREPARED')
        if not args.prepare_only:
            print(f'RUN {case["name"]} (timeout: {args.timeout}s)', flush=True)
            started = time.monotonic()
            try:
                with (folder/'run.log').open('w') as log:
                    subprocess.run(command, cwd=folder, env={**clean_env, **settings},
                                   stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
                result = dict(status='PASS', **check_outputs(folder, name, case, version))
                if case['name'].startswith('pxn-reducescatter-'):
                    baseline = output/case['name'].replace('pxn-', 'cross-', 1)
                    def signature(directory):
                        with (directory/'ncclFlowModel_detailed_flows.csv').open(newline='') as handle:
                            return Counter((r['src'], r['dest'], r['flow_size'], r['channel_id'],
                                            r['parent_flow_ids']) for r in csv.DictReader(handle))
                    if signature(baseline) == signature(folder):
                        raise ValueError('PXN ReduceScatter generated the same schedule as PXN-off')
                    result['pxn_schedule_differs_from_baseline'] = True
            except KeyboardInterrupt:
                summary['cases'][case['name']] = dict(status='INTERRUPTED',
                    reason='Interrupted by user', elapsed_seconds=round(time.monotonic() - started, 3))
                save()
                print(f'\nINTERRUPTED {case["name"]}; results saved in {output}', file=sys.stderr)
                return 130
            except (subprocess.SubprocessError, ValueError, OSError, KeyError, csv.Error) as exc:
                result = dict(status='FAIL', reason=str(exc))
            result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        summary['cases'][case['name']] = result
        save()
        print(f'{result["status"]} {case["name"]}' + (': ' + result['reason'] if 'reason' in result else ''), flush=True)
    counts = Counter(r['status'] for r in summary['cases'].values())
    print(f'Results: {output}\n{dict(counts)}')
    return 1 if counts['FAIL'] else 0


if __name__ == '__main__':
    sys.exit(main())
