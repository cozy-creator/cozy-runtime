"""Read-only ordinary-Cozy proof collector. No install, inference, rental or teardown actions."""
from pathlib import Path
from decimal import Decimal
import argparse
import hashlib
import json
import shutil
import subprocess
import time

LEGACY = {'room', 'offload', 'offload_begin', 'ended_for_room',
          'gpu_pressure', 'gpu_pressure_observation'}
ROTATE_BYTES = 4 << 20  # exact frozen MemoryLog source; it retains current + prior file


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def command(out, label, argv):
    assert not (out / (label + '.command.json')).exists(), 'preserve the old collection'
    began = time.time_ns() // 1_000_000
    try:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=60)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        code, stdout, stderr = None, error.stdout or '', error.stderr or ''
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors='replace')
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors='replace')
    (out / (label + '.stdout')).write_text(stdout)
    (out / (label + '.stderr')).write_text(stderr)
    row = {'command': argv, 'started_unix_ms': began, 'finished_unix_ms': time.time_ns() // 1_000_000,
           'returncode': code, 'stdout_sha256': hashlib.sha256(stdout.encode()).hexdigest(),
           'stdout_bytes': len(stdout.encode()), 'stderr_bytes': len(stderr.encode())}
    save(out / (label + '.command.json'), row)
    return row, stdout


def parse_log(text, machine):
    records, lines, errors = [], [], []
    for at, line in enumerate(text.splitlines()):
        try:
            wrapper = json.loads(line)
            if not isinstance(wrapper, dict) or wrapper.get('log') != 'memory':
                errors.append({'line': at, 'reason': 'not a memory-log wrapper'})
                continue
            assert wrapper.get('machine') == machine
            raw = wrapper['text']
            assert isinstance(raw, str)
            stamp, note = raw.split(' ', 1)
            milliseconds = int(Decimal(stamp) * 1000)
            body = json.loads(note)
            assert isinstance(body, dict) and isinstance(body.get('event'), str)
            records.append({'at_ms': milliseconds, 'note': body, 'line': len(lines)})
            lines.append(raw)
        except (AssertionError, ValueError, TypeError, KeyError, ArithmeticError) as error:
            errors.append({'line': at, 'reason': type(error).__name__})
    return {'records': records, 'lines': lines, 'parse_errors': errors,
            'retained_text_bytes': sum(len(line.encode()) + 1 for line in lines)}


def expected(cohort):
    return {'version': cohort['tensord']['version'], 'commit': cohort['creator_commit'],
            'runtime': cohort['runtime']['version'], 'tensorfs': cohort['tensorfs']['tensorfs_version']}


def state_errors(document, rental_id, cohort):
    errors = []
    if document.get('rental_id') != rental_id:
        errors.append('machine rental identity mismatch')
    if document.get('state') != 'ready' or document.get('unreachable'):
        errors.append('machine is not a currently reachable ready worker')
    status = document.get('status') or {}
    for key, value in expected(cohort).items():
        if status.get(key) != value:
            errors.append(f'worker {key} differs: {status.get(key)!r} != {value!r}')
    if len(status.get('gpus', [])) != 1:
        errors.append('first owner cohort requires exactly one worker-visible GPU')
    return errors


def capture(out, args, cohort):
    out.mkdir(parents=True, exist_ok=False)
    cli = shutil.which('cozy')
    assert cli is not None
    cli_digest = hashlib.sha256(Path(cli).read_bytes()).hexdigest()
    show_call, show_text = command(out, 'machine', [cli, 'machine', 'show', args.machine, '--json'])
    log_call, log_text = command(out, 'memory', [cli, 'machine', 'logs', args.machine,
                                              'memory', '--lines', '0', '--json'])
    try:
        show = json.loads(show_text)
    except ValueError:
        show = {}
    parsed = parse_log(log_text, args.machine)
    errors = state_errors(show, args.rental_id, cohort)
    if show_call['returncode'] != 0 or log_call['returncode'] != 0:
        errors.append('a read-only collection command failed; see raw response')
    if parsed['parse_errors']:
        errors.append('memory snapshot has malformed/non-log records; absence cannot be qualified')
    result = {'machine': args.machine, 'rental_id': args.rental_id, 'cli': cli,
              'cli_sha256': cli_digest, 'worker': show, 'memory': parsed,
              'memory_read': log_call, 'read_commands_succeeded': not errors,
              'errors': errors, 'query_boundary': 'all retained current + prior memory files; no CLI line cap and no timestamp prefilter'}
    save(out / 'capture.json', result)
    return result


def run_show(out, label, run):
    call, text = command(out, label, ['cozy', 'run', 'show', str(run), '--once', '--json'])
    try:
        document = json.loads(text)
    except ValueError:
        document = {}
    save(out / (label + '.json'), document)
    return call, document


def detail(run):
    value = (run.get('result') or {}).get('detail')
    return json.loads(value) if isinstance(value, str) else value


def run_errors(run, parsed, payload, cohort):
    errors = []
    if run.get('memoized') is True:
        errors.append('a memoized result is not automatic-owner execution')
    if run.get('status') != 'completed' or not (run.get('result') or {}).get('passed'):
        errors.append('ordinary run or fixture did not pass')
    if not isinstance(parsed, dict) or parsed.get('passed') is not True:
        return errors + ['fixture detail missing or unsuccessful']
    provenance = parsed.get('provenance') or {}
    for key in ('runtime_version', 'runtime_owner_sha256', 'tensorfs_version',
                'extension_sha256', 'shim_sha256', 'gpu_uuid'):
        if provenance.get(key) != payload[key]:
            errors.append(f'fixture provenance mismatch for {key}')
    software = run.get('measurements', {}).get('software', {})
    for key, value in {'runtime': cohort['runtime']['version'],
                       'tensorfs': cohort['tensorfs']['tensorfs_version'],
                       'tensord': cohort['tensord']['version'],
                       'tensord_commit': cohort['creator_commit']}.items():
        if software.get(key) != value:
            errors.append(f'actual run software mismatch for {key}')
    gpus = run.get('measurements', {}).get('execution', {}).get('gpus', [])
    if len(gpus) != 1 or gpus[0].get('pid') != provenance.get('pid'):
        errors.append('actual executor GPU PID differs from the fixture owner')
    if len(gpus) == 1 and gpus[0].get('uuid', '').removeprefix('GPU-').lower() != payload['gpu_uuid'].removeprefix('GPU-').lower():
        errors.append('actual execution GPU differs from the owned payload')
    return errors


def anchor(before, after):
    old, new = before['lines'], after['lines']
    if not old:
        # With no old retained notes and <4MiB afterward, the frozen rotation rule could
        # not have rotated away a prior file during this interval. Still scope to returned
        # memory notes, never all driver events or an arbitrary whole-machine history.
        return {'covered': bool(new) and after['retained_text_bytes'] < ROTATE_BYTES,
                'method': 'empty baseline; returned retained text below the frozen rotation threshold',
                'records': 0, 'claim': 'bounded logged-action coverage only'}
    count = min(32, len(old)); needle = old[-count:]
    found = [at for at in range(len(new) - count + 1) if new[at:at + count] == needle]
    return {'covered': len(found) == 1, 'method': 'exact pre-run suffix retained in post-run all-lines read',
            'records': count, 'matches': found,
            'claim': 'bounded logged-action coverage only; missing/ambiguous anchor is inconclusive'}


def analyze(before, after, gpu, observed, payload, cohort):
    first, second = detail(gpu), detail(observed)
    errors = list(before['errors']) + list(after['errors'])
    errors += run_errors(gpu, first, payload, cohort)
    errors += run_errors(observed, second, payload, cohort)
    if not isinstance(first, dict) or not isinstance(second, dict):
        return {'passed': False, 'errors': errors, 'scope': 'missing real fixture result'}
    pid = first.get('provenance', {}).get('pid')
    same_owner = (second.get('provenance', {}).get('pid') == pid
                  and second.get('owner_row') == first.get('owner_row'))
    if not same_owner:
        errors.append('observe was not the same PID and participant incarnation')
    if first.get('automatic_owner_after', {}).get('active') is not True or second.get('facts', {}).get('active') is not True:
        errors.append('actual Runtime owner was not active in both requests')
    bs, ads = before['worker'].get('status', {}), after['worker'].get('status', {})
    stable_boot = all(bs.get(k) == ads.get(k) and bs.get(k) is not None
                      for k in ('worker_id', 'started_at_ms'))
    if not stable_boot:
        errors.append('worker boot changed across the test')
    gpus = gpu.get('measurements', {}).get('execution', {}).get('gpus', [])
    device = gpus[0].get('gpu') if len(gpus) == 1 else None
    notes = after['memory']['records']
    policy = [r for r in notes if r['note'].get('event') == 'gpu_memory_policy'
              and r['note'].get('pid') == pid and r['note'].get('gpus') == [device]
              and r['at_ms'] >= bs.get('started_at_ms', 0)]
    if len(policy) != 1 or policy[0]['note'].get('mode') != 'Cooperative':
        errors.append('exact PID/device Cooperative policy binding missing, duplicated or different')
    coverage = anchor(before['memory'], after['memory'])
    if not coverage['covered']:
        errors.append('log retention anchor cannot establish bounded absence')
    start = policy[0]['at_ms'] if len(policy) == 1 else None
    execution = observed.get('measurements', {}).get('execution', {})
    finish = execution.get('reply_sent_unix_ms')
    if not isinstance(finish, (int, float)) or start is None or finish < start:
        errors.append('same-worker policy-to-reply timestamp window is unavailable')
        window = []
    else:
        window = [r for r in notes if start <= r['at_ms'] <= finish]
    # This first cohort has exactly one worker-visible device. Any legacy GPU action in
    # this window is therefore relevant, even an ended_for_room note lacking a PID.
    legacy = [r for r in window if r['note'].get('event') in LEGACY]
    if legacy:
        errors.append('legacy GPU byte-policy actions were logged in the owned interval')
    return {'passed': not errors, 'errors': errors, 'gpu_run': gpu.get('number'),
            'observe_run': observed.get('number'), 'pid': pid, 'owner_row': first.get('owner_row'),
            'gpu_uuid': payload['gpu_uuid'], 'device_index': device, 'same_pid_and_incarnation': same_owner,
            'worker_boot_stable': stable_boot, 'policy_bindings': policy,
            'logged_action_window_ms': [start, finish], 'coverage': coverage,
            'scoped_notes': window, 'legacy_actions': legacy,
            'actual_versions': gpu.get('measurements', {}).get('software'),
            'start_active_boundary': 'raw Start frame is not exposed by runshow; actual owner active in both requests plus exact worker/run versions and TensorD source acknowledgement are checked',
            'scope': 'automatic same-domain owner mechanism and bounded logged policy isolation only; not all unlogged paths, machine shared-mode propagation, game coexistence or complete P3'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('before', 'observe-input', 'after'))
    parser.add_argument('--machine')
    parser.add_argument('--rental-id')
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--payload', type=Path)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--run', type=int)
    parser.add_argument('--observe-run', type=int)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); cohort = json.loads(args.cohort.read_text())
    if args.action == 'before':
        assert args.machine and args.rental_id
        row = capture(args.out, args, cohort)
        print(json.dumps({'capture': str(args.out / 'capture.json'), 'valid': not row['errors'], 'errors': row['errors']}))
        return bool(row['errors'])
    assert args.payload and args.run
    payload = json.loads(args.payload.read_text())
    if args.action == 'observe-input':
        args.out.mkdir(parents=True, exist_ok=False)
        _, run = run_show(args.out, 'gpu-run', args.run); parsed = detail(run)
        errors = run_errors(run, parsed, payload, cohort)
        save(args.out / 'validation.json', {'passed': not errors, 'errors': errors})
        if errors:
            print(json.dumps({'passed': False, 'errors': errors})); return 1
        result = {**payload, 'expected_pid': parsed['provenance']['pid'], 'owner_row': parsed['owner_row']}
        save(args.out / 'observe-input.json', result)
        print(json.dumps({'payload': str(args.out / 'observe-input.json'),
                          'expected_pid': result['expected_pid'], 'owner_row': result['owner_row']}))
        return 0
    assert args.machine and args.rental_id and args.before and args.observe_run
    after = capture(args.out, args, cohort)
    _, gpu = run_show(args.out, 'gpu-run', args.run)
    _, observed = run_show(args.out, 'observe-run', args.observe_run)
    before = json.loads((args.before / 'capture.json').read_text())
    report = analyze(before, after, gpu, observed, payload, cohort)
    save(args.out / 'qualification.json', report)
    print(json.dumps({k: report.get(k) for k in ('passed', 'errors', 'gpu_run', 'observe_run', 'pid', 'same_pid_and_incarnation')}))
    return not report['passed']


if __name__ == '__main__':
    raise SystemExit(main())
