"""Bounded local experiment runner; every run gets its own output directory."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'reports' / 'runs'
PROFILES = {'fixture_lr': 'logistic_regression', 'fixture_rf': 'random_forest',
            'oulad_lr': 'logistic_regression', 'oulad_rf': 'random_forest'}
for _family in ('hist_gradient_boosting', 'extra_trees', 'knn', 'gaussian_nb'):
    PROFILES['oulad_' + _family] = _family
    PROFILES['fixture_' + _family] = _family
for _profile,_family in list(PROFILES.items()):
    PROFILES[_profile+'_separate'] = _family
_pool = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()
_active = None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def _execute(directory, record):
    global _active
    try:
        fixture = record['profile'].startswith('fixture_')
        command = [sys.executable, '-m', 'edurisk.oulad_external', '--raw',
                   str(ROOT / 'data' / 'oulad' / ('fixture' if fixture else 'raw')), '--out', str(directory),
                   '--family', PROFILES[record['profile']], '--bootstrap', '200' if fixture else '600']
        if fixture:
            command.append('--fixture')
        if record['profile'].endswith('_separate'):
            command.append('--separate-calibration')
        with (directory / 'run.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=300 if fixture else 1200, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        record['status'] = 'completed' if completed.returncode == 0 else 'failed'
        record['exit_code'] = completed.returncode
        if completed.returncode:
            record['error'] = 'Эксперимент завершился с ошибкой. Подробности в журнале.'
    except subprocess.TimeoutExpired:
        record.update(status='failed', error='Превышено ограничение времени запуска.')
    except Exception as error:
        record.update(status='failed', error=str(error))
    finally:
        record['finished_at'] = _now()
        try:
            _write(directory / 'run.json', record)
        finally:
            with _lock:
                _active = None


def start(profile):
    global _active
    if profile not in PROFILES:
        raise ValueError('Неизвестный профиль эксперимента')
    if profile.startswith('oulad_'):
        from .oulad_external import REQUIRED_FILES
        if not all((ROOT / 'data' / 'oulad' / 'raw' / name).is_file() for name in REQUIRED_FILES):
            raise ValueError('Для внешней проверки нужны все семь оригинальных CSV в data/oulad/raw')
    with _lock:
        if _active:
            raise RuntimeError('Уже выполняется эксперимент. Дождитесь завершения.')
        run_id = uuid.uuid4().hex
        directory = RUNS / run_id
        directory.mkdir(parents=True)
        record = dict(id=run_id, profile=profile, status='running', started_at=_now(),
                      mode='SYNTHETIC_SCHEMA_FIXTURE' if profile.startswith('fixture_') else 'REAL_OULAD_EXTERNAL')
        _write(directory / 'run.json', record)
        _active = run_id
        try:
            _pool.submit(_execute, directory, record.copy())
        except Exception:
            _active = None
            record.update(status='failed', error='Не удалось запустить рабочий процесс')
            _write(directory / 'run.json', record)
            raise
        return record


def history(limit=30):
    records = []
    for path in RUNS.glob('*/run.json'):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if record['status'] == 'running' and record['id'] != _active:
            record['status'] = 'interrupted'
        if record['status'] == 'completed':
            try:
                summary=json.loads((path.parent / 'external_oulad_summary.json').read_text(encoding='utf-8'))
                record['metrics']=summary['metrics']['selected_calibrated']
            except (ValueError, OSError, KeyError):
                pass
        records.append(record)
    ordered = sorted(records, key=lambda r: r['started_at'], reverse=True)
    return ordered if limit is None else ordered[:limit]
