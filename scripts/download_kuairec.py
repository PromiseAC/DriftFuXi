"""Download the authors' KuaiRec archive and verify the Zenodo MD5.

Only dataset preparation. Files remain in ignored tmp/ and are never redistributed.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5
from pathlib import Path
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://zenodo.org/records/18164998/files/KuaiRec.zip'
SIZE = 431964858
MD5 = '261550d472c48eff4990fb13c0e5bcf7'
CHUNK = 8 * 1024 * 1024
WORKERS = 12
TARGET = ROOT / 'tmp/KuaiRec.zip'
PARTS = ROOT / 'tmp/.kuairec_parts'


def checksum(path):
    h = md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(index):
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    part = PARTS / f'{index:03d}.part'
    if part.exists() and part.stat().st_size == end - start + 1:
        return index
    pending = part.with_suffix('.pending')
    for attempt in range(3):
        result = subprocess.run([
            'curl', '--fail', '--silent', '--show-error', '--location', '--retry', '2',
            '--connect-timeout', '20', '--max-time', '900', '--range', f'{start}-{end}',
            '--output', str(pending), '--write-out', '%{http_code}', URL,
        ], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout == '206' and pending.stat().st_size == end - start + 1:
            os.replace(pending, part)
            return index
        pending.unlink(missing_ok=True)
        print(f'chunk {index} retry {attempt+1}: status={result.stdout} error={result.stderr[-180:]}', flush=True)
    raise RuntimeError(f'Failed range {start}-{end}')


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size == SIZE and checksum(TARGET) == MD5:
        print('Already verified:', TARGET)
        return
    n = (SIZE + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, i) for i in range(n)]
        for count, future in enumerate(as_completed(futures), 1):
            print(f'chunk {future.result()+1}/{n} verified, complete {count}/{n}', flush=True)
    assembled = TARGET.with_suffix('.assembling')
    with assembled.open('wb') as out:
        for i in range(n):
            with (PARTS/f'{i:03d}.part').open('rb') as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                    out.write(block)
    assert assembled.stat().st_size == SIZE
    actual = checksum(assembled)
    if actual != MD5:
        raise RuntimeError(f'MD5 mismatch: {actual} != {MD5}')
    os.replace(assembled, TARGET)
    print('KuaiRec archive verified:', TARGET, actual, flush=True)

if __name__ == '__main__':
    main()
