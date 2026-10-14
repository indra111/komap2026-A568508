#!/usr/bin/env bash
# 학습된 가중치(GitHub Release)를 받아 runs/ 아래에 배치하고 SHA256을 검증한다.
# 사용법: ./download_weights.sh   (weights-manifest.json 기준, 이미 받은 파일은 해시가 맞으면 건너뜀)
set -euo pipefail
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "python3 필요"; exit 1; }
python3 - <<'EOF'
import hashlib, json, subprocess, sys
from pathlib import Path
m = json.loads(Path('weights-manifest.json').read_text())
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''): h.update(c)
    return h.hexdigest()
import time
n = len(m['files'])
for k, it in enumerate(m['files'], 1):
    p = Path(it['path']); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and sha(p) == it['sha256']:
        print(f'[{k}/{n}] ok (이미 있음) {p}', flush=True); continue
    print(f"[{k}/{n}] download {it['asset']} ({it['bytes'] / 1e6:.0f} MB) ...", flush=True)
    t = time.time()
    part = p.with_name(p.name + '.part')   # only a verified file gets the final name
    subprocess.run(['curl', '-sS', '-L', '--fail', '--retry', '5', '-o', str(part), it['url']], check=True)
    if sha(part) != it['sha256']:
        sys.exit(f'SHA256 mismatch: {p}')
    part.replace(p)
    print(f'[{k}/{n}] ok {p}  (SHA256 일치, {time.time() - t:.0f}초)', flush=True)
print('all weights verified')
EOF
