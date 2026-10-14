#!/usr/bin/env bash
# 실행 파일: test 10장(또는 val 20장)의 분할 마스크를 생성한다.
# 사용법: ./run_predict.sh [test|val] [출력폴더]
set -euo pipefail
cd "$(dirname "$0")"
SPLIT="${1:-test}"
OUT="${2:-outputs/${SPLIT}_masks}"
python3 -c "import json,os,sys; m=[f['path'] for f in json.load(open('weights-manifest.json'))['files'] if not os.path.exists(f['path'])]; sys.exit('가중치 없음: ./download_weights.sh 를 먼저 실행하세요\n' + '\n'.join(m)) if m else None"
WORK="$(mktemp -d)"
python -u predict_opus_champion.py --recipe review/opus/recipe-lov16-2head-al3ni110-crfbw05-psi100.json \
  --split "$SPLIT" --work "$WORK" --output "$OUT"
rm -rf "$WORK"
echo "masks: $OUT"
