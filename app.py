"""Local web interface for the final segmentation model (no extra packages: Python stdlib HTTP server).

    python app.py                 # then open http://localhost:7860
    python app.py --port 8000 --recipe review/opus/<recipe>.json

Everything run_predict.sh does can be run from the browser:
  1. readiness check (GPU, weights, data) and weight download (download_weights.sh, SHA256 verified)
  2. reproduce the submission: test 10 masks, or val 20 masks with the mIoU score
  3. segment your own microscope photos
All inference goes through predict_opus_champion.py (same code as run_predict.sh, byte-identical masks), one job at a
time on the GPU. Results: phase mask (official RGB colours), overlay, phase area fractions and particle counts
(8-connected components >= 10 px). Everything stays on this machine under app_jobs/."""
import argparse, base64, io, json, re, shutil, subprocess, sys, threading, time, uuid, zipfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / 'app_jobs'
PHASES = [('Al 기지', (153, 127, 76)), ('초정 Si', (76, 178, 76)), ('Al₃Ni', (25, 76, 153)), ('공정 Si', (204, 204, 204))]
jobs, lock, queue = {}, threading.Lock(), []
weights_job = {'state': 'idle', 'progress': ''}
RECIPE = None


def default_recipe():
    found = sorted((ROOT / 'review/opus').glob('recipe-*.json'))
    if len(found) != 1:
        sys.exit('Pass --recipe (expected exactly one review/opus/recipe-*.json)')
    return found[0]


def manifest_rows(split):
    m = json.loads((ROOT / 'data/manifest.json').read_text())
    return [r for r in m['items'] if r['split'] == split]


def readiness():
    gpu = None
    if shutil.which('nvidia-smi'):
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True)
        gpu = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    wm = ROOT / 'weights-manifest.json'
    files = json.loads(wm.read_text())['files'] if wm.exists() else []
    have = sum((ROOT / f['path']).exists() for f in files)
    data = {s: sum((ROOT / 'data' / r['image']).exists() for r in manifest_rows(s)) for s in ('test', 'val')}
    expected = {s: len(manifest_rows(s)) for s in ('test', 'val')}
    return {'gpu': gpu, 'weights': [have, len(files)], 'data': data, 'data_expected': expected,
            'recipe': Path(RECIPE).name, 'weights_job': weights_job}


def download_weights():
    weights_job.update(state='running', progress='시작')
    proc = subprocess.Popen(['bash', str(ROOT / 'download_weights.sh')], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    ok = 0
    with (JOBS / 'download_weights.log').open('w') as log:
        for line in proc.stdout:
            log.write(line)
            if '] ok' in line:
                ok += 1; weights_job['progress'] = f'{ok}개 검증 완료'
            elif '] download' in line:
                weights_job['progress'] = f'{ok}개 완료 · 받는 중: {line.split()[2]}'
    proc.wait()
    weights_job.update(state='done' if proc.returncode == 0 else 'error',
                       progress='모든 가중치 검증 완료' if proc.returncode == 0 else '실패 — app_jobs/download_weights.log 확인')


def analyse(image_path, mask_path, out_dir, truth_path=None):
    from image_processing import cv2
    rgb = np.array(Image.open(image_path).convert('RGB'))
    mrgb = np.array(Image.open(mask_path).convert('RGB'))
    cls = np.zeros(mrgb.shape[:2], np.uint8)
    for i, (_, c) in enumerate(PHASES):
        cls[(mrgb == np.array(c, np.uint8)).all(2)] = i
    stem = Path(mask_path).stem.removesuffix('_mask')
    Image.fromarray((0.55 * rgb + 0.45 * mrgb).astype(np.uint8)).save(out_dir / f'{stem}_overlay.png')
    Image.fromarray(rgb).save(out_dir / f'{stem}_input.png')
    phases = []
    for i, (name, c) in enumerate(PHASES):
        n = None
        if i:
            _, _, st, _ = cv2.connectedComponentsWithStats((cls == i).astype(np.uint8), connectivity=8)
            n = int((st[1:, 4] >= 10).sum())
        phases.append({'name': name, 'color': '#%02x%02x%02x' % c, 'fraction': round(float((cls == i).mean() * 100), 2), 'particles': n})
    res = {'name': stem, 'size': f'{rgb.shape[1]}×{rgb.shape[0]}', 'phases': phases,
           'input': f'{stem}_input.png', 'mask': Path(mask_path).name, 'overlay': f'{stem}_overlay.png'}
    if truth_path:
        shutil.copy(truth_path, out_dir / f'{stem}_truth.png'); res['truth'] = f'{stem}_truth.png'
    return res


def worker():
    while True:
        with lock:
            jid = queue.pop(0) if queue else None
        if jid is None:
            time.sleep(1); continue
        job = jobs[jid]; d = JOBS / jid
        job.update(state='running', started=time.time(), progress='모델 불러오는 중')
        n_members = len(json.loads(Path(RECIPE).read_text())['members'])
        cmd = [sys.executable, '-u', str(ROOT / 'predict_opus_champion.py'), '--recipe', str(RECIPE),
               '--work', str(d / 'work'), '--output', str(d / 'out')]
        cmd += ['--split', job['split']] if job['kind'] == 'split' else ['--split', 'val', '--images', str(d / 'in')]
        (d / 'work').mkdir(exist_ok=True)
        with (d / 'log.txt').open('w') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            done = 0
            for line in proc.stdout:
                log.write(line); log.flush()
                if 'teacher_member_done' in line:
                    done += 1
                    job['progress'] = f'모델 {done}/{n_members} 추론 완료' if done < n_members else '보정기·후처리 중'
            proc.wait()
        if proc.returncode:
            job.update(state='error', message=f'추론 실패 — app_jobs/{jid}/log.txt 확인'); continue
        res = []
        if job['kind'] == 'split':
            for r in manifest_rows(job['split']):
                stem = Path(r['image']).stem.removesuffix('_image')
                truth = ROOT / 'data' / r['mask'] if r.get('mask') else None
                res.append(analyse(ROOT / 'data' / r['image'], d / 'out' / f'{stem}_mask.png', d / 'out', truth))
            pm = json.loads((d / 'out' / 'prediction-manifest.json').read_text())
            if pm.get('mean_image_miou') is not None:
                job['miou'] = round(pm['mean_image_miou'] * 100, 4)
                per = {s['id']: s['miou'] for s in pm.get('images', [])}
                for x in res:
                    x['miou'] = round(per.get(x['name'], 0) * 100, 1) if x['name'] in per else None
        else:
            for img in sorted((d / 'in').iterdir()):
                res.append(analyse(img, d / 'out' / f"{img.stem.removesuffix('_image')}_mask.png", d / 'out'))
        job.update(state='done', results=res, seconds=round(time.time() - job['started']), progress='완료')


def enqueue(kind, **extra):
    jid = uuid.uuid4().hex[:12]
    (JOBS / jid).mkdir(parents=True)
    with lock:
        jobs[jid] = {'id': jid, 'kind': kind, 'state': 'queued', 'progress': '대기 중', **extra}
        queue.append(jid)
    return jid


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype='application/json; charset=utf-8'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            return self.send(200, PAGE, 'text/html; charset=utf-8')
        if self.path == '/api/status':
            return self.send(200, readiness())
        m = re.fullmatch(r'/api/jobs/([0-9a-f]{12})', self.path)
        if m and m.group(1) in jobs:
            jid = m.group(1)
            return self.send(200, {**jobs[jid], 'position': queue.index(jid) + 1 if jid in queue else 0})
        m = re.fullmatch(r'/api/jobs/([0-9a-f]{12})/zip', self.path)
        if m and m.group(1) in jobs:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w') as z:
                for p in sorted((JOBS / m.group(1) / 'out').glob('*_mask.png')):
                    z.write(p, p.name)
            return self.send(200, buf.getvalue(), 'application/zip')
        m = re.fullmatch(r'/files/([0-9a-f]{12})/([\w\-.]+\.png)', self.path)
        if m and (JOBS / m.group(1) / 'out' / m.group(2)).is_file():
            return self.send(200, (JOBS / m.group(1) / 'out' / m.group(2)).read_bytes(), 'image/png')
        self.send(404, {'error': 'not found'})

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        data = json.loads(self.rfile.read(n)) if n else {}
        if self.path == '/api/weights':
            if weights_job['state'] != 'running':
                threading.Thread(target=download_weights, daemon=True).start()
            return self.send(200, weights_job)
        if self.path == '/api/split':
            if data.get('split') not in ('test', 'val'):
                return self.send(400, {'error': 'split은 test 또는 val'})
            st = readiness()
            if st['weights'][0] < st['weights'][1]:
                return self.send(400, {'error': '가중치가 없습니다. 먼저 [가중치 받기]를 실행하세요.'})
            if st['data'][data['split']] < st['data_expected'][data['split']]:
                return self.send(400, {'error': 'data/extracted 에 주최 측 데이터가 없습니다 (README 참고).'})
            return self.send(200, {'id': enqueue('split', split=data['split'], count=st['data_expected'][data['split']])})
        if self.path == '/api/jobs':
            files = [f for f in data.get('files', []) if Path(f['name']).suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')]
            if not files:
                return self.send(400, {'error': 'PNG/JPG/TIF 이미지를 올려 주세요'})
            st = readiness()
            if st['weights'][0] < st['weights'][1]:
                return self.send(400, {'error': '가중치가 없습니다. 먼저 [가중치 받기]를 실행하세요.'})
            jid = uuid.uuid4().hex[:12]; d = JOBS / jid / 'in'; d.mkdir(parents=True)
            for f in files:
                (d / re.sub(r'[^\w\-.]', '_', Path(f['name']).name)).write_bytes(base64.b64decode(f['data'].split(',', 1)[-1]))
            with lock:
                jobs[jid] = {'id': jid, 'kind': 'images', 'state': 'queued', 'progress': '대기 중', 'count': len(files)}
                queue.append(jid)
            return self.send(200, {'id': jid})
        self.send(404, {'error': 'not found'})


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>합금 조직 분할 — 팀서유기 A568508</title>
<style>
:root{--ink:#1f2328;--mute:#5a6370;--line:#d9dee4;--bg:#f6f7f9;--acc:#1957a6;--ok:#1d7a3e;--warn:#9a5b00}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 -apple-system,"Malgun Gothic","Apple SD Gothic Neo",sans-serif;color:var(--ink);background:var(--bg)}
header{padding:28px 32px 6px;max-width:1180px;margin:auto}h1{font-size:22px;margin:0 0 4px}header p{margin:0;color:var(--mute)}
main{max-width:1180px;margin:auto;padding:12px 32px 48px}
.panel{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-top:16px}
.panel h2{font-size:16px;margin:0 0 10px}.panel h2 small{font-weight:400;color:var(--mute);margin-left:6px}
.checks{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.check{border:1px solid var(--line);border-radius:10px;padding:10px 12px}.check b{display:block;font-size:13px;color:var(--mute);font-weight:500}
.check span{font-weight:600}.ok{color:var(--ok)}.warn{color:var(--warn)}
button{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:9px 16px;font:inherit;font-weight:600;cursor:pointer}
button.alt{background:#eef1f5;color:var(--ink)}button:disabled{background:#9db3cf;color:#fff;cursor:default}
.row{display:flex;gap:12px;align-items:center;margin:12px 0 0;flex-wrap:wrap}
#drop{border:2px dashed #aab4c0;border-radius:12px;background:#fafbfc;padding:28px;text-align:center;cursor:pointer;transition:.15s}
#drop.on{border-color:var(--acc);background:#eef4fc}#drop b{color:var(--acc)}#files{margin:8px 0 0;color:var(--mute);font-size:14px}
.status{color:var(--mute)}.bar{height:6px;background:#e3e8ee;border-radius:3px;overflow:hidden;flex:1;min-width:160px;max-width:320px}.bar i{display:block;height:100%;background:var(--acc);width:0;transition:width .4s}
.score{font-size:15px;margin-top:14px;padding:12px 14px;background:#eef6f0;border-radius:10px}.score b{font-size:22px;color:var(--ok)}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:18px}
.card h3{font-size:15px;margin:0 0 2px}.card .meta{color:var(--mute);font-size:13px;margin-bottom:8px}
.tabs{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap}.tabs button{background:#eef1f5;color:var(--ink);padding:5px 11px;font-weight:500}.tabs button.sel{background:var(--ink);color:#fff}
.view img{width:100%;border-radius:6px;display:block;image-rendering:pixelated}
.stack{display:flex;height:14px;border-radius:4px;overflow:hidden;margin:6px 0 12px;border:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:5px 4px;border-bottom:1px solid var(--line);text-align:right}td:first-child,th:first-child{text-align:left}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px;border:1px solid #0002}
.note{color:var(--mute);font-size:13px;margin-top:8px}@media(max-width:860px){.card{grid-template-columns:1fr}.checks{grid-template-columns:1fr}}
</style></head><body>
<header><h1>고탄성 알루미늄 합금 조직 분할</h1><p>Al 기지 · 초정 Si · Al₃Ni · 공정 Si를 구분하고 조직별 면적 분율을 계산합니다. (KoMaP 2026 · 팀서유기 A568508 · 검증 mIoU 79.90%)</p></header>
<main>
<section class="panel"><h2>1. 준비 상태 <small id="recipe"></small></h2>
  <div class="checks">
    <div class="check"><b>GPU</b><span id="c-gpu">확인 중…</span></div>
    <div class="check"><b>학습된 가중치</b><span id="c-w">확인 중…</span></div>
    <div class="check"><b>주최 측 데이터 (test / val)</b><span id="c-d">확인 중…</span></div>
  </div>
  <div class="row"><button id="dl" class="alt">가중치 받기 (GitHub Release, 5.3GB)</button><span class="status" id="dl-s"></span></div>
</section>
<section class="panel"><h2>2. 제출 결과 재현 <small>run_predict.sh 와 같은 결과</small></h2>
  <div class="row"><button data-split="test">test 10장 마스크 만들기</button><button data-split="val">val 20장 점수 확인 (mIoU)</button>
    <div class="bar"><i id="p-split"></i></div><span class="status" id="s-split"></span></div>
  <div id="out-split"></div>
</section>
<section class="panel"><h2>3. 내 사진 분석</h2>
  <div id="drop"><b>이미지를 끌어다 놓거나 클릭</b>해서 선택하세요<br><span style="color:var(--mute);font-size:13px">PNG · JPG · TIF, 여러 장 가능 · 사진은 이 컴퓨터 밖으로 나가지 않습니다</span><input id="pick" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff" multiple hidden></div>
  <div id="files"></div>
  <div class="row"><button id="go" disabled>분할 시작</button><div class="bar"><i id="p-img"></i></div><span class="status" id="s-img"></span></div>
  <div id="out-img"></div>
</section>
</main>
<script>
const $=s=>document.querySelector(s);let picked=[];
async function status(){
  const s=await (await fetch('/api/status')).json();
  $('#recipe').textContent='모델 구성: '+s.recipe;
  $('#c-gpu').innerHTML=s.gpu?'<span class="ok">✓ '+s.gpu+'</span>':'<span class="warn">✗ NVIDIA GPU를 찾지 못함 (추론에 필요)</span>';
  const w=s.weights;$('#c-w').innerHTML=w[0]===w[1]?'<span class="ok">✓ '+w[0]+'/'+w[1]+'개</span>':'<span class="warn">'+w[0]+'/'+w[1]+'개 — 가중치 받기 필요</span>';
  const d=s.data,e=s.data_expected;$('#c-d').innerHTML=(d.test===e.test&&d.val===e.val?'<span class="ok">✓ ':'<span class="warn">')+d.test+'/'+e.test+' · '+d.val+'/'+e.val+'장</span>';
  const j=s.weights_job;$('#dl-s').textContent=j.state==='idle'?'':j.progress;$('#dl').disabled=j.state==='running'||w[0]===w[1];
  if(j.state==='running')setTimeout(status,3000);}
$('#dl').onclick=async()=>{await fetch('/api/weights',{method:'POST'});status()};
document.querySelectorAll('[data-split]').forEach(b=>b.onclick=async()=>{
  const r=await fetch('/api/split',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({split:b.dataset.split})});
  const j=await r.json();if(!r.ok){$('#s-split').textContent=j.error;return}
  document.querySelectorAll('[data-split]').forEach(x=>x.disabled=true);$('#out-split').innerHTML='';poll(j.id,'split')});
const drop=$('#drop'),pick=$('#pick');drop.onclick=()=>pick.click();pick.onchange=()=>setFiles([...pick.files]);
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,v=>{v.preventDefault();drop.classList.add('on')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,v=>{v.preventDefault();drop.classList.remove('on')}));
drop.addEventListener('drop',v=>setFiles([...v.dataTransfer.files]));
function setFiles(f){picked=f.filter(x=>/\.(png|jpe?g|tiff?)$/i.test(x.name));$('#files').textContent=picked.length?picked.length+'장 선택: '+picked.map(x=>x.name).join(', '):'';$('#go').disabled=!picked.length}
const read=f=>new Promise(r=>{const fr=new FileReader();fr.onload=()=>r({name:f.name,data:fr.result});fr.readAsDataURL(f)});
$('#go').onclick=async()=>{$('#go').disabled=true;$('#s-img').textContent='업로드 중…';
  const files=await Promise.all(picked.map(read));
  const r=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({files})});
  const j=await r.json();if(!r.ok){$('#s-img').textContent=j.error;$('#go').disabled=false;return}
  $('#out-img').innerHTML='';poll(j.id,'img')};
async function poll(id,k){
  const j=await (await fetch('/api/jobs/'+id)).json();const m=/모델 (\d+)\/(\d+)/.exec(j.progress||'');
  $('#p-'+k).style.width=(j.state==='done'?100:m?5+85*m[1]/m[2]:j.progress==='보정기·후처리 중'?93:4)+'%';
  $('#s-'+k).textContent=j.state==='queued'?'대기 중 ('+j.position+'번째)':j.state==='running'?j.progress+' — 모델 8개를 차례로 불러와 수 분 걸립니다':j.state==='done'?j.count+'장 완료 ('+j.seconds+'초)':j.message;
  if(j.state==='done'||j.state==='error'){$('#go').disabled=!picked.length;document.querySelectorAll('[data-split]').forEach(x=>x.disabled=false)}
  if(j.state==='done')render(id,j,k);else if(j.state!=='error')setTimeout(()=>poll(id,k),2000)}
function render(id,j,k){
  const out=$('#out-'+k);
  out.innerHTML=(j.miou!=null?'<div class="score">검증 20장 이미지별 mIoU 평균 <b>'+j.miou.toFixed(2)+'%</b></div>':'')+
    '<div class="row"><a href="/api/jobs/'+id+'/zip"><button>마스크 전체 내려받기 (zip)</button></a></div>';
  for(const r of j.results){
    const c=document.createElement('div');c.className='card';const src=x=>'/files/'+id+'/'+r[x];
    const tabs=[['overlay','겹쳐 보기'],['mask','분할 결과'],['input','원본']].concat(r.truth?[['truth','정답']]:[]);
    c.innerHTML=`<div><h3>${r.name}</h3><div class="meta">${r.size} px${r.miou!=null?' · mIoU '+r.miou+'%':''}</div>
      <div class="tabs">${tabs.map(([x,t],i)=>`<button data-k="${x}" class="${i?'':'sel'}">${t}</button>`).join('')}</div>
      <div class="view"><img loading="lazy" src="${src('overlay')}" alt=""></div></div>
      <div><h3>조직 정량</h3><div class="stack">${r.phases.map(p=>`<span style="width:${p.fraction}%;background:${p.color}" title="${p.name} ${p.fraction}%"></span>`).join('')}</div>
      <table><tr><th>조직</th><th>면적 분율</th><th>입자 수*</th></tr>${r.phases.map(p=>`<tr><td><span class="sw" style="background:${p.color}"></span>${p.name}</td><td>${p.fraction.toFixed(2)}%</td><td>${p.particles??'—'}</td></tr>`).join('')}</table>
      <div class="note">* 8-이웃으로 연결된 영역 중 10px 이상. 붙어 있는 결정은 하나로 셉니다.</div>
      <div class="row"><a href="${src('mask')}" download="${r.mask}"><button class="alt">마스크 PNG</button></a></div></div>`;
    c.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{c.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('sel'));b.classList.add('sel');c.querySelector('.view img').src=src(b.dataset.k)});
    out.appendChild(c)}}
status();
</script></body></html>"""


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=7860); p.add_argument('--recipe', default=None)
    a = p.parse_args()
    RECIPE = Path(a.recipe) if a.recipe else default_recipe()
    JOBS.mkdir(exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    print(f'모델 구성: {RECIPE.name}\n브라우저에서 http://localhost:{a.port} 를 여세요 (종료: Ctrl+C)', flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
