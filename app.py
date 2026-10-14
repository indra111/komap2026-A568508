"""Local web interface for the final segmentation model (no extra packages: Python stdlib HTTP server).

    python app.py                 # then open http://localhost:7860
    python app.py --port 8000 --recipe review/opus/<recipe>.json

Uploaded images are segmented with exactly the submission pipeline (predict_opus_champion.py --images),
one job at a time on the GPU. Results: phase mask (official RGB colours), overlay, phase area fractions and
particle counts (8-connected components >= 10 px). Everything stays on this machine under app_jobs/."""
import argparse, base64, io, json, re, subprocess, sys, threading, time, uuid, zipfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / 'app_jobs'
PHASES = [('Al 기지', (153, 127, 76)), ('초정 Si', (76, 178, 76)), ('Al₃Ni', (25, 76, 153)), ('공정 Si', (204, 204, 204))]
jobs, lock, queue = {}, threading.Lock(), []


def default_recipe():
    found = sorted((ROOT / 'review/opus').glob('recipe-*.json'))
    if len(found) != 1:
        sys.exit('Pass --recipe (expected exactly one review/opus/recipe-*.json)')
    return found[0]


def analyse(image_path, mask_path, out_dir):
    from image_processing import cv2
    rgb = np.array(Image.open(image_path).convert('RGB'))
    mrgb = np.array(Image.open(mask_path).convert('RGB'))
    cls = np.full(mrgb.shape[:2], 0, np.uint8)
    for i, (_, c) in enumerate(PHASES):
        cls[(mrgb == np.array(c, np.uint8)).all(2)] = i
    overlay = (0.55 * rgb + 0.45 * mrgb).astype(np.uint8)
    stem = Path(mask_path).stem.removesuffix('_mask')
    Image.fromarray(overlay).save(out_dir / f'{stem}_overlay.png')
    Image.fromarray(rgb).save(out_dir / f'{stem}_input.png')
    phases = []
    for i, (name, c) in enumerate(PHASES):
        n = None
        if i:
            k, _, st, _ = cv2.connectedComponentsWithStats((cls == i).astype(np.uint8), connectivity=8)
            n = int((st[1:, 4] >= 10).sum())
        phases.append({'name': name, 'color': '#%02x%02x%02x' % c, 'fraction': round(float((cls == i).mean() * 100), 2), 'particles': n})
    return {'name': stem, 'size': f'{rgb.shape[1]}×{rgb.shape[0]}', 'phases': phases,
            'input': f'{stem}_input.png', 'mask': Path(mask_path).name, 'overlay': f'{stem}_overlay.png'}


def worker(recipe):
    while True:
        with lock:
            jid = queue.pop(0) if queue else None
        if jid is None:
            time.sleep(1); continue
        job = jobs[jid]; d = JOBS / jid
        job.update(state='running', started=time.time())
        n_members = len(json.loads(Path(recipe).read_text())['members'])
        cmd = [sys.executable, '-u', str(ROOT / 'predict_opus_champion.py'), '--recipe', str(recipe), '--split', 'val',
               '--images', str(d / 'in'), '--work', str(d / 'work'), '--output', str(d / 'out')]
        (d / 'work').mkdir(exist_ok=True)
        log = (d / 'log.txt').open('w')
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        done = 0
        for line in proc.stdout:
            log.write(line); log.flush()
            if 'teacher_member_done' in line:
                done += 1; job['progress'] = f'모델 {done}/{n_members} 추론 완료'
                if done == n_members: job['progress'] = '보정기·후처리 중'
        proc.wait(); log.close()
        if proc.returncode:
            job.update(state='error', message='추론 실패 — app_jobs/%s/log.txt 확인' % jid); continue
        res = []
        for img in sorted((d / 'in').iterdir()):
            stem = img.stem[:-len('_image')] if img.stem.endswith('_image') else img.stem
            res.append(analyse(img, d / 'out' / f'{stem}_mask.png', d / 'out'))
        job.update(state='done', results=res, seconds=round(time.time() - job['started']), progress='완료')


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
        m = re.fullmatch(r'/api/jobs/([0-9a-f]{12})', self.path)
        if m and m.group(1) in jobs:
            job = jobs[m.group(1)]
            return self.send(200, {**job, 'position': queue.index(m.group(1)) + 1 if m.group(1) in queue else 0})
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
        if self.path != '/api/jobs':
            return self.send(404, {'error': 'not found'})
        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        jid = uuid.uuid4().hex[:12]; d = JOBS / jid / 'in'; d.mkdir(parents=True)
        for f in data['files']:
            name = re.sub(r'[^\w\-.]', '_', Path(f['name']).name)
            if Path(name).suffix.lower() not in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                continue
            (d / name).write_bytes(base64.b64decode(f['data'].split(',', 1)[-1]))
        if not any(d.iterdir()):
            return self.send(400, {'error': 'PNG/JPG/TIF 이미지를 올려 주세요'})
        with lock:
            jobs[jid] = {'id': jid, 'state': 'queued', 'progress': '대기 중', 'count': len(list(d.iterdir()))}
            queue.append(jid)
        self.send(200, {'id': jid})


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>합금 조직 분할 — 팀서유기 A568508</title>
<style>
:root{--ink:#1f2328;--mute:#5a6370;--line:#d9dee4;--bg:#f6f7f9;--acc:#1957a6}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 -apple-system,"Malgun Gothic","Apple SD Gothic Neo",sans-serif;color:var(--ink);background:var(--bg)}
header{padding:28px 32px 8px;max-width:1180px;margin:auto}h1{font-size:22px;margin:0 0 4px}header p{margin:0;color:var(--mute)}
main{max-width:1180px;margin:auto;padding:16px 32px 48px}
#drop{border:2px dashed #aab4c0;border-radius:12px;background:#fff;padding:36px;text-align:center;cursor:pointer;transition:.15s}
#drop.on{border-color:var(--acc);background:#eef4fc}#drop b{color:var(--acc)}
#files{margin:10px 0 0;color:var(--mute);font-size:14px}
button{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:10px 18px;font:inherit;font-weight:600;cursor:pointer}
button:disabled{background:#9db3cf;cursor:default}.row{display:flex;gap:12px;align-items:center;margin:14px 0}
#status{color:var(--mute)}.bar{height:6px;background:#e3e8ee;border-radius:3px;overflow:hidden;flex:1;max-width:320px}.bar i{display:block;height:100%;background:var(--acc);width:0;transition:width .4s}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:18px;display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:20px}
.card h2{font-size:16px;margin:0 0 2px}.card .meta{color:var(--mute);font-size:13px;margin-bottom:10px}
.tabs{display:flex;gap:6px;margin-bottom:8px}.tabs button{background:#eef1f5;color:var(--ink);padding:6px 12px;font-weight:500}.tabs button.sel{background:var(--ink);color:#fff}
.view img{width:100%;border-radius:6px;display:block;image-rendering:pixelated}
.stack{display:flex;height:16px;border-radius:4px;overflow:hidden;margin:6px 0 14px;border:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:6px 4px;border-bottom:1px solid var(--line);text-align:right}td:first-child,th:first-child{text-align:left}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px;border:1px solid #0002}
.note{color:var(--mute);font-size:13px;margin-top:10px}@media(max-width:860px){.card{grid-template-columns:1fr}}
</style></head><body>
<header><h1>고탄성 알루미늄 합금 조직 분할</h1><p>현미경 사진을 올리면 Al 기지 · 초정 Si · Al₃Ni · 공정 Si를 구분하고 조직별 면적 분율을 계산합니다. (KoMaP 2026 · 팀서유기 A568508)</p></header>
<main>
<div id="drop"><b>이미지를 끌어다 놓거나 클릭</b>해서 선택하세요<br><span style="color:var(--mute);font-size:13px">PNG · JPG · TIF, 여러 장 가능 · 사진은 이 컴퓨터 밖으로 나가지 않습니다</span><input id="pick" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff" multiple hidden></div>
<div id="files"></div>
<div class="row"><button id="go" disabled>분할 시작</button><div class="bar"><i id="prog"></i></div><span id="status"></span></div>
<div id="out"></div>
</main>
<script>
const $=s=>document.querySelector(s);let picked=[];
const drop=$('#drop'),pick=$('#pick');
drop.onclick=()=>pick.click();pick.onchange=()=>setFiles([...pick.files]);
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,v=>{v.preventDefault();drop.classList.add('on')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,v=>{v.preventDefault();drop.classList.remove('on')}));
drop.addEventListener('drop',v=>setFiles([...v.dataTransfer.files]));
function setFiles(f){picked=f.filter(x=>/\\.(png|jpe?g|tiff?)$/i.test(x.name));$('#files').textContent=picked.length?picked.length+'장 선택: '+picked.map(x=>x.name).join(', '):'';$('#go').disabled=!picked.length}
const read=f=>new Promise(r=>{const fr=new FileReader();fr.onload=()=>r({name:f.name,data:fr.result});fr.readAsDataURL(f)});
$('#go').onclick=async()=>{
  $('#go').disabled=true;$('#status').textContent='업로드 중…';$('#prog').style.width='3%';
  const files=await Promise.all(picked.map(read));
  const r=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({files})});
  const j=await r.json();if(!r.ok){$('#status').textContent=j.error;$('#go').disabled=false;return}
  poll(j.id)};
async function poll(id){
  const j=await (await fetch('/api/jobs/'+id)).json();
  const m=/모델 (\\d+)\\/(\\d+)/.exec(j.progress||'');
  $('#prog').style.width=(j.state==='done'?100:m?5+85*m[1]/m[2]:j.progress==='보정기·후처리 중'?93:4)+'%';
  $('#status').textContent=j.state==='queued'?'대기 중 ('+j.position+'번째)':j.state==='running'?j.progress+' — 모델 8개를 차례로 불러와 수 분 걸립니다':j.state==='done'?j.count+'장 완료 ('+j.seconds+'초)':j.message;
  if(j.state==='done'){render(id,j.results);$('#go').disabled=false;return}
  if(j.state==='error'){$('#go').disabled=false;return}
  setTimeout(()=>poll(id),2000)}
function render(id,res){
  const out=$('#out');out.innerHTML='<div class="row"><a href="/api/jobs/'+id+'/zip"><button>마스크 전체 내려받기 (zip)</button></a></div>';
  for(const r of res){
    const c=document.createElement('div');c.className='card';
    const src=k=>'/files/'+id+'/'+r[k];
    c.innerHTML=`<div><h2>${r.name}</h2><div class="meta">${r.size} px</div>
      <div class="tabs">${[['overlay','겹쳐 보기'],['mask','분할 결과'],['input','원본']].map(([k,t],i)=>`<button data-k="${k}" class="${i?'':'sel'}">${t}</button>`).join('')}</div>
      <div class="view"><img src="${src('overlay')}" alt=""></div></div>
      <div><h2>조직 정량</h2><div class="stack">${r.phases.map(p=>`<span style="width:${p.fraction}%;background:${p.color}" title="${p.name} ${p.fraction}%"></span>`).join('')}</div>
      <table><tr><th>조직</th><th>면적 분율</th><th>입자 수*</th></tr>${r.phases.map(p=>`<tr><td><span class="sw" style="background:${p.color}"></span>${p.name}</td><td>${p.fraction.toFixed(2)}%</td><td>${p.particles??'—'}</td></tr>`).join('')}</table>
      <div class="note">* 8-이웃으로 연결된 영역 중 10px 이상. 붙어 있는 결정은 하나로 셉니다.</div>
      <div class="row"><a href="${src('mask')}" download="${r.mask}"><button>마스크 PNG</button></a></div></div>`;
    c.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{c.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('sel'));b.classList.add('sel');c.querySelector('.view img').src=src(b.dataset.k)});
    out.appendChild(c)}}
</script></body></html>"""


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=7860); p.add_argument('--recipe', default=None)
    a = p.parse_args()
    recipe = Path(a.recipe) if a.recipe else default_recipe()
    JOBS.mkdir(exist_ok=True)
    threading.Thread(target=worker, args=(recipe,), daemon=True).start()
    print(f'모델 구성: {recipe.name}\n브라우저에서 http://localhost:{a.port} 를 여세요 (종료: Ctrl+C)', flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
