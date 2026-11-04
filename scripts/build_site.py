"""Build the static results site (GitHub Pages / preview) from samples/.

    python scripts/build_site.py            # -> site/index.html + site/img/
The page is fully static: the sample images, their traced overlays and the
scores in samples/results.json. The live demo runs in a Codespace.
"""
import json
import os
import shutil
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")
SITE = os.path.join(ROOT, "site")
REPO = os.environ.get("FPVEC_REPO", "GS-Manoj-Kumar/floor-plan-parser")


def thumb(src, dst, width=900):
    img = cv2.imread(src)
    if img is None:
        return False
    k = min(1.0, width / img.shape[1])
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return True


def main(fragment_out=None):
    rows = json.load(open(os.path.join(SAMPLES, "results.json")))
    if os.path.isdir(SITE):
        shutil.rmtree(SITE)
    os.makedirs(os.path.join(SITE, "img"))
    samples = {}
    for r in rows:
        name, var = r["sample"], r["variant"]
        ext = "png" if var == "clean" else "jpg"
        thumb(os.path.join(SAMPLES, "input", f"{name}_{var}.{ext}"), os.path.join(SITE, "img", f"{name}_{var}_in.jpg"))
        thumb(os.path.join(SAMPLES, "output", f"{name}_{var}_overlay.jpg"), os.path.join(SITE, "img", f"{name}_{var}_out.jpg"))
        s = samples.setdefault(name, {"name": name, "style": r["style"], "dims": r["dims"], "units": r["units"], "v": {}})
        s["v"]["clean" if var == "clean" else "distorted"] = dict(
            kind=var, method=r["scale_method"], votes=r["votes"], size=r["size_err_pct"],
            wr=r["wall_recall"], wp=r["wall_precision"], swr=r["shape_wall_recall"], swp=r["shape_wall_precision"],
            dt=r["doors_true"], df=r["doors_found"], dx=r["doors_false"],
            wt=r["windows_true"], wf=r["windows_found"], wx=r["windows_false"],
            fix=r["rectify"], secs=r["seconds"])

    def agg(var):
        rs = [r for r in rows if (r["variant"] == "clean") == (var == "clean")]
        n = len(rs)
        size = sorted(r["size_err_pct"] for r in rs)
        return dict(n=n, auto=sum(r["scale_method"] in ("dimensions", "room areas") for r in rs),
                    size=size[n // 2],
                    swr=sum(r["shape_wall_recall"] for r in rs) / n, swp=sum(r["shape_wall_precision"] for r in rs) / n,
                    wr=sum(r["wall_recall"] for r in rs) / n,
                    doors=sum(r["doors_found"] for r in rs) / max(1, sum(r["doors_true"] for r in rs)),
                    xd=sum(r["doors_false"] for r in rs),
                    wins=sum(r["windows_found"] for r in rs) / max(1, sum(r["windows_true"] for r in rs)),
                    xw=sum(r["windows_false"] for r in rs))

    data = {"samples": list(samples.values()), "summary": {"clean": agg("clean"), "distorted": agg("distorted")},
            "repo": REPO}
    body = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__REPO__", REPO)
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as f:
        f.write("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                + body.replace("<!--BODY-->", "</head>\n<body>") + "\n</body>\n</html>\n")
    if fragment_out:
        with open(fragment_out, "w", encoding="utf-8") as f:
            f.write(body.replace("<!--BODY-->", ""))
    open(os.path.join(SITE, ".nojekyll"), "w").close()
    print("site written:", SITE, len(samples), "samples")


TEMPLATE = r"""<title>fpvec Sample Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: drafting sheet. A title block, the score table, then a 2-column
   sheet of plan cards; each card flips clean/distorted and input/traced. */
:root{
  --paper:#f5f6f3; --sheet:#ffffff; --ink:#1b2230; --muted:#5d6676; --rule:#d8dcd5;
  --wall:#c4262e; --door:#b07800; --window:#1f6fd1; --ok:#1d7a4a; --warn:#a65f00; --bad:#b3261e;
  --chip:#eef0eb;
  --display:"Archivo","Arial Narrow",system-ui,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#14171c; --sheet:#1b1f26; --ink:#e7e9ee; --muted:#9aa3b2; --rule:#2c323c;
    --wall:#ff6b6b; --door:#f2b632; --window:#6aa8ff; --ok:#4fc58a; --warn:#f2a54a; --bad:#ff7a70; --chip:#242a33;
    color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --paper:#14171c; --sheet:#1b1f26; --ink:#e7e9ee; --muted:#9aa3b2; --rule:#2c323c;
  --wall:#ff6b6b; --door:#f2b632; --window:#6aa8ff; --ok:#4fc58a; --warn:#f2a54a; --bad:#ff7a70; --chip:#242a33;
  color-scheme:dark;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font:15px/1.55 var(--body);margin:0}
.wrap{max-width:1180px;margin:0 auto;padding-block:28px 56px;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
  display:grid;gap:28px}
.wrap>*,header.title>*,.card,.grid>*{min-width:0}
header.title{display:grid;grid-template-columns:1fr auto;gap:18px 28px;align-items:end;border-bottom:2px solid var(--ink);padding-bottom:18px}
.eyebrow{font:500 12px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
h1{font:800 clamp(30px,4.6vw,52px)/1.02 var(--display);letter-spacing:-.01em;margin:8px 0 10px;text-wrap:balance}
h1 em{font-style:normal;color:var(--wall)}
.lede{max-width:64ch;color:var(--muted);margin:0}
.actions{display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end}
.btn{display:inline-flex;align-items:center;gap:8px;padding:9px 14px;border:1.5px solid var(--ink);border-radius:4px;
  color:var(--ink);text-decoration:none;font:600 14px/1 var(--body);background:var(--sheet)}
.btn.primary{background:var(--ink);color:var(--paper)}
.btn:focus-visible,button:focus-visible{outline:2px solid var(--window);outline-offset:2px}
h2{font:700 20px/1.2 var(--display);margin:0 0 10px;text-wrap:balance}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;background:var(--sheet);border:1px solid var(--rule)}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font:500 11.5px/1.2 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);background:var(--chip)}
td.num{font-family:var(--mono)}
td b{font-family:var(--display);font-weight:700;font-size:15px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--muted);font-size:13px;margin-top:10px}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-1px;margin-right:6px}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.seg{display:inline-flex;border:1.5px solid var(--ink);border-radius:4px;overflow:hidden}
.seg button{font:500 13px/1 var(--body);padding:8px 12px;border:0;background:var(--sheet);color:var(--ink);cursor:pointer}
.seg button+button{border-left:1.5px solid var(--ink)}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--paper)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,520px),1fr));gap:18px}
.card{background:var(--sheet);border:1px solid var(--rule);display:grid;grid-template-rows:auto auto 1fr}
.card .hd{display:flex;gap:10px;align-items:baseline;justify-content:space-between;padding:12px 14px;border-bottom:1px solid var(--rule)}
.card .hd h3{font:700 16px/1.2 var(--display);margin:0}
.tag{font:500 11.5px/1 var(--mono);color:var(--muted);letter-spacing:.04em}
.fig{position:relative;background:var(--chip);aspect-ratio:4/3;max-width:100%;overflow:hidden}
.fig img{width:100%;height:100%;object-fit:contain;display:block}
.fig .ctl{position:absolute;left:10px;top:10px;display:flex;gap:6px;flex-wrap:wrap}
.fig .ctl .seg button{padding:6px 9px;font-size:12px}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;border-top:1px solid var(--rule)}
.stat{padding:10px 12px;min-width:0}
.stat+.stat{border-left:1px solid var(--rule)}
.stat .k{font:500 10.5px/1.2 var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.stat .v{font:600 15px/1.3 var(--mono);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat .s{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.good{color:var(--ok)}.mid{color:var(--warn)}.poor{color:var(--bad)}
.notes{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:18px}
.notes p{margin:0;color:var(--muted);max-width:62ch}
.notes h3{font:700 15px/1.3 var(--display);margin:0 0 4px;color:var(--ink)}
code{font-family:var(--mono);font-size:.92em;background:var(--chip);padding:1px 5px;border-radius:3px}
footer{color:var(--muted);font-size:13px;border-top:1px solid var(--rule);padding-top:14px}
@media (max-width:640px){header.title{grid-template-columns:1fr}.actions{justify-content:flex-start}
  .stats{grid-template-columns:repeat(2,minmax(0,1fr))}.stat:nth-child(3){border-left:0}.stat:nth-child(n+3){border-top:1px solid var(--rule)}}
@media (prefers-reduced-motion:no-preference){.fig img{transition:opacity .15s}}
</style>
<!--BODY-->
<div class="wrap">
  <header class="title">
    <div>
      <div class="eyebrow">fpvec · raster floor plan → vector</div>
      <h1>15 plans, clean and <em>distorted</em></h1>
      <p class="lede">Coloured floor plans in five drawing styles, each also as a phone photo taken at an angle or a crooked scan.
      Every image went through the same call: correct the photo, read the scale from the dimensions or room areas, trace walls, doors and windows.
      Scores are against the generator's exact ground truth.</p>
    </div>
    <div class="actions">
      <a class="btn primary" href="https://codespaces.new/__REPO__?quickstart=1" target="_blank" rel="noopener">Try the live demo</a>
      <a class="btn" href="https://github.com/__REPO__" target="_blank" rel="noopener">Source on GitHub</a>
    </div>
  </header>

  <section>
    <h2>Scores</h2>
    <div class="scroll"><table id="sum"></table></div>
    <div class="legend">
      <span><span class="sw" style="background:var(--wall)"></span>walls (dark outline = exterior)</span>
      <span><span class="sw" style="background:var(--door)"></span>doors with swing</span>
      <span><span class="sw" style="background:var(--window)"></span>windows</span>
      <span>Walls: share of true wall length matched within 150 mm, after fitting the overall size. Doors / windows: centre within 350 mm.</span>
    </div>
  </section>

  <section>
    <div class="filters" style="justify-content:space-between;margin-bottom:12px">
      <h2 style="margin:0">Plans</h2>
      <div class="filters">
        <div class="seg" role="group" aria-label="Show all cards as">
          <button type="button" id="allClean" aria-pressed="false">All clean</button>
          <button type="button" id="allDist" aria-pressed="false">All distorted</button>
        </div>
        <div class="seg" role="group" aria-label="Image">
          <button type="button" id="allIn" aria-pressed="false">Inputs</button>
          <button type="button" id="allOut" aria-pressed="true">Traced</button>
        </div>
      </div>
    </div>
    <div class="grid" id="grid"></div>
  </section>

  <section class="notes">
    <div><h3>Scale, three ways</h3><p>Dimension numbers are read by OCR (or from a PDF's text) and measured against their dimension lines. Without dimensions, room-area labels such as <code>38.4 m2</code> are matched to the rooms the walls enclose. As a last resort the main wall is assumed to be 200 mm.</p></div>
    <div><h3>Photo correction</h3><p>The sheet's four corners are found and the perspective removed, light is evened out, and the drawing is rotated and sheared until its walls are exactly horizontal and vertical. If the dimensions show the x and y scales still differ, the image is stretched to square pixels.</p></div>
    <div><h3>Into the editor</h3><p>Every result also comes out as a <code>floor_plan.json</code> that the Floor Plan editor opens directly: walls with thickness, doors with hinge and swing, windows, and the traced image as an underlay.</p></div>
  </section>
  <footer>Generated by <code>samples/run_samples.py</code> and <code>scripts/build_site.py</code>. The plans are synthetic, so the scores can be exact; real plans vary more.</footer>
</div>
<script>
(function(){
var D=__DATA__;
function pct(x){return Math.round(x*100)+'%';}
function cls(x,a,b){return x>=a?'good':x>=b?'mid':'poor';}
var S=D.summary,t=document.getElementById('sum');
t.innerHTML='<thead><tr><th>Set</th><th>Plans</th><th>Scale found automatically</th><th>Median size error</th><th>Walls found</th><th>Wall precision</th><th>Doors found</th><th>Windows found</th><th>False doors / windows</th></tr></thead><tbody>'+
['clean','distorted'].map(function(k){var a=S[k];return '<tr><td><b>'+(k==='clean'?'Clean':'Distorted')+'</b></td><td class="num">'+a.n+'</td><td class="num">'+a.auto+' / '+a.n+'</td><td class="num">'+a.size.toFixed(2)+' %</td><td class="num '+cls(a.swr,.95,.85)+'">'+pct(a.swr)+'</td><td class="num">'+pct(a.swp)+'</td><td class="num '+cls(a.doors,.85,.7)+'">'+pct(a.doors)+'</td><td class="num '+cls(a.wins,.9,.75)+'">'+pct(a.wins)+'</td><td class="num">'+a.xd+' / '+a.xw+'</td></tr>';}).join('')+'</tbody>';
var STY={sales:'Sales plan',wood:'Wood floors',blue:'Blue walls',grey_partitions:'Grey partitions',brown_walls:'Brown walls'};
var METHOD={dimensions:'dimensions','room areas':'room areas',wall:'wall assumed',given:'given',none:'none'};
var grid=document.getElementById('grid'),cards=[];
D.samples.forEach(function(s){
  var st={v:'distorted',img:'out'};
  var el=document.createElement('article');el.className='card';
  el.innerHTML='<div class="hd"><h3></h3><span class="tag"></span></div>'+
    '<div class="fig"><img alt="" loading="lazy" decoding="async"><div class="ctl">'+
    '<div class="seg"><button type="button" data-v="clean">Clean</button><button type="button" data-v="distorted">Distorted</button></div>'+
    '<div class="seg"><button type="button" data-i="in">Input</button><button type="button" data-i="out">Traced</button></div></div></div>'+
    '<div class="stats"></div>';
  grid.appendChild(el);
  var num=s.name.slice(0,2);
  el.querySelector('h3').textContent=num+' · '+(STY[s.style]||s.style);
  el.querySelector('.tag').textContent=(s.dims?'dimensions in '+(s.units==='ftin'?'ft-in':'mm'):'room areas only');
  function render(){
    var r=s.v[st.v],img=el.querySelector('img');
    img.src='img/'+s.name+'_'+r.kind+'_'+st.img+'.jpg';
    img.alt=(STY[s.style]||s.style)+' plan, '+(st.v==='clean'?'clean image':r.kind)+', '+(st.img==='in'?'input':'traced result');
    el.querySelectorAll('[data-v]').forEach(function(b){b.setAttribute('aria-pressed',String(b.dataset.v===st.v));b.textContent=b.dataset.v==='clean'?'Clean':(s.v.distorted.kind==='photo'?'Photo':'Scan');});
    el.querySelectorAll('[data-i]').forEach(function(b){b.setAttribute('aria-pressed',String(b.dataset.i===st.img));});
    var fix=(r.fix||[]).join(', ');
    el.querySelector('.stats').innerHTML=
      '<div class="stat"><div class="k">Scale</div><div class="v '+cls(1-r.size/100,.99,.97)+'">'+r.size.toFixed(2)+' %</div><div class="s">'+METHOD[r.method]+(r.votes?' ('+r.votes+')':'')+'</div></div>'+
      '<div class="stat"><div class="k">Walls</div><div class="v '+cls(r.swr,.95,.85)+'">'+pct(r.swr)+'</div><div class="s">precision '+pct(r.swp)+'</div></div>'+
      '<div class="stat"><div class="k">Doors</div><div class="v '+cls(r.df/Math.max(1,r.dt),.85,.6)+'">'+r.df+' / '+r.dt+'</div><div class="s">'+r.dx+' false</div></div>'+
      '<div class="stat"><div class="k">Windows</div><div class="v '+cls(r.wf/Math.max(1,r.wt),.9,.7)+'">'+r.wf+' / '+r.wt+'</div><div class="s">'+(fix?'fixed: '+fix:r.wx+' false')+'</div></div>';
  }
  el.querySelectorAll('[data-v]').forEach(function(b){b.onclick=function(){st.v=b.dataset.v;render();};});
  el.querySelectorAll('[data-i]').forEach(function(b){b.onclick=function(){st.img=b.dataset.i;render();};});
  cards.push({st:st,render:render});render();
});
function setAll(k,v,ids){cards.forEach(function(c){c.st[k]=v;c.render();});ids.forEach(function(p){document.getElementById(p[0]).setAttribute('aria-pressed',String(p[1]));});}
document.getElementById('allClean').onclick=function(){setAll('v','clean',[['allClean',true],['allDist',false]]);};
document.getElementById('allDist').onclick=function(){setAll('v','distorted',[['allClean',false],['allDist',true]]);};
document.getElementById('allIn').onclick=function(){setAll('img','in',[['allIn',true],['allOut',false]]);};
document.getElementById('allOut').onclick=function(){setAll('img','out',[['allIn',false],['allOut',true]]);};
})();
</script>"""

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
