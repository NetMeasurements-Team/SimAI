#!/usr/bin/env python3
"""Generate a standalone teaching view that expands collectives into transfers."""

from __future__ import annotations

import json
from pathlib import Path

from plotly.offline.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "plots" / "collective_vs_p2p.html"

SCENARIOS = {
    "DP": {
        "kind": "allreduce",
        "payload": 248_879_616,
        "operations": 1,
        "title": "DP gradient AllReduce",
    },
    "TP": {
        "kind": "allreduce",
        "payload": 50_331_648,
        "operations": 48,
        "title": "TP activation/gradient AllReduce",
    },
    "PP": {
        "kind": "p2p",
        "payload": 12_582_912,
        "operations": 24,
        "title": "PP stage-to-stage transfer",
    },
}

MODEL_PROFILES = {
    "gpt2": {
        "label": "GPT-2-like 124M",
        "parameters": 124_439_808,
        "hidden": 768,
        "layers": 12,
        "sequence": 1024,
        "global_batch": 32,
        "microbatch": 8,
        "bytes_per_value": 2,
    },
    "oneb": {
        "label": "Llama-like theoretical 1B",
        "parameters": 1_000_000_000,
        "hidden": 2048,
        "layers": 16,
        "sequence": 1024,
        "global_batch": 32,
        "microbatch": 8,
        "bytes_per_value": 2,
    },
}


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inside DP, TP, and PP communication</title>
<script>__PLOTLY_JS__</script>
<style>
:root{--ink:#17202a;--muted:#667085;--line:#d9e0e9;--blue:#277da1;--red:#d62828;--purple:#6a4c93;--panel:#f7f9fc}
*{box-sizing:border-box} body{margin:0;padding:24px;font-family:Inter,system-ui,sans-serif;color:var(--ink);background:white}
h1{margin:0 0 4px;font-size:29px}.subtitle{color:var(--muted);margin-bottom:18px}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.tab,button{border:0;border-radius:8px;padding:9px 15px;font:inherit;font-weight:700;cursor:pointer}.tab{background:#e8edf4;color:#344054}.tab.active{background:#295f98;color:white}
.main{display:grid;grid-template-columns:minmax(480px,1.3fr) minmax(310px,.7fr);gap:16px}.panel{border:1px solid var(--line);border-radius:12px;background:white;padding:16px}.panel.soft{background:var(--panel)}
.panel h2{font-size:20px;margin:0 0 4px}.description{color:var(--muted);margin-bottom:8px}.diagram-wrap{min-height:390px;display:flex;align-items:center;justify-content:center}.diagram-wrap svg{width:100%;max-width:620px;height:370px}
.controls{display:flex;align-items:center;justify-content:center;gap:8px}.controls button{background:#295f98;color:white}.controls button.secondary{background:#dfe8f3;color:#24364b}.step{min-width:230px;text-align:center;font-weight:750}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:12px}.card{border:1px solid var(--line);border-radius:9px;padding:10px}.card .label{font-size:12px;color:var(--muted)}.card .value{font-size:20px;font-weight:780;margin-top:2px}
.formula{font-family:ui-monospace,SFMono-Regular,monospace;background:#fff;border:1px solid var(--line);padding:11px;border-radius:8px;margin-top:12px;line-height:1.55}.explain{margin-top:12px;font-size:14px;line-height:1.55}
.plots{margin-top:16px}.plot{height:460px;border:1px solid var(--line);border-radius:12px}.takeaway{margin-top:14px;border-left:5px solid var(--purple);background:#faf8fc;padding:12px 14px;line-height:1.55}
.assumption{margin-top:12px;color:var(--muted);font-size:12px;line-height:1.45}.node{stroke:#fff;stroke-width:3}.edge{stroke:#b7c0cd;stroke-width:4;fill:none}.flow{stroke:#ed8b23;stroke-width:7;fill:none;stroke-dasharray:12 9;animation:dash .65s linear infinite}.backflow{stroke:#9c6ade}.p2pflow{stroke:#6a4c93;stroke-width:9;stroke-dasharray:14 10;animation:dash .65s linear infinite}.node-label{font-weight:800;font-size:16px;fill:white;text-anchor:middle;dominant-baseline:middle}.edge-label{font-size:13px;fill:#475467;text-anchor:middle}.phase-label{font-size:19px;font-weight:800;fill:#25364a;text-anchor:middle}.small-label{font-size:13px;fill:#667085;text-anchor:middle}@keyframes dash{to{stroke-dashoffset:-42}}
.explore{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:10px;margin:0 0 8px;padding:12px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}.explore label{display:flex;flex-direction:column;gap:5px;font-size:13px;font-weight:700}.explore select,.explore input{width:100%}.model-note{margin:0 0 14px;padding:8px 11px;background:#eef5fb;color:#295f98;border-radius:8px;font-size:13px}.ledger{margin-top:14px;border-top:1px solid var(--line);padding-top:10px}.ledger h3{margin:0 0 8px;font-size:15px}.ledger table{width:100%;border-collapse:collapse;font-size:13px}.ledger th,.ledger td{text-align:left;padding:6px 8px;border-bottom:1px solid #e8ecf2}.ledger th{color:var(--muted);font-size:11px;text-transform:uppercase}.step-slider{width:min(520px,100%);display:block;margin:9px auto 0}
.topnav{display:flex;align-items:center;gap:12px;margin-bottom:14px}.backlink{display:inline-block;text-decoration:none;border-radius:8px;padding:8px 12px;background:#e4edf7;color:#254f7a;font-weight:750}.context{display:none;flex:1;border-left:4px solid var(--purple);background:#faf8fc;padding:9px 12px;border-radius:0 8px 8px 0;color:#4b3767}.context.visible{display:block}.context b{color:#34234d}
.compute-estimate{display:grid;grid-template-columns:minmax(220px,1.35fr) 1fr 1fr;gap:10px;align-items:stretch;margin:0 0 14px}.compute-intro,.compute-metric{padding:11px 13px;border:1px solid var(--line);border-radius:9px;background:white}.compute-intro b,.compute-metric b{display:block}.compute-intro span,.compute-metric span{display:block;color:var(--muted);font-size:12px;margin-top:3px}.compute-metric strong{display:block;font-size:23px;margin-top:3px}.compute-metric small{display:block;color:var(--muted);margin-top:2px}
@media(max-width:900px){body{padding:12px}.main,.plots{grid-template-columns:1fr}.explore{grid-template-columns:1fr 1fr}.compute-estimate{grid-template-columns:1fr}.diagram-wrap{min-height:340px}.cards{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="topnav"><a id="backLink" class="backlink" href="communication_fingerprints_interactive.html">← Back to fingerprints</a><div id="selectionContext" class="context"></div></div>
<h1>What happens inside each communication operation?</h1>
<div class="subtitle">A logical AllReduce expands into many point-to-point chunk transfers; a PP message remains one adjacent-stage transfer.</div>
<div class="tabs"><button class="tab active" data-s="DP">DP AllReduce</button><button class="tab" data-s="TP">TP AllReduce</button><button class="tab" data-s="PP">PP point-to-point</button></div>
<div class="explore">
  <label>Model profile<select id="modelProfile"><option value="gpt2">GPT-2-like 124M</option><option value="oneb">Llama-like theoretical 1B</option></select></label>
  <label>AllReduce participants: <span id="worldValue">4 GPUs</span><input id="worldSize" type="range" min="2" max="8" step="1" value="4"></label>
  <label>Animation speed: <span id="speedValue">1.0×</span><input id="speed" type="range" min="0.25" max="2" step="0.25" value="1"></label>
  <label>PP boundary<select id="ppBoundary"><option value="1">Stage 1 ↔ Stage 2</option><option value="2" selected>Stage 2 ↔ Stage 3</option><option value="3">Stage 3 ↔ Stage 4</option></select></label>
  <label>PP direction<select id="ppDirection"><option value="forward">Forward activation</option><option value="backward">Backward gradient</option></select></label>
</div>
<div id="modelNote" class="model-note"></div>
<section class="compute-estimate" aria-label="Ideal H100 compute estimate">
  <div class="compute-intro"><b>Ideal single-H100 compute estimate</b><span id="computeAssumptions"></span></div>
  <div class="compute-metric"><b>Forward, one transformer layer</b><strong id="layerForwardTime"></strong><small id="layerForwardFlops"></small></div>
  <div class="compute-metric"><b>Complete training step</b><strong id="trainingStepTime"></strong><small id="trainingStepFlops"></small></div>
</section>

<section class="main">
  <div class="panel">
    <h2 id="opTitle"></h2>
    <div class="diagram-wrap" id="diagram"></div>
    <div class="controls"><button id="previous" class="secondary">← Previous</button><button id="play">▶ Play</button><div class="step" id="stepLabel"></div><button id="next" class="secondary">Next →</button></div>
    <input id="stepSlider" class="step-slider" type="range" min="0" max="5" step="1" value="0">
    <div class="ledger"><h3>Transfers in the current algorithmic step</h3><div id="ledger"></div></div>
  </div>
  <aside class="panel soft">
    <h2>Operation accounting</h2>
    <div class="cards">
      <div class="card"><div class="label">Logical payload</div><div class="value" id="payload"></div></div>
      <div class="card"><div class="label">Participants</div><div class="value" id="participants"></div></div>
      <div class="card"><div class="label">Algorithmic steps</div><div class="value" id="steps"></div></div>
      <div class="card"><div class="label">Chunk transmissions</div><div class="value" id="transfers"></div></div>
      <div class="card"><div class="label">Traffic per rank</div><div class="value" id="perRank"></div></div>
      <div class="card"><div class="label">Aggregate link traffic</div><div class="value" id="aggregate"></div></div>
    </div>
    <div class="formula" id="formula"></div>
    <div class="explain" id="explanation"></div>
  </aside>
</section>

<section class="plots"><div id="trafficPlot" class="plot"></div></section>
<script>
const scenarios=__SCENARIOS__, profiles=__PROFILES__, params=new URLSearchParams(window.location.search);
const H100_BF16_DENSE_FLOPS=989.4e12;
let N=4;
let selected=['DP','TP','PP'].includes(params.get('strategy'))?params.get('strategy'):'DP', step=0, timer=null;
const el=id=>document.getElementById(id);
const fmtMB=b=>(b/1e6).toFixed(2)+' MB';
const fmtGB=b=>(b/1e9).toFixed(3)+' GB';

function accounting(s){
 const d=scenarios[s];
 if(d.kind==='p2p') return {participants:2,steps:1,transfers:1,perRank:d.payload,aggregate:d.payload,chunk:d.payload};
 const ringSteps=2*(N-1),ringTransfers=ringSteps*N;
 return {participants:N,steps:ringSteps,transfers:ringTransfers,
   perRank:2*(N-1)/N*d.payload,aggregate:2*(N-1)*d.payload,chunk:d.payload/N};
}

function applyModelProfile(){
 const key=el('modelProfile').value,p=profiles[key];
 scenarios.DP.payload=p.parameters*p.bytes_per_value;
 scenarios.DP.operations=1;
 scenarios.TP.payload=p.global_batch*p.sequence*p.hidden*p.bytes_per_value;
 scenarios.TP.operations=p.layers*4;
 scenarios.PP.payload=p.microbatch*p.sequence*p.hidden*p.bytes_per_value;
 scenarios.PP.operations=2*(4-1)*(p.global_batch/p.microbatch);
 if(key==='gpt2'){
  el('modelNote').textContent=`Baseline: ${p.label} · ${p.layers} layers · hidden ${p.hidden} · DP ${fmtMB(scenarios.DP.payload)} · TP ${fmtMB(scenarios.TP.payload)} · PP ${fmtMB(scenarios.PP.payload)}`;
 }else{
  const base=profiles.gpt2;
  const dpRatio=p.parameters/base.parameters;
  const tensorRatio=p.hidden/base.hidden;
  const tpRatio=tensorRatio*(p.layers/base.layers);
  el('modelNote').textContent=`Compared with GPT-2-like 124M: DP iteration ${dpRatio.toFixed(2)}× · TP iteration ${tpRatio.toFixed(2)}× · PP iteration ${tensorRatio.toFixed(2)}×. Current payloads: DP ${fmtMB(scenarios.DP.payload)}, TP ${fmtMB(scenarios.TP.payload)}, PP ${fmtMB(scenarios.PP.payload)}.`;
 }
  updateComputeEstimate(p);
}

function formatFlops(value){return value>=1e12?(value/1e12).toFixed(2)+' TFLOP':(value/1e9).toFixed(2)+' GFLOP'}
function formatComputeTime(seconds){return seconds<0.001?(seconds*1e6).toFixed(1)+' µs':(seconds*1e3).toFixed(2)+' ms'}
function updateComputeEstimate(p){
 const tokens=p.global_batch*p.sequence;
 const layerForwardFlops=24*tokens*p.hidden*p.hidden+4*p.global_batch*p.sequence*p.sequence*p.hidden;
 const trainingStepFlops=6*p.parameters*tokens;
 el('layerForwardTime').textContent=formatComputeTime(layerForwardFlops/H100_BF16_DENSE_FLOPS);
 el('trainingStepTime').textContent=formatComputeTime(trainingStepFlops/H100_BF16_DENSE_FLOPS);
 el('layerForwardFlops').textContent=formatFlops(layerForwardFlops)+' compute';
 el('trainingStepFlops').textContent=formatFlops(trainingStepFlops)+' compute';
 el('computeAssumptions').textContent=`H100 SXM BF16 dense peak 989.4 TFLOP/s · batch ${p.global_batch} · sequence ${p.sequence} · 100% utilization`;
}

function arrowMarker(color){return `<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="${color}"/></marker></defs>`}
function ringSVG(d,a){
 const pts=Array.from({length:N},(_,i)=>{const angle=-Math.PI/2+2*Math.PI*i/N;return [300+205*Math.cos(angle),185+137*Math.sin(angle)]});
 const phase=step<N-1?'Reduce-scatter':'All-gather', local=step%(N-1)+1;
 const color=phase==='Reduce-scatter'?'#ed8b23':'#9c6ade';
 let svg=`<svg viewBox="0 0 600 370" role="img">${arrowMarker(color)}<text x="300" y="142" class="phase-label">${phase} · step ${local}/${N-1}</text>`;
 for(let i=0;i<N;i++){
   const [x1,y1]=pts[i],[x2,y2]=pts[(i+1)%N];
   svg+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="edge"/>`;
 }
 for(let i=0;i<N;i++){
   const [x1,y1]=pts[i],[x2,y2]=pts[(i+1)%N];
   const dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy),pad=35;
   const sx=x1+dx/len*pad,sy=y1+dy/len*pad,ex=x2-dx/len*pad,ey=y2-dy/len*pad;
   svg+=`<line x1="${sx}" y1="${sy}" x2="${ex}" y2="${ey}" class="flow ${phase==='All-gather'?'backflow':''}" marker-end="url(#arrow)"/>`;
 }
 pts.forEach(([x,y],i)=>{svg+=`<circle cx="${x}" cy="${y}" r="31" fill="${selected==='DP'?'#d62828':'#277da1'}" class="node"/><text x="${x}" y="${y}" class="node-label">GPU ${i}</text>`});
 svg+=`<text x="300" y="176" class="small-label">${N} simultaneous sends per step</text><text x="300" y="198" class="small-label">each chunk = ${fmtMB(a.chunk)}</text></svg>`;
 return svg;
}
function p2pSVG(d){
 const boundary=+el('ppBoundary').value,forward=el('ppDirection').value==='forward';
 const src=forward?boundary:boundary+1,dst=forward?boundary+1:boundary;
 const phase=forward?'Forward activation':'Backward gradient';
 return `<svg viewBox="0 0 600 370" role="img">${arrowMarker('#6a4c93')}
 <text x="300" y="42" class="phase-label">${phase}: one adjacent-stage transfer</text>
 <rect x="70" y="125" width="155" height="110" rx="18" fill="#6a4c93"/><text x="147" y="171" class="node-label">Stage ${src}</text><text x="147" y="197" class="node-label">GPU ${src-1}</text>
 <rect x="375" y="125" width="155" height="110" rx="18" fill="#9c6ade"/><text x="452" y="171" class="node-label">Stage ${dst}</text><text x="452" y="197" class="node-label">GPU ${dst-1}</text>
 <line x1="230" y1="180" x2="365" y2="180" class="p2pflow" marker-end="url(#arrow)"/>
 <text x="300" y="145" class="edge-label">activation or gradient</text><text x="300" y="225" class="edge-label">${fmtMB(d.payload)} sent once</text></svg>`;
}

function chunkFor(rank,phase,local){
 if(phase==='Reduce-scatter') return (rank-(local-1)+N)%N;
 return (rank-(N-1)-(local-1)+2*N)%N;
}
function renderLedger(d,a){
 if(d.kind==='p2p'){
  const boundary=+el('ppBoundary').value,forward=el('ppDirection').value==='forward';
  const src=forward?boundary:boundary+1,dst=forward?boundary+1:boundary;
  el('ledger').innerHTML=`<table><thead><tr><th>Sender</th><th>Receiver</th><th>Tensor</th><th>Bytes</th></tr></thead><tbody><tr><td>Stage ${src}</td><td>Stage ${dst}</td><td>${forward?'activation':'gradient'}</td><td>${fmtMB(d.payload)}</td></tr></tbody></table>`;
  return;
 }
 const phase=step<N-1?'Reduce-scatter':'All-gather',local=step%(N-1)+1;
 const rows=Array.from({length:N},(_,rank)=>`<tr data-rank="${rank}"><td>GPU ${rank}</td><td>GPU ${(rank+1)%N}</td><td>chunk ${chunkFor(rank,phase,local)}</td><td>${fmtMB(a.chunk)}</td></tr>`).join('');
 el('ledger').innerHTML=`<table><thead><tr><th>Sender</th><th>Receiver</th><th>Chunk</th><th>Bytes</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function render(){
 N=+el('worldSize').value;el('worldValue').textContent=N+' GPUs';el('speedValue').textContent=(+el('speed').value).toFixed(2).replace(/0$/,'')+'×';
 const d=scenarios[selected],a=accounting(selected),maxStep=d.kind==='p2p'?0:2*(N-1)-1;
 step=Math.max(0,Math.min(step,maxStep)); el('opTitle').textContent=d.title;
 el('diagram').innerHTML=d.kind==='p2p'?p2pSVG(d):ringSVG(d,a);
 el('stepSlider').max=maxStep;el('stepSlider').value=step;el('stepSlider').disabled=d.kind==='p2p';
 renderLedger(d,a);
 el('payload').textContent=fmtMB(d.payload);el('participants').textContent=a.participants;el('steps').textContent=a.steps;
 el('transfers').textContent=a.transfers;el('perRank').textContent=fmtMB(a.perRank);el('aggregate').textContent=fmtMB(a.aggregate);
 if(d.kind==='p2p'){
  el('stepLabel').textContent='Direct transfer';
  el('formula').innerHTML=`1 message × ${fmtMB(d.payload)}<br>= ${fmtMB(a.aggregate)} aggregate traffic`;
  el('explanation').innerHTML=`The sender transmits the tensor to one adjacent stage. There is no collective decomposition and no reduce-scatter/all-gather phase.`;
 }else{
  el('stepLabel').textContent=`Step ${step+1}/${a.steps} · ${step<N-1?'reduce-scatter':'all-gather'}`;
  el('formula').innerHTML=`2 × (N − 1) × payload<br>= 2 × ${N-1} × ${fmtMB(d.payload)}<br>= ${fmtGB(a.aggregate)} aggregate traffic`;
  el('explanation').innerHTML=`The logical AllReduce is split into ${N} chunks of ${fmtMB(a.chunk)}. Every algorithmic step performs ${N} simultaneous neighbor sends. ${a.steps} steps produce ${a.transfers} point-to-point chunk transmissions.`;
 }
 document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.s===selected));
}

function applyDrillDownContext(){
 const strategy=params.get('strategy');
 if(!['DP','TP','PP'].includes(strategy))return;
 const source=params.get('source'),destination=params.get('destination'),phase=params.get('phase');
 if(strategy==='PP'&&source&&destination){
  el('ppBoundary').value=String(Math.min(+source,+destination));
  el('ppDirection').value=phase&&phase.includes('backward')?'backward':'forward';
 }
 const parts=[`${strategy} message ${params.get('message')||''}`.trim()];
 if(params.get('layer'))parts.push(params.get('layer').replaceAll('_',' '));
 if(phase)parts.push(phase.replaceAll('_',' '));
 if(params.get('microbatch'))parts.push(`microbatch ${params.get('microbatch')}`);
 if(source&&destination)parts.push(`Stage ${source} → Stage ${destination}`);
 if(params.get('payload'))parts.push(`${(+params.get('payload')/1e6).toFixed(2)} MB`);
 el('selectionContext').innerHTML=`<b>Inspecting:</b> ${parts.join(' · ')}`;
 el('selectionContext').classList.add('visible');
}

function drawPlots(){
 const names=['DP','TP','PP'],logicalPayload=[],oneOperation=[],wholeIteration=[];
 for(const s of names){
  const d=scenarios[s],a=accounting(s);
  logicalPayload.push(d.payload/1e6);
  oneOperation.push(a.aggregate/1e6);
  wholeIteration.push(a.aggregate*d.operations/1e6);
 }
 const label=value=>value>=1000?(value/1000).toFixed(3)+' GB':value.toFixed(2)+' MB';
 Plotly.newPlot('trafficPlot',[
  {type:'bar',name:'Logical payload',x:names,y:logicalPayload,marker:{color:'#5d6b7a'},text:logicalPayload.map(label),textposition:'outside',hovertemplate:'%{x}<br>logical workload payload: %{text}<extra></extra>'},
  {type:'bar',name:'One complete operation, all GPUs',x:names,y:oneOperation,marker:{color:'#ed8b23'},text:oneOperation.map(label),textposition:'outside',hovertemplate:'%{x}<br>one complete operation: %{text}<extra></extra>'},
  {type:'bar',name:'Whole iteration',x:names,y:wholeIteration,marker:{color:['#d62828','#277da1','#6a4c93']},text:wholeIteration.map(label),textposition:'outside',hovertemplate:'%{x}<br>whole iteration: %{text}<extra></extra>'}
 ],{title:`${profiles[el('modelProfile').value].label}: logical payload → one operation → whole iteration`,barmode:'group',
   yaxis:{title:'Data transmitted (MB, decimal; logarithmic scale)',type:'log'},margin:{l:85,r:25,t:80,b:60},legend:{orientation:'h',y:1.16,x:0}},
   {responsive:true,displaylogo:false});
}
function stop(){if(timer){clearInterval(timer);timer=null}el('play').textContent='▶ Play'}
function restartTimer(){if(!timer)return;clearInterval(timer);timer=setInterval(()=>{step=(step+1)%(2*(N-1));render()},900/(+el('speed').value))}
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{stop();selected=b.dataset.s;step=0;render()}));
el('previous').addEventListener('click',()=>{stop();step--;render()});el('next').addEventListener('click',()=>{stop();step++;render()});
el('stepSlider').addEventListener('input',()=>{stop();step=+el('stepSlider').value;render()});
el('worldSize').addEventListener('input',()=>{stop();step=0;render();drawPlots()});
el('modelProfile').addEventListener('input',()=>{stop();step=0;applyModelProfile();render();drawPlots()});
el('speed').addEventListener('input',()=>{render();restartTimer()});
for(const id of ['ppBoundary','ppDirection'])el(id).addEventListener('input',render);
el('play').addEventListener('click',()=>{if(scenarios[selected].kind==='p2p')return;if(timer){stop();return}el('play').textContent='❚❚ Pause';timer=setInterval(()=>{step=(step+1)%(2*(N-1));render()},900/(+el('speed').value))});
el('backLink').addEventListener('click',event=>{
 if(document.referrer&&document.referrer.includes('communication_fingerprints_interactive.html')){event.preventDefault();history.back();}
});
if(params.get('model')&&profiles[params.get('model')])el('modelProfile').value=params.get('model');
applyModelProfile();applyDrillDownContext();render();drawPlots();
</script>
</body></html>"""


def main() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    html = HTML.replace("__PLOTLY_JS__", get_plotlyjs())
    html = html.replace("__SCENARIOS__", json.dumps(SCENARIOS, separators=(",", ":")))
    html = html.replace("__PROFILES__", json.dumps(MODEL_PROFILES, separators=(",", ":")))
    OUTPUT.write_text(html)
    print(OUTPUT)


if __name__ == "__main__":
    main()
