#!/usr/bin/env python3
"""Generate a self-contained interactive communication-fingerprint viewer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from plotly.offline.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "plots" / "communication_fingerprints_interactive.html"


def load_events() -> list[dict]:
    with (ROOT / "parsed" / "events.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "scenario": row["scenario"],
            "event_order": int(row["event_order"]),
            "layer": row["layer"],
            "phase": row["phase"],
            "operation": row["operation"],
            "payload_bytes": int(row["payload_bytes"]),
            "estimated_comm_time_us": (
                float(row["estimated_comm_time_us"])
                if row["estimated_comm_time_us"]
                else None
            ),
        }
        for row in rows
        if row["scenario"] in {"DP", "TP"}
    ]


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive communication fingerprints</title>
<script>__PLOTLY_JS__</script>
<style>
  :root { --ink:#17202a; --muted:#667085; --line:#d8dee8; --panel:#f7f9fc; }
  * { box-sizing:border-box; }
  body { margin:0; padding:24px; color:var(--ink); font-family:Inter,system-ui,sans-serif; background:#fff; }
  h1 { margin:0 0 4px; font-size:28px; }
  .subtitle { color:var(--muted); margin-bottom:18px; }
  .controls { display:grid; grid-template-columns:repeat(auto-fit,minmax(185px,1fr)); gap:12px; padding:14px; border:1px solid var(--line); border-radius:12px; background:var(--panel); }
  .control { display:flex; flex-direction:column; gap:6px; font-size:13px; font-weight:650; }
  .control small { color:var(--muted); font-weight:400; }
  select,input,button { font:inherit; }
  select,input[type=number] { width:100%; padding:7px; border:1px solid #b9c2d0; border-radius:7px; background:white; }
  input[type=range] { width:100%; }
  .checks { display:flex; gap:12px; flex-wrap:wrap; padding-top:7px; }
  .checks label { font-weight:600; }
  .buttons { display:flex; gap:8px; align-items:end; }
  button { cursor:pointer; border:0; border-radius:7px; padding:8px 13px; background:#295f98; color:white; }
  button.secondary { color:#24364b; background:#dfe8f3; }
  .cards { display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:10px; margin:14px 0 4px; }
  .card { border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
  .card .label { color:var(--muted); font-size:12px; }
  .card .value { font-size:21px; font-weight:750; margin-top:2px; }
  #plot { width:100%; height:650px; }
  #plot .scatterlayer .point { cursor:pointer; }
  .drill-hint { margin:10px 0 0; padding:9px 12px; border-radius:8px; background:#eef5fb; color:#295f98; font-size:13px; font-weight:650; }
  .note { border-left:4px solid #6a4c93; padding:8px 12px; color:#4a5568; background:#faf8fc; font-size:13px; }
  @media(max-width:700px){ body{padding:12px}.cards{grid-template-columns:1fr 1fr}#plot{height:560px} }
</style>
</head>
<body>
<h1>Communication fingerprints of DP, TP, and PP</h1>
<div class="subtitle">Explore message count, payload, phase ordering, and pipeline scheduling.</div>
<div class="drill-hint">Click any message marker to inspect how that operation is implemented on the network.</div>

<section class="controls">
  <div class="control">
    <span>Visible strategies</span>
    <div class="checks">
      <label><input id="showDP" type="checkbox" checked> DP</label>
      <label><input id="showTP" type="checkbox" checked> TP</label>
      <label><input id="showPP" type="checkbox" checked> PP</label>
    </div>
  </div>
  <label class="control">PP schedule
    <select id="schedule"><option value="gpipe">Forward then backward</option><option value="1f1b">1F1B</option></select>
    <small>1F1B uses dependency-aware stage ordering.</small>
  </label>
  <label class="control">PP microbatches: <span id="mbValue">4</span>
    <input id="microbatches" type="range" min="1" max="16" step="1" value="4">
    <small>Microbatch size remains 8 sequences.</small>
  </label>
  <label class="control">Marker scale: <span id="scaleValue">1.0×</span>
    <input id="markerScale" type="range" min="0.5" max="2" step="0.1" value="1">
    <small>Marker area represents payload.</small>
  </label>
  <label class="control">Bandwidth (GB/s)
    <input id="bandwidth" type="number" min="0.1" step="0.1" value="360">
    <small>Used only for the serialization estimate.</small>
  </label>
  <label class="control">Per-message latency (µs)
    <input id="latency" type="number" min="0" step="0.1" value="1">
    <small>Used only for the serialization estimate.</small>
  </label>
  <label class="control">Execution progress: <span id="progressValue">100%</span>
    <input id="progress" type="range" min="0" max="100" step="1" value="100">
    <small>Play the schematic event sequence.</small>
  </label>
  <div class="buttons"><button id="play">▶ Play</button><button id="reset" class="secondary">Reset</button></div>
</section>

<section class="cards">
  <div class="card"><div class="label">Visible messages</div><div class="value" id="messageCount">–</div></div>
  <div class="card"><div class="label">Logical payload</div><div class="value" id="totalPayload">–</div></div>
  <div class="card"><div class="label">Median message</div><div class="value" id="medianPayload">–</div></div>
  <div class="card"><div class="label">Serialized comm. sum</div><div class="value" id="serializedTime">–</div></div>
</section>

<div id="plot"></div>
<div class="note"><b>Interpretation:</b> PP=4 creates three communication boundaries, shown as S1↔S2, S2↔S3, and S3↔S4. Purple right-pointing markers are forward activations; pink left-pointing markers are backward gradients. In 1F1B, identical horizontal positions indicate concurrent schedule slots. Horizontal position is normalized schedule progress, not a SimAI timestamp. The bandwidth/latency result is a sum of isolated message serialization costs; it is not pipeline step time and does not model contention or overlap.</div>

<script>
const baseEvents = __EVENTS_JSON__;
const PP_PAYLOAD = 12582912;
const STAGES = 4;
const showDP=document.getElementById('showDP'), showTP=document.getElementById('showTP'), showPP=document.getElementById('showPP');
const schedule=document.getElementById('schedule'), microbatches=document.getElementById('microbatches');
const markerScale=document.getElementById('markerScale'), bandwidth=document.getElementById('bandwidth'), latency=document.getElementById('latency');
const progress=document.getElementById('progress'), play=document.getElementById('play'), reset=document.getElementById('reset');
const mbValue=document.getElementById('mbValue'), scaleValue=document.getElementById('scaleValue'), progressValue=document.getElementById('progressValue');
const messageCount=document.getElementById('messageCount'), totalPayload=document.getElementById('totalPayload');
const medianPayload=document.getElementById('medianPayload'), serializedTime=document.getElementById('serializedTime');
const colors = {
  forward:'#277DA1', backward_input_gradient:'#F8961E',
  optimizer_weight_gradient:'#D62828', forward_activation:'#6A4C93',
  backward_gradient:'#E76FAD'
};
const labels = {
  forward:'TP forward', backward_input_gradient:'TP backward',
  optimizer_weight_gradient:'DP gradient sync', forward_activation:'PP forward activation',
  backward_gradient:'PP backward gradient'
};
let timer = null;

function taskQueue(stage, microbatches, schedule) {
  const q = [];
  if (schedule === 'gpipe') {
    for (let m=0;m<microbatches;m++) q.push({kind:'F',m});
    for (let m=microbatches-1;m>=0;m--) q.push({kind:'B',m});
    return q;
  }
  const warmup = Math.min(STAGES-stage-1, microbatches);
  for (let m=0;m<warmup;m++) q.push({kind:'F',m});
  const remaining = microbatches-warmup;
  for (let i=0;i<remaining;i++) {
    q.push({kind:'F',m:warmup+i});
    q.push({kind:'B',m:i});
  }
  for (let m=remaining;m<microbatches;m++) q.push({kind:'B',m});
  return q;
}

function makePP(microbatches, schedule) {
  const queues = Array.from({length:STAGES},(_,s)=>taskQueue(s,microbatches,schedule));
  const ptr = Array(STAGES).fill(0), doneF = new Set(), doneB = new Set(), events=[];
  let slot=0;
  while (ptr.some((p,s)=>p<queues[s].length)) {
    const ready=[];
    for (let s=0;s<STAGES;s++) {
      if (ptr[s]>=queues[s].length) continue;
      const task=queues[s][ptr[s]], key=`${s}:${task.m}`;
      if (task.kind==='F') {
        if (s===0 || doneF.has(`${s-1}:${task.m}`)) ready.push({s,...task,key});
      } else if (doneF.has(key) && (s===STAGES-1 || doneB.has(`${s+1}:${task.m}`))) {
        ready.push({s,...task,key});
      }
    }
    if (!ready.length) throw new Error('Pipeline schedule deadlock');
    slot++;
    for (const task of ready) {
      ptr[task.s]++;
      if (task.kind==='F') {
        doneF.add(task.key);
        if (task.s<STAGES-1) events.push({scenario:'PP',phase:'forward_activation',slot,
          microbatch:task.m+1,src:task.s+1,dst:task.s+2,payload_bytes:PP_PAYLOAD,
          operation:'SEND',layer:`stage ${task.s+1} → ${task.s+2}`});
      } else {
        doneB.add(task.key);
        if (task.s>0) events.push({scenario:'PP',phase:'backward_gradient',slot,
          microbatch:task.m+1,src:task.s+1,dst:task.s,payload_bytes:PP_PAYLOAD,
          operation:'SEND',layer:`stage ${task.s+1} → ${task.s}`});
      }
    }
    if (slot>10000) throw new Error('Pipeline schedule exceeded safety bound');
  }
  const maxSlot=Math.max(...events.map(e=>e.slot));
  events.forEach((e,i)=>{ e.progress=maxSlot===1?1:(e.slot-1)/(maxSlot-1); e.index=i; });
  return events;
}

function allEvents() {
  const counters={DP:0,TP:0};
  const fixed=baseEvents.map(e=>{
    let progress;
    if (e.scenario==='DP') progress=1;
    else progress=e.event_order/47;
    return {...e,progress,index:counters[e.scenario]++};
  });
  return fixed.concat(makePP(+microbatches.value,schedule.value));
}

function durationUs(e) {
  const bw=Math.max(0.1,+bandwidth.value), lat=Math.max(0,+latency.value);
  return lat + e.payload_bytes/(bw*1000);
}

function selected(e) {
  return (e.scenario==='DP'&&showDP.checked)||(e.scenario==='TP'&&showTP.checked)||(e.scenario==='PP'&&showPP.checked);
}

function update() {
  mbValue.textContent=microbatches.value;
  scaleValue.textContent=(+markerScale.value).toFixed(1)+'×';
  progressValue.textContent=progress.value+'%';
  const full=allEvents().filter(selected), cutoff=+progress.value/100;
  const visible=full.filter(e=>e.progress<=cutoff+1e-9);
  const maxPayload=Math.max(...full.map(e=>e.payload_bytes),1), traces=[];
  const phases=['optimizer_weight_gradient','forward','backward_input_gradient','forward_activation','backward_gradient'];
  for (const phase of phases) {
    const rows=visible.filter(e=>e.phase===phase);
    if (!rows.length) continue;
    traces.push({
      type:'scatter',mode:'markers',name:labels[phase],
      x:rows.map(e=>e.progress),
      y:rows.map(e=>{
        if(e.scenario==='DP') return 3;
        if(e.scenario==='TP') return 2;
        const boundaryCenter=1+(Math.min(e.src,e.dst)-2)*0.12;
        return boundaryCenter+(e.phase==='forward_activation'?0.018:-0.018);
      }),
      marker:{
        color:colors[phase],symbol:eSymbol(phase),
        size:rows.map(e=>(10+28*Math.sqrt(e.payload_bytes/maxPayload))*(+markerScale.value)),
        line:{color:'white',width:1},sizemode:'diameter'
      },
      customdata:rows.map(e=>[
        e.scenario,e.layer,e.operation,e.payload_bytes/1e6,durationUs(e),
        e.microbatch||'–',e.src||'–',e.dst||'–',e.phase.replaceAll('_',' '),
        e.index+1,e.payload_bytes,e.phase
      ]),
      hovertemplate:'<b>%{customdata[0]}</b><br>phase: %{customdata[8]}<br>location: %{customdata[1]}<br>operation: %{customdata[2]}<br>payload: %{customdata[3]:.2f} MB<br>serialization estimate: %{customdata[4]:.2f} µs<br>microbatch: %{customdata[5]}<br>source/destination: %{customdata[6]} → %{customdata[7]}<br><b>Click to inspect this operation</b><extra></extra>'
    });
  }
  Plotly.react('plot',traces,{
    margin:{l:90,r:25,t:42,b:75},paper_bgcolor:'white',plot_bgcolor:'#fbfcfe',
    hovermode:'closest',legend:{orientation:'h',y:1.1,x:0},
    xaxis:{title:'Normalized schematic step progress (not simulated time)',range:[-0.025,1.025],tickvals:[0,.5,1],ticktext:['forward start','backward / steady state','step end'],gridcolor:'#e5e9f0'},
    yaxis:{range:[0.70,3.28],tickvals:[0.88,1,1.12,2,3],ticktext:['PP S1↔S2','PP S2↔S3','PP S3↔S4','TP=4','DP=4'],gridcolor:'#e5e9f0',fixedrange:true},
    shapes:[{type:'line',x0:.5,x1:.5,y0:.75,y1:3.25,line:{color:'#b8c0cc',dash:'dash'}}],
    annotations:[
      {x:1,y:3.18,text:'DP: 1 × 248.88 MB',showarrow:false,xanchor:'right'},
      {x:1,y:2.18,text:'TP: 48 × 50.33 MB',showarrow:false,xanchor:'right'},
      {x:1,y:1.22,text:`PP: ${2*(STAGES-1)*(+microbatches.value)} × 12.58 MB`,showarrow:false,xanchor:'right'}
    ]
  },{responsive:true,displaylogo:false});
  const payloads=full.map(e=>e.payload_bytes).sort((a,b)=>a-b);
  const median=payloads.length?(payloads.length%2?payloads[(payloads.length-1)/2]:(payloads[payloads.length/2-1]+payloads[payloads.length/2])/2):0;
  messageCount.textContent=full.length.toLocaleString();
  totalPayload.textContent=(full.reduce((s,e)=>s+e.payload_bytes,0)/1e6).toFixed(2)+' MB';
  medianPayload.textContent=(median/1e6).toFixed(2)+' MB';
  serializedTime.textContent=(full.reduce((s,e)=>s+durationUs(e),0)/1000).toFixed(3)+' ms';
}

function eSymbol(phase){
  if(phase==='forward_activation') return 'triangle-right';
  if(phase==='backward_gradient') return 'triangle-left';
  return 'circle';
}
function stop(){ if(timer){clearInterval(timer);timer=null;} play.textContent='▶ Play'; }
function saveOverviewState(){
  const state={schedule:schedule.value,microbatches:microbatches.value,markerScale:markerScale.value,
    bandwidth:bandwidth.value,latency:latency.value,progress:progress.value,
    showDP:showDP.checked,showTP:showTP.checked,showPP:showPP.checked};
  try{sessionStorage.setItem('experimentAOverviewState',JSON.stringify(state));}catch(_error){}
}
function restoreOverviewState(){
  try{
    const state=JSON.parse(sessionStorage.getItem('experimentAOverviewState')||'null');
    if(!state)return;
    for(const id of ['schedule','microbatches','markerScale','bandwidth','latency','progress'])if(state[id]!==undefined)document.getElementById(id).value=state[id];
    for(const id of ['showDP','showTP','showPP'])if(state[id]!==undefined)document.getElementById(id).checked=state[id];
  }catch(_error){}
}

for (const id of ['showDP','showTP','showPP','schedule','microbatches','markerScale','bandwidth','latency','progress']) {
  document.getElementById(id).addEventListener('input',update);
}
play.addEventListener('click',()=>{
  if(timer){stop();return;}
  if(+progress.value>=100) progress.value=0;
  play.textContent='❚❚ Pause';
  timer=setInterval(()=>{ progress.value=Math.min(100,+progress.value+2); update(); if(+progress.value>=100)stop(); },80);
});
reset.addEventListener('click',()=>{stop();progress.value=100;microbatches.value=4;schedule.value='gpipe';markerScale.value=1;bandwidth.value=360;latency.value=1;showDP.checked=showTP.checked=showPP.checked=true;update();});
restoreOverviewState();
update();
document.getElementById('plot').on('plotly_click',event=>{
  const cd=event.points[0].customdata;
  saveOverviewState();
  const target=new URL('collective_vs_p2p.html',window.location.href);
  target.searchParams.set('strategy',cd[0]);
  target.searchParams.set('message',cd[9]);
  target.searchParams.set('layer',cd[1]);
  target.searchParams.set('phase',cd[11]);
  target.searchParams.set('payload',cd[10]);
  if(cd[5]!=='–')target.searchParams.set('microbatch',cd[5]);
  if(cd[6]!=='–')target.searchParams.set('source',cd[6]);
  if(cd[7]!=='–')target.searchParams.set('destination',cd[7]);
  target.searchParams.set('schedule',schedule.value);
  window.location.href=target.href;
});
</script>
</body>
</html>
"""


def main() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    html = HTML.replace("__PLOTLY_JS__", get_plotlyjs())
    html = html.replace("__EVENTS_JSON__", json.dumps(load_events(), separators=(",", ":")))
    OUTPUT.write_text(html)
    print(OUTPUT)


if __name__ == "__main__":
    main()
