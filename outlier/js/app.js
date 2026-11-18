/* Outlier game logic. Depends on data.js and recognizer.js. */
(() => {
"use strict";
const { POOL, ROUNDS, LIMIT, UNLISTED, RARE_SHARE, UNCLEAR } = window.OUTLIER_DATA;
const R = window.Recognizer;

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const norm = s => String(s||"").toLowerCase().trim().replace(/^(a|an|the)\s+/,"").replace(/[^a-z0-9 \-']/g,"").replace(/\s+/g," ").slice(0,40);
const day = (d => d.getFullYear()+"-"+(d.getMonth()+1)+"-"+d.getDate())(new Date());
const store = { get(k){try{return JSON.parse(localStorage.getItem(k))}catch{return null}}, set(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch{}} };

/* ---------- rarity: surprisal from the precomputed tables ---------- */
const MAXBITS = -Math.log2(UNLISTED);
const rarity = p => Math.max(1, Math.min(100, Math.round(100 * -Math.log2(p) / MAXBITS)));
function shareOf(pi, label){
  const p = POOL[pi];
  if (label in p.common) return {share:p.common[label], kind:"common"};
  if (p.rare.includes(label)) return {share:RARE_SHARE, kind:"rare"};
  return {share:UNLISTED, kind:"unlisted"};
}

/* ---------- daily prompt set ---------- */
function pick(seed, n){ let h=7; for(const c of seed){h=(h*31+c.charCodeAt(0))>>>0}
  const r=()=>{h^=h<<13;h>>>=0;h^=h>>>17;h^=h<<5;h>>>=0;return h/4294967296};
  const idx=POOL.map((_,i)=>i); for(let i=idx.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]]} return idx.slice(0,n); }

let runs = store.get("outlier-runs-"+day) || 0;
let set = pick(day+"#"+runs, ROUNDS);
let st = {i:0, res:[]};
let cur = null, locked = false, tEnd = 0, tRaf = 0;
let modelOK = null; // null = loading, true = ready, false = failed

/* ---------- recognizer loading ---------- */
R.load((stage, p) => {
  $("loadTxt").textContent = stage === "Ready" ? "Recognizer ready" : `${stage}… ${Math.round(p*100)}%`;
  $("loadBar").style.transform = `scaleX(${p})`;
}).then(() => {
  modelOK = true; $("load").classList.add("done");
  setTimeout(() => { $("load").hidden = true; }, 1200);
}).catch(e => {
  console.error(e); modelOK = false;
  $("loadTxt").textContent = "Couldn't load the recognizer. You'll type your answers instead.";
  $("loadBar").style.transform = "scaleX(0)";
  if (!$("round").hidden && !cur) $("manual").hidden = false;
});

/* ---------- timer ---------- */
function startTimer(ms = LIMIT*1000){
  stopTimer(); tEnd = performance.now() + ms;
  const tick = () => {
    const left = Math.max(0, tEnd - performance.now());
    $("bar").style.transform = `scaleX(${left/(LIMIT*1000)})`;
    const t = Math.ceil(left/1000); $("secs").textContent = Math.floor(t/60) + ":" + String(t%60).padStart(2,"0");
    $("timer").classList.toggle("low", left <= 10000);
    if (left <= 0) { tRaf = 0; return timeUp(); }
    tRaf = requestAnimationFrame(tick);
  };
  tick();
}
function stopTimer(){ if (tRaf) cancelAnimationFrame(tRaf); tRaf = 0; }
function resumeTimer(){ startTimer(Math.max(10000, tEnd - performance.now())); }
function timeUp(){
  if (locked) return;
  if (cur && (manualHidden() || norm($("mIn").value))) return submit();
  locked = true;
  st.res[st.i] = {pi:set[st.i], q:POOL[set[st.i]].q, label:"time's up", fits:false, timeout:true, score:0, thumb:cur?cur.thumb:null};
  result();
}
const manualHidden = () => $("manual").hidden;

/* ---------- screens ---------- */
function show(id){ for (const s of ["home","round","result","end"]) $(s).hidden = s!==id; }
function total(){ return st.res.reduce((a,r)=>a+(r?r.score:0),0); }
function hud(){
  $("chip").textContent = total();
  $("steps").innerHTML = Array.from({length:ROUNDS},(_,k)=>`<i class="${k<st.i?"done":k===st.i&&$("home").hidden?"now":""}"></i>`).join("");
}

function round(){
  if (st.i >= ROUNDS) return end();
  $("prompt").textContent = POOL[set[st.i]].q;
  cur = null; $("pv").hidden = true; $("plus").hidden = false; $("file").value = "";
  $("send").disabled = true; $("err").textContent = "";
  $("manual").hidden = modelOK !== false; $("mIn").value = "";
  locked = false; $("timer").classList.remove("low");
  show("round"); hud(); startTimer();
}

async function loadImage(file){
  if (!file || !file.type.startsWith("image/")) return;
  try{
    const bmp = await createImageBitmap(file);
    const draw = (max,q) => new Promise(res=>{ const s=Math.min(1,max/Math.max(bmp.width,bmp.height));
      const c=document.createElement("canvas"); c.width=bmp.width*s|0; c.height=bmp.height*s|0;
      const x=c.getContext("2d"); x.fillStyle="#fff"; x.fillRect(0,0,c.width,c.height); x.drawImage(bmp,0,0,c.width,c.height);
      c.toBlob(b=>res({b,u:c.toDataURL("image/jpeg",q)}),"image/jpeg",q); });
    const big = await draw(640,.9), small = await draw(220,.75);
    cur = {blob:big.b, thumb:small.u};
    $("pv").src = big.u; $("pv").hidden = false; $("plus").hidden = true; $("err").textContent = "";
    $("send").disabled = !manualHidden() && !$("mIn").value.trim();
  }catch{ $("err").textContent = "Couldn't read that image. Try a JPEG or PNG."; }
}

function makeEntry(pi, label, fits, extra = {}){
  const {share, kind} = shareOf(pi, label);
  return Object.assign({pi, q:POOL[pi].q, label, fits, kind, share, score: fits ? rarity(share) : 0}, extra);
}

async function submit(){
  if (!cur || locked) return;
  const pi = set[st.i];
  locked = true; stopTimer(); $("send").disabled = true; $("err").textContent = "";
  if (manualHidden()){
    $("busy").hidden = false;
    $("busy").firstElementChild.textContent = modelOK === null ? "Loading recognizer…" : "Looking…";
    try{
      const r = await R.classify(cur.blob);
      const label = r.ranked[0].label;
      const alts = r.ranked.slice(1).map(x=>x.label).filter(l=>!UNCLEAR.includes(l)).slice(0,4);
      st.res[st.i] = makeEntry(pi, label, R.fits(pi, label, r.promptScores),
        {thumb:cur.thumb, conf:r.ranked[0].p, alts, promptScores:r.promptScores});
    }catch(e){
      console.error(e); $("busy").hidden = true; locked = false; resumeTimer();
      $("manual").hidden = false; $("err").textContent = "Couldn't recognize it. Type what it is.";
      return;
    }
    $("busy").hidden = true;
  } else {
    const label = norm($("mIn").value);
    if (!label){ locked = false; resumeTimer(); $("send").disabled = false; return; }
    // no model: trust the player
    st.res[st.i] = makeEntry(pi, label, true, {thumb:cur.thumb, typed:true, alts:[]});
  }
  result();
}

/* Player corrects the label ("Not it?") */
function relabel(label){
  const old = st.res[st.i]; label = norm(label); if (!label || label === old.label) return;
  const fits = old.promptScores ? R.fits(old.pi, label, old.promptScores) : true;
  st.res[st.i] = makeEntry(old.pi, label, fits, {thumb:old.thumb, alts:old.alts.filter(a=>a!==label).concat(old.label).slice(0,4), promptScores:old.promptScores, corrected:true});
  result();
}

function field(r){
  const W=520,H=120,out=[]; let s=11; const rnd=()=>{s=(s*16807)%2147483647;return s/2147483647};
  for (const share of Object.values(POOL[r.pi].common)){
    const sc = rarity(share), n = Math.max(1, Math.round(share*100));
    for (let k=0;k<n;k++){ const x=14+(sc/100)*(W-40)+(rnd()-.5)*26, y=H/2+(rnd()-.5)*(H-30);
      out.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" fill="var(--dot)"/>`); }
  }
  if (r.fits){ const x=14+(r.score/100)*(W-40);
    out.push(`<circle cx="${x}" cy="${H/2}" r="16" fill="none" stroke="#ff4f1f" stroke-width="2" opacity=".35"/><circle cx="${x}" cy="${H/2}" r="9" fill="#ff4f1f"/>`); }
  return out.join("");
}

function result(){
  const r = st.res[st.i];
  $("rImg").hidden = !r.thumb; if (r.thumb) $("rImg").src = r.thumb;
  $("rQ").textContent = r.q;
  $("rLabel").innerHTML = r.timeout ? `<span class="bad">Time's up</span>`
    : r.fits ? esc(r.label)
    : `<span class="bad">${UNCLEAR.includes(r.label) ? "Can't tell what it is" : esc(r.label)+" ✕"}</span>`;
  $("rNum").innerHTML = r.score + "<small>/100</small>";
  $("field").innerHTML = field(r);
  const top = Object.entries(POOL[r.pi].common).sort((a,b)=>b[1]-a[1]).slice(0,3).map(e=>`<b>${esc(e[0])}</b>`).join(" · ");
  $("herd").innerHTML = r.timeout ? `You had ${LIMIT/60} minutes. Most say ${top}.`
    : !r.fits ? (UNCLEAR.includes(r.label) ? "Try one object, close up." : "Doesn't fit the prompt.")
    : r.kind === "common" ? `About ${Math.max(1,Math.round(r.share*100))}% of people say this. Most say ${top}.`
    : r.kind === "rare" ? `Few people think of this. Most say ${top}.`
    : `Almost nobody says this. Most say ${top}.`;
  // correction chips
  const fix = $("fix");
  if (!r.timeout && r.alts && r.thumb && !r.typed){
    fix.innerHTML = `Not it?` + r.alts.map(a=>`<button class="chip" data-l="${esc(a)}">${esc(a)}</button>`).join("") +
      `<button class="chip" data-other="1">other…</button>`;
    fix.hidden = false;
  } else fix.hidden = true;
  $("next").textContent = st.i+1 >= ROUNDS ? "Finish" : "Next";
  show("result"); hud();
}

function end(){
  $("eNum").innerHTML = total() + `<small>/${ROUNDS*100}</small>`;
  $("tiles").innerHTML = st.res.map(r => `<div class="tile">${r&&r.thumb?`<img src="${r.thumb}" alt="">`:""}<b>${r?r.score:0}</b></div>`).join("");
  $("toast").textContent = "";
  show("end"); st.i = ROUNDS; hud();
}
function shareText(){
  const bar = st.res.map(r => !r ? "⬛" : r.score>=90?"🟥":r.score>=60?"🟧":r.score>=30?"🟨":r.score>0?"⬜":"⬛").join("");
  return `Outlier ${day}\n${bar} ${total()}/${ROUNDS*100}`;
}

/* ---------- wiring ---------- */
$("go").onclick = round;
$("file").onchange = e => loadImage(e.target.files[0]);
const snap = $("snap");
["dragenter","dragover"].forEach(t=>snap.addEventListener(t,e=>{e.preventDefault();snap.classList.add("over")}));
["dragleave","drop"].forEach(t=>snap.addEventListener(t,e=>{e.preventDefault();snap.classList.remove("over")}));
snap.addEventListener("drop",e=>{const f=e.dataTransfer&&e.dataTransfer.files[0]; if(f) loadImage(f);});
document.addEventListener("paste",e=>{ if ($("round").hidden) return; const it=[...(e.clipboardData?.items||[])].find(i=>i.type.startsWith("image/")); if (it) loadImage(it.getAsFile()); });
$("mIn").oninput = () => { $("send").disabled = !cur || !$("mIn").value.trim(); };
$("mIn").onkeydown = e => { if (e.key==="Enter" && !$("send").disabled) submit(); };
$("send").onclick = submit;
$("skip").onclick = () => { stopTimer(); locked = true; const pi=set[st.i]; st.res[st.i]={pi,q:POOL[pi].q,label:"skipped",fits:false,score:0,thumb:null}; st.i++; round(); };
$("next").onclick = () => { st.i++; round(); };
$("fix").onclick = e => {
  const b = e.target.closest("button"); if (!b) return;
  if (b.dataset.other){
    $("fix").insertAdjacentHTML("beforeend", `<input id="fixIn" placeholder="What is it? Press Enter" maxlength="40">`);
    b.remove(); const i = $("fixIn"); i.focus(); i.onkeydown = ev => { if (ev.key==="Enter") relabel(i.value); };
  } else relabel(b.dataset.l);
};
$("copy").onclick = async () => { try{ await navigator.clipboard.writeText(shareText()); $("toast").textContent="Copied"; }catch{ $("toast").textContent=shareText().replace("\n"," "); } };
$("reset").onclick = () => { runs++; store.set("outlier-runs-"+day, runs); set = pick(day+"#"+runs, ROUNDS); st={i:0,res:[]}; round(); };
hud();
})();
