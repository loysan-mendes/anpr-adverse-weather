const $ = id => document.getElementById(id);
let file = null, bitmap = null, result = null, selected = -1, busy = false, selectionVersion = 0;
const percent = value => Number.isFinite(value) ? `${Math.round(value * 100)}%` : '—';
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
function resetResults() { result = null; selected = -1; $('results').hidden = true; $('empty').hidden = false; $('result-count').textContent = 'Ready to analyze'; }
function draw() {
  if (!bitmap) return;
  const canvas = $('canvas'), ctx = canvas.getContext('2d');
  const scale = Math.min(1, 1600 / bitmap.width);
  canvas.width = Math.round(bitmap.width * scale); canvas.height = Math.round(bitmap.height * scale);
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  (result?.vehicles || []).forEach(vehicle => {
    const [x1,y1,x2,y2] = vehicle.box.map(x => x * scale);
    ctx.strokeStyle = '#8b9cff';
    ctx.lineWidth = result.plates[selected]?.vehicle_id === vehicle.id ? 5 : 2;
    ctx.setLineDash([8,4]); ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.setLineDash([]);
    ctx.font = 'bold 14px Segoe UI';
    const label = `V${vehicle.id} ${vehicle.vehicle_type} ${percent(vehicle.confidence)}`;
    const y = Math.max(20,y1);
    ctx.fillStyle = '#202944'; ctx.fillRect(x1,y-20,ctx.measureText(label).width+12,20);
    ctx.fillStyle = '#ffffff'; ctx.fillText(label,x1+6,y-5);
  });
  (result?.plates || []).forEach((plate, index) => {
    const [x1,y1,x2,y2] = plate.box.map(x => x * scale);
    ctx.strokeStyle = plate.status === 'accepted' ? '#65ed9c' : '#ffcc64';
    ctx.lineWidth = index === selected ? 5 : 2;
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.font = 'bold 14px Segoe UI'; ctx.fillStyle = ctx.strokeStyle;
    const label = `${index+1}`; const y = Math.max(20,y1);
    ctx.fillRect(x1,y-20,26,20); ctx.fillStyle = '#153225';ctx.fillText(label,x1+7,y-5);
  });
}
async function choose(candidate) {
  if (busy || !candidate) return;
  const version = ++selectionVersion;
  error('');
  if (!['image/jpeg','image/png','image/webp'].includes(candidate.type) || candidate.size > 12*1024*1024) {
    error('Choose a JPEG, PNG or WebP image up to 12 MB.'); return;
  }
  try {
    const next = await createImageBitmap(candidate, {imageOrientation:'from-image'});
    if (version !== selectionVersion) { next.close(); return; }
    if (next.width*next.height > 12000000) { next.close(); throw new Error('Choose an image up to 12 megapixels.'); }
    bitmap?.close(); bitmap = next; file = candidate; resetResults(); draw();
    $('drop-zone').hidden = true; $('preview').hidden = false; $('file-info').hidden = false; $('change').hidden = false;
    $('file-name').textContent = candidate.name; $('dimensions').textContent = `${bitmap.width} × ${bitmap.height}`;
    $('analyze').disabled = false;
  } catch(e) { error(e.message || 'This image could not be opened.'); }
}
$('browse').onclick = $('change').onclick = () => $('file').click();
$('file').onchange = e => { choose(e.target.files[0]); e.target.value = ''; };
['dragenter','dragover'].forEach(event => $('drop-zone').addEventListener(event,e => {e.preventDefault();$('drop-zone').classList.add('drag');}));
['dragleave','drop'].forEach(event => $('drop-zone').addEventListener(event,e => {e.preventDefault();$('drop-zone').classList.remove('drag');}));
$('drop-zone').addEventListener('drop', e => choose(e.dataTransfer.files[0]));
window.addEventListener('dragover',e=>e.preventDefault()); window.addEventListener('drop',e=>e.preventDefault());
function element(tag, text, className) { const node = document.createElement(tag); if(text !== undefined) node.textContent = text; if(className) node.className = className; return node; }
function render() {
  $('results').hidden = false; $('empty').hidden = true;
  $('result-count').textContent = 'Analysis complete';
  $('detected').textContent = result.num_plates_detected; $('accepted').textContent = result.num_plates_accepted;
  $('duration').textContent = `${result.timings_seconds.total.toFixed(1)}s`;
  $('quality').replaceChildren(...[`${result.quality.condition} · ${percent(result.quality.condition_confidence)}`,`Severity: ${result.quality.severity}`,result.restoration_applied ? `Restored: ${result.restoration_applied}` : 'Original image',result.profile].map(s=>element('span',s,'chip')));
  $('plates').replaceChildren();
  $('vehicles').replaceChildren();
  $('vehicles').append(element('h3', `Vehicles (${result.num_vehicles_detected || 0})`));
  if (!result.vehicles?.length) $('vehicles').append(element('p','No supported vehicles detected.','plate-meta'));
  (result.vehicles || []).forEach(vehicle => {
    const matched = result.plates.some(plate => plate.vehicle_id === vehicle.id);
    $('vehicles').append(element('p',`V${vehicle.id} · ${vehicle.vehicle_type} · ${percent(vehicle.confidence)}${matched ? '' : ' · No plate matched'}`,'plate-meta'));
  });
  if (!result.plates.length) $('plates').append(element('p','No plates detected. Try a closer, clearer image or thorough mode.','plate-meta'));
  result.plates.forEach((plate,index) => {
    const card=element('article',undefined,'plate'), button=element('button',undefined,'plate-main');
    const text=element('span'); text.append(element('span',`PLATE ${String(index+1).padStart(2,'0')}`,'plate-label'));
    text.append(element('strong',plate.plate_text || plate.proposed_text || 'Unreadable','plate-text'));
    button.append(text,element('span',plate.status === 'accepted' ? 'Accepted' : 'Needs review',`status ${plate.status}`));
    button.setAttribute('aria-pressed','false');
    button.onclick=()=>{ selected=index; document.querySelectorAll('.plate').forEach(n=>n.classList.remove('selected')); document.querySelectorAll('.plate-main').forEach(n=>n.setAttribute('aria-pressed','false'));card.classList.add('selected');button.setAttribute('aria-pressed','true');draw();};
    card.append(button,element('div',`OCR ${percent(plate.ocr_confidence)} · Detection ${percent(plate.detection_confidence)} · ${plate.ocr_source || 'original'} view${plate.status !== 'accepted' ? ' · Proposed text only' : ''}`,'plate-meta'));
    const vehicleLabel = plate.vehicle_match_status === 'matched'
      ? `Vehicle: ${plate.vehicle_type} · V${plate.vehicle_id} · ${percent(plate.vehicle_confidence)}`
      : `Vehicle: unknown · Needs review (${(plate.vehicle_match_status || 'unmatched').replaceAll('_',' ')})`;
    card.append(element('div',vehicleLabel,'plate-meta'));
    const details=element('details');details.append(element('summary','Reading evidence'));
    const lines=(plate.readings || []).map(r=>`${r.ocr_source} / ${r.ocr_variant || 'original'}: ${r.plate_text || r.proposed_text || 'unreadable'} · ${r.status} · OCR ${percent(r.ocr_confidence)}`);
    details.append(element('p',lines.join('\n'),'evidence'));card.append(details);$('plates').append(card);
  }); draw();
}
$('analyze').onclick = async () => {
  if (!file || busy) return;
  busy=true; error('');resetResults(); draw(); $('empty').hidden=true;$('loading').hidden=false;
  ['analyze','change','profile','browse'].forEach(id=>$(id).disabled=true);
  $('result-count').textContent='Processing'; const start=Date.now();$('elapsed').textContent='0 seconds';
  const timer=setInterval(()=>$('elapsed').textContent=`${Math.floor((Date.now()-start)/1000)} seconds`,1000);
  try {
    const response=await fetch(`/api/analyze?profile=${encodeURIComponent($('profile').value)}`,{method:'POST',headers:{'Content-Type':file.type},body:file});
    const data=await response.json();if(!response.ok) throw new Error(data.error || 'Analysis failed.');
    result=data;render();
  } catch(e) {error(e.message || 'Could not connect to the local server.');$('empty').hidden=false;$('result-count').textContent='Try again';}
  finally {clearInterval(timer);busy=false;$('loading').hidden=true;['analyze','change','profile','browse'].forEach(id=>$(id).disabled=false);}
};
$('download').onclick=()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify({filename:file.name,...result},null,2)],{type:'application/json'}));const link=element('a');link.href=url;link.download='clearplate-results.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
fetch('/api/health').then(r=>{if(!r.ok)throw new Error();$('connection').textContent='Local server online';}).catch(()=>$('connection').textContent='Server unavailable');

$('sample').onclick = async () => {
  $('sample').disabled = true; error('');
  try {
    const response = await fetch('/api/sample');
    if (!response.ok) throw new Error('Sample unavailable. Please choose your own image.');
    await choose(new File([await response.blob()], 'sample-vehicles.jpg', {type:'image/jpeg'}));
  } catch(e) { error(e.message); }
  finally { $('sample').disabled = false; }
};
