/**
 * Clearplate Vision AI · Client Controller
 * Adverse-Weather ANPR & Multi-Scale Vehicle Association
 */

const $ = id => document.getElementById(id);
let file = null, bitmap = null, result = null, selectedPlate = -1, selectedVehicle = -1, busy = false, selectionVersion = 0;

const percent = val => Number.isFinite(val) ? `${Math.round(val * 100)}%` : '—';

function error(message) {
  const el = $('error');
  el.textContent = message;
  el.hidden = !message;
}

function resetResults() {
  result = null;
  selectedPlate = -1;
  selectedVehicle = -1;
  $('results').hidden = true;
  $('empty').hidden = false;
  $('result-count').textContent = 'Ready for Scene';
  $('download').hidden = true;
  $('header-reset').hidden = !file;
}

function fullReset() {
  file = null;
  bitmap?.close();
  bitmap = null;
  resetResults();
  $('preview').hidden = true;
  $('drop-zone').hidden = false;
  $('file-info').hidden = true;
  $('change').hidden = true;
  $('header-reset').hidden = true;
  $('analyze').disabled = true;
  error('');
}

function draw() {
  if (!bitmap) return;
  const canvas = $('canvas'), ctx = canvas.getContext('2d');
  const scale = Math.min(1, 1600 / bitmap.width);
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);

  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);

  const showVehicles = $('toggle-vehicles')?.checked ?? true;
  const showPlates = $('toggle-plates')?.checked ?? true;

  // 1. Draw Vehicle Bounding Boxes (Purple)
  if (showVehicles && result?.vehicles) {
    result.vehicles.forEach(vehicle => {
      const isSelected = selectedVehicle === vehicle.id || (selectedPlate >= 0 && result.plates[selectedPlate]?.vehicle_id === vehicle.id);
      const [x1, y1, x2, y2] = vehicle.box.map(coord => coord * scale);
      const w = x2 - x1, h = y2 - y1;

      ctx.save();
      ctx.strokeStyle = isSelected ? '#d946ef' : '#8b5cf6';
      ctx.lineWidth = isSelected ? 4 : 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x1, y1, w, h);
      ctx.setLineDash([]);

      // Vehicle Tag Header
      const label = `V${vehicle.id} ${vehicle.vehicle_type.toUpperCase()} ${percent(vehicle.confidence)}`;
      ctx.font = 'bold 12px Consolas, monospace';
      const textWidth = ctx.measureText(label).width;
      const tagY = Math.max(18, y1);

      ctx.fillStyle = isSelected ? '#a21caf' : 'rgba(30, 27, 75, 0.9)';
      ctx.fillRect(x1, tagY - 18, textWidth + 12, 18);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x1 + 6, tagY - 4);
      ctx.restore();
    });
  }

  // 2. Draw License Plate Bounding Boxes (Neon Green / Amber)
  if (showPlates && result?.plates) {
    result.plates.forEach((plate, index) => {
      const isSelected = index === selectedPlate;
      const [x1, y1, x2, y2] = plate.box.map(coord => coord * scale);
      const w = x2 - x1, h = y2 - y1;

      ctx.save();
      const isAccepted = plate.status === 'accepted';
      const mainColor = isAccepted ? '#10b981' : '#f59e0b';

      // Subtle shadow/glow for selected plate
      if (isSelected) {
        ctx.shadowColor = mainColor;
        ctx.shadowBlur = 12;
      }

      ctx.strokeStyle = mainColor;
      ctx.lineWidth = isSelected ? 4 : 2;
      ctx.strokeRect(x1, y1, w, h);

      // Plate Index Badge
      ctx.shadowBlur = 0;
      const badgeText = `${index + 1}`;
      ctx.font = 'bold 12px Consolas, monospace';
      const tagY = Math.max(18, y1);

      ctx.fillStyle = mainColor;
      ctx.fillRect(x1, tagY - 18, 22, 18);
      ctx.fillStyle = '#090d16';
      ctx.fillText(badgeText, x1 + 6, tagY - 4);
      ctx.restore();
    });
  }
}

async function choose(candidate) {
  if (busy || !candidate) return;
  const version = ++selectionVersion;
  error('');

  if (!['image/jpeg', 'image/png', 'image/webp'].includes(candidate.type) || candidate.size > 12 * 1024 * 1024) {
    error('Please select a JPEG, PNG or WebP image under 12 MB.');
    return;
  }

  try {
    const next = await createImageBitmap(candidate, { imageOrientation: 'from-image' });
    if (version !== selectionVersion) {
      next.close();
      return;
    }
    if (next.width * next.height > 12000000) {
      next.close();
      throw new Error('Image dimensions exceed the 12-megapixel maximum.');
    }

    bitmap?.close();
    bitmap = next;
    file = candidate;
    resetResults();
    draw();

    $('drop-zone').hidden = true;
    $('preview').hidden = false;
    $('file-info').hidden = false;
    $('change').hidden = false;
    $('header-reset').hidden = false;

    $('file-name').textContent = candidate.name;
    $('dimensions').textContent = `${bitmap.width} × ${bitmap.height} px`;
    $('analyze').disabled = false;
  } catch (e) {
    error(e.message || 'Unable to decode this image format safely.');
  }
}

// UI Event Listeners
$('browse').onclick = $('change').onclick = () => $('file').click();
$('file').onchange = e => {
  choose(e.target.files[0]);
  e.target.value = '';
};

$('toggle-vehicles')?.addEventListener('change', draw);
$('toggle-plates')?.addEventListener('change', draw);
$('header-reset')?.addEventListener('click', fullReset);

['dragenter', 'dragover'].forEach(evt => {
  $('drop-zone').addEventListener(evt, e => {
    e.preventDefault();
    $('drop-zone').classList.add('drag');
  });
});

['dragleave', 'drop'].forEach(evt => {
  $('drop-zone').addEventListener(evt, e => {
    e.preventDefault();
    $('drop-zone').classList.remove('drag');
  });
});

$('drop-zone').addEventListener('drop', e => choose(e.dataTransfer.files[0]));
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());

function makeElement(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function render() {
  $('results').hidden = false;
  $('empty').hidden = true;
  $('download').hidden = false;

  const plateCount = result.num_plates_detected || 0;
  const acceptedCount = result.num_plates_accepted || 0;
  $('result-count').textContent = `${acceptedCount} of ${plateCount} Accepted`;

  $('detected').textContent = plateCount;
  $('accepted').textContent = acceptedCount;
  $('duration').textContent = `${result.timings_seconds.total.toFixed(2)}s`;

  // Atmospheric & Restoration Diagnostics Chips
  const qualityEl = $('quality');
  qualityEl.replaceChildren();

  const condChip = makeElement('div', undefined, 'chip-diag');
  condChip.innerHTML = `<span class="chip-diag-badge badge-condition">${result.quality.condition.toUpperCase()}</span> ${percent(result.quality.condition_confidence)} Confidence`;
  qualityEl.append(condChip);

  const sevChip = makeElement('div', undefined, 'chip-diag');
  sevChip.innerHTML = `<span class="chip-diag-badge badge-severity">SEVERITY</span> ${result.quality.severity.toUpperCase()}`;
  qualityEl.append(sevChip);

  const restoreChip = makeElement('div', undefined, 'chip-diag');
  if (result.restoration_applied) {
    restoreChip.innerHTML = `<span class="chip-diag-badge badge-restored">RESTORED</span> ${result.restoration_applied}`;
  } else {
    restoreChip.innerHTML = `<span class="chip-diag-badge badge-original">SENSOR PIXELS</span> Native Pass`;
  }
  qualityEl.append(restoreChip);

  const modeChip = makeElement('div', undefined, 'chip-diag');
  modeChip.innerHTML = `<span class="chip-diag-badge badge-condition">PROFILE</span> ${result.profile.toUpperCase()}`;
  qualityEl.append(modeChip);

  // Full-Scene Vehicles Grid
  const vehiclesEl = $('vehicles');
  vehiclesEl.replaceChildren();
  const vCount = result.vehicles?.length || 0;
  $('vehicle-count-badge').textContent = `${vCount} found`;

  if (!vCount) {
    const noV = makeElement('div', 'No supported vehicles identified in this scene.', 'section-help');
    vehiclesEl.append(noV);
  } else {
    result.vehicles.forEach(vehicle => {
      const card = makeElement('div', undefined, 'vehicle-mini-card');
      card.onclick = () => {
        selectedVehicle = (selectedVehicle === vehicle.id) ? -1 : vehicle.id;
        selectedPlate = -1;
        draw();
      };

      const matchedPlate = (result.plates || []).find(p => p.vehicle_id === vehicle.id);
      const matchedText = matchedPlate 
        ? `Linked to Plate #${(result.plates.indexOf(matchedPlate) + 1)}` 
        : 'No plate linked';

      card.innerHTML = `
        <div class="vehicle-mini-top">
          <span class="vehicle-tag-id">V${vehicle.id}</span>
          <span class="vehicle-type-label">${vehicle.vehicle_type}</span>
        </div>
        <div class="vehicle-conf-bar">
          <div class="vehicle-conf-fill" style="width: ${Math.round(vehicle.confidence * 100)}%"></div>
        </div>
        <span class="vehicle-matched-note">${percent(vehicle.confidence)} · ${matchedText}</span>
      `;
      vehiclesEl.append(card);
    });
  }

  // Plates Feed
  const platesEl = $('plates');
  platesEl.replaceChildren();

  if (!result.plates || !result.plates.length) {
    platesEl.append(makeElement('div', 'No license plates localized. Try thorough mode or a higher-resolution frame.', 'section-help'));
  } else {
    result.plates.forEach((plate, index) => {
      const card = makeElement('article', undefined, 'plate-card');
      card.id = `plate-card-${index}`;

      card.onclick = () => {
        selectedPlate = (selectedPlate === index) ? -1 : index;
        selectedVehicle = -1;
        document.querySelectorAll('.plate-card').forEach(c => c.classList.remove('selected'));
        if (selectedPlate >= 0) {
          card.classList.add('selected');
          const detailsEl = card.querySelector('.plate-evidence-drawer');
          if (detailsEl) detailsEl.open = true;
        }
        draw();
      };

      const isAccepted = plate.status === 'accepted';
      const statusPillClass = isAccepted ? 'pill-accepted' : (plate.status === 'unreadable' ? 'pill-unreadable' : 'pill-review');
      const statusPillText = isAccepted ? '● VERIFIED' : (plate.status === 'unreadable' ? '■ UNREADABLE' : '▲ REVIEW REQUIRED');
      const displayText = plate.plate_text || plate.proposed_text || 'UNREADABLE';

      // Generate High-Res Plate Crop Thumbnail from source bitmap
      let cropDataUrl = '';
      if (bitmap && plate.box) {
        try {
          const [bx1, by1, bx2, by2] = plate.box;
          const cropW = Math.max(1, Math.round(bx2 - bx1));
          const cropH = Math.max(1, Math.round(by2 - by1));
          const offCanvas = document.createElement('canvas');
          offCanvas.width = cropW;
          offCanvas.height = cropH;
          const offCtx = offCanvas.getContext('2d');
          offCtx.drawImage(bitmap, bx1, by1, cropW, cropH, 0, 0, cropW, cropH);
          cropDataUrl = offCanvas.toDataURL('image/jpeg', 0.9);
        } catch (e) {}
      }

      const cropBlock = cropDataUrl ? `
        <div class="plate-crop-container">
          <img src="${cropDataUrl}" class="plate-crop-thumb" alt="Sensor Crop">
          <span class="plate-crop-label">SENSOR CROP</span>
        </div>
      ` : '';

      // Indian High Security Registration Plate (HSRP) Styled Badge
      const previewRow = `
        <div class="plate-preview-row">
          ${cropBlock}
          <div class="hsrp-plate-box">
            <div class="hsrp-ind-strip">
              <div class="hsrp-chakra-dot"></div>
              <span class="hsrp-ind-text">IND</span>
            </div>
            <div class="hsrp-rivet rivet-tl"></div>
            <div class="hsrp-rivet rivet-tr"></div>
            <div class="hsrp-rivet rivet-bl"></div>
            <div class="hsrp-rivet rivet-br"></div>
            <div class="hsrp-plate-number ${!isAccepted && !plate.proposed_text ? 'unreadable' : ''}">
              ${displayText}
            </div>
          </div>
        </div>
      `;

      // Status, Confidence Gauges, and Evidence Drawer
      const ocrScore = plate.ocr_confidence ?? 0;
      const detScore = plate.detection_confidence ?? 0;
      const fmtScore = plate.format_score ?? 0;

      const vehicleMatchLabel = plate.vehicle_match_status === 'matched'
        ? `<span class="vehicle-match-status match-success">Linked to Vehicle V${plate.vehicle_id} (${plate.vehicle_type} · ${percent(plate.vehicle_confidence)})</span>`
        : `<span class="vehicle-match-status match-review">Vehicle association unverified (${(plate.vehicle_match_status || 'unmatched').replace('_', ' ')})</span>`;

      const reviewNotice = plate.review_reason === 'low_final_ocr_confidence'
        ? `<div class="alert-banner alert-error" style="margin: 0 0 8px 0; font-size: 11px;">⚠ Reading requires review: OCR confidence falls below the 0.90 safety acceptance threshold.</div>`
        : '';

      const evidenceLines = (plate.readings || []).map(r => 
        `• [${r.ocr_source} / ${r.ocr_variant || 'standard'}] → "${r.plate_text || r.proposed_text || 'unreadable'}" (OCR: ${percent(r.ocr_confidence)}, Status: ${r.status})`
      ).join('\n');

      card.innerHTML = `
        <div class="plate-card-body" style="padding-top: 12px;">
          <div class="plate-status-row">
            <span class="plate-seq-badge">DETECTION #0${index + 1}</span>
            <span class="plate-pill-status ${statusPillClass}">${statusPillText}</span>
          </div>
        </div>
        ${previewRow}
        <div class="plate-card-body">
          ${reviewNotice}
          <div class="plate-gauges-grid">
            <div class="gauge-item">
              <span class="gauge-label">OCR CONFIDENCE</span>
              <span class="gauge-val">${percent(ocrScore)}</span>
              <div class="gauge-bar"><div class="gauge-fill fill-ocr" style="width: ${Math.round(ocrScore * 100)}%"></div></div>
            </div>
            <div class="gauge-item">
              <span class="gauge-label">DETECTOR</span>
              <span class="gauge-val">${percent(detScore)}</span>
              <div class="gauge-bar"><div class="gauge-fill fill-detect" style="width: ${Math.round(detScore * 100)}%"></div></div>
            </div>
            <div class="gauge-item">
              <span class="gauge-label">RTO FORMAT</span>
              <span class="gauge-val">${percent(fmtScore)}</span>
              <div class="gauge-bar"><div class="gauge-fill fill-format" style="width: ${Math.round(fmtScore * 100)}%"></div></div>
            </div>
          </div>
          <div class="plate-vehicle-match">
            ${vehicleMatchLabel}
          </div>
          <details class="plate-evidence-drawer">
            <summary>Optical Evidence & Variants (${(plate.readings || []).length})</summary>
            <div class="evidence-content">${evidenceLines || 'No secondary variants generated.'}</div>
          </details>
        </div>
      `;

      platesEl.append(card);
    });
  }

  draw();
}

$('analyze').onclick = async () => {
  if (!file || busy) return;
  busy = true;
  error('');
  resetResults();
  draw();

  $('empty').hidden = true;
  $('loading').hidden = false;
  ['analyze', 'change', 'profile', 'browse', 'header-sample'].forEach(id => {
    const el = $(id);
    if (el) el.disabled = true;
  });

  $('result-count').textContent = 'Analyzing…';
  const start = Date.now();
  $('elapsed').textContent = '0.0s elapsed';
  const timer = setInterval(() => {
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    $('elapsed').textContent = `${elapsed}s elapsed`;
  }, 100);

  try {
    const profileVal = encodeURIComponent($('profile').value);
    const response = await fetch(`/api/analyze?profile=${profileVal}`, {
      method: 'POST',
      headers: { 'Content-Type': file.type },
      body: file
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Pipeline execution failed.');
    result = data;
    render();
  } catch (e) {
    error(e.message || 'Connection to local inference engine failed.');
    $('empty').hidden = false;
    $('result-count').textContent = 'Analysis Failed';
  } finally {
    clearInterval(timer);
    busy = false;
    $('loading').hidden = true;
    ['analyze', 'change', 'profile', 'browse', 'header-sample'].forEach(id => {
      const el = $(id);
      if (el) el.disabled = false;
    });
  }
};

$('download').onclick = () => {
  if (!result) return;
  const jsonStr = JSON.stringify({ filename: file?.name, ...result }, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = makeElement('a');
  link.href = url;
  link.download = `clearplate-${Date.now()}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

// Health Monitor
fetch('/api/health')
  .then(r => {
    if (!r.ok) throw new Error();
    const pill = $('connection');
    pill.className = 'status-pill';
    pill.querySelector('.status-text').textContent = 'Engine Online · Ready';
  })
  .catch(() => {
    const pill = $('connection');
    pill.className = 'status-pill status-error';
    pill.querySelector('.status-text').textContent = 'Engine Offline';
  });

// Sample Image Handler
const loadSample = async () => {
  if (busy) return;
  const btn = $('sample');
  const hBtn = $('header-sample');
  if (btn) btn.disabled = true;
  if (hBtn) hBtn.disabled = true;
  error('');

  try {
    const response = await fetch('/api/sample');
    if (!response.ok) throw new Error('Demo sample is not installed on this local workstation.');
    const blob = await response.blob();
    await choose(new File([blob], 'demo-sample-scene.jpg', { type: 'image/jpeg' }));
  } catch (e) {
    error(e.message);
  } finally {
    if (btn) btn.disabled = false;
    if (hBtn) hBtn.disabled = false;
  }
};

$('sample').onclick = loadSample;
$('header-sample').onclick = loadSample;
