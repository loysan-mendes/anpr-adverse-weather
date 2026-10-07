/**
 * Clearplate Vision AI · Client Controller
 * Adverse-Weather ANPR & Multi-Scale Vehicle Association
 */

const $ = id => document.getElementById(id);
let file = null, bitmap = null, result = null, selectedPlate = -1, selectedVehicle = -1, busy = false, selectionVersion = 0;

const percent = val => Number.isFinite(val) ? `${Math.round(val * 100)}%` : '—';

function makeElement(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}

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

  // 1. Draw Vehicle Bounding Boxes (Monochrome Crisp White Dashed)
  if (showVehicles && result?.vehicles) {
    result.vehicles.forEach(vehicle => {
      const isSelected = selectedVehicle === vehicle.id || (selectedPlate >= 0 && result.plates[selectedPlate]?.vehicle_id === vehicle.id);
      const [x1, y1, x2, y2] = vehicle.box.map(coord => coord * scale);
      const w = x2 - x1, h = y2 - y1;

      ctx.save();
      ctx.strokeStyle = isSelected ? '#ffffff' : 'rgba(255, 255, 255, 0.45)';
      ctx.lineWidth = isSelected ? 3 : 1.5;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(x1, y1, w, h);
      ctx.setLineDash([]);

      // Vehicle Tag Header
      const label = `V${vehicle.id} ${vehicle.vehicle_type.toUpperCase()} ${percent(vehicle.confidence)}`;
      ctx.font = '600 11px "JetBrains Mono", Consolas, monospace';
      const textWidth = ctx.measureText(label).width;
      const tagY = Math.max(16, y1);

      ctx.fillStyle = isSelected ? '#ffffff' : 'rgba(0, 0, 0, 0.85)';
      ctx.fillRect(x1, tagY - 16, textWidth + 10, 16);
      ctx.fillStyle = isSelected ? '#000000' : '#ffffff';
      ctx.fillText(label, x1 + 5, tagY - 4);
      ctx.restore();
    });
  }

  // 2. Draw License Plate Bounding Boxes (Crisp Solid White)
  if (showPlates && result?.plates) {
    result.plates.forEach((plate, index) => {
      const isSelected = index === selectedPlate;
      const [x1, y1, x2, y2] = plate.box.map(coord => coord * scale);
      const w = x2 - x1, h = y2 - y1;

      ctx.save();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(x1, y1, w, h);

      // Plate Index Badge
      const badgeText = `${index + 1}`;
      ctx.font = '700 11px "JetBrains Mono", Consolas, monospace';
      const tagY = Math.max(16, y1);

      ctx.fillStyle = '#ffffff';
      ctx.fillRect(x1, tagY - 16, 20, 16);
      ctx.fillStyle = '#000000';
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
  const link = document.createElement('a');
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

/* ==========================================================================
   Multi-Frame Video Tracking & Mode Switcher
   ========================================================================== */

let currentMode = 'image';
let videoFile = null;
let isDemoVideo = false;
let videoResult = null;
let videoBusy = false;

function switchMode(mode) {
  currentMode = mode;
  const imgTab = $('tab-mode-image');
  const vidTab = $('tab-mode-video');
  const imgWorkspace = $('workspace-image');
  const vidWorkspace = $('workspace-video');

  if (mode === 'image') {
    imgTab.classList.add('active');
    vidTab.classList.remove('active');
    imgWorkspace.hidden = false;
    vidWorkspace.hidden = true;
  } else {
    vidTab.classList.add('active');
    imgTab.classList.remove('active');
    imgWorkspace.hidden = true;
    vidWorkspace.hidden = false;
  }
}

$('tab-mode-image').onclick = () => switchMode('image');
$('tab-mode-video').onclick = () => switchMode('video');

function videoError(msg) {
  const el = $('video-error');
  el.textContent = msg;
  el.hidden = !msg;
}

function resetVideoWorkspace() {
  videoFile = null;
  isDemoVideo = false;
  videoResult = null;
  const player = $('video-display-element');
  player.pause();
  player.removeAttribute('src');
  player.load();

  $('video-drop-zone').hidden = false;
  $('video-stage-container').hidden = true;
  $('video-file-info').hidden = true;
  $('video-change').hidden = true;
  $('video-analyze-btn').disabled = true;
  $('video-empty').hidden = false;
  $('video-loading').hidden = true;
  $('video-results').hidden = true;
  $('video-download-btn').hidden = true;
  $('video-result-count').textContent = 'Awaiting Video';
  $('video-overlay-badge').textContent = 'Source Video Preview';
  videoError('');
}

$('video-change').onclick = resetVideoWorkspace;

async function setVideoSource(fileOrDemo, isDemo = false) {
  videoError('');
  isDemoVideo = isDemo;
  const player = $('video-display-element');

  if (isDemo) {
    videoFile = null;
    player.src = '/api/sample-video';
    $('video-file-name').textContent = 'demo_vehicle_passage.mp4';
    $('video-telemetry-badge').textContent = 'DEMO · 33 Frames';
  } else {
    videoFile = fileOrDemo;
    player.src = URL.createObjectURL(videoFile);
    $('video-file-name').textContent = videoFile.name;
    const mb = (videoFile.size / (1024 * 1024)).toFixed(1);
    $('video-telemetry-badge').textContent = `${mb} MB`;
  }

  $('video-drop-zone').hidden = true;
  $('video-stage-container').hidden = false;
  $('video-file-info').hidden = false;
  $('video-change').hidden = false;
  $('video-analyze-btn').disabled = false;
  $('video-empty').hidden = false;
  $('video-results').hidden = true;
  $('video-result-count').textContent = 'Ready to Track';
  $('video-overlay-badge').textContent = isDemo ? 'DEMO PASSAGE CLIP' : 'UPLOADED CLIP';
  player.load();
}

$('video-browse-btn').onclick = () => $('video-file-input').click();
$('video-file-input').onchange = e => {
  const f = e.target.files?.[0];
  if (f) setVideoSource(f, false);
};

const vDrop = $('video-drop-zone');
['dragenter', 'dragover'].forEach(name => {
  vDrop.addEventListener(name, e => {
    e.preventDefault();
    vDrop.classList.add('drag-over');
  });
});
['dragleave', 'drop'].forEach(name => {
  vDrop.addEventListener(name, e => {
    e.preventDefault();
    vDrop.classList.remove('drag-over');
  });
});
vDrop.ondrop = e => {
  const f = e.dataTransfer?.files?.[0];
  if (f && (f.type.startsWith('video/') || f.name.match(/\.(mp4|webm|mov)$/i))) {
    setVideoSource(f, false);
  } else {
    videoError('Please drop an MP4, WebM, or MOV video file.');
  }
};

$('video-demo-btn').onclick = () => setVideoSource(null, true);

// Video Analysis Runner
$('video-analyze-btn').onclick = async () => {
  if (videoBusy) return;
  videoBusy = true;
  videoError('');
  $('video-analyze-btn').disabled = true;
  $('video-empty').hidden = true;
  $('video-results').hidden = true;
  $('video-loading').hidden = false;

  const timerEl = $('video-elapsed');
  const start = performance.now();
  const timer = setInterval(() => {
    timerEl.textContent = `${((performance.now() - start) / 1000).toFixed(1)}s elapsed`;
  }, 100);

  try {
    const stride = $('video-stride').value || '1';
    let url = `/api/analyze-video?stride=${stride}`;
    let body = null;
    let headers = {};

    if (isDemoVideo) {
      url += '&demo=1';
      body = '';
    } else {
      if (!videoFile) throw new Error('No video selected.');
      body = await videoFile.arrayBuffer();
      headers['Content-Type'] = 'video/mp4';
    }

    const response = await fetch(url, { method: 'POST', body, headers });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.error || `Server responded with ${response.status}`);
    }

    const data = await response.json();
    videoResult = data;
    renderVideoResults(data);
  } catch (err) {
    videoError(err.message);
    $('video-empty').hidden = false;
  } finally {
    clearInterval(timer);
    videoBusy = false;
    $('video-loading').hidden = true;
    $('video-analyze-btn').disabled = false;
  }
};

function renderVideoResults(data) {
  $('video-results').hidden = false;
  $('video-download-btn').hidden = false;
  $('video-result-count').textContent = `${data.num_accepted_plates || 0} Accepted Plates`;

  // Update telemetry dashboard
  $('metric-video-vehicles').textContent = data.num_vehicles_tracked ?? data.passages?.length ?? 0;
  $('metric-video-accepted').textContent = data.num_accepted_plates ?? 0;
  const frames = data.video_metadata?.processed_frames ?? 0;
  $('metric-video-frames').textContent = frames;
  const fps = (frames / (data.processing_time_seconds || 1)).toFixed(1);
  $('metric-video-fps').textContent = `${fps} FPS processing speed`;
  $('metric-video-time').textContent = `${(data.processing_time_seconds || 0).toFixed(2)}s`;

  // Switch video player to annotated video feed if available
  if (data.video_stream_url) {
    const player = $('video-display-element');
    player.src = data.video_stream_url;
    $('video-overlay-badge').textContent = '🎯 ANNOTATED BYTETRACK FEED';
    player.load();
    player.play().catch(() => {});
  }

  // Render passages list
  const listEl = $('video-passages-list');
  listEl.innerHTML = '';
  $('video-passages-badge').textContent = `${data.passages?.length || 0} events`;

  if (!data.passages || data.passages.length === 0) {
    listEl.innerHTML = '<div class="state-empty" style="padding:24px"><p>No vehicle passages detected in this clip.</p></div>';
    return;
  }

  data.passages.forEach(p => {
    const card = document.createElement('div');
    card.className = 'passage-card';
    card.id = `passage-card-${p.track_id}`;

    const isAccepted = p.status === 'accepted';
    const statusClass = isAccepted ? 'badge-accepted' : 'badge-review';
    const statusLabel = isAccepted ? 'ACCEPTED' : (p.status || 'REVIEW').toUpperCase();
    const confPercent = Math.round((p.fused_confidence || 0) * 100);

    let alternativesHtml = '';
    if (p.alternatives && p.alternatives.length > 1) {
      alternativesHtml = `
        <div class="passage-alternatives">
          <strong style="color:var(--text-muted); font-size:0.75rem;">Multi-Frame Candidate Breakdown:</strong>
          ${p.alternatives.map(a => `
            <div class="alternative-item">
              <span style="font-weight:600;">${a.text}</span>
              <span>${a.frame_count} frames · vote weight ${a.votes.toFixed(2)}</span>
            </div>
          `).join('')}
        </div>
      `;
    }

    card.innerHTML = `
      <div class="passage-card-top">
        <span class="passage-track-badge">#${p.track_id} · ${(p.vehicle_type || 'VEHICLE').toUpperCase()}</span>
        <span class="passage-time-range">⏱ ${p.first_seen_sec}s ➔ ${p.last_seen_sec}s (${p.total_plate_frames} frames)</span>
      </div>

      <div class="passage-body">
        <div class="passage-plate-display">
          <span class="passage-plate-text">${p.plate_text || p.proposed_text || '—'}</span>
          <div class="passage-meta-line">
            <span class="passage-agreement-tag">🎯 ${p.agreement_count || 1} unanimous frame votes</span>
            <span>Confidence: ${confPercent}%</span>
            <span>HSRP Format: ${(p.format_score || 0).toFixed(1)}</span>
          </div>
        </div>

        <div style="display:flex; flex-direction:column; align-items:flex-end; gap:8px;">
          <span class="badge-status ${statusClass}">${statusLabel}</span>
          <button class="btn-seek-passage" type="button" data-time="${p.first_seen_sec}">
            <span>▶ Seek (${p.first_seen_sec}s)</span>
          </button>
        </div>
      </div>
      ${alternativesHtml}
    `;

    card.querySelector('.btn-seek-passage').onclick = e => {
      e.stopPropagation();
      const time = parseFloat(e.currentTarget.getAttribute('data-time') || 0);
      const player = $('video-display-element');
      player.currentTime = time;
      player.play().catch(() => {});
      document.querySelectorAll('.passage-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
    };

    listEl.appendChild(card);
  });
}

$('video-download-btn').onclick = () => {
  if (!videoResult) return;
  const jsonStr = JSON.stringify(videoResult, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `passage-events-${Date.now()}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
