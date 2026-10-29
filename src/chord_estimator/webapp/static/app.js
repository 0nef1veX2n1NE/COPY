const $ = (selector) => document.querySelector(selector);
const form = $('#uploadForm');
const input = $('#audioFile');
const dropZone = $('#dropZone');
const analyzeButton = $('#analyzeButton');
const progressPanel = $('#progressPanel');
const resultSection = $('#resultSection');
const audioPlayer = $('#audioPlayer');
let currentJob = null;
let currentSegments = [];
let pollTimer = null;

const stageNames = {
  upload: '업로드 준비', queued: '분석 대기', starting: '모델 준비', decode: '오디오 디코딩',
  'log-cqt': 'Log-CQT 계산', inference: 'Transformer 추론', decoding: '구간 해석',
  writing: '결과 저장', complete: '분석 완료', failed: '분석 실패'
};

function formatTime(seconds, precise = false) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const rest = safe - minutes * 60;
  return `${minutes}:${rest.toFixed(precise ? 2 : 0).padStart(precise ? 5 : 2, '0')}`;
}

function showToast(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => { toast.hidden = true; }, 5200);
}

function selectFile(file) {
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  $('#dropTitle').textContent = file.name;
  $('#fileMeta').textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · 분석 준비됨`;
  analyzeButton.disabled = false;
}

['dragenter', 'dragover'].forEach((event) => dropZone.addEventListener(event, (e) => {
  e.preventDefault(); dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((event) => dropZone.addEventListener(event, (e) => {
  e.preventDefault(); dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', (e) => selectFile(e.dataTransfer.files[0]));
input.addEventListener('change', () => selectFile(input.files[0]));

function updateProgress(stage, value) {
  progressPanel.hidden = false;
  $('#progressStage').textContent = stageNames[stage] || stage;
  $('#progressValue').textContent = `${value}%`;
  $('#progressBar').style.width = `${value}%`;
}

async function pollJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`, { cache: 'no-store' });
    if (!response.ok) throw new Error('작업 상태를 불러오지 못했습니다.');
    const job = await response.json();
    updateProgress(job.stage, job.progress);
    if (job.status === 'succeeded') {
      window.clearTimeout(pollTimer);
      renderResult(job);
      return;
    }
    if (job.status === 'failed') throw new Error(job.error || '분석에 실패했습니다.');
    pollTimer = window.setTimeout(() => pollJob(jobId), 900);
  } catch (error) {
    analyzeButton.disabled = false;
    showToast(error.message);
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!input.files[0]) return;
  analyzeButton.disabled = true;
  resultSection.hidden = true;
  updateProgress('upload', 2);
  const body = new FormData();
  body.append('file', input.files[0]);
  try {
    const response = await fetch('/api/jobs', { method: 'POST', body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '업로드하지 못했습니다.');
    currentJob = data.id;
    pollJob(data.id);
  } catch (error) {
    analyzeButton.disabled = false;
    showToast(error.message);
  }
});

function chordClass(chord) {
  if (chord === 'N' || chord === 'UNKNOWN') return 'neutral';
  return chord.endsWith(':min') ? 'minor' : 'major';
}

function renderResult(job) {
  const result = job.result;
  const metadata = result.metadata;
  currentSegments = result.segments;
  $('#resultTitle').textContent = job.filename;
  $('#durationMetric').textContent = formatTime(metadata.audio.duration_sec);
  $('#segmentMetric').textContent = String(result.segment_count);
  $('#frameMetric').textContent = String(result.frame_count);
  $('#inferenceMetric').textContent = `${metadata.timing.inference_sec.toFixed(2)}s`;
  $('#axisEnd').textContent = formatTime(metadata.audio.duration_sec);
  audioPlayer.src = job.audio_url;

  const timeline = $('#timeline');
  timeline.replaceChildren();
  currentSegments.forEach((segment, index) => {
    const button = document.createElement('button');
    const ratio = (segment.end_sec - segment.start_sec) / metadata.audio.duration_sec;
    button.className = `timeline-segment ${chordClass(segment.chord)}`;
    button.style.flex = `${Math.max(ratio, .001)} 0 0`;
    button.textContent = ratio > .025 ? segment.display : '';
    button.title = `${formatTime(segment.start_sec, true)}–${formatTime(segment.end_sec, true)} · ${segment.display}`;
    button.setAttribute('role', 'listitem');
    button.dataset.index = index;
    button.addEventListener('click', () => { audioPlayer.currentTime = segment.start_sec; audioPlayer.play().catch(() => {}); });
    timeline.appendChild(button);
  });

  const rows = $('#segmentRows');
  rows.replaceChildren();
  currentSegments.forEach((segment, index) => {
    const row = document.createElement('tr');
    row.innerHTML = `<td>${String(index + 1).padStart(3, '0')}</td>
      <td>${formatTime(segment.start_sec, true)} — ${formatTime(segment.end_sec, true)}</td>
      <td class="${segment.needs_review ? 'review' : ''}">${segment.display}</td>
      <td>${segment.candidates.map((candidate) => `${candidate.display} ${(candidate.score * 100).toFixed(1)}`).join(' · ')}</td>
      <td>${(segment.score * 100).toFixed(1)}</td><td>${segment.uncertainty.toFixed(3)}</td>`;
    row.addEventListener('click', () => { audioPlayer.currentTime = segment.start_sec; audioPlayer.play().catch(() => {}); });
    rows.appendChild(row);
  });

  const labels = { json: 'JSON', csv: 'CSV', lab: 'LAB', frames: 'RAW NPZ' };
  const downloads = $('#downloads');
  downloads.replaceChildren();
  Object.entries(job.downloads).forEach(([kind, url]) => {
    const link = document.createElement('a'); link.href = url; link.textContent = `↓ ${labels[kind]}`; downloads.appendChild(link);
  });
  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  analyzeButton.disabled = false;
}

audioPlayer.addEventListener('timeupdate', () => {
  const time = audioPlayer.currentTime;
  const index = currentSegments.findIndex((segment) => time >= segment.start_sec && time < segment.end_sec);
  document.querySelectorAll('.timeline-segment.active').forEach((node) => node.classList.remove('active'));
  if (index >= 0) {
    const node = document.querySelector(`.timeline-segment[data-index="${index}"]`);
    node?.classList.add('active');
    $('#nowChord').textContent = currentSegments[index].display;
  }
});

$('#newAnalysis').addEventListener('click', () => {
  resultSection.hidden = true; progressPanel.hidden = true; input.value = ''; analyzeButton.disabled = true;
  $('#dropTitle').textContent = '오디오 파일을 놓으세요'; $('#fileMeta').textContent = '또는 클릭해 MP3/WAV 선택';
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

Promise.all([fetch('/healthz'), fetch('/api/info')]).then(async ([healthResponse, infoResponse]) => {
  if (!healthResponse.ok || !infoResponse.ok) throw new Error();
  const health = await healthResponse.json(); const info = await infoResponse.json();
  const status = $('#systemStatus'); status.classList.add('ready');
  status.innerHTML = `<span class="pulse"></span>${health.profile.toUpperCase()} READY`;
  $('#limitText').textContent = `MAX ${info.max_upload_mb} MB`;
}).catch(() => { $('#systemStatus').innerHTML = '<span class="pulse"></span>MODEL OFFLINE'; });

