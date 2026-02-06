const API_BASE = import.meta.env.VITE_API_URL || '';

export async function predictImage(file) {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${API_BASE}/api/predict/image`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }

  return res.json();
}

export async function predictVideo(file, sampleRate = 25, maxFrames = 200) {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(
    `${API_BASE}/api/predict/video?sample_rate=${sampleRate}&max_frames=${maxFrames}`,
    { method: 'POST', body: form }
  );

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }

  return res.json();
}

export async function predictBatch(files) {
  const form = new FormData();
  files.forEach((f) => form.append('files', f));

  const res = await fetch(`${API_BASE}/api/predict/batch`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }

  return res.json();
}

export async function getPhases() {
  const res = await fetch(`${API_BASE}/api/phases`);
  return res.json();
}

export async function healthCheck() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}
