"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const videos = [$('cozy-video'), $('comfy-video')];
  const covers = [...document.querySelectorAll('[data-play-cover]')];
  const hardwareSelect = $('hardware-select');
  const inputSelect = $('input-select');
  const playButton = $('play-button');
  const restartButton = $('restart-button');
  const seek = $('seek-control');
  const speed = $('speed-select');
  const audio = $('audio-select');
  const status = $('player-status');
  const internalSeek = new WeakSet();
  let data;
  let selection = [];
  let loaded = false;
  let playPending = false;
  let selectionVersion = 0;
  let animation = 0;
  let dragging = false;
  let settingAudio = false;
  let pendingSeek = null;

  const engineKey = (name) => String(name).toLowerCase().includes('comfy') ? 'comfy' : String(name).toLowerCase().includes('cozy') ? 'cozy' : '';
  const hardwareName = (hardware, count) => `${count} × ${hardware === 'h100' ? 'H100 SXM 80 GB' : 'RTX 5090'}`;
  const formatSeconds = (seconds) => {
    if (!Number.isFinite(seconds)) return '—';
    const rounded = Math.round(seconds * 10);
    const minutes = Math.floor(rounded / 600);
    const remainder = ((rounded % 600) / 10).toFixed(1);
    return minutes ? `${minutes}m ${remainder.padStart(4, '0')}s` : `${remainder}s`;
  };
  const clock = (seconds) => `${Math.floor(Math.max(0, seconds) / 60)}:${String(Math.floor(Math.max(0, seconds) % 60)).padStart(2, '0')}`;
  const assetPath = (path, directory) => {
    if (typeof path !== 'string' || !path.startsWith(`${directory}/`) || path.split('/').includes('..') || !/^[a-zA-Z0-9_./-]+$/.test(path)) throw new Error('Invalid media path in comparison data.');
    return path;
  };
  const report = (message, error = false) => {
    status.textContent = message;
    status.classList.toggle('is-error', error);
  };
  const duration = () => {
    const lengths = videos.map((video) => video.duration).filter((value) => Number.isFinite(value) && value > 0);
    return lengths.length ? Math.min(...lengths) : 362 / 24;
  };
  const updateTime = () => {
    const end = duration();
    const current = videos[0].currentTime || 0;
    seek.max = String(end);
    if (!dragging) seek.value = String(Math.min(current, end));
    seek.setAttribute('aria-valuetext', `${current.toFixed(1)} seconds of ${end.toFixed(1)} seconds`);
    $('time-display').textContent = `${clock(current)} / ${clock(end)}`;
  };
  const pauseTogether = () => {
    videos.forEach((video) => { if (!video.paused) video.pause(); });
    playButton.textContent = 'Play together';
    if (animation) cancelAnimationFrame(animation);
    animation = 0;
  };
  const seekVideo = (video, time) => {
    if (video.readyState < 1 || Math.abs(video.currentTime - time) < 0.015) return;
    internalSeek.add(video);
    video.currentTime = time;
  };
  const seekTogether = (time) => {
    const target = Math.max(0, Math.min(time, duration()));
    videos.forEach((video) => seekVideo(video, target));
    updateTime();
  };
  const applyAudio = () => {
    settingAudio = true;
    videos[0].muted = audio.value !== 'cozy';
    videos[1].muted = audio.value !== 'comfy';
    settingAudio = false;
  };
  const tick = () => {
    animation = 0;
    if (videos[0].paused || videos[1].paused) return;
    if (!dragging && !videos.some((video) => video.seeking) && Math.abs(videos[0].currentTime - videos[1].currentTime) > 0.12) seekVideo(videos[1], videos[0].currentTime);
    updateTime();
    animation = requestAnimationFrame(tick);
  };
  const loadSelected = () => {
    if (loaded) return;
    loaded = true;
    videos.forEach((video, index) => {
      video.src = assetPath(selection[index].video, 'media');
      video.controls = true;
      video.playbackRate = Number(speed.value);
      covers[index].hidden = true;
    });
    applyAudio();
  };
  const playTogether = async () => {
    if (playPending || selection.length !== 2) return;
    const version = selectionVersion;
    playPending = true;
    loadSelected();
    if (videos.some((video) => video.ended)) seekTogether(0);
    report('Loading the selected two clips…');
    const result = await Promise.allSettled(videos.map((video) => video.play()));
    if (version !== selectionVersion) return;
    playPending = false;
    if (result.some((item) => item.status === 'rejected')) {
      pauseTogether();
      report('Playback could not start. Try Play together again or download the original clips.', true);
      return;
    }
    playButton.textContent = 'Pause together';
    report('Playing together. Audio comes from the selected source only.');
    if (!animation) animation = requestAnimationFrame(tick);
  };

  function selectPair() {
    selectionVersion += 1;
    playPending = false;
    pauseTogether();
    loaded = false;
    pendingSeek = null;
    seek.disabled = true;
    seek.value = '0';
    const [hardware, count] = hardwareSelect.value.split(':');
    const input = Number(inputSelect.value);
    const pair = data.artifacts.filter((item) => item.hardware === hardware && Number(item.gpu_count) === Number(count) && Number(item.input) === input && item.mode === 'turbo');
    selection = ['cozy', 'comfy'].map((engine) => pair.find((item) => engineKey(item.engine) === engine));
    if (selection.some((item) => !item)) {
      selection = [];
      playButton.disabled = restartButton.disabled = true;
      covers.forEach((cover) => { cover.disabled = true; });
      report('This comparison pair is not available.', true);
      return;
    }
    videos.forEach((video, index) => {
      video.pause();
      video.removeAttribute('src');
      video.load();
      video.controls = false;
      internalSeek.delete(video);
      const item = selection[index];
      const prefix = index ? 'comfy' : 'cozy';
      video.poster = assetPath(item.poster, 'posters');
      covers[index].hidden = false;
      covers[index].disabled = false;
      $(`${prefix}-time`).textContent = `${formatSeconds(Number(item.seconds))} to generate`;
      $(`${prefix}-attention`).textContent = item.attention || 'Attention settings in data.json';
      $(`${prefix}-encoder`).textContent = item.text_encoder || 'See methodology';
      $(`${prefix}-hash`).textContent = item.sha256 || 'See data.json';
      const link = $(`${prefix}-download`);
      link.href = assetPath(item.video, 'media');
      link.hidden = false;
      $(`${prefix}-size`).textContent = Number.isFinite(item.bytes) ? `${(item.bytes / 1024 / 1024).toFixed(1)} MiB` : '';
    });
    const inputData = data.inputs.find((item) => Number(item.id) === input);
    $('prompt-text').textContent = inputData?.prompt || '';
    $('seed-label').textContent = inputData ? `Seed ${inputData.seed}` : '';
    playButton.disabled = restartButton.disabled = false;
    applyAudio();
    updateTime();
    report('Ready. Only the selected two clips load when you press play.');
  }

  function renderResults() {
    if (!Array.isArray(data.results) || data.results.length !== 8) return;
    const fragment = document.createDocumentFragment();
    const order = (row) => (row.hardware === 'h100' ? 100 : Number(row.gpu_count) * 2) + (row.mode === 'regular' ? 1 : 0);
    [...data.results].sort((a, b) => order(a) - order(b)).forEach((row) => {
      const cozy = Number(row.cozy_seconds), comfy = Number(row.comfy_seconds);
      if (!(cozy > 0 && comfy > 0)) throw new Error('Invalid result timing.');
      const tr = document.createElement('tr');
      [hardwareName(row.hardware, row.gpu_count), row.mode === 'turbo' ? 'Turbo' : 'Regular', formatSeconds(cozy), formatSeconds(comfy), `${((comfy - cozy) / comfy * 100).toFixed(1)}%`].forEach((value, index) => {
        const cell = document.createElement(index === 0 ? 'th' : 'td');
        cell.textContent = value;
        if (index === 0) cell.scope = 'row';
        if (index > 1) cell.classList.add('numeric');
        if (index === 2 && cozy <= comfy) cell.classList.add('winner');
        if (index === 4) cell.classList.add('saving');
        tr.append(cell);
      });
      fragment.append(tr);
    });
    $('results-body').replaceChildren(fragment);
    const hero = data.results.find((row) => row.hardware === 'rtx5090' && Number(row.gpu_count) === 8 && row.mode === 'turbo');
    if (hero) {
      $('hero-seconds').textContent = Number(hero.cozy_seconds).toFixed(1);
      $('hero-comfy').textContent = `${Number(hero.comfy_seconds).toFixed(1)}s`;
      $('hero-saving').textContent = `${((hero.comfy_seconds - hero.cozy_seconds) / hero.comfy_seconds * 100).toFixed(1)}%`;
    }
    if (data.transport?.cozy && data.transport?.comfy) {
      const body = document.createDocumentFragment();
      for (const [key, name] of [['cozy', 'Cozy'], ['comfy', 'ComfyUI']]) {
        for (const denoise of [false, true]) {
          const arm = data.transport[key];
          const peer = Number(arm[denoise ? 'peer_denoise_seconds' : 'peer_seconds']);
          const host = Number(arm[denoise ? 'host_denoise_seconds' : 'host_seconds']);
          if (!(peer > 0 && host > 0)) continue;
          const tr = document.createElement('tr');
          [`${name} · ${denoise ? 'denoising' : 'complete inference'}`, `${peer.toFixed(3)}s`, `${host.toFixed(3)}s`, `${((host - peer) / host * 100).toFixed(1)}%`].forEach((value, index) => {
            const cell = document.createElement(index ? 'td' : 'th');
            cell.textContent = value;
            if (!index) cell.scope = 'row'; else cell.classList.add('numeric');
            if (index === 3) cell.classList.add('saving');
            tr.append(cell);
          });
          body.append(tr);
        }
      }
      if (body.childNodes.length === 4) $('transport-body').replaceChildren(body);
    }
  }

  hardwareSelect.addEventListener('change', selectPair);
  inputSelect.addEventListener('change', selectPair);
  playButton.addEventListener('click', () => videos.some((video) => !video.paused) ? pauseTogether() : playTogether());
  covers.forEach((cover) => cover.addEventListener('click', playTogether));
  restartButton.addEventListener('click', () => {
    if (!loaded) return playTogether();
    seekTogether(0);
    playTogether();
  });
  seek.addEventListener('pointerdown', () => { dragging = true; });
  seek.addEventListener('input', () => {
    pendingSeek = Number(seek.value);
    seekTogether(pendingSeek);
  });
  seek.addEventListener('change', () => { dragging = false; pendingSeek = null; updateTime(); });
  seek.addEventListener('pointerup', () => { dragging = false; });
  seek.addEventListener('pointercancel', () => { dragging = false; });
  speed.addEventListener('change', () => videos.forEach((video) => { video.playbackRate = Number(speed.value); }));
  audio.addEventListener('change', applyAudio);
  videos.forEach((video, index) => {
    video.addEventListener('loadedmetadata', () => {
      if (videos.every((item) => item.readyState >= 1)) seek.disabled = false;
      if (pendingSeek !== null) seekVideo(video, pendingSeek);
      updateTime();
    });
    video.addEventListener('timeupdate', updateTime);
    video.addEventListener('play', () => { if (!playPending && videos[1 - index].paused) playTogether(); });
    video.addEventListener('pause', () => {
      if (!playPending) {
        pauseTogether();
        if (loaded) report(video.ended ? 'Finished. Press Restart to compare again.' : 'Paused. Seek or press Play together to continue.');
      }
    });
    video.addEventListener('ended', () => { pauseTogether(); report('Finished. Press Restart to compare again.'); });
    video.addEventListener('seeking', () => {
      if (!internalSeek.has(video) && loaded) seekVideo(videos[1 - index], video.currentTime);
    });
    video.addEventListener('seeked', () => { internalSeek.delete(video); updateTime(); });
    video.addEventListener('volumechange', () => {
      if (!settingAudio && !video.muted) { audio.value = index ? 'comfy' : 'cozy'; applyAudio(); }
    });
    video.addEventListener('error', () => {
      if (!loaded) return;
      pauseTogether();
      report('A selected clip could not load. Check your connection or download the original video.', true);
    });
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pauseTogether(); });

  async function initialize() {
    try {
      const response = await fetch('data.json', { credentials: 'same-origin' });
      if (!response.ok) throw new Error('Comparison data is unavailable.');
      data = await response.json();
      if (!Array.isArray(data.inputs) || !Array.isArray(data.artifacts)) throw new Error('Invalid comparison data.');
      renderResults();
      inputSelect.replaceChildren(...data.inputs.map((item) => {
        const option = document.createElement('option');
        option.value = String(item.id); option.textContent = item.label; return option;
      }));
      inputSelect.value = '1';
      hardwareSelect.value = 'rtx5090:1';
      hardwareSelect.disabled = inputSelect.disabled = false;
      selectPair();
    } catch (error) {
      report('The comparison data could not load. Refresh to try again; the timing tables and data download remain available.', true);
      console.error('Benchmark data:', error);
    }
  }
  initialize();
})();
