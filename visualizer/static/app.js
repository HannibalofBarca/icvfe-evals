(() => {
  "use strict";

  const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];
  const SOURCES = ["human", "robodopamine", "robometer", "icvfe"];
  const SOURCE_LABELS = { human: "Human", robodopamine: "RoboDopamine", robometer: "RoboMeter", icvfe: "ICVFE" };

  const S = {
    tasks: [],
    currentTaskIndex: null,
    taskEpisodes: [],
    episode: null,
    fps: 30,
    numFrames: 0,
    duration: 0,
    visible: { human: true, robodopamine: true, robometer: true, icvfe: true },
    playbackRate: 1,
    isPlaying: false,
    currentTime: 0,
    rafId: null,
    lastTs: null,
  };

  // ---------------------------------------------------------------- DOM refs
  const $ = (id) => document.getElementById(id);
  const taskSelect = $("taskSelect");
  const searchBox = $("searchBox");
  const posLabel = $("posLabel");
  const prevBtn = $("prevBtn");
  const nextBtn = $("nextBtn");
  const datasetOutcomeTag = $("datasetOutcomeTag");
  const episodeUidTag = $("episodeUidTag");
  const themeToggleBtn = $("themeToggleBtn");

  const legendRow = $("legendRow");
  const valueChartSvg = $("valueChart");
  const scrubber = $("scrubber");
  const scrubFill = $("scrubFill");
  const scrubPlayhead = $("scrubPlayhead");
  const timeText = $("timeText");
  const frameText = $("frameText");

  const playBtn = $("playBtn");
  const speedButtonsEl = $("speedButtons");
  const toastEl = $("toast");

  // ---------------------------------------------------------------- theme
  function applyTheme(theme) {
    if (theme === "light") document.documentElement.setAttribute("data-theme", "light");
    else document.documentElement.removeAttribute("data-theme");
    themeToggleBtn.textContent = theme === "light" ? "☀️" : "🌙";
    if (S.episode) renderChart();
  }
  applyTheme(localStorage.getItem("theme") === "light" ? "light" : "dark");
  themeToggleBtn.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
    localStorage.setItem("theme", next);
    applyTheme(next);
  });

  // -------------------------------------------------------------------- API
  async function api(path) {
    const res = await fetch(path);
    if (!res.ok) {
      let msg = res.statusText;
      try {
        const j = await res.json();
        msg = j.detail || msg;
      } catch (_) {}
      throw new Error(msg);
    }
    return res.json();
  }

  function toast(msg, isError) {
    toastEl.textContent = msg;
    toastEl.classList.toggle("error", !!isError);
    toastEl.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => toastEl.classList.remove("show"), 2200);
  }

  function seriesColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(`--s-${name}`).trim();
  }

  function formatTime(seconds) {
    if (!isFinite(seconds)) seconds = 0;
    const totalTenths = Math.round(Math.max(0, seconds) * 10);
    const m = Math.floor(totalTenths / 600);
    const s = (totalTenths % 600) / 10;
    return `${m}:${s.toFixed(1).padStart(4, "0")}`;
  }

  // ------------------------------------------------------------- tasks list
  async function loadTasks() {
    S.tasks = await api("/api/tasks");
    renderTaskSelect();
    if (S.tasks.length) await selectTask(S.tasks[0].task_index);
  }

  function renderTaskSelect() {
    taskSelect.innerHTML = S.tasks
      .map((t) => `<option value="${t.task_index}">${escapeHtml(t.task)} (${t.n_episodes})</option>`)
      .join("");
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  async function selectTask(taskIndex) {
    S.currentTaskIndex = taskIndex;
    taskSelect.value = String(taskIndex);
    S.taskEpisodes = await api(`/api/tasks/${taskIndex}/episodes`);
    await loadEpisode(taskIndex, 1);
  }

  taskSelect.addEventListener("change", (e) => {
    if (e.target.value === "") return;
    selectTask(parseInt(e.target.value, 10)).catch((err) => toast(err.message, true));
  });

  // ----------------------------------------------------------------- nav
  async function navigateByOffset(delta) {
    if (!S.episode) return;
    const newPos = Math.max(1, Math.min(S.episode.position + delta, S.episode.total_in_task));
    if (newPos === S.episode.position) return;
    await loadEpisode(S.currentTaskIndex, newPos);
  }
  prevBtn.addEventListener("click", () => navigateByOffset(-1));
  nextBtn.addEventListener("click", () => navigateByOffset(1));

  function jumpToSearch() {
    if (!S.episode) return;
    const raw = parseInt(searchBox.value, 10);
    if (isNaN(raw)) return;
    const pos = Math.max(1, Math.min(raw, S.episode.total_in_task));
    loadEpisode(S.currentTaskIndex, pos).catch((err) => toast(err.message, true));
  }
  searchBox.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      jumpToSearch();
    }
  });

  // ----------------------------------------------------------- episode load
  async function loadEpisode(taskIndex, position) {
    stopPlayback();
    const ep = await api(`/api/episode/${taskIndex}/${position}`);
    S.episode = ep;
    S.fps = ep.fps;
    S.numFrames = ep.num_frames;
    S.duration = ep.num_frames / ep.fps;
    S.currentTime = 0;

    posLabel.textContent = `${ep.position} / ${ep.total_in_task}`;
    searchBox.value = ep.position;
    searchBox.placeholder = String(ep.total_in_task);
    episodeUidTag.textContent = ep.episode_uid;

    datasetOutcomeTag.classList.remove("ds-success", "ds-fail");
    if (ep.dataset_success === "success") {
      datasetOutcomeTag.textContent = "Dataset: success";
      datasetOutcomeTag.classList.add("ds-success");
    } else if (ep.dataset_success === "fail") {
      datasetOutcomeTag.textContent = "Dataset: fail";
      datasetOutcomeTag.classList.add("ds-fail");
    } else {
      datasetOutcomeTag.textContent = "Dataset: unknown";
    }

    renderLegend();
    renderChart();
    seekTo(0);
  }

  // ------------------------------------------------------------- legend
  function renderLegend() {
    legendRow.innerHTML = SOURCES.map((src) => {
      const s = S.episode.series[src];
      const off = !S.visible[src] ? " off" : "";
      if (!s.available) {
        return `<div class="source-chip disabled" data-src="${src}">
          <span class="swatch" style="background:${seriesColor(src)}"></span>
          <span class="name">${SOURCE_LABELS[src]}</span>
          <span class="unavailable">not available for this episode</span>
        </div>`;
      }
      const note = s.n_reps ? `<span class="note">avg of ${s.n_reps} in-context replicates</span>` : "";
      return `<div class="source-chip${off}" data-src="${src}">
        <span class="swatch" style="background:${seriesColor(src)}"></span>
        <span class="name">${SOURCE_LABELS[src]}</span>
        <span class="value" data-value-for="${src}">–</span>
        ${note}
      </div>`;
    }).join("");

    legendRow.querySelectorAll(".source-chip:not(.disabled)").forEach((chip) => {
      chip.addEventListener("click", () => {
        const src = chip.dataset.src;
        S.visible[src] = !S.visible[src];
        chip.classList.toggle("off", !S.visible[src]);
        renderChart();
        updateReadouts(S.currentTime);
      });
    });
  }

  // ------------------------------------------------------------------ chart
  const CHART_PAD_LEFT = 46, CHART_PAD_RIGHT = 16, CHART_PAD_TOP = 14, CHART_PAD_BOTTOM = 30;
  let chartDims = { w: 1000, h: 400, plotW: 1000 - CHART_PAD_LEFT - CHART_PAD_RIGHT, plotH: 400 - CHART_PAD_TOP - CHART_PAD_BOTTOM };

  function measureChart() {
    const rect = valueChartSvg.getBoundingClientRect();
    const w = Math.max(200, Math.round(rect.width) || 1000);
    const h = Math.max(160, Math.round(rect.height) || 400);
    chartDims = { w, h, plotW: w - CHART_PAD_LEFT - CHART_PAD_RIGHT, plotH: h - CHART_PAD_TOP - CHART_PAD_BOTTOM };
    return chartDims;
  }

  function niceTimeStep(duration) {
    const target = Math.max(duration / 8, 0.5);
    const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900];
    for (const s of steps) if (target <= s) return s;
    return 1800;
  }

  function chartXForTime(t) {
    const frac = S.duration ? t / S.duration : 0;
    return CHART_PAD_LEFT + frac * chartDims.plotW;
  }
  function chartYForValue(v) {
    return CHART_PAD_TOP + (1 - Math.max(0, Math.min(100, v)) / 100) * chartDims.plotH;
  }
  function timeAtClientX(clientX, rectEl) {
    const rect = rectEl.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return frac * S.duration;
  }

  function renderChart() {
    if (!S.episode) return;
    measureChart();
    valueChartSvg.setAttribute("viewBox", `0 0 ${chartDims.w} ${chartDims.h}`);
    const chartW = chartDims.w, chartH = chartDims.h;
    const parts = [];

    for (const gv of [0, 25, 50, 75, 100]) {
      const y = chartYForValue(gv);
      parts.push(`<line class="chart-grid" x1="${CHART_PAD_LEFT}" y1="${y.toFixed(1)}" x2="${chartW - CHART_PAD_RIGHT}" y2="${y.toFixed(1)}" />`);
      parts.push(`<text class="chart-axis-label" x="${CHART_PAD_LEFT - 8}" y="${(y + 3).toFixed(1)}" text-anchor="end">${gv}</text>`);
    }
    if (S.duration > 0) {
      const step = niceTimeStep(S.duration);
      for (let t = 0; t <= S.duration + 1e-6; t += step) {
        const x = chartXForTime(t);
        parts.push(`<line class="chart-grid" x1="${x.toFixed(1)}" y1="${CHART_PAD_TOP}" x2="${x.toFixed(1)}" y2="${chartH - CHART_PAD_BOTTOM}" />`);
        parts.push(`<text class="chart-axis-label" x="${x.toFixed(1)}" y="${chartH - CHART_PAD_BOTTOM + 18}" text-anchor="middle">${formatTime(t)}</text>`);
      }
    }
    parts.push(`<line class="chart-axis" x1="${CHART_PAD_LEFT}" y1="${CHART_PAD_TOP}" x2="${CHART_PAD_LEFT}" y2="${chartH - CHART_PAD_BOTTOM}" />`);
    parts.push(`<line class="chart-axis" x1="${CHART_PAD_LEFT}" y1="${chartH - CHART_PAD_BOTTOM}" x2="${chartW - CHART_PAD_RIGHT}" y2="${chartH - CHART_PAD_BOTTOM}" />`);

    for (const src of SOURCES) {
      const s = S.episode.series[src];
      if (!s.available || !S.visible[src]) continue;
      const progress = s.progress;
      const n = progress.length;
      const fps = S.fps;
      const pts = [];
      for (let i = 0; i < n; i++) {
        const t = i / fps;
        pts.push(`${chartXForTime(t).toFixed(1)},${chartYForValue(progress[i]).toFixed(1)}`);
      }
      parts.push(`<polyline class="chart-line" points="${pts.join(" ")}" stroke="${seriesColor(src)}" />`);
    }

    parts.push(`<line id="chartPlayhead" class="chart-playhead" x1="${CHART_PAD_LEFT}" y1="${CHART_PAD_TOP}" x2="${CHART_PAD_LEFT}" y2="${chartH - CHART_PAD_BOTTOM}" />`);
    for (const src of SOURCES) {
      const s = S.episode.series[src];
      if (!s.available || !S.visible[src]) continue;
      parts.push(`<circle id="dot-${src}" class="chart-playhead-dot" r="4" fill="${seriesColor(src)}" cx="${CHART_PAD_LEFT}" cy="${CHART_PAD_TOP}" />`);
    }

    valueChartSvg.innerHTML = parts.join("");
    updateChartPlayhead(S.currentTime);
  }

  let chartResizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(chartResizeTimer);
    chartResizeTimer = setTimeout(() => {
      if (S.episode) renderChart();
    }, 150);
  });

  function valueAtTime(progress, t) {
    const idx = Math.max(0, Math.min(progress.length - 1, Math.round(t * S.fps)));
    return progress[idx];
  }

  function updateChartPlayhead(t) {
    const line = document.getElementById("chartPlayhead");
    if (!line) return;
    const x = chartXForTime(t).toFixed(1);
    line.setAttribute("x1", x);
    line.setAttribute("x2", x);
    for (const src of SOURCES) {
      const dot = document.getElementById(`dot-${src}`);
      if (!dot) continue;
      const s = S.episode.series[src];
      const y = chartYForValue(valueAtTime(s.progress, t));
      dot.setAttribute("cx", x);
      dot.setAttribute("cy", y.toFixed(1));
    }
  }

  function updateReadouts(t) {
    for (const src of SOURCES) {
      const el = legendRow.querySelector(`[data-value-for="${src}"]`);
      if (!el) continue;
      const s = S.episode.series[src];
      el.textContent = valueAtTime(s.progress, t).toFixed(1);
    }
  }

  let chartScrubbing = false;
  valueChartSvg.addEventListener("pointerdown", (e) => {
    if (!S.episode) return;
    chartScrubbing = true;
    valueChartSvg.setPointerCapture(e.pointerId);
    seekTo(timeAtClientX(e.clientX, valueChartSvg));
  });
  valueChartSvg.addEventListener("pointermove", (e) => {
    if (chartScrubbing) seekTo(timeAtClientX(e.clientX, valueChartSvg));
  });
  valueChartSvg.addEventListener("pointerup", () => (chartScrubbing = false));

  // -------------------------------------------------------------- scrubber
  let scrubbing = false;
  scrubber.addEventListener("pointerdown", (e) => {
    if (!S.episode) return;
    scrubbing = true;
    scrubber.setPointerCapture(e.pointerId);
    seekTo(timeAtClientX(e.clientX, scrubber));
  });
  scrubber.addEventListener("pointermove", (e) => {
    if (scrubbing) seekTo(timeAtClientX(e.clientX, scrubber));
  });
  scrubber.addEventListener("pointerup", () => (scrubbing = false));

  // ----------------------------------------------------------- playhead / seek
  function seekTo(t) {
    S.currentTime = Math.max(0, Math.min(t, S.duration || t));
    const frac = S.duration ? S.currentTime / S.duration : 0;
    scrubPlayhead.style.left = `${frac * 100}%`;
    scrubFill.style.width = `${frac * 100}%`;
    const frame = Math.round(S.currentTime * S.fps);
    timeText.textContent = formatTime(S.currentTime);
    frameText.textContent = `frame ${frame} / ${Math.max(0, S.numFrames - 1)}`;
    updateChartPlayhead(S.currentTime);
    updateReadouts(S.currentTime);
  }

  function stepFrames(n) {
    stopPlayback();
    seekTo(S.currentTime + n / S.fps);
  }
  $("gotoStartBtn").addEventListener("click", () => { stopPlayback(); seekTo(0); });
  $("gotoEndBtn").addEventListener("click", () => { stopPlayback(); seekTo(S.duration); });
  $("stepBack10").addEventListener("click", () => stepFrames(-10));
  $("stepBackFrame").addEventListener("click", () => stepFrames(-1));
  $("stepFwdFrame").addEventListener("click", () => stepFrames(1));
  $("stepFwd10").addEventListener("click", () => stepFrames(10));

  // ------------------------------------------------------ playback (no video)
  // There's no video to sync against in this export, so "play" advances a
  // virtual clock at wall-clock speed (scaled by S.playbackRate) via rAF,
  // driving the same seekTo() the scrubber and chart clicks use.
  function playTick(ts) {
    if (!S.isPlaying) return;
    if (S.lastTs != null) {
      const dt = (ts - S.lastTs) / 1000;
      seekTo(S.currentTime + dt * S.playbackRate);
    }
    S.lastTs = ts;
    if (S.currentTime >= S.duration) {
      stopPlayback();
      return;
    }
    S.rafId = requestAnimationFrame(playTick);
  }

  function togglePlay() {
    if (!S.episode) return;
    if (S.isPlaying) {
      stopPlayback();
      return;
    }
    if (S.currentTime >= S.duration) seekTo(0);
    S.isPlaying = true;
    S.lastTs = null;
    updatePlayButton();
    S.rafId = requestAnimationFrame(playTick);
  }
  function stopPlayback() {
    S.isPlaying = false;
    if (S.rafId != null) cancelAnimationFrame(S.rafId);
    S.rafId = null;
    updatePlayButton();
  }
  function updatePlayButton() {
    playBtn.textContent = S.isPlaying ? "⏸" : "▶";
  }
  playBtn.addEventListener("click", togglePlay);

  speedButtonsEl.innerHTML = SPEEDS.map((s) => `<button data-speed="${s}">${s}x</button>`).join("");
  speedButtonsEl.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      S.playbackRate = parseFloat(btn.dataset.speed);
      speedButtonsEl.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
    });
    if (parseFloat(btn.dataset.speed) === 1) btn.classList.add("active");
  });

  // -------------------------------------------------------------- keyboard
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
    if (e.code === "Space") {
      e.preventDefault();
      togglePlay();
    } else if (e.code === "ArrowLeft") {
      stepFrames(e.shiftKey ? -10 : -1);
    } else if (e.code === "ArrowRight") {
      stepFrames(e.shiftKey ? 10 : 1);
    } else if (e.code === "Home") {
      stopPlayback();
      seekTo(0);
    } else if (e.code === "End") {
      stopPlayback();
      seekTo(S.duration);
    }
  });

  // ------------------------------------------------------------------ boot
  loadTasks().catch((e) => toast("Failed to load tasks: " + e.message, true));
})();
