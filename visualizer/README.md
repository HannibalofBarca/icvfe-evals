# Visualizer

A read-only, multi-source clone of the "value vs time" chart from the
[ICL Manual Annotation tool](../../Manual%20Annotation) (`server.py` +
`static/{index.html,app.js,style.css}` there), adapted to show all five
evaluator curves for an episode at once — **Human, RoboDopamine, RoboMeter
(online FT), ICVFE, pi0.6-recap** — as toggleable overlaid lines, instead of
one editable curve.

Differences from the tool it's cloned from:

- **Toggles instead of editing.** Each source is a clickable legend chip
  (checkbox-like) that shows/hides its line on the chart and its live
  value readout at the current playhead position. Nothing here writes
  annotations — there's no save/delete/vocab/outcome-correction UI.
- **RoboMeter is the online fine-tuned export**
  (`finetuned_robometer_online_icl_demo_dataset_continuous`), not the base
  one `../analysis/pipeline.py` uses elsewhere — ICVFE and pi0.6-recap are
  both online-only models (`../analysis/a3_pipeline.py`'s docstring), so
  this keeps the comparison consistent.
- **Video is streamed, not local.** This project's own exports don't carry
  video, only the progress-value curves already used by
  `../analysis/pipeline.py` — but the episodes are drawn from a Hugging
  Face lerobot dataset that does. Only the single "main cam" (zed) feed is
  shown, streamed directly from HF (byte-range requests, same technique
  `../../Manual Annotation/server.py` uses — no local video files or
  proxying). When video is available for an episode it's the playback
  clock: the scrubber/chart playhead follow the `<video>` element's own
  `timeupdate`. Episodes with no resolvable video (see
  `server.py`'s `load_video_episode_index_map()`) fall back to a synthetic
  playhead animated at wall-clock speed via `requestAnimationFrame`.
- **ICVFE and pi0.6-recap show raw + EMA.** Both draw two lines under one
  legend chip: a faded, thin raw curve underlaid beneath an EMA-smoothed
  one on top, since their raw per-frame predictions are too noisy to read
  at full-episode width on their own. The smoothing constant is a live
  slider (`EMA α`, default 0.05) — dragging it recomputes both curves
  client-side from their `raw_progress` arrays, no server round-trip.
- **ICVFE is an average.** ICVFE scores each query episode once per
  in-context example drawn from its task (9 replicates per episode in this
  export). The curve shown is the mean prediction across those replicates
  — the legend chip says how many. 3 of the 243 episodes have no ICVFE
  curve at all (outside its eval set); their chip is greyed out.
- **pi0.6-recap is single-pass.** Unlike ICVFE it has no in-context
  replicates (one query pass per episode), so its raw curve is the model's
  own single prediction. It covers the same 240-episode subset as ICVFE;
  episodes outside that subset show a greyed-out chip too.
- **Own tiny backend, no new dependency.** `server.py` uses only the Python
  standard library (`http.server`) — it imports `../analysis/pipeline.py`
  for archive extraction and cache paths, so no FastAPI/uvicorn install is
  needed just to look at the data.

## Running it

```
cd visualizer
python server.py            # http://127.0.0.1:8000 (extracts data/*.zip on first run if analysis/cache/ isn't populated yet)
python server.py --port 8123 --host 0.0.0.0
```

Open the printed URL, pick a task, and use the legend chips to toggle
sources on/off. Click or drag on the chart (or the scrubber underneath) to
scrub; space bar plays/pauses; arrow keys step a frame (shift+arrow steps
10). Drag the `EMA α` slider in the transport row to retune how heavily
ICVFE/recap are smoothed.
