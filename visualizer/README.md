# Visualizer

A read-only, multi-source clone of the "value vs time" chart from the
[ICL Manual Annotation tool](../../Manual%20Annotation) (`server.py` +
`static/{index.html,app.js,style.css}` there), adapted to show all four
evaluator curves for an episode at once — **Human, RoboDopamine, RoboMeter,
ICVFE** — as toggleable overlaid lines, instead of one editable curve.

Differences from the tool it's cloned from:

- **Toggles instead of editing.** Each source is a clickable legend chip
  (checkbox-like) that shows/hides its line on the chart and its live
  value readout at the current playhead position. Nothing here writes
  annotations — there's no save/delete/vocab/outcome-correction UI.
- **No video.** This project's exports don't carry video, only the
  progress-value curves already used by `../analysis/pipeline.py`. "Play"
  instead animates a virtual playhead over the chart at wall-clock speed
  (via `requestAnimationFrame`), scaled by the speed buttons — same
  scrubber/frame-step/keyboard-shortcut feel, just not synced to a
  `<video>` element.
- **ICVFE is an average.** ICVFE scores each query episode once per
  in-context example drawn from its task (9 replicates per episode in this
  export). The curve shown is the mean prediction across those replicates
  — the legend chip says how many. 3 of the 243 episodes have no ICVFE
  curve at all (outside its eval set); their chip is greyed out.
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
10).
