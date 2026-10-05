# ICVFE Evals — Analysis

This repo holds the raw evaluation exports (`data/`), an analysis pipeline
(`analysis/`) that measures agreement between three automatic progress
evaluators (and each other) against human ground-truth annotation across 27
manipulation tasks, and a browser visualizer (`visualizer/`) for inspecting
any single episode's four curves side by side.

## Data sources (`data/`)

| Archive | What it is | Role |
|---|---|---|
| `manual_icl_demo_dataset_continuous.zip` | Human-annotated per-frame task progress (0–100) | **Human ground truth** |
| `zs_robodopamine_icl_demo_dataset_continuous.zip` | Zero-shot RoboDopamine per-frame progress (0–100) | Evaluator: **robodopamine** |
| `finetuned_robometer_icl_demo_dataset_continuous.zip` | Fine-tuned RoboMeter per-frame progress (0–100) | Evaluator: **robometer** |
| `eval_on_vfe_icl_keep_true_annotation.tar.gz` | ICVFE in-context-learning eval output (per-trajectory target/prediction curves + precomputed metrics) | Evaluator: **icvfe** |
| `eval_recap_20000.tar.gz` | pi0.6-recap eval output at checkpoint step 20000 (per-trajectory target/prediction curves + precomputed metrics, split into `seen`/`unseen` task-novelty subdirs) | Evaluator: **recap** |

Each of the first three archives contains one JSON file per episode
(`{task_slug}_{timestamp}__{episode:06d}.json`), with a `points` array of
per-frame `progress` values (0–100 scale) keyed by a stable `episode_uid`
(`"{task_slug}_{timestamp}:{episode:06d}"`) shared across all three sources.

The ICVFE archive is structured differently: it evaluates each episode
**in-context**, i.e. it runs the model once per (context episode, query
episode) pair within a task (leave-one-out over that task's episodes). Each
such run produces a `query_eval_episode_XXXXXX.npz` curve file containing a
`target_progress` and `prediction_progress` array (0–1 scale, same frame
count as the source episode), plus a task-level/checkpoint-level
`metrics.json` listing every trajectory's context/query pairing and task.

The recap archive is structured similarly to ICVFE (per-trajectory `.npz`
curve files with `target_progress`/`prediction_progress`, plus a
`metrics.json` per split), but it's a **single no-context pass** per episode
(`context_demo_id` is always empty — leave-one-out in-context conditioning is
off), and its 240 trajectories are split into two independent `seen/` and
`unseen/` subdirs (task-novelty splits, each with its own `metrics.json` and
`curves/` dir) rather than one tree. Curve files live at
`<split>/curves/task_{task_id:03d}_{demo_id}.npz`.

## Methodology (`analysis/pipeline.py`)

For every pair of evaluators, and at three levels of aggregation, we compute
**Pearson r**, **Kendall's tau-b**, and **MAE** between the two progress
curves. MAE is reported as a **0-1 fraction of task progress**, not "points
out of 100" — the raw archives happen to store progress on a 0-100 scale,
but that's itself just one normalization choice (a percentage), not a
non-normalized "real" unit, so we divide by 100 rather than let an arbitrary
choice of scale inflate the number 100x:

- **episode** — one number per episode, computed over that episode's own
  frames (this *is* the natural unit: within one episode you have a paired
  time series of two curves, and it's the smallest unit where pooling raw
  points is appropriate — frames within an episode are highly autocorrelated,
  so they aren't independent observations of anything beyond that one episode).
- **task** — the *average of the episode-level numbers* within the task, not
  a pool of raw frames across episodes. Pooling frames across episodes would
  treat every frame as an independent observation when the episode, not the
  frame, is the actual independent sampling unit — under pooling, a longer
  episode dominates the number just by contributing more (non-independent)
  points, which isn't the intended notion of "agreement across a
  heterogeneous set of episodes."
- **total** — the average of all episode-level numbers across every task
  (every episode weighted equally, regardless of which task it belongs to).

Seven evaluator pairs are computed:

1. `robodopamine_vs_human`
2. `robometer_vs_human`
3. `icvfe_vs_human`
4. `recap_vs_human`
5. `robometer_vs_robodopamine` (cross-correlation between the two automatic
   scorers, no human curve involved)
6. `icvfe_vs_robodopamine` (cross-correlation between icvfe's prediction and
   robodopamine's curve — see "ICVFE's target is not the human curve" below,
   this is in fact what ICVFE's own eval natively scores against)
7. `recap_vs_robodopamine` (same story as `icvfe_vs_robodopamine`, see
   "recap's target is not the human curve either" below)

### Episode matching

`manual`, `finetuned_robometer`, and `zs_robodopamine` are joined on
`episode_uid`. 243 episodes are common to all three (26 episodes present in
the two automatic-scorer archives have no human annotation and are excluded
from every pairwise table, for consistency across pairs). Progress arrays
are truncated to the shorter length when frame counts disagree (in practice
they never do — 0 mismatches observed).

### ICVFE's target is not the human curve — read this before comparing raw numbers

The ICVFE npz curve files (`query_eval_episode_XXXXXX.npz`) contain a
`target_progress` array that looks like it should be the human ground truth
(the archive is literally named `eval_on_vfe_icl_keep_true_annotation`, and
its own `metrics.json` calls this `ground_truth_version:
annotation_progress_clamped_v1`). **It is not the `manual` human curve.**
Each npz's embedded `metadata_json.annotation_path` points at a file under
`zs_robodopamine_icl_demo_dataset_continuous` (verified across a random
sample spanning multiple tasks — always robodopamine, never manual), and
`target_progress` matches the corresponding `zs_robodopamine` episode's raw
progress curve to ~1e-6 (i.e. float32 rounding only — not merely correlated,
numerically the same curve). In other words: this ICVFE eval run was scored
against RoboDopamine's annotations as its "true annotation," not against the
human annotations in `manual`.

Consequences for `analysis/pipeline.py` (`build_icvfe_rows`):

- **`icvfe_vs_robodopamine`** uses `target_progress` vs `prediction_progress`
  directly from the npz — this is exactly what ICVFE was natively evaluated
  against, no extra alignment needed.
- **`icvfe_vs_human`** is computed separately: `prediction_progress` is
  aligned against the actual `manual` curve for the same `episode_uid`, using
  the npz's own `timesteps` array as the frame index into `manual`'s
  `points`. This is *not* what ICVFE's own `metrics.json`/`aggregate` numbers
  reflect — those are all against the robodopamine-derived target — so don't
  quote ICVFE's self-reported Pearson/MAE as an "ICVFE vs human" number.

Both pairs share the same ICVFE-specific aggregation, because ICVFE
evaluates each query episode multiple times (once per in-context example
drawn from the same task — 9 context replicates per query episode in this
export):

- **episode level**: for each query episode we average Pearson/Kendall/MAE
  across its context replicates. `n` = number of replicates averaged, not
  frame count.
- **task level**: the *average of the episode-level numbers* within the
  task (not frame-pooling — pooling would repeat the same target curve once
  per replicate, artificially inflating the effective sample size).
- **total level**: the average of all episode-level numbers.

This is the same episode-average aggregation rule used everywhere else in
the pipeline (see the granularity section above) — ICVFE/recap just have an
extra averaging step first (across context replicates) to get down to one
number per episode, since their npz export scores each query episode
multiple times.

### recap's target is not the human curve either

Same caveat, same verification method: `eval_recap_20000.tar.gz`'s npz
`target_progress` is robodopamine's curve for that episode (not `manual`),
per each npz's embedded `metadata_json.annotation_path`, which points at
`zs_robodopamine_icl_demo_dataset_continuous` for every trajectory. So
`recap_vs_robodopamine` uses `target_progress` vs `prediction_progress`
directly (what recap was natively scored against), and `recap_vs_human` is
computed separately by aligning `prediction_progress` against `manual` via
the npz's `timesteps`, same as `icvfe_vs_human`.

Unlike ICVFE, recap has no in-context replicates to average per episode
(`context_demo_id` is always `""`, one pass per query episode) — but
`build_recap_rows` still routes both pairs through the same
`_aggregate_icvfe_pair` aggregation (averaging episode-level numbers into
task/total, not frame-pooling) so recap's task/total numbers are aggregated
identically to ICVFE's and comparable in derivation, even though there's
nothing to average within an episode (`n`=1 replicate).

`target_progress`/`prediction_progress` already ship at 0-1, so no rescaling
is needed there; the `manual` curve (0-100) is divided by 100 wherever it's
compared against them, so all seven pairs' MAE end up on the same 0-1 scale.
ICVFE and recap each cover 240 of the 243 human-annotated episodes (3 were
not part of either eval set) and the same 27 tasks.

### Kendall's tau-b

Computed with `scipy.stats.kendalltau(..., variant="b")`, which corrects for
ties — progress curves frequently plateau (constant progress across several
frames), so tau-b rather than tau-a is the appropriate variant.

## Outputs (`analysis/output/`)

- `metrics_long.csv` — the master table: one row per
  `(level, task, group_id, pair, metric)`, all seven pairs. Filter this for
  anything.
- `episode_level_wide.csv` — one row per episode, one column per
  `pair × metric` (21 columns, 7 pairs x 3 metrics) at episode granularity.
- `task_level_wide.csv` — one row per task (27 rows), same column layout, at
  task granularity.
- `total_level_wide.csv` — one row per pair (7 rows), at total granularity.
- `report.md` — static document stitching the master figures + total/task
  tables together (see "Graphs" below).
- `vs_robodopamine/`, `vs_human/` — the same three tables/figures, scoped to
  one baseline at a time (see "Baseline-specific reports" below).
- `progress_curves/` — actual overlaid progress curves, not just agreement
  numbers (see "Progress curves" below).

## Running it

```
cd analysis
python pipeline.py            # extracts archives into analysis/cache/ (cached after first run), writes analysis/output/*.csv
python plotting.py            # reads analysis/output/*.csv, writes analysis/output/figures/*.png
python build_report.py        # writes analysis/output/report.md, embedding the figures + summary tables
python baseline_reports.py    # writes analysis/output/{vs_robodopamine,vs_human}/ (filtered tables + figures + report.md)
python progress_curves.py     # writes analysis/output/progress_curves/*.png + curve_data_*.csv

python pipeline.py --no-cache # force re-extraction (e.g. if data/ archives changed)
```

Requires `numpy`, `pandas`, `scipy`, `matplotlib`. `pipeline.py` takes
~20–30s (dominated by loading ~2,000 ICVFE + 240 recap `.npz` curve files);
the other steps are fast. `baseline_reports.py` and `progress_curves.py` both
need `pipeline.py` to have run first (they read its `output/*.csv` /
`cache/`); `progress_curves.py` re-extracts archives itself if `cache/` is
missing, same as `pipeline.py`.

**Cache staleness:** `extract_archives()` only re-extracts an archive whose
`analysis/cache/<name>/` subdir is missing or empty — it does **not** check
whether the `data/*.zip`/`*.tar.gz` file itself changed. If you replace one
of the source archives (e.g. a new `manual_icl_demo_dataset_continuous.zip`
with updated annotations) without clearing its cache subdir first, every
downstream number will silently keep using the old extracted data. Either
delete the specific stale subdir (`rm -rf analysis/cache/manual`) or run
`python pipeline.py --no-cache` to force a full re-extraction, then re-run
the rest of the chain.

## Ablations (A1–A3) and retrieval evals (B1–B2)

All five read archives from `data/demo_set_annotations/demo_set_annotations/`
(download from the
[icl-vfe-eval-results](https://huggingface.co/datasets/Hannibal52Barca/icl-vfe-eval-results)
HF dataset). Value-model ablations (A) are scored against the human (manual)
progress curves with Pearson r, Kendall tau-b and MAE at episode / task / total
granularity. Sparse evaluators are aligned by `frame_index` and linearly
interpolated onto the dense human curve, not zipped by list position.

| Script | What it tests | Output |
|---|---|---|
| `analysis/a1_pipeline.py` | **A1 — reward model roster.** Zero-shot TOPReward, GVL, RoboMeter and RoboDopamine vs. human. The only place TOPReward and GVL are scored. | `output/A1-reward model ablation/` |
| `analysis/a2_pipeline.py` | **A2 — fine-tuned vs. zero-shot.** RoboMeter FT vs. ZS and RoboDopamine FT vs. ZS (non-online exports), i.e. whether in-domain adaptation helps each family. | `output/A2-finetuned vs zeroshot/` |
| `analysis/a3_pipeline.py` | **A3 — online-eligible models vs. human.** RoboMeter FT (online), RECAP and IC-VFE (step 8800, plus causal-EMA α=0.5), i.e. the models usable at rollout time without full-episode/goal context. | `output/A3-online ft models vs human/` |
| `analysis/b1_full_run.py` (+ `b1_retrieval.py`, `b1_full_run_{tasklist,shard,merge}.py` for sharded runs) | **B1 — retrieval methods.** Per task, a fixed demo-bank episode (lowest `episode_index`) is searched for the chunk that best matches every frame of every other episode, by value (RoboMeter FT online, RECAP, IC-VFE, IC-VFE EMA; RoboDopamine ZS as an upper-bound reference), by DINO visual distance, and by a naive unweighted value + vision sum. | `output/b1_full_run/` |
| `analysis/b2_lambda_sweep.py` | **B2 — vision/value mixing weight.** Retrieval score `λ·value_dist + (1−λ)·vision_dist` for λ ∈ {0, 0.25, 0.5, 0.75, 1} over the B1 online-eligible value sources (no RoboDopamine). | `output/B2-vision value lambda sweep/`, `output/B2-lambda sweep summary/` |

B1/B2 retrieval quality is measured with the metric functions in
`analysis/b_metrics.py`. **TTC error** is the difference in normalized
time-to-completion between query and retrieved frame. **videoDTW error** is
the path-normalized DTW distance between the 30-frame DINO chunks.
**BC error** is the mean L1 distance between the two 30-step raw action chunks, and **DINO error** is the
single-frame embedding distance.

`analysis/seen_unseen_split.py` re-splits every A-series method's total-level
metrics into seen (15) vs. unseen (12) tasks, using RECAP's own fine-tuning
split. It writes `output/seen-unseen split/`.

```
cd analysis
python a1_pipeline.py && python a2_pipeline.py && python a3_pipeline.py
python b1_full_run.py && python b2_lambda_sweep.py
python seen_unseen_split.py
```

## Graphs (`analysis/plotting.py`, `analysis/build_report.py`)

Plain matplotlib PNGs, matched to what each granularity can actually show:

- **Total level** (7 pairs): one grouped bar chart per metric, all in
  `figures/total_level_summary.png`.
- **Task level** (27 tasks x 7 pairs): a grouped horizontal bar chart per
  metric (`figures/task_level_{pearson,kendall,mae}.png`) — readable at that
  count; the full numbers are also in `task_level_wide.csv`.
- **Episode level** (243 episodes, 240 for the ICVFE/recap pairs): too
  granular for a per-episode bar chart, so this is a **table**
  (`episode_level_wide.csv` / `metrics_long.csv`) with a box-plot of the
  per-episode distribution per pair as a graph complement
  (`figures/episode_distribution_{pearson,kendall,mae}.png`).

`analysis/output/report.md` stitches the figures and the total/task tables
into one static document; open it directly (any Markdown viewer renders the
relative image links) or read `analysis/output/*.csv` / `figures/*.png`
directly.

## Baseline-specific reports (`analysis/baseline_reports.py`)

The master report above puts all seven pairs on one chart, which answers
"how does everything compare to everything" but buries the two questions
that actually matter for picking an evaluator: *given robodopamine as the
reference, how do the candidate evaluators compare?* and *given human as the
reference, how do they compare?* `baseline_reports.py` re-slices the same
`pipeline.py` output into two self-contained subdirectories, each with its
own `metrics_long.csv` / `task_level_wide.csv` / `episode_level_wide.csv` /
`total_level_wide.csv` / `figures/*.png` / `report.md`:

- **`output/vs_robodopamine/`** — robometer, icvfe, recap, each vs
  robodopamine (3 pairs). Robodopamine itself isn't in here — it *is* the
  baseline in this subdir.
- **`output/vs_human/`** — robodopamine, robometer, icvfe, recap, each vs
  human (4 pairs). Robodopamine is included here: once human is the
  baseline, robodopamine is just another candidate evaluator like the other
  three, so its own agreement with human belongs alongside them for
  reference.

## Progress curves (`analysis/progress_curves.py`)

Everything above is agreement *statistics* (Pearson/Kendall/MAE) — it never
shows what a progress curve actually looks like. `progress_curves.py`
produces three tiers of granularity for two comparisons — (a)
`human_vs_robodopamine` and (b) `all_evaluators` (robodopamine / robometer /
icvfe / recap together) — each written as a `<comparison>__<tier>` path
under `progress_curves/`:

- **`*__global.png`** — mean curve per evaluator (±1 std band for (a); (b)
  omits the band since 4 overlapping bands aren't legible), pooled over all
  episodes. Each episode is resampled onto a common 101-point normalized-time
  grid (0–100% of that episode's frames, linear interpolation) so curves of
  different lengths can be averaged; x-axis is therefore "% of episode",
  not frame number.
- **`*__per_task/<task_slug>/<episode_uid>.png`** — every single episode
  gets its own plot, actual (non-resampled, real frame-index x-axis) curves,
  organized one subdirectory per task: 243 episode plots for (a), 240 for
  (b) — i.e. every episode in the respective comparison's episode set, not
  just a representative sample.
- **`*__representative/<task_slug>.png`** — one plot per task (27 files),
  the same actual-curve rendering as above but for a single representative
  episode per task: the **longest** episode (every episode in this dataset
  is annotated `"success"`, so "longest success episode" reduces to
  "longest episode" — see the `success` field in the `manual` archive's
  per-episode JSON).
- **`curve_data_human_vs_robodopamine.csv`, `curve_data_all_evaluators.csv`**
  — the underlying resampled mean/std arrays behind the `*__global.png` /
  per-task-mean plots (long format: level, task, series, pct_progress, mean,
  std, n_episodes), for reuse without re-running the resampling.

Episode sets are matched per comparison so the averaged/representative
curves aren't skewed by one evaluator having more/fewer episodes than
another — e.g. `all_evaluators` restricts robodopamine/robometer (269 raw
episodes each) down to the 240 icvfe/recap also cover, rather than averaging
over a larger, different episode set for two of the four lines.

## Visualizer (`visualizer/`)

The aggregate tables above answer "how well does X track Y, overall / per
task" — they don't show what any single episode's curves actually look like.
`visualizer/` is a small local web app for that: pick a task and episode and
see Human, RoboDopamine, RoboMeter, and ICVFE's progress curves overlaid on
one chart, each toggleable via a legend chip. Run `python visualizer/server.py`
and open the printed URL. See `visualizer/README.md` for what it is (a clone
of the video-annotation tool's chart, minus editing and video) and its
scrubber/keyboard controls.
