# ICVFE Evals

Evaluation of value/reward models against human progress annotations on the ICL
demo set (27 tasks, 243 human-annotated episodes), plus value-guided demo
retrieval experiments.

## Data

Download the archives from the
[icl-vfe-eval-results](https://huggingface.co/datasets/Hannibal52Barca/icl-vfe-eval-results)
HF dataset into this repo (paths match):

```bash
hf download Hannibal52Barca/icl-vfe-eval-results --repo-type dataset --local-dir .
```

Everything lives under `data/demo_set_annotations/demo_set_annotations/`:

| Folder | Archive | Model | Type |
|---|---|---|---|
| `manual/` | `manual_icl_demo_dataset_{continuous,milestone}.zip` | Human annotation (reference) | – |
| `robo_dopamine/` | `zs_robodopamine_…`, `finetuned_robodopamine_…` | Robo-Dopamine GRM, zero-shot / fine-tuned | Offline |
| `robometer/` | `zeroshot_robometer_…`, `finetuned_robometer_…` | RoboMeter-4B, zero-shot / fine-tuned | Offline |
| `topreward/` | `zs_topreward_…` | TOPReward, zero-shot | Offline |
| `gvl/` | `zs_gvl_…` | GVL, zero-shot | Offline |
| `robometer/` | `zeroshot_robometer_online_…`, `finetuned_robometer_online_…` | RoboMeter-4B online, zero-shot / fine-tuned | Online |
| `RECAP/` | `recap_20000_…` | π0.6 RECAP-style VFE, step 20000 (no context) | Online |
| `ICVFE/` | `icvfe_8800_…`, `icvfe_ema_0.5_…` | IC-VFE step 8800, raw / causal EMA α=0.5 | Online |
| `dino/` | `dino_base_…` | DINOv2 embeddings (retrieval only) | – |
| `SARM/` | `sarm_…_subtasks.zip` | SARM subtask annotations | – |

The manual, zero-shot Robo-Dopamine, fine-tuned RoboMeter and DINO archives are
hosted in sibling datasets; the HF dataset card lists where to place them.

## Scoring

- **Metrics:** Pearson r, Kendall τ-b, MAE (progress on a 0–1 scale), each vs. the human curve.
- **Alignment:** curves are matched to the human curve by `frame_index`. Sparse
  curves (TOPReward, GVL and fine-tuned Robo-Dopamine sample every 10th frame or
  irregularly) are linearly interpolated onto every frame.
- **IC-VFE / RECAP:** the `target_progress` stored in their `.npz` files is
  Robo-Dopamine's curve, not the human one, so it is ignored. `prediction_progress` is
  re-scored against the human curve on the file's own `timesteps`.
- **Aggregation:** episode-level metrics use that episode's frames. Task-, split- and
  total-level metrics are **frame-pooled**: all aligned frames from all episodes in
  the group, including every IC-VFE context replicate, go into one correlation.

### Pooling vs. episode averaging

The aggregation choice changes the numbers a lot. Pooling also checks whether a
model's values mean the same thing across episodes. Episode averaging only checks
ranking within each episode, so it rewards degenerate curves that just rise with
elapsed time (e.g. TOPReward). Seen-task Kendall τ-b / Pearson r:

| Model | Frame-pooled (used) | Episode-averaged |
|---|---|---|
| Robo-Dopamine (ZS) | 0.66 / 0.83 | 0.80 / 0.88 |
| TOPReward | 0.52 / 0.60 | 0.80 / 0.74 |
| GVL | 0.20 / 0.25 | 0.35 / 0.43 |
| RoboMeter FT (online) | 0.15 / 0.30 | 0.32 / 0.45 |
| RECAP | 0.37 / 0.50 | 0.48 / 0.61 |
| IC-VFE (EMA 0.5) | 0.36 / 0.50 | 0.53 / 0.65 |

Compare models only within one aggregation scheme.

## Analyses (`analysis/`)

| Script | Experiment | Output (`analysis/output/`) |
|---|---|---|
| `a1_pipeline.py` | **A1:** zero-shot roster (TOPReward, GVL, RoboMeter, Robo-Dopamine) vs. human | `A1-reward model ablation/` |
| `a2_pipeline.py` | **A2:** fine-tuned vs. zero-shot (RoboMeter, Robo-Dopamine) | `A2-finetuned vs zeroshot/` |
| `a3_pipeline.py` | **A3:** online models (RoboMeter ZS/FT online, RECAP, IC-VFE) vs. human | `A3-online ft models vs human/` |
| `seen_unseen_split.py` | A1–A3 metrics split into seen (15) / unseen (12) tasks, using RECAP's fine-tuning split | `seen-unseen split/` |
| `b1_full_run.py` | **B1:** retrieval from a per-task demo-bank episode by value, DINO distance, or value + DINO; sharded via `b1_full_run_{tasklist,shard,merge}.py` | `b1_full_run/` |
| `b2_lambda_sweep.py` | **B2:** sweep of λ in `λ·value_dist + (1−λ)·vision_dist`, λ ∈ {0, 0.25, 0.5, 0.75, 1} | `B2-*` |

B1/B2 retrieval metrics (`b_metrics.py`):
- **TTC error:** difference in normalized time-to-completion between the query and the retrieved frame.
- **videoDTW error:** DTW distance between the 30-frame DINO chunks.
- **BC error:** L1 distance between the 30-step action chunks.
- **DINO error:** single-frame embedding distance.

```bash
cd analysis
python a1_pipeline.py && python a2_pipeline.py && python a3_pipeline.py && python seen_unseen_split.py
python b1_full_run.py && python b2_lambda_sweep.py
```

Requires `numpy`, `pandas`, `scipy`, `matplotlib`. Archives are extracted into
`analysis/cache/` on first run. Delete the relevant subdirectory (or pass `--no-cache`)
after replacing an archive.

## Other tools

- `analysis/pipeline.py`, `plotting.py`, `build_report.py`, `baseline_reports.py`,
  `progress_curves.py`: earlier pairwise study (Robo-Dopamine / RoboMeter / IC-VFE /
  RECAP vs. human and vs. Robo-Dopamine) with figures and per-episode curve plots.
- `rescore_vfe_paper_ablation.py`, `rescore_vfe_ema.py`: CPU rescoring of saved IC-VFE
  ablation predictions and causal-EMA smoothing.
- `visualizer/`: local web app to overlay one episode's curves (`python visualizer/server.py`).
