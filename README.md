# ICVFE Evals

Evaluation of value/reward models against human progress annotations on the ICL
demo set (27 tasks, 243 human-annotated episodes), plus value-guided demo
retrieval experiments.

## Data

Download the archives from the
[icl-vfe-eval-results](https://huggingface.co/datasets/Hannibal52Barca/icl-vfe-eval-results)
HF dataset into this repo:

```bash
hf download Hannibal52Barca/icl-vfe-eval-results --repo-type dataset --local-dir .
```

Everything lives under `data/demo_set_annotations/demo_set_annotations/`:

| Method | Model | Type |
|---|---|---|
| Manual | Human annotation (reference) | – |
| Robo-Dopamine (zero-shot) | Robo-Dopamine GRM | Offline |
| Robo-Dopamine (fine-tuned) | Robo-Dopamine GRM | Offline |
| RoboMeter (zero-shot) | RoboMeter-4B | Offline |
| RoboMeter (fine-tuned) | RoboMeter-4B | Offline |
| TOPReward (zero-shot) | TOPReward | Offline |
| GVL (zero-shot) | GVL | Offline |
| RoboMeter online (zero-shot) | RoboMeter-4B | Online |
| RoboMeter online (fine-tuned) | RoboMeter-4B | Online |
| RECAP | π0.6 RECAP-style VFE | Online |
| IC-VFE | IC-VFE | Online |
| DINO\* | DINOv2 embeddings | – |
| SARM\* | SARM subtask annotations | – |

\* Not scored as a value model. DINO is used only for the embedding distance in
B1/B2 retrieval.

\* SARM is not used in these analyses.

Some archives are hosted in sibling datasets; the HF dataset card lists where to place them.

## Links

**Data**
- Source episodes: [adityx23/icl-demo-dataset](https://huggingface.co/datasets/adityx23/icl-demo-dataset)
- Model predictions and analysis outputs: [Hannibal52Barca/icl-vfe-eval-results](https://huggingface.co/datasets/Hannibal52Barca/icl-vfe-eval-results)
- Human annotations: [Hannibal52Barca/icl_project_manual_annotation](https://huggingface.co/datasets/Hannibal52Barca/icl_project_manual_annotation) (tool: [icl-manual-annotation](https://github.com/HannibalofBarca/icl-manual-annotation))

**Models**

| Method | Weights | Code |
|---|---|---|
| Robo-Dopamine (zero-shot) | [tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview](https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview) | [HannibalofBarca/Robo-Dopamine](https://github.com/HannibalofBarca/Robo-Dopamine) |
| Robo-Dopamine (fine-tuned) | Per-task LoRAs: [Hannibal52Barca/robo-dopamine-lora-checkpoints](https://huggingface.co/Hannibal52Barca/robo-dopamine-lora-checkpoints) | [HannibalofBarca/Robo-Dopamine](https://github.com/HannibalofBarca/Robo-Dopamine) |
| RoboMeter (zero-shot, incl. online) | [robometer/Robometer-4B](https://huggingface.co/robometer/Robometer-4B) | [HannibalofBarca/robometer](https://github.com/HannibalofBarca/robometer) |
| RoboMeter (fine-tuned, incl. online) | [Hannibal52Barca/robometer-4b-icl-finetuned](https://huggingface.co/Hannibal52Barca/robometer-4b-icl-finetuned) | [HannibalofBarca/robometer](https://github.com/HannibalofBarca/robometer) |
| TOPReward (zero-shot) | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | [huggingface/lerobot](https://github.com/huggingface/lerobot) (`lerobot.rewards.topreward`) |
| GVL (zero-shot) | GPT-4o (OpenAI API) | [robometer/robometer](https://github.com/robometer/robometer) (`robometer/evals/baselines/gvl.py`) |
| DINO\* | [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) | – |
| IC-VFE, RECAP | Not yet released | – |

## Scoring

- **Metrics:** Pearson r, Kendall τ-b, MAE (progress on a 0–1 scale), each vs. the human curve.
- **Alignment:** curves are matched to the human curve by `frame_index`. Sparse
  curves (TOPReward, GVL and fine-tuned Robo-Dopamine sample every 10th frame or
  irregularly) are linearly interpolated onto every frame.
- **IC-VFE / RECAP:** the `target_progress` stored in their `.npz` files is
  Robo-Dopamine's curve, not the human one, so it is ignored. `prediction_progress` is
  re-scored against the human curve on the file's own `timesteps`.
- **Aggregation:** every metric is computed **per episode**, over that episode's own
  aligned frames, then averaged: task, split and total values are the mean over
  episodes, each episode counting once. Frames are not pooled across episodes, except
  in the frame-pooled comparison columns of the seen/unseen split.
  IC-VFE's context replicates (about 9 per episode) are averaged into one episode
  value first (`analysis/aggregation.py`).
- **Flat curves:** Pearson and Kendall are undefined when either curve is constant
  (e.g. a failed episode the human marked 0 throughout, or a model that outputs 0
  everywhere). Those episodes score 0 rather than being skipped, matching
  `evaluate_ref.py`.

Episode averaging only checks ranking within each episode, so curves that simply
rise with elapsed time (e.g. TOPReward) score well.

### Results

Kendall τ-b / Pearson r vs. human, from `seen-unseen split/seen_unseen_split.csv`.

**Episode-averaged** (`aggregation = episode`; what the A1–A3 pipelines report):

| Model | Seen τ | Seen r | Unseen τ | Unseen r |
|---|---|---|---|---|
| Robo-Dopamine | 0.79 | 0.88 | 0.79 | 0.90 |
| Robo-Dopamine FT | 0.64 | 0.72 | 0.81 | 0.88 |
| RoboMeter | 0.56 | 0.70 | 0.58 | 0.73 |
| RoboMeter FT | 0.65 | 0.78 | 0.68 | 0.81 |
| TOPReward | 0.80 | 0.74 | 0.81 | 0.79 |
| GVL | 0.34 | 0.41 | 0.29 | 0.37 |
| IC-VFE (EMA 0.5) | 0.53 | 0.64 | 0.54 | 0.67 |
| IC-VFE (step 8800) | 0.51 | 0.62 | 0.52 | 0.65 |
| RECAP | 0.48 | 0.60 | 0.43 | 0.57 |
| RoboMeter online | 0.26 | 0.34 | 0.27 | 0.38 |
| RoboMeter FT online | 0.32 | 0.44 | 0.33 | 0.46 |

**Frame-pooled** (`aggregation = frame`; every aligned frame from every episode in the
split, including each IC-VFE context replicate, in one correlation):

| Model | Seen τ | Seen r | Unseen τ | Unseen r |
|---|---|---|---|---|
| Robo-Dopamine | 0.66 | 0.83 | 0.69 | 0.86 |
| Robo-Dopamine FT | 0.61 | 0.73 | 0.74 | 0.87 |
| RoboMeter | 0.46 | 0.60 | 0.53 | 0.69 |
| RoboMeter FT | 0.52 | 0.69 | 0.59 | 0.75 |
| TOPReward | 0.52 | 0.60 | 0.57 | 0.71 |
| GVL | 0.20 | 0.25 | 0.24 | 0.30 |
| IC-VFE (EMA 0.5) | 0.36 | 0.50 | 0.34 | 0.50 |
| IC-VFE (step 8800) | 0.35 | 0.49 | 0.34 | 0.48 |
| RECAP | 0.37 | 0.50 | 0.25 | 0.37 |
| RoboMeter online | 0.13 | 0.23 | 0.23 | 0.35 |
| RoboMeter FT online | 0.15 | 0.30 | 0.23 | 0.37 |

Pooling also checks whether a model's values mean the same thing across episodes;
episode averaging only checks ranking within each episode. Compare models only within
one aggregation scheme.

Fine-tuned Robo-Dopamine covers 109 of the 121 unseen episodes.

## Analyses (`analysis/`)

| Script | Experiment | Output (`analysis/output/`) |
|---|---|---|
| `a1_pipeline.py` | **A1:** zero-shot roster (TOPReward, GVL, RoboMeter, Robo-Dopamine) vs. human | `A1-reward model ablation/` |
| `a2_pipeline.py` | **A2:** fine-tuned vs. zero-shot (RoboMeter, Robo-Dopamine) | `A2-finetuned vs zeroshot/` |
| `a3_pipeline.py` | **A3:** online models (RoboMeter ZS/FT online, RECAP, IC-VFE) vs. human | `A3-online ft models vs human/` |
| `seen_unseen_split.py` | A1–A3 metrics split into seen (15) / unseen (12) tasks, using RECAP's fine-tuning split, both episode-averaged and frame-pooled | `seen-unseen split/` |
| `b1_full_run.py` | **B1:** for each frame of an episode, retrieve the matching point in a reference demo of the same task, using value estimates, DINO visual similarity, or both, and measure how well the retrieved point matches | `b1_full_run/` |

B1 retrieval metrics (`b_metrics.py`):
- **TTC error:** difference in normalized time-to-completion between the query and the retrieved frame.
- **DINO error:** single-frame embedding distance.

```bash
cd analysis
python a1_pipeline.py && python a2_pipeline.py && python a3_pipeline.py && python seen_unseen_split.py
python b1_full_run.py
```

Requires `numpy`, `pandas`, `scipy`, `matplotlib`. Archives are extracted into
`analysis/cache/` on first run. Delete the relevant subdirectory (or pass `--no-cache`)
after replacing an archive.

## Additional functionalities

Not used in the reported results.

**Extra B1 metrics and retrieval methods**
- **Video chunk error** (formerly "videoDTW"): L1 distance between the retrieved and
  query 10-frame DINO-embedding chunks.
- **BC error:** L1 distance between the retrieved and query action chunks.
- `b1_vep_retrieval.py`: B1 retrieval by VEP embedding nearest neighbor instead of DINO
  (`b_metrics.VepIndex`), output in `b1_vep_retrieval/`.

`b_metrics.py` computes both extra metrics, and `b1_full_run.py` / `b2_lambda_sweep.py`
still write them as the `video_chunk_error` and `bc_error` columns.

**Scripts**
- `b2_lambda_sweep.py`: B2, a sweep of the value/vision mixing weight λ in B1 retrieval
  (`λ·value_dist + (1−λ)·vision_dist`).
- `rescore_vfe_paper_ablation.py`: CPU rescoring of saved IC-VFE ablation predictions.
- `analysis/pipeline.py`, `plotting.py`, `build_report.py`, `baseline_reports.py`,
  `progress_curves.py`: earlier pairwise study (Robo-Dopamine / RoboMeter / IC-VFE /
  RECAP vs. human and vs. Robo-Dopamine) with figures and per-episode curve plots.
- `visualizer/`: local web app to overlay one episode's curves (`python visualizer/server.py`).
