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
| IC-VFE | 0.35 / 0.49 | 0.51 / 0.63 |

Compare models only within one aggregation scheme.

## Analyses (`analysis/`)

| Script | Experiment | Output (`analysis/output/`) |
|---|---|---|
| `a1_pipeline.py` | **A1:** zero-shot roster (TOPReward, GVL, RoboMeter, Robo-Dopamine) vs. human | `A1-reward model ablation/` |
| `a2_pipeline.py` | **A2:** fine-tuned vs. zero-shot (RoboMeter, Robo-Dopamine) | `A2-finetuned vs zeroshot/` |
| `a3_pipeline.py` | **A3:** online models (RoboMeter ZS/FT online, RECAP, IC-VFE) vs. human | `A3-online ft models vs human/` |
| `seen_unseen_split.py` | A1–A3 metrics split into seen (15) / unseen (12) tasks, using RECAP's fine-tuning split | `seen-unseen split/` |
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
