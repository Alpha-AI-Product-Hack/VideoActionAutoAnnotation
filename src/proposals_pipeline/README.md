# Egocentric action-boundary proposals

Boundary proposals for egocentric video (Assembly101, EPIC-Kitchens) from three
sources, merged with weighted temporal NMS; all thresholds tuned on ground truth.

```
video ──┬─ abd     V-JEPA 2.1 embeddings + DDM-Net GEBD scores ─┐
        ├─ motion  optical flow + homography residual           ├─ scored proposals ─ weighted NMS ─ boundaries
        └─ sam3    SAM3 hand/object masks → contact events      ┘
```

- `abd`: peaks of the z-scored cosine-valley curve of 1 s V-JEPA windows (every 0.125 s), optionally fused with DDM-Net boundary probabilities.
- `motion`: peaks or valleys of the smoothed foreground flow residual (or whole-frame flow, or their derivative) at 8 fps.
- `sam3`: hand-object contact on/off and held-object switch events from batched bf16 SAM3 masks at 4 fps.
- Fusion: per-source min-max scores times a weight in [0.25, 2], greedy NMS within `nms_window_s`, agreement of other sources added to the kept score, cutoff `min_fused_score`.
- Evaluation: one-to-one Hungarian matching within 0.5/1/2 s, micro P/R/F1 per benchmark (`a101-coarse`, `a101-fine`, `epic`), proposals outside annotated spans ignored.
- Tuning (`tune`): per-source grid, fusion grid, coordinate descent; objective F1@1 s; an oracle on the tuned videos, with `--per-video-oracle` as the ceiling.

## Setup, run, evaluate

```bash
bash setup.sh                    # venv, torch (CUDA=cu126 to override), deps, DDM-Net checkpoint, tests
huggingface-cli login            # facebook/sam3 is gated

python -m boundary_pipeline extract                                   # cache signals, all sources, all videos
python -m boundary_pipeline --videos a101/9046-a29 extract --sources motion,sam3
python -m boundary_pipeline tune --benchmark a101-coarse --per-video-oracle --out configs/a101-coarse.json
python -m boundary_pipeline evaluate --config configs/a101-coarse.json --plots .cache/plots/a101-coarse
python -m boundary_pipeline run --config configs/a101-coarse.json --video clip.mp4 --out clip.json --plot clip.png
```

Full local data (2.3 h of video), sequentially so the GPU is not shared:

```bash
nohup sh -c 'python -m boundary_pipeline extract --sources motion,sam3 && \
             python -m boundary_pipeline extract --sources abd --batch-size 16 && \
             for b in a101-coarse a101-fine epic; do \
               python -m boundary_pipeline tune --benchmark $b --per-video-oracle --out configs/$b.json && \
               python -m boundary_pipeline evaluate --config configs/$b.json --plots .cache/plots/$b; \
             done' > overnight.log 2>&1 &
```

Data: `data/assembly101/{recordings,annotations}` (`download_assembly101.py`),
`data/epic_kitchens/{EPIC-KITCHENS,annotations}`. `extract` skips cached
signals; `--sam3-fps` (global option) selects the SAM3 rate, caches are per rate.

## Quality

Source ablation on `a101/9052-b06a` (318 s), every subset tuned on that
recording. Bold = best per column, underlined = second.

`a101-coarse` (12 GT boundaries)

| sources | F1@0.5s | F1@1.0s | F1@2.0s | P@1s | R@1s | det/gt | weights |
|---|---|---|---|---|---|---|---|
| motion | 0.133 | **0.467** | **0.600** | <u>0.389</u> | 0.583 | <u>1.50</u> | motion 0.25 |
| motion+sam3 | 0.133 | **0.467** | **0.600** | <u>0.389</u> | 0.583 | <u>1.50</u> | motion 2, sam3 0.25 |
| abd+motion | 0.087 | <u>0.435</u> | <u>0.435</u> | **0.455** | 0.417 | **0.92** | abd 0.25, motion 1 |
| abd+motion+sam3 | 0.087 | <u>0.435</u> | <u>0.435</u> | **0.455** | 0.417 | **0.92** | abd 0.25, motion 1, sam3 0.25 |
| sam3 | **0.174** | 0.348 | 0.348 | 0.235 | <u>0.667</u> | 2.83 | sam3 0.25 |
| abd+sam3 | <u>0.139</u> | 0.278 | 0.278 | 0.167 | **0.833** | 5.00 | abd 0.25, sam3 1 |
| abd | 0.070 | 0.246 | 0.351 | 0.156 | 0.583 | 3.75 | abd 0.25 |

`a101-fine` (184 GT boundaries)

| sources | F1@0.5s | F1@1.0s | F1@2.0s | P@1s | R@1s | det/gt | weights |
|---|---|---|---|---|---|---|---|
| abd+motion+sam3 | 0.589 | **0.794** | 0.822 | **0.812** | 0.777 | <u>0.96</u> | abd 2, motion 0.25, sam3 0.25 |
| motion+sam3 | **0.629** | <u>0.782</u> | 0.812 | 0.718 | <u>0.859</u> | 1.20 | motion 2, sam3 0.25 |
| abd+motion | 0.571 | 0.769 | 0.808 | <u>0.778</u> | 0.761 | **0.98** | abd 0.5, motion 0.25 |
| motion | 0.600 | 0.758 | 0.800 | 0.663 | **0.886** | 1.34 | motion 0.25 |
| abd+sam3 | <u>0.615</u> | 0.750 | **0.828** | 0.720 | 0.783 | 1.09 | abd 1, sam3 0.25 |
| abd | 0.590 | 0.748 | <u>0.825</u> | 0.670 | 0.848 | 1.27 | abd 0.25 |
| sam3 | 0.564 | 0.715 | 0.771 | 0.736 | 0.696 | 0.95 | sam3 0.25 |

Three-recording smoke tuning (9014-b02a, 9034-b04c, 9046-a29; `configs/smoke_*.json`,
tuned before source weights were restricted to [0.25, 2]), F1@1 s:

| benchmark | fused | best single source | per-video oracle |
|---|---|---|---|
| a101-coarse | 0.455 | 0.422 (motion) | 0.526 |
| a101-fine | 0.744 | 0.740 (abd) | 0.764 |

## Performance

`a101/9052-b06a`, 318 s at 636x480 / 60 fps, RTX 5080, nothing cached, model loading included:

| stage | wall | vs. real time | peak RAM |
|---|---|---|---|
| motion (CPU, 8 fps) | 43 s | 7.4x | 0.2 GB |
| sam3 (GPU, 4 fps, bf16) | 218 s | 1.5x | 4.2 GB |
| abd (GPU, V-JEPA + DDM-Net) | 348 s | 0.9x | 8.4 GB |
| propose + fuse, all sources | 0.24 s | | |

## Layout

```
boundary_pipeline/     datasets, sources/{abd,motion,sam3}, proposals (NMS), fusion, evaluation, tuning, plotting, cli
action_boundaries/     V-JEPA encoder, DDM-Net (vendored, 3 forward-pass fixes), ABD curve, camera-motion signal, embedding store
sam3_pipeline/         batched SAM3 detector, mask ops, tracklet linking
configs/               tuned configurations and results
tests/                 pytest, no GPU or data needed
```

Notes: `Dev-Jahn/vjepa2.1-vitl-fpc64-384` uses `trust_remote_code` (reviewed);
DDM-Net replaces DyBDet/MOSS, which have no released weights; Assembly101
annotations are at 30 fps on 60 fps video, one ego view per recording (first
`HMC_*` file); cache keys ignore absolute paths (`EmbeddingStore.migrate()` re-keys
older stores); Assembly101 is CC BY-NC 4.0.
