# Egocentric action-boundary proposals

Action-boundary proposals for egocentric video (Assembly101, EPIC-Kitchens)
from three independent signals, merged with temporal NMS. Every threshold
is tuned against ground truth.

```
video ──┬─ abd     V-JEPA 2.1 embeddings + DDM-Net GEBD scores ─┐
        ├─ motion  optical flow + homography residual           ├─ scored proposals ─ weighted NMS ─ boundaries
        └─ sam3    SAM3 hand/object masks → contact events      ┘
```

Each source has an expensive `extract` step (cached per video) and a cheap
`propose` step that turns the cached signal into scored proposals. Since
proposing is cheap, all source thresholds and the fusion parameters are
tuned by direct search (`tune`); configs live in `configs/`.

## Sources

| source | cached signal | proposals | tuned |
|---|---|---|---|
| `abd` | V-JEPA 2.1 embeddings of 1 s windows every 0.125 s; DDM-Net boundary probability `b_t` | peaks of `d_z = -z(cos(f_t, f_{t+δ}))`, of `b_t`, or of `max(b_t, w·d_z)`; score = prominence | `mode`, `gebd_weight`, `delta`, `sigma`, `prominence`, `min_distance_s` |
| `motion` | at 8 fps: whole-frame flow magnitude `m`, residual `r` of a RANSAC background homography (hand/object motion) | peaks or valleys of smoothed, z-scored `log1p(r)` / `log1p(m)`, or peaks of their derivative magnitude; score = prominence | `signal`, `mode`, `sigma`, `prominence`, `min_distance_s` |
| `sam3` | at 4 fps: hand masks; hand-adjacent object masks with score, tracklet id and hand overlap at dilation 0/5/15 px | contact on/off events (with hysteresis) and held-object switches; score = persistence of the new state | `dilate_px`, `min_obj_score`, `min_overlap_frac`, `min_hold_frames`, `min_track_len`, `use_switch` |

## Fusion

Scores are min-max normalized per source and video and multiplied by a
source weight (0 drops the source). Greedy NMS keeps a proposal when no
kept proposal lies within `nms_window_s`; a kept proposal's score adds the
best suppressed score of each other source, and `min_fused_score` drops
the rest.

## Evaluation

One-to-one Hungarian matching within a tolerance, micro P/R/F1 over the
videos of a benchmark, proposals outside the annotated spans ignored.

| benchmark | videos | ground truth |
|---|---|---|
| `a101-coarse` | 10 Assembly101 recordings, one ego view each | edges between consecutive coarse segments, per phase |
| `a101-fine` | same | starts and ends of fine-grained actions, merged within 0.2 s |
| `epic` | EPIC-Kitchens P02_123, P04_113 (train annotations) | narration starts and stops, merged within 0.2 s |

`tune` maximizes F1@1.0 s: each source alone over its grid, then the
fusion grid, then coordinate descent over everything. It is tuned and
scored on the same videos (an oracle, not a held-out estimate);
`--per-video-oracle` re-tunes per video for the ceiling.

## Results

Smoke run on three recordings (9014-b02a, 9034-b04c, 9046-a29), tuned on
those same recordings, SAM3 at 5 fps. `solo` = one source with its tuned
parameters; `without` = fused with that source's weight set to 0.

`a101-coarse`

| variant | F1@0.5s | F1@1.0s | F1@2.0s | P@1s | R@1s | det/gt |
|---|---|---|---|---|---|---|
| fused | 0.258 | 0.455 | 0.515 | 0.417 | 0.500 | 1.20 |
| solo:abd | 0.097 | 0.324 | 0.453 | 0.214 | 0.667 | 3.12 |
| solo:motion | 0.245 | 0.422 | 0.471 | 0.299 | 0.717 | 2.40 |
| solo:sam3 | 0.151 | 0.212 | 0.264 | 0.123 | 0.750 | 6.08 |
| without:sam3 | 0.213 | 0.393 | 0.459 | 0.387 | 0.400 | 1.03 |
| per-video oracle | 0.321 | 0.526 | 0.615 | 0.427 | 0.683 | 1.60 |

`a101-fine`

| variant | F1@0.5s | F1@1.0s | F1@2.0s | P@1s | R@1s | det/gt |
|---|---|---|---|---|---|---|
| fused | 0.635 | 0.744 | 0.825 | 0.674 | 0.830 | 1.23 |
| solo:abd | 0.593 | 0.740 | 0.826 | 0.714 | 0.767 | 1.07 |
| solo:motion | 0.585 | 0.712 | 0.808 | 0.725 | 0.698 | 0.96 |
| solo:sam3 | 0.490 | 0.589 | 0.685 | 0.523 | 0.675 | 1.29 |
| without:motion | 0.521 | 0.629 | 0.724 | 0.718 | 0.560 | 0.78 |
| per-video oracle | 0.631 | 0.764 | 0.816 | 0.706 | 0.832 | 1.18 |

The tuner gave ABD weight 0 in both fused configs. Parameters are in
`configs/smoke_*.json`, plots in `.cache/plots/`.

## Usage

```bash
bash setup.sh                    # venv, torch (CUDA=cu126 to override), deps, checkpoint, tests
huggingface-cli login            # facebook/sam3 is gated

python -m boundary_pipeline extract                                   # all sources, all videos
python -m boundary_pipeline --videos a101/9046-a29 extract --sources motion,sam3
python -m boundary_pipeline tune --benchmark a101-coarse --per-video-oracle --out configs/a101-coarse.json
python -m boundary_pipeline evaluate --config configs/a101-coarse.json --plots .cache/plots/a101-coarse
python -m boundary_pipeline run --config configs/a101-coarse.json --video clip.mp4 --out clip.json --plot clip.png
```

`extract` skips cached signals. `--sam3-fps` (global option) changes the
SAM3 rate; caches are kept per rate and configs record theirs. On an
RTX 5080 the full local data (2.3 h of video) takes about 3 h for `abd`
and 1.3 h for `sam3`; run them sequentially:

```bash
nohup sh -c 'python -m boundary_pipeline extract --sources motion,sam3 && \
             python -m boundary_pipeline extract --sources abd --batch-size 16 && \
             for b in a101-coarse a101-fine epic; do \
               python -m boundary_pipeline tune --benchmark $b --per-video-oracle --out configs/$b.json && \
               python -m boundary_pipeline evaluate --config configs/$b.json --plots .cache/plots/$b; \
             done' > overnight.log 2>&1 &
```

Data: `data/assembly101/{recordings,annotations}` (`download_assembly101.py`)
and `data/epic_kitchens/{EPIC-KITCHENS,annotations}`.

## Layout

```
boundary_pipeline/     datasets, sources/{abd,motion,sam3}, proposals (NMS), fusion, evaluation, tuning, plotting, cli
action_boundaries/     V-JEPA encoder, DDM-Net (vendored), ABD curve, camera-motion signal, embedding store
sam3_pipeline/         batched SAM3 detector, mask ops, tracklet linking
configs/               tuned configurations and results
tests/                 pytest, no GPU or data needed
```

Caches: `.cache/embedding_store`, `.cache/sam3`, `.cache/motion`,
`.cache/ddm_net`.

## Notes

- V-JEPA 2.1 is not in `transformers` upstream; `Dev-Jahn/vjepa2.1-vitl-fpc64-384`
  ships its own modeling code (`trust_remote_code=True`), reviewed before use:
  plain PyTorch modules, safetensors weights. To avoid it, point
  `action_boundaries/constants.py` at an official `facebook/vjepa2-*` checkpoint.
- DDM-Net (CVPR 2022, MIT) stands in for DyBDet / MOSS BasicGEBD, which have no
  released weights. The vendored copy fixes three forward-pass bugs (removed
  torchvision import, position-embedding slice on the wrong dimension, bare
  `.squeeze()` at batch size 1); the checkpoint loads with no key mismatch.
  Trained on third-person Kinetics, so the `fused` mode keeps the ABD valley.
- SAM3 runs as a bf16 batched image model with one backbone pass for both
  prompts (~5x faster than fp32, identical detections); the video tracker is
  not used since hand identity is not needed and its memory grows with length.
- Assembly101 annotations are at 30 fps, videos at 60 fps. The manifest takes the
  first `HMC_*` file per recording, i.e. view e3 for `HMC_2...` and e1 for
  `HMC_8...` recordings. Coarse segments are contiguous, so a coarse boundary is a
  shared segment edge. Only the EPIC train split is local. Assembly101 is CC BY-NC 4.0.
- Dropped after evaluation: a WiLoR hand-aperture channel on top of the SAM3
  contact events (EPIC train segments < 2 s: boundary recall@0.5 s 0.25 → 0.59,
  F1 0.27 → 0.46, many more false positives; needs WiLoR and a registered MANO
  model, CC BY-NC-ND) and the camera-motion suppression rule (whole frame moved,
  nothing moved relative to it), which changed F1 by under 0.01 on every clip.
  Single-source calibrations also failed to transfer between videos (pure ABD
  F1@1 s 0.72 on one EPIC clip, 0.43 on another at identical settings), which is
  why every threshold is tuned per benchmark and the per-video oracle is reported.
