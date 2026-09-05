# Open-Vocabulary Action Classification

Frozen open-vocabulary action classification for pre-segmented video clips.

This repository does not segment a video by itself. It expects action intervals
from an upstream temporal segmentation system, for example
`VideoActionAutoAnnotation`, and assigns each interval to the closest label from
an action bank.

The selected production candidate is `microsoft/xclip-base-patch32` used as a
frozen video-text ranker. It is local, deterministic enough for batch inference,
GPU-enabled, does not require API calls, and produces both a Top-k prediction and
a full ranked list of action labels for downstream integration.

## Method

For every input segment:

1. Sample 8 uniformly spaced RGB frames inside `[start_sec, end_sec]`.
2. Render every candidate action label with the prompt
   `the action is: {action_label}`.
3. Encode video and text with frozen X-CLIP.
4. Score labels through the X-CLIP video-conditioned text path
   (`prompts_generator`), then rank by cosine similarity.
5. Write one JSON object per segment with `pred_action`, `topk_labels`,
   `topk_scores`, timing fields, and metadata.

No weights are trained or updated. The action vocabulary is open in the practical
sense: pass any `.csv` or `.txt` label bank at inference time.

## Best Experiment

The best reproducible full-action experiment is:

- run: `artifacts/runs/a101-20-dissimilar-xclip-action20`
- model: `microsoft/xclip-base-patch32`
- task: Assembly101 coarse full action classification
- bank: 20 deliberately dissimilar `verb-object` actions
- input: 20 GT-cut clips from local official Assembly101 coarse annotations
- GPU: RTX 5070 Laptop GPU, CUDA path verified

Metrics:

| run | n | labels | Top-1 | Top-3 | Top-5 | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| X-CLIP B/32 | 20 | 20 | 0.2000 | 0.3000 | 0.4000 | 0.1410 |
| random seed 0 | 20 | 20 | 0.0500 | 0.1500 | 0.4000 | 0.0333 |

Reproduce it:

```bash
.venv/bin/python -m action_ranker.run clips \
  --encoder xclip \
  --clips-json artifacts/runs/a101-20-dissimilar/clips.json \
  --media-root data/raw/assembly101 \
  --bank a101_dissimilar_action20 \
  --bank-file artifacts/runs/a101-20-dissimilar/action_bank.csv \
  --dataset assembly101 \
  --run-id repro-a101-20-dissimilar-xclip-action20
```

The broader full Assembly101 coarse bank experiment is still hard: on 202 labels
and 10 clips, X-CLIP B/32 gives Top-1 0.0000, Top-3 0.2000, Top-5 0.2000,
Top-20 0.4000, Top-50 0.6000. This is not good enough as a final action chooser,
but it is useful as a fast shortlist generator.

## Experiment Summary

| experiment | model | n | labels | Top-1 | Top-3 | Top-5 | Top-20 | Top-50 | macro-F1 | conclusion |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A101 dissimilar full actions | X-CLIP B/32 | 20 | 20 | 0.2000 | 0.3000 | 0.4000 | 1.0000 | 1.0000 | 0.1410 | best full-action run |
| A101 full coarse bank | X-CLIP B/32 | 10 | 202 | 0.0000 | 0.2000 | 0.2000 | 0.4000 | 0.6000 | 0.0000 | useful Top-50 shortlist |
| A101 full coarse bank | X-CLIP B/32 T=32 | 10 | 202 | 0.0000 | 0.2000 | 0.2000 | 0.5000 | 0.5000 | 0.0000 | no clear gain |
| A101 full coarse bank | X-CLIP B/16 zero-shot | 10 | 202 | 0.0000 | 0.2000 | 0.2000 | 0.4000 | 0.5000 | 0.0000 | no clear gain |
| A101 full coarse bank | FAES + X-CLIP top2 | 10 | 202 | 0.0000 | 0.2000 | 0.2000 | 0.3000 | 0.3000 | 0.0000 | worse shortlist |
| A101 fine official verbs | X-CLIP B/32 | 20 | 10 | 0.0500 | 0.4000 | 0.5500 | 1.0000 | 1.0000 | 0.0125 | strong Top-5, weak Top-1 |
| A101 fine official verbs | Qwen3-VL 2B | 20 | 10 | 0.1000 | 0.3000 | 0.5000 | 1.0000 | 1.0000 | 0.0182 | slower, small Top-1 gain |
| A101 coarse official verbs | X-CLIP B/32 | 20 | 10 | 0.1000 | 0.3000 | 0.5500 | 1.0000 | 1.0000 | 0.0200 | local, fast baseline |
| A101 coarse official verbs | Qwen3-VL 2B | 20 | 10 | 0.1500 | 0.3000 | 0.4000 | 1.0000 | 1.0000 | 0.0737 | higher small-n Top-1, lower Top-5 |
| A101 fine diverse verbs | X-CLIP B/32 | 200 | 10 | 0.1150 | 0.3100 | 0.5050 | 1.0000 | 1.0000 | 0.0610 | near random, GPU verified |
| A101 fine diverse verbs | Qwen3-VL 2B | 200 | 10 | 0.1100 | 0.2850 | 0.4900 | 1.0000 | 1.0000 | 0.0450 | not better than X-CLIP |
| EPIC local full bank | X-CLIP B/32 | 10 | 3806 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 | only 10 local clips available |

Gemini was implemented as a constrained VLM runner, but the sandbox blocked the
network call and the run has zero predictions. Qwen3-VL 8B with Unsloth 4-bit was
able to load, but generation on an 8-frame contact sheet ran out of GPU memory.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[torch,video,hf]'
```

If CUDA PyTorch is not already installed, install the correct PyTorch wheel for
the target machine first, then run the editable install above.

## Input From Segmentation

The classifier expects a JSON file with a `clips` array. Each row is one segment
from the upstream segmentation pipeline.

```json
{
  "clips": [
    {
      "dataset": "custom",
      "video_id": "example_video.mp4",
      "start_sec": 12.40,
      "end_sec": 16.10
    }
  ]
}
```

`video_id` is resolved under `--media-root`. It may also be a relative nested
path such as `recording_001/C10095_rgb.mp4`.

For evaluation, add `gold_action`, `gold_verb_id`, and `gold_noun_id` when they
are available. For production inference they are optional.

## Action Bank

Use a built-in bank:

- `assembly101_coarse`: 202 Assembly101 coarse actions
- `epic_kitchens_observed`: 3806 observed EPIC-KITCHENS actions

Or pass a custom bank with `--bank-file`. CSV banks need one of these label
columns: `label`, `action_cls`, `prompt_action_label`, or `gold_action`.
Optional `verb_id` and `noun_id` columns are preserved for evaluation.

Example custom CSV:

```csv
label,verb_id,noun_id
open fridge,1,10
close fridge,2,10
pick up cup,3,20
put down cup,4,20
```

## Batch Inference

```bash
.venv/bin/python -m action_ranker.run clips \
  --encoder xclip \
  --clips-json path/to/segments.json \
  --media-root path/to/videos \
  --bank custom_actions \
  --bank-file path/to/action_bank.csv \
  --dataset custom \
  --run-id my-xclip-run
```

Outputs are written to:

```text
artifacts/runs/my-xclip-run/
```

Important files:

- `predictions.jsonl`: one compact prediction object per segment
- `rankings.jsonl`: full ranked label list and similarity scores per segment
- `metrics_<bank>.json`: evaluation metrics when gold labels are present
- `skip_log.jsonl`: invalid, missing, or unreadable intervals
- `run_meta.json`: model, prompt, taxonomy, and timing metadata

## JSON Output Contract

`predictions.jsonl` contains one JSON object per input segment:

```json
{
  "dataset": "assembly101",
  "video_id": "9012-c14b",
  "start_sec": 55.3,
  "end_sec": 64.0666666667,
  "gold_action": "inspect toy",
  "pred_action": "attempt to attach water tank",
  "topk_labels": [
    "attempt to attach water tank",
    "unscrew chassis",
    "attach turntable top",
    "detach clamp arm",
    "screw body"
  ],
  "topk_scores": [0.15195179, 0.14894506, 0.14376263, 0.14360198, 0.14303316],
  "dictionary_id": "a101_dissimilar_action20",
  "encoder_id": "microsoft/xclip-base-patch32",
  "prompt_id": "the_action_is",
  "gold_rank": 13,
  "frame_count": 8,
  "inference_s": 1.224526,
  "decode_s": 1.021132,
  "encode_s": 0.203394,
  "rank_s": 0.0
}
```

`topk_scores` are cosine similarities; higher is better. `gold_action` and
`gold_rank` are evaluation fields. In production, `gold_action` can be empty and
`gold_rank` will be `null`.

`rankings.jsonl` has the same segment key plus all labels sorted by similarity:

```json
{
  "clip_id": "9012-c14b:55.3:64.0666666667",
  "dictionary_id": "a101_dissimilar_action20",
  "pred_action": "attempt to attach water tank",
  "labels": ["attempt to attach water tank", "unscrew chassis"],
  "cosine_similarity": [0.15195179, 0.14894506],
  "cosine_distance": [0.84804821, 0.85105491]
}
```

Downstream systems should usually consume `pred_action`, `topk_labels`, and
`topk_scores`. Use `rankings.jsonl` when a second-stage reranker or human review
needs a larger shortlist.

## Single Clip Inference

```bash
.venv/bin/python -m action_ranker.run one-clip \
  --encoder xclip \
  --bank assembly101_coarse \
  --video data/raw/assembly101/9012-c14b.mp4 \
  --start 55.3 \
  --end 64.0667 \
  --run-id one-clip-demo
```

This writes `artifacts/runs/one-clip-demo/ranking.json`.

## Integration Notes

- Keep segmentation and classification separate. The upstream segmenter owns
  `[start_sec, end_sec]`; this project owns action label ranking inside those
  intervals.
- Use a domain-specific action bank whenever possible. Full verb-object banks
  with many near-synonyms are much harder than compact task-specific banks.
- Prefer consuming Top-k or Top-50 shortlists over only Top-1 when the downstream
  system can rerank with context.
- The first X-CLIP run downloads Hugging Face weights and builds a text cache.
  Later runs reuse cached text embeddings.
- Check `run_meta.json` to verify GPU inference. A good run should report CUDA
  model/device metadata in the experiment-specific helpers, or use an X-CLIP
  encoder id with normal inference timings in `action_ranker.run`.
