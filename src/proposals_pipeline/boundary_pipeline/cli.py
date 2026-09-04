"""Command-line entry point: `python -m boundary_pipeline <command>`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from boundary_pipeline import datasets
from boundary_pipeline.sources import SOURCE_NAMES, get_source


def _select_items(args) -> list[datasets.VideoItem]:
    items = datasets.discover(args.assembly101_root, args.epic_root)
    if args.videos:
        items = [datasets.find_item(items, v) for v in args.videos.split(",")]
    return items


def cmd_extract(args) -> None:
    items = _select_items(args)
    sources = args.sources.split(",")
    print(f"{len(items)} video(s), sources={sources}", flush=True)
    if "motion" in sources:
        motion = get_source("motion")
        for it in items:
            if motion.load(it) is None or args.recompute:
                print(f"[motion] {it.video_id}", flush=True)
                motion.extract(it)
    if "sam3" in sources:
        import torch

        from sam3_pipeline.sam3_tracker import Sam3FrameDetector

        sam3 = get_source("sam3")
        todo = [it for it in items if sam3.load(it, target_fps=args.sam3_fps) is None or args.recompute]
        if todo:
            detector = Sam3FrameDetector(model_id=args.sam3_model, batch_size=args.batch_size, dtype=torch.bfloat16)
            for it in todo:
                print(f"[sam3] {it.video_id} @ {args.sam3_fps:g} fps", flush=True)
                sam3.extract(it, detector, target_fps=args.sam3_fps)
            del detector
            torch.cuda.empty_cache()
    if "abd" in sources:
        abd = get_source("abd")
        todo = [it for it in items if abd.load(it) is None or args.recompute]
        if todo:
            from action_boundaries.gebd import DDMNetScorer
            from action_boundaries.vjepa_encoder import VJepa21Encoder

            encoder, scorer = VJepa21Encoder(), DDMNetScorer()
            for it in todo:
                print(f"[abd] {it.video_id}", flush=True)
                abd.extract(it, encoder, scorer, batch_size=args.batch_size)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="boundary_pipeline", description=__doc__)
    ap.add_argument("--assembly101-root", default="data/assembly101")
    ap.add_argument("--epic-root", default="data/epic_kitchens")
    ap.add_argument("--videos", help="comma-separated video ids (e.g. a101/9014-b02a,epic/P02_123); default: all")
    ap.add_argument("--sam3-fps", type=float, default=4.0, help="SAM3 sampling rate; caches are per rate")
    sub = ap.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="compute and cache per-video signals")
    ex.add_argument("--sources", default=",".join(SOURCE_NAMES))
    ex.add_argument("--recompute", action="store_true")
    ex.add_argument("--batch-size", type=int, default=8)
    ex.add_argument("--sam3-model", default="facebook/sam3")
    ex.set_defaults(func=cmd_extract)
    _add_commands(sub)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


def _load_config(path: str):
    from boundary_pipeline.tuning import Config

    doc = json.loads(Path(path).read_text())
    return Config.from_dict(doc), doc


def _load_kwargs(sam3_fps: float) -> dict:
    return {"sam3": {"target_fps": sam3_fps}}


def _per_video_table(videos, config, tol: float) -> list[dict]:
    from boundary_pipeline.tuning import fused_times
    from boundary_pipeline.evaluation import evaluate

    rows = []
    for v in videos:
        counts = evaluate([p.time_s for p in fused_times(v, config)], v.gt)
        rows.append({"video": v.item.video_id, **{str(t): c.summary() for t, c in counts.items()}})
    return rows


def cmd_tune(args) -> None:
    from boundary_pipeline import tuning

    items = _select_items(args)
    sources = args.sources.split(",")
    videos = tuning.load_videos(datasets.benchmark_items(items, args.benchmark), sources, _load_kwargs(args.sam3_fps))
    print(f"benchmark={args.benchmark}: {len(videos)} videos, sources={sources}, objective=F1@{args.tolerance}s", flush=True)
    config, info = tuning.tune(videos, sources, tol=args.tolerance, budget=args.budget, rounds=args.rounds, seed=args.seed)
    results = tuning.ablation(videos, config, args.tolerance)
    doc = {
        "benchmark": args.benchmark,
        "objective": {"metric": "f1", "tolerance_s": args.tolerance},
        "sam3_fps": args.sam3_fps,
        "videos": [v.item.video_id for v in videos],
        **config.to_dict(),
        "results": results,
        "per_video": _per_video_table(videos, config, args.tolerance),
    }
    if args.per_video_oracle:
        doc["per_video_oracle"] = _per_video_oracle(videos, sources, args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2))
    _print_results(results, args.tolerance)
    print(f"config -> {args.out}")


def _per_video_oracle(videos, sources, args) -> dict:
    """Upper bound: every parameter re-tuned on each video separately."""
    from boundary_pipeline import tuning
    from boundary_pipeline.evaluation import aggregate

    per_video = []
    for v in videos:
        cfg, _ = tuning.tune([v], sources, tol=args.tolerance, budget=args.budget, rounds=1, seed=args.seed, log=lambda *_: None)
        counts = tuning.score([v], cfg)
        per_video.append(counts)
        print(f"  [per-video oracle] {v.item.video_id}: F1@{args.tolerance}s = {counts[args.tolerance].f1:.3f}", flush=True)
    total = aggregate(per_video)
    return {str(t): c.summary() for t, c in total.items()}


def _print_results(results: dict, tol: float) -> None:
    print(f"{'variant':16s} " + "  ".join(f"F1@{t}s" for t in ("0.5", "1.0", "2.0")) + "   P@{0}s  R@{0}s  det/gt".format(tol))
    for name, res in results.items():
        r = res[str(tol)]
        f1s = "  ".join(f"{res[t]['f1']:.3f} " for t in ("0.5", "1.0", "2.0"))
        print(f"{name:16s} {f1s}  {r['precision']:.3f}  {r['recall']:.3f}  {r['over_segmentation']:.2f}")


def cmd_evaluate(args) -> None:
    from boundary_pipeline import tuning

    config, doc = _load_config(args.config)
    items = _select_items(args)
    benchmark = args.benchmark or doc["benchmark"]
    tol = doc["objective"]["tolerance_s"]
    videos = tuning.load_videos(datasets.benchmark_items(items, benchmark), list(config.sources), _load_kwargs(doc.get("sam3_fps", args.sam3_fps)))
    results = tuning.ablation(videos, config, tol)
    _print_results(results, tol)
    print("\nper video (F1@1.0s):")
    for row in _per_video_table(videos, config, tol):
        print(f"  {row['video']:16s} {row['1.0']['f1']:.3f}  (det {row['1.0']['n_det']}, gt {row['1.0']['n_gt']})")
    if args.plots:
        from boundary_pipeline.plotting import plot_timeline

        out_dir = Path(args.plots)
        out_dir.mkdir(parents=True, exist_ok=True)
        for v in videos:
            per_source = {s: v.proposals(s, p) for s, p in config.sources.items()}
            fused = tuning.fused_times(v, config)
            lo = v.gt.spans[0][0]
            plot_timeline(out_dir / f"{v.item.slug}.png", per_source, fused, v.gt, tol, window=(lo, min(lo + args.plot_seconds, v.item.duration_s)), title=v.item.video_id)
        print(f"plots -> {out_dir}")


def cmd_run(args) -> None:
    """Proposals for one video file (no ground truth needed)."""
    import torch

    from boundary_pipeline import fusion
    from boundary_pipeline.datasets import VideoItem, _video_meta

    config, doc = _load_config(args.config)
    sam3_fps = doc.get("sam3_fps", args.sam3_fps)
    path = Path(args.video)
    fps, duration = _video_meta(path)
    item = VideoItem(f"run/{path.stem}", "custom", path, fps, duration)
    per_source = {}
    for name, params in config.sources.items():
        src = get_source(name)
        kwargs = _load_kwargs(sam3_fps).get(name, {})
        if src.load(item, **kwargs) is None:
            if name == "motion":
                src.extract(item)
            elif name == "sam3":
                from sam3_pipeline.sam3_tracker import Sam3FrameDetector

                src.extract(item, Sam3FrameDetector(model_id=args.sam3_model, batch_size=args.batch_size, dtype=torch.bfloat16), target_fps=sam3_fps)
            else:
                from action_boundaries.gebd import DDMNetScorer
                from action_boundaries.vjepa_encoder import VJepa21Encoder

                src.extract(item, VJepa21Encoder(), DDMNetScorer(), batch_size=args.batch_size)
        per_source[name] = src.propose(src.load(item, **kwargs), params)
    fused = fusion.fuse(per_source, config.fusion)
    out = {
        "video": str(path), "config": args.config,
        "boundaries": [p.to_dict() for p in fused],
        "per_source": {k: [p.to_dict() for p in v] for k, v in per_source.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"{len(fused)} boundaries -> {args.out}")
    if args.plot:
        from boundary_pipeline.plotting import plot_timeline

        plot_timeline(args.plot, per_source, fused, title=path.name)
        print(f"plot -> {args.plot}")


def _add_commands(sub) -> None:
    tu = sub.add_parser("tune", help="oracle-tune all thresholds on a benchmark")
    tu.add_argument("--benchmark", choices=list(datasets.BENCHMARKS), required=True)
    tu.add_argument("--sources", default=",".join(SOURCE_NAMES))
    tu.add_argument("--tolerance", type=float, default=1.0, help="objective tolerance (seconds)")
    tu.add_argument("--budget", type=int, default=1500, help="max configurations per source grid")
    tu.add_argument("--rounds", type=int, default=3, help="coordinate-descent rounds")
    tu.add_argument("--seed", type=int, default=0)
    tu.add_argument("--per-video-oracle", action="store_true", help="also compute the per-video upper bound")
    tu.add_argument("--out", required=True)
    tu.set_defaults(func=cmd_tune)

    ev = sub.add_parser("evaluate", help="score a config, with ablations and optional plots")
    ev.add_argument("--config", required=True)
    ev.add_argument("--benchmark", choices=list(datasets.BENCHMARKS))
    ev.add_argument("--plots", help="directory for per-video timeline PNGs")
    ev.add_argument("--plot-seconds", type=float, default=120.0)
    ev.set_defaults(func=cmd_evaluate)

    ru = sub.add_parser("run", help="proposals for an arbitrary video file")
    ru.add_argument("--config", required=True)
    ru.add_argument("--video", required=True)
    ru.add_argument("--out", required=True)
    ru.add_argument("--plot")
    ru.add_argument("--batch-size", type=int, default=8)
    ru.add_argument("--sam3-model", default="facebook/sam3")
    ru.set_defaults(func=cmd_run)
