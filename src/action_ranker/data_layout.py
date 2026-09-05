from __future__ import annotations

from pathlib import Path

from action_ranker.taxonomies import REPO_ROOT

SPLITS_DIR = REPO_ROOT / "data" / "splits"
RAW_A101 = REPO_ROOT / "data" / "raw" / "assembly101"
RAW_EPIC = REPO_ROOT / "data" / "raw" / "epic_kitchens"

DEFAULT_PATHS = {
    "epic_kitchens": {
        "video_dir": RAW_EPIC,
        "val_list": SPLITS_DIR / "epic_kitchens_val_ids.txt",
        "labels": RAW_EPIC / "intervals.csv",
    },
    "assembly101": {
        "video_dir": RAW_A101,
        "val_list": SPLITS_DIR / "assembly101_val_ids.txt",
        "labels": RAW_A101 / "intervals.csv",
    },
}

ENV_EXPORT = {
    "ACTION_RANKER_EPIC_VIDEO_DIR": DEFAULT_PATHS["epic_kitchens"]["video_dir"],
    "ACTION_RANKER_EPIC_VAL_LIST": DEFAULT_PATHS["epic_kitchens"]["val_list"],
    "ACTION_RANKER_EPIC_LABELS": DEFAULT_PATHS["epic_kitchens"]["labels"],
    "ACTION_RANKER_A101_VIDEO_DIR": DEFAULT_PATHS["assembly101"]["video_dir"],
    "ACTION_RANKER_A101_VAL_LIST": DEFAULT_PATHS["assembly101"]["val_list"],
    "ACTION_RANKER_A101_LABELS": DEFAULT_PATHS["assembly101"]["labels"],
}


def rel_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
