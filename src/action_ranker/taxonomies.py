from __future__ import annotations

import csv
from pathlib import Path

from action_ranker.prompts import PROMPT_ID
from action_ranker.types import DictionaryRow

REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_DIR = REPO_ROOT / "data" / "action_taxonomies"

DICTIONARIES = {
    "assembly101_coarse": {
        "path": TAXONOMY_DIR / "assembly101_coarse_actions.csv",
        "label_column": "action_cls",
        "identity": "label",
    },
    "epic_kitchens_observed": {
        "path": TAXONOMY_DIR / "epic_kitchens_actions_observed.csv",
        "label_column": "prompt_action_label",
        "identity": "verb_noun_id",
    },
}


def load_dictionary_rows(dictionary_id: str) -> list[DictionaryRow]:
    spec = DICTIONARIES.get(dictionary_id)
    if spec is None:
        known = ", ".join(sorted(DICTIONARIES))
        raise ValueError(f"Unknown dictionary_id={dictionary_id!r}. Known: {known}")
    path: Path = spec["path"]
    column: str = spec["label_column"]
    if not path.is_file():
        raise FileNotFoundError(f"Taxonomy file missing: {path}")
    rows: list[DictionaryRow] = []
    seen: set[tuple] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no column {column!r}")
        for raw in reader:
            label = (raw.get(column) or "").strip()
            verb = _opt_int(raw.get("verb_id"))
            noun = _opt_int(raw.get("noun_id"))
            row = DictionaryRow(label=label, verb_id=verb, noun_id=noun)
            if spec["identity"] == "verb_noun_id" and verb is not None and noun is not None:
                key: tuple = ("id", verb, noun)
            else:
                if not label:
                    continue
                key = ("label", label)
            if key in seen:
                continue
            seen.add(key)
            if spec["identity"] == "verb_noun_id" and not label:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty dictionary after unique collapse: {path}")
    return rows


def load_dictionary_labels(dictionary_id: str) -> list[str]:
    return [row.label for row in load_dictionary_rows(dictionary_id)]


def _opt_int(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def taxonomy_path(dictionary_id: str) -> Path:
    return DICTIONARIES[dictionary_id]["path"]


def default_prompt_id() -> str:
    return PROMPT_ID
