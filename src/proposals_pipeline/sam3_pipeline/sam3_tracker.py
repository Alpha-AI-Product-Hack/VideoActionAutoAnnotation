"""Batched per-frame SAM3 detection of hands and hand-adjacent objects.

One vision-backbone pass per frame is decoded under two text prompts
("hand", "object being held or manipulated"). Objects are kept only when
they overlap a dilated hand mask. bf16 is several times faster than fp32
on recent GPUs with identical detections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from sam3_pipeline.mask_ops import dilate_mask, overlap_frac

HAND_PROMPT = "hand"
OBJECT_PROMPT = "object being held or manipulated"


@dataclass
class FrameMasks:
    frame_idx: int
    hand_masks: list[np.ndarray] = field(default_factory=list)
    object_masks: list[np.ndarray] = field(default_factory=list)
    object_scores: list[float] = field(default_factory=list)


class Sam3FrameDetector:
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: str | None = None,
        hand_score_threshold: float = 0.5,
        object_score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        max_objects_per_frame: int = 20,
        hand_filter_dilate_px: int = 15,
        batch_size: int = 8,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from transformers import Sam3Model, Sam3Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hand_score_threshold = hand_score_threshold
        self.object_score_threshold = object_score_threshold
        self.mask_threshold = mask_threshold
        self.max_objects_per_frame = max_objects_per_frame
        self.hand_filter_dilate_px = hand_filter_dilate_px
        self.batch_size = batch_size
        self.model = Sam3Model.from_pretrained(model_id, device_map=self.device, dtype=dtype).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self._prompt_ids: dict[str, dict] = {}

    def _text_inputs(self, prompt: str, n: int) -> dict:
        if prompt not in self._prompt_ids:
            tok = self.processor.tokenizer([prompt], return_tensors="pt", padding="max_length", max_length=32)
            self._prompt_ids[prompt] = {k: v.to(self.device) for k, v in tok.items()}
        return {k: v.expand(n, -1) for k, v in self._prompt_ids[prompt].items()}

    def _keep_near_hands(self, masks: np.ndarray, scores: np.ndarray, hand_masks: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        if not hand_masks or len(masks) == 0:
            return [], []
        dilated = [dilate_mask(h, self.hand_filter_dilate_px) for h in hand_masks]
        keep = [i for i in range(len(masks)) if any(overlap_frac(masks[i], h) > 0 for h in dilated)]
        keep.sort(key=lambda i: -scores[i])
        keep = keep[: self.max_objects_per_frame]
        return [masks[i] for i in keep], [float(scores[i]) for i in keep]

    @torch.no_grad()
    def detect(self, frames: list[np.ndarray], show_progress_bar: bool = False) -> list[FrameMasks]:
        """`frames`: RGB uint8 arrays. Returns one `FrameMasks` per frame."""
        results: list[FrameMasks] = []
        batches = range(0, len(frames), self.batch_size)
        if show_progress_bar:
            from tqdm import tqdm

            batches = tqdm(batches, desc="SAM3 frames")
        for start in batches:
            batch = frames[start:start + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            pixel_values = inputs["pixel_values"].to(next(self.model.parameters()).dtype)
            sizes = inputs["original_sizes"].tolist()
            vision = self.model.get_vision_features(pixel_values=pixel_values)
            per_prompt = {}
            for prompt, thr in ((HAND_PROMPT, self.hand_score_threshold), (OBJECT_PROMPT, self.object_score_threshold)):
                out = self.model(vision_embeds=vision, **self._text_inputs(prompt, len(batch)))
                per_prompt[prompt] = self.processor.post_process_instance_segmentation(
                    out, threshold=thr, mask_threshold=self.mask_threshold, target_sizes=sizes
                )
            for i in range(len(batch)):
                hands = per_prompt[HAND_PROMPT][i]
                objs = per_prompt[OBJECT_PROMPT][i]
                fm = FrameMasks(frame_idx=start + i, hand_masks=list(hands["masks"].detach().cpu().numpy().astype(bool)))
                fm.object_masks, fm.object_scores = self._keep_near_hands(
                    objs["masks"].detach().cpu().numpy().astype(bool), objs["scores"].detach().float().cpu().numpy(), fm.hand_masks
                )
                results.append(fm)
        return results
