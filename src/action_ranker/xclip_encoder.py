from __future__ import annotations

import numpy as np


class XClipEncoder:
    encoder_id = "microsoft/xclip-base-patch32"
    num_frames = 8

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        num_frames: int | None = None,
    ):
        try:
            import torch
            from transformers import XCLIPModel, XCLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "X-CLIP requires extras: pip install -e '.[torch]'"
            ) from exc
        self.torch = torch
        self.encoder_id = model_id or self.encoder_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = XCLIPProcessor.from_pretrained(self.encoder_id)
        self.model = XCLIPModel.from_pretrained(self.encoder_id)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.dim = int(getattr(self.model.config, "projection_dim", None) or 512)
        trained_t = int(self.model.mit.position_embedding.shape[1])
        self._set_num_frames(trained_t if num_frames is None else int(num_frames))

    def _set_num_frames(self, num_frames: int) -> None:
        """Use T frames. If T differs from the checkpoint, interpolate MIT pos emb.

        Cross-frame layers also store `num_frames` and reshape B*T with that value.
        Frozen interpolation is not training.
        """
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        torch = self.torch
        model = self.model
        trained_t = int(model.mit.position_embedding.shape[1])
        self.num_frames = int(num_frames)
        vision_cfg = model.config.vision_config
        vision_cfg.num_frames = self.num_frames
        for layer in model.vision_model.encoder.layers:
            layer.num_frames = self.num_frames
        if self.num_frames != trained_t:
            pe = model.mit.position_embedding.detach()
            stretched = torch.nn.functional.interpolate(
                pe.permute(0, 2, 1),
                size=self.num_frames,
                mode="linear",
                align_corners=True,
            ).permute(0, 2, 1)
            model.mit.position_embedding = torch.nn.Parameter(stretched, requires_grad=False)

    def encode_clips(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[0] == 0:
            raise ValueError("Empty clip batch is not allowed")
        if frames.shape[1] != self.num_frames:
            raise ValueError(f"X-CLIP expects T={self.num_frames} frames")
        videos = []
        for clip in frames:
            thwc = np.transpose(clip, (0, 2, 3, 1))
            uint8 = np.clip(thwc * 255.0, 0, 255).astype(np.uint8)
            videos.append([frame for frame in uint8])
        with self.torch.no_grad():
            inputs = self._process_videos(videos)
            pixel = inputs["pixel_values"].to(self.device)
            feats = self.model.get_video_features(pixel_values=pixel)
        return self._to_numpy(feats)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Empty text batch is not allowed")
        with self.torch.no_grad():
            inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
            kwargs = {k: v.to(self.device) for k, v in inputs.items() if k in {"input_ids", "attention_mask"}}
            feats = self.model.get_text_features(**kwargs)
        return self._to_numpy(feats)

    def score_clip_texts(self, frames: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        """Full X-CLIP logits path: MIT video embed vs video-conditioned text prompts.

        `text_embeddings` are projected text features [M, D] from `encode_texts`.
        Returns cosine similarity [B, M] after `prompts_generator` (same order as
        `model.logits_per_video / logit_scale`).
        """
        if frames.shape[0] == 0:
            raise ValueError("Empty clip batch is not allowed")
        if frames.shape[1] != self.num_frames:
            raise ValueError(f"X-CLIP expects T={self.num_frames} frames")
        texts = np.asarray(text_embeddings, dtype=np.float32)
        if texts.ndim != 2:
            raise ValueError("text_embeddings must be [M, D]")
        videos = []
        for clip in frames:
            thwc = np.transpose(clip, (0, 2, 3, 1))
            uint8 = np.clip(thwc * 255.0, 0, 255).astype(np.uint8)
            videos.append([frame for frame in uint8])
        model = self.model
        torch = self.torch
        with torch.no_grad():
            inputs = self._process_videos(videos)
            pixel_values = inputs["pixel_values"].to(self.device)
            batch_size, num_frames, num_channels, height, width = pixel_values.shape
            flat = pixel_values.reshape(-1, num_channels, height, width)
            vision_outputs = model.vision_model(pixel_values=flat)
            video_embeds = model.visual_projection(vision_outputs[1])
            cls_features = video_embeds.view(batch_size, num_frames, -1)
            mit_outputs = model.mit(cls_features)
            video_embeds = mit_outputs[1]

            img_features = vision_outputs[0][:, 1:, :]
            img_features = model.prompts_visual_layernorm(img_features)
            img_features = img_features @ model.prompts_visual_projection
            img_features = img_features.view(batch_size, num_frames, -1, video_embeds.shape[-1])
            img_features = img_features.mean(dim=1, keepdim=False)

            text_embeds = torch.as_tensor(texts, device=self.device, dtype=video_embeds.dtype)
            text_embeds = text_embeds.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            text_embeds = text_embeds + model.prompts_generator(text_embeds, img_features)

            video_embeds = video_embeds / video_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            cosine = torch.einsum("bd,bkd->bk", video_embeds, text_embeds)
        return cosine.detach().cpu().numpy().astype(np.float32)

    def score_frames_texts(self, frames: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        """FAES-style frame-action cosine matrix.

        Returns [B, T, M], where each sampled frame feature is compared to each
        cached action text embedding before any temporal aggregation.
        """
        if frames.shape[0] == 0:
            raise ValueError("Empty clip batch is not allowed")
        if frames.shape[1] != self.num_frames:
            raise ValueError(f"X-CLIP expects T={self.num_frames} frames")
        texts = np.asarray(text_embeddings, dtype=np.float32)
        if texts.ndim != 2:
            raise ValueError("text_embeddings must be [M, D]")
        videos = []
        for clip in frames:
            thwc = np.transpose(clip, (0, 2, 3, 1))
            uint8 = np.clip(thwc * 255.0, 0, 255).astype(np.uint8)
            videos.append([frame for frame in uint8])
        model = self.model
        torch = self.torch
        with torch.no_grad():
            inputs = self._process_videos(videos)
            pixel_values = inputs["pixel_values"].to(self.device)
            batch_size, num_frames, num_channels, height, width = pixel_values.shape
            flat = pixel_values.reshape(-1, num_channels, height, width)
            vision_outputs = model.vision_model(pixel_values=flat)
            frame_embeds = model.visual_projection(vision_outputs[1])
            frame_embeds = frame_embeds.view(batch_size, num_frames, -1)
            text_embeds = torch.as_tensor(texts, device=self.device, dtype=frame_embeds.dtype)

            frame_embeds = frame_embeds / frame_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            cosine = torch.einsum("btd,md->btm", frame_embeds, text_embeds)
        return cosine.detach().cpu().numpy().astype(np.float32)

    def score_clip_texts_joint(self, frames: np.ndarray, prompts: list[str]) -> np.ndarray:
        """Run video and text through one `model.forward` (prompts_generator inside).

        `prompts` are already rendered strings. Returns cosine [B, M].
        """
        if frames.shape[0] == 0:
            raise ValueError("Empty clip batch is not allowed")
        if frames.shape[1] != self.num_frames:
            raise ValueError(f"X-CLIP expects T={self.num_frames} frames")
        if not prompts:
            raise ValueError("Empty prompt list is not allowed")
        rows = []
        torch = self.torch
        with torch.no_grad():
            for clip in frames:
                thwc = np.transpose(clip, (0, 2, 3, 1))
                uint8 = np.clip(thwc * 255.0, 0, 255).astype(np.uint8)
                video = [frame for frame in uint8]
                inputs = self.processor(
                    text=list(prompts),
                    videos=video,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                if "pixel_values" not in inputs:
                    inputs = self.processor(
                        text=list(prompts),
                        images=video,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                kwargs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                    if key in {"input_ids", "attention_mask", "pixel_values"}
                }
                out = self.model(**kwargs)
                logits = out.logits_per_video
                if not torch.is_tensor(logits):
                    logits = logits[0] if isinstance(logits, (tuple, list)) else logits
                scale = self.model.logit_scale.exp()
                cosine = logits / scale
                rows.append(cosine.reshape(-1).detach().cpu().numpy().astype(np.float32))
        return np.stack(rows, axis=0)

    def _process_videos(self, videos: list[list[np.ndarray]]):
        inputs = self.processor(videos=videos, return_tensors="pt")
        if "pixel_values" not in inputs:
            inputs = self.processor(images=videos, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        if pixel_values.ndim == 4:
            if pixel_values.shape[0] % self.num_frames != 0:
                raise ValueError("X-CLIP processor returned an unexpected frame batch shape")
            pixel_values = pixel_values.reshape(-1, self.num_frames, *pixel_values.shape[1:])
            inputs["pixel_values"] = pixel_values
        if pixel_values.ndim != 5:
            raise ValueError("X-CLIP processor must return pixel_values with shape [B,T,C,H,W]")
        return inputs

    def _to_numpy(self, feats) -> np.ndarray:
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        return feats.detach().cpu().numpy().astype(np.float32)
