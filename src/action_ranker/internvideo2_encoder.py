from __future__ import annotations

import contextlib
import importlib.machinery
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


INTERNVIDEO2_1B_MODEL_ID = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
INTERNVIDEO2_6B_MODEL_ID = "OpenGVLab/InternVideo2-Stage2_6B"
INTERNVIDEO2_1B_CHECKPOINT = "InternVideo2-stage2_1b-224p-f4.pt"


@dataclass(frozen=True)
class InternVideo2Profile:
    model_id: str
    source: str
    num_frames: int
    frame_size: int
    embed_dim: int
    checkpoint_bytes: int | None
    parameter_count: int | None
    gated: bool
    note: str

    def weight_bytes_for_dtype(self, dtype_name: str) -> int | None:
        if self.parameter_count is not None:
            bytes_per_param = 2 if dtype_name in {"float16", "bfloat16"} else 4
            return int(self.parameter_count * bytes_per_param)
        return self.checkpoint_bytes


INTERNVIDEO2_PROFILES: dict[str, InternVideo2Profile] = {
    INTERNVIDEO2_1B_MODEL_ID: InternVideo2Profile(
        model_id=INTERNVIDEO2_1B_MODEL_ID,
        source="gated HF checkpoint + OpenGVLab/InternVideo2 stage2 code",
        num_frames=4,
        frame_size=224,
        embed_dim=512,
        checkpoint_bytes=2_820_610_931,
        parameter_count=None,
        gated=True,
        note="HF repository contains a .pt checkpoint only; architecture code is reused from the OpenGVLab 6B custom-code package.",
    ),
    INTERNVIDEO2_6B_MODEL_ID: InternVideo2Profile(
        model_id=INTERNVIDEO2_6B_MODEL_ID,
        source="HF AutoModel custom_code package",
        num_frames=4,
        frame_size=224,
        embed_dim=512,
        checkpoint_bytes=None,
        parameter_count=6_366_500_794,
        gated=False,
        note="HF metadata reports 6.37B F32 parameters; fp16 weights alone exceed 8 GB VRAM.",
    ),
}


class InternVideo2Encoder:
    """Frozen InternVideo2 video-text encoder.

    The 6B HF repository is packaged as `AutoModel` custom code. The lighter 1B
    repository is gated and exposes only the `.pt` checkpoint, so this adapter
    instantiates the same Stage2 architecture from the OpenGVLab custom-code
    package and then loads the 1B checkpoint.
    """

    encoder_id = INTERNVIDEO2_1B_MODEL_ID
    num_frames = 4

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        dtype: str = "auto",
        num_frames: int | None = None,
        text_batch_size: int = 32,
        checkpoint_path: str | Path | None = None,
        checkpoint_repo: str | None = None,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "InternVideo2 requires torch plus the internvideo2 extra: "
                "pip install -e '.[internvideo2]'"
            ) from exc

        self.torch = torch
        self.encoder_id = model_id or self.encoder_id
        if self.encoder_id not in INTERNVIDEO2_PROFILES:
            raise ValueError(f"Unsupported InternVideo2 model id: {self.encoder_id}")
        self.profile = INTERNVIDEO2_PROFILES[self.encoder_id]
        self.num_frames = int(num_frames or self.profile.num_frames)
        if self.num_frames != self.profile.num_frames:
            raise ValueError(
                f"{self.encoder_id} is configured for {self.profile.num_frames} frames; "
                f"got --num-frames {self.num_frames}"
            )
        self.frame_size = int(self.profile.frame_size)
        self.dim = int(self.profile.embed_dim)
        self.device = _resolve_device(torch, device)
        self.dtype = _resolve_dtype(torch, dtype, self.device)
        self.text_batch_size = int(text_batch_size)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else _env_path("INTERNVIDEO2_1B_CHECKPOINT_PATH")
        self.checkpoint_repo = checkpoint_repo or os.environ.get("INTERNVIDEO2_1B_CHECKPOINT_REPO") or INTERNVIDEO2_1B_MODEL_ID
        if self.text_batch_size < 1:
            raise ValueError("text_batch_size must be >= 1")

        if self.encoder_id == INTERNVIDEO2_1B_MODEL_ID:
            self.model = self._load_1b_model()
        else:
            self.model = self._load_hf_automodel()

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode_clips(self, frames: np.ndarray) -> np.ndarray:
        array = np.asarray(frames, dtype=np.float32)
        if array.ndim != 5 or array.shape[0] == 0:
            raise ValueError("frames must be float32 [B, T, C, H, W]")
        if array.shape[1] != self.num_frames:
            raise ValueError(f"InternVideo2 expects T={self.num_frames} frames")
        tensor = self._frames_to_tensor(array)
        with self.torch.inference_mode():
            feats = self.model.get_vid_feat(tensor)
        return _to_numpy(feats)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Empty text batch is not allowed")
        rows = []
        with self.torch.inference_mode():
            for offset in range(0, len(texts), self.text_batch_size):
                batch = list(texts[offset : offset + self.text_batch_size])
                try:
                    feats = self.model.get_txt_feat(batch)
                except Exception:  # noqa: BLE001 - some remote tokenizers accept only str
                    feats = self.torch.cat([self.model.get_txt_feat(text) for text in batch], dim=0)
                rows.append(feats)
        return _to_numpy(self.torch.cat(rows, dim=0))

    def _frames_to_tensor(self, frames: np.ndarray):
        # Project decoder already returns [0, 1] RGB CHW; InternVideo2 expects
        # ImageNet-normalized frames in the same [B, T, C, H, W] layout.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3, 1, 1)
        normalized = (np.clip(frames, 0.0, 1.0) - mean) / std
        tensor = self.torch.from_numpy(np.ascontiguousarray(normalized)).to(self.device, non_blocking=True)
        if self.dtype is not None:
            tensor = tensor.to(dtype=self.dtype)
        return tensor

    def _load_hf_automodel(self):
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("InternVideo2 requires transformers: pip install -e '.[internvideo2]'") from exc

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.dtype is not None:
            kwargs["torch_dtype"] = self.dtype
        with _flash_attn_import_shim():
            model = AutoModel.from_pretrained(self.encoder_id, **kwargs)
        return model.to(self.device)

    def _load_1b_model(self):
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as exc:
            raise ImportError("InternVideo2 1B requires huggingface_hub: pip install -e '.[internvideo2]'") from exc

        code_dir = Path(
            snapshot_download(
                INTERNVIDEO2_6B_MODEL_ID,
                allow_patterns=["*.py", "configs/*.json", "config.json"],
            )
        )
        checkpoint_path = self._resolve_1b_checkpoint(hf_hub_download)
        model = self._build_1b_architecture(code_dir)
        checkpoint = self.torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = _checkpoint_state_dict(checkpoint)
        state_dict, adapt_info = _adapt_1b_state_dict_for_remote_code(state_dict, model)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        self.load_info = {
            "checkpoint_path": str(checkpoint_path),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            **adapt_info,
        }
        del checkpoint, state_dict
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        if self.dtype is not None:
            model = model.to(dtype=self.dtype)
        model = model.to(self.device)
        _set_remote_config_device(model, self.device)
        return model

    def _resolve_1b_checkpoint(self, hf_hub_download) -> Path:
        if self.checkpoint_path is not None:
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(f"InternVideo2 1B checkpoint is missing: {self.checkpoint_path}")
            return self.checkpoint_path
        try:
            return Path(
                hf_hub_download(
                    repo_id=self.checkpoint_repo,
                    filename=INTERNVIDEO2_1B_CHECKPOINT,
                    token=True,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep HF errors readable at CLI boundary
            message = str(exc)
            if "gated repo" in message.lower() or "not in the authorized list" in message.lower():
                raise RuntimeError(
                    "Cannot download InternVideo2 1B checkpoint from "
                    f"{self.checkpoint_repo!r}: the current Hugging Face token is not authorized. "
                    f"Request/accept access at https://huggingface.co/{INTERNVIDEO2_1B_MODEL_ID}, "
                    "or pass --internvideo2-checkpoint-path to a local checkpoint."
                ) from exc
            raise

    def _build_1b_architecture(self, code_dir: Path):
        self._ensure_bert_large_cached()
        _patch_transformers_modeling_utils()
        with _flash_attn_import_shim(), _prepend_sys_path(code_dir):
            import modeling_internvideo2 as modeling  # type: ignore[import-not-found]

        cfg_path = code_dir / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["device"] = self.device
        cfg["num_frames"] = self.num_frames
        cfg["num_frames_test"] = self.num_frames
        cfg["origin_num_frames"] = self.num_frames
        cfg["size_t"] = self.frame_size
        cfg["torch_dtype"] = _dtype_name(self.dtype)
        cfg["use_half_precision"] = self.dtype == self.torch.float16
        cfg["use_bf16"] = self.dtype == self.torch.bfloat16
        cfg["use_flash_sdp"] = False
        cfg["use_mem_efficient_sdp"] = False
        cfg["gradient_checkpointing"] = False
        cfg.setdefault("model", {})
        cfg["model"].setdefault("vision_encoder", {})
        cfg["model"]["vision_encoder"].update(
            {
                "name": "pretrain_internvideo2_1b_patch14_224",
                "num_frames": self.num_frames,
                "use_flash_attn": False,
                "use_fused_rmsnorm": False,
                "use_fused_mlp": False,
            }
        )
        cfg["model"].setdefault("text_encoder", {})
        cfg["model"]["text_encoder"]["config"] = str(code_dir / "configs" / "config_bert_large.json")
        config = modeling.InternVideo2_Stage2_Config(**cfg)
        model = modeling.InternVideo2_Stage2(config=config, is_pretrain=True)
        _set_remote_config_device(model, self.device)
        return model

    def _ensure_bert_large_cached(self) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("InternVideo2 1B requires huggingface_hub: pip install -e '.[internvideo2]'") from exc
        snapshot_download(
            "bert-large-uncased",
            allow_patterns=[
                "config.json",
                "pytorch_model.bin",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
            ],
        )


def internvideo2_preflight(model_id: str, *, dtype: str = "auto", device: str | None = None) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"model_id": model_id, "ok": False, "reason": "torch is not installed"}
    if model_id not in INTERNVIDEO2_PROFILES:
        raise ValueError(f"Unsupported InternVideo2 model id: {model_id}")
    profile = INTERNVIDEO2_PROFILES[model_id]
    resolved_device = _resolve_device(torch, device)
    resolved_dtype = _resolve_dtype(torch, dtype, resolved_device)
    dtype_name = _dtype_name(resolved_dtype)
    cuda_available = bool(torch.cuda.is_available())
    total_bytes = 0
    free_bytes = 0
    gpu_name = None
    if cuda_available:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        gpu_name = torch.cuda.get_device_name(0)
    weight_bytes = profile.weight_bytes_for_dtype(dtype_name)
    # Leave conservative room for activations, tokenizer/text encoder work, CUDA
    # context, allocator fragmentation, and temporary checkpoint tensors.
    recommended_bytes = int(weight_bytes * 1.35) if weight_bytes is not None else None
    runnable_on_gpu = bool(
        resolved_device == "cuda"
        and cuda_available
        and recommended_bytes is not None
        and free_bytes >= recommended_bytes
    )
    return {
        "model_id": model_id,
        "source": profile.source,
        "gated": profile.gated,
        "device": resolved_device,
        "dtype": dtype_name,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "total_vram_bytes": int(total_bytes),
        "free_vram_bytes": int(free_bytes),
        "estimated_weight_bytes": int(weight_bytes) if weight_bytes is not None else None,
        "recommended_vram_bytes": recommended_bytes,
        "runnable_on_gpu": runnable_on_gpu,
        "note": profile.note,
    }


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model", "module", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _adapt_1b_state_dict_for_remote_code(state_dict: dict[str, Any], model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    target_keys = set(model.state_dict().keys())
    adapted: dict[str, Any] = {}
    remapped_gamma = 0
    dropped: list[str] = []
    for key, value in state_dict.items():
        target_key = key
        if key.endswith(".gamma"):
            candidate = f"{key[:-6]}.weight"
            if candidate in target_keys:
                target_key = candidate
                remapped_gamma += 1
        if target_key in target_keys:
            adapted[target_key] = value
        else:
            dropped.append(key)
    return adapted, {
        "remapped_gamma_keys": remapped_gamma,
        "dropped_unexpected_keys": len(dropped),
        "dropped_unexpected_key_examples": dropped[:8],
    }


def _resolve_device(torch: Any, device: str | None) -> str:
    if device in {None, "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    return device


def _resolve_dtype(torch: Any, dtype: str, device: str):
    if dtype == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError("dtype must be one of: auto, float16, bfloat16, float32") from exc


def _dtype_name(dtype: Any) -> str:
    if dtype is None:
        return "float32"
    text = str(dtype)
    return text.rsplit(".", 1)[-1]


def _to_numpy(feats: Any) -> np.ndarray:
    if isinstance(feats, (tuple, list)):
        feats = feats[-1]
    return feats.detach().float().cpu().numpy().astype(np.float32)


def _set_remote_config_device(model: Any, device: str) -> None:
    for attr in ("_config", "config"):
        cfg = getattr(model, attr, None)
        if cfg is not None:
            with contextlib.suppress(Exception):
                setattr(cfg, "device", device)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _patch_transformers_modeling_utils() -> None:
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
    except ImportError:
        return
    for name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))


@contextlib.contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    text = str(path)
    sys.path.insert(0, text)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(text)


@contextlib.contextmanager
def _flash_attn_import_shim() -> Iterator[None]:
    """Let OpenGVLab code import when flash-attn is not installed.

    InternVideo2 imports flash-attn modules at module import time even when the
    selected config disables flash attention. The shim is intentionally tiny and
    raises if a disabled path is accidentally used.
    """

    installed = "flash_attn" in sys.modules
    if installed:
        yield
        return

    def unavailable(*_args, **_kwargs):
        raise ImportError("flash-attn is not installed; disable use_flash_attn/use_fused_* or install flash-attn")

    modules = {
        "flash_attn": types.ModuleType("flash_attn"),
        "flash_attn.modules": types.ModuleType("flash_attn.modules"),
        "flash_attn.modules.mlp": types.ModuleType("flash_attn.modules.mlp"),
        "flash_attn.ops": types.ModuleType("flash_attn.ops"),
        "flash_attn.ops.rms_norm": types.ModuleType("flash_attn.ops.rms_norm"),
        "flash_attn.flash_attn_interface": types.ModuleType("flash_attn.flash_attn_interface"),
        "flash_attn.bert_padding": types.ModuleType("flash_attn.bert_padding"),
    }
    for name, module in modules.items():
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    modules["flash_attn.modules.mlp"].FusedMLP = unavailable
    modules["flash_attn.ops.rms_norm"].DropoutAddRMSNorm = unavailable
    modules["flash_attn.flash_attn_interface"].flash_attn_varlen_qkvpacked_func = unavailable
    modules["flash_attn.bert_padding"].unpad_input = unavailable
    modules["flash_attn.bert_padding"].pad_input = unavailable
    previous = {name: sys.modules.get(name) for name in modules}
    os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "TRUE")
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
