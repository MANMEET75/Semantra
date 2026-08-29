from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class OnnxEmbeddingModel:
    """Local ONNX embedding adapter. Model assets are intentionally package data."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = (
            Path(model_dir) if model_dir else Path(__file__).parent / "assets" / "all-MiniLM-L6-v2"
        )
        self._session = None
        self.dimension = 384

    def _load(self):
        if self._session is not None:
            return
        model = self.model_dir / "onnx" / "model.onnx"
        if not model.exists():
            raise RuntimeError(
                "Bundled ONNX model assets are missing. Install a complete Semantra wheel "
                "or pass a custom EmbeddingModel."
            )
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime and tokenizers are required for the default embedding model"
            ) from exc
        self._session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        tokenizer_file = self.model_dir / "tokenizer.json"
        if not tokenizer_file.exists():
            raise RuntimeError("Bundled tokenizer assets are missing")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.enable_truncation(max_length=256)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        encodings = self._tokenizer.encode_batch(list(texts))
        max_len = max((len(item.ids) for item in encodings), default=1)
        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention = np.zeros_like(input_ids)
        token_type = np.zeros_like(input_ids)
        for row, item in enumerate(encodings):
            length = len(item.ids)
            input_ids[row, :length] = item.ids
            attention[row, :length] = item.attention_mask
            token_type[row, :length] = item.type_ids
        inputs = {
            name: value
            for name, value in {
                "input_ids": input_ids,
                "attention_mask": attention,
                "token_type_ids": token_type,
            }.items()
            if name in {x.name for x in self._session.get_inputs()}
        }
        hidden = self._session.run(None, inputs)[0]
        mask = attention[..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return pooled.astype(np.float32)
