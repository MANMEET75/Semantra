from pathlib import Path
from threading import Lock
from typing import Optional, Sequence

import numpy as np


def _multilingual_asset_dir() -> Path:
    try:
        from semantra_multilingual import get_asset_dir
    except ImportError as exc:
        raise RuntimeError(
            "Multilingual assets are not installed. Install semantra-classify-multilingual "
            "or install Semantra directly from GitHub."
        ) from exc
    return Path(get_asset_dir())


class OnnxEmbeddingModel:
    """Local ONNX embedding adapter. Model assets are intentionally package data."""

    def __init__(self, model_dir: Optional[str] = None, model_name: str = "english"):
        if model_name not in {"english", "multilingual"}:
            raise ValueError("model_name must be 'english' or 'multilingual'")
        self.model_dir = (
            Path(model_dir)
            if model_dir
            else Path(__file__).parent
            / "assets"
            / ("all-MiniLM-L6-v2" if model_name == "english" else "multilingual-e5-small")
        )
        self.model_name = model_name
        self.query_prefix = "query: " if model_name == "multilingual" else ""
        self.document_prefix = "passage: " if model_name == "multilingual" else ""
        self._session = None
        self._load_lock = Lock()
        self.dimension = 384

    def _load(self):
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return
            model = self.model_dir / "onnx" / "model.onnx"
            if self.model_name == "multilingual" and not model.exists():
                self.model_dir = _multilingual_asset_dir()
                model = self.model_dir / "onnx" / "model.onnx"
            if not model.exists():
                raise RuntimeError(
                    "Bundled ONNX model assets are missing. Install the matching Semantra "
                    "model package or pass a custom EmbeddingModel."
                )
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "onnxruntime and tokenizers are required for the default embedding model"
                ) from exc
            session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            tokenizer_file = self.model_dir / "tokenizer.json"
            if not tokenizer_file.exists():
                raise RuntimeError("Bundled tokenizer assets are missing")
            tokenizer = Tokenizer.from_file(str(tokenizer_file))
            tokenizer.enable_truncation(max_length=256)
            self._session = session
            self._tokenizer = tokenizer

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
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

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]
