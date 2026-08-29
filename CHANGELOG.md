# Changelog

## 0.1.0 - 2026-08-29

- Initial offline hybrid semantic classifier.
- Supports Python 3.9 through 3.13, including current Google Colab runtimes.
- Bundled Apache-2.0 MiniLM ONNX embedding model.
- BM25 lexical matching, hybrid ranking, Unknown detection, top-k results, and persistence.
## 0.2.0 - 2026-08-29

- Add the bundled multilingual E5-small ONNX model via `Classifier(model="multilingual")`.
- Support Unicode-aware BM25 matching for Hindi, Hinglish, and other scripts.
- Persist and restore the selected built-in model mode.
- Document language coverage, latency characteristics, and multilingual usage.
- Expand README guidance with language coverage, benchmarking, and production use cases.
