# Semantra

Offline few-shot classification using local open-source embeddings and BM25 lexical matching.

```python
from semantra import Classifier

classifier = Classifier()
classifier.add_class("speaker", [
    "my speaker is not working",
    "no sound is coming from my device",
])
result = classifier.predict("I cannot hear any sound")
print(result.class_name, result.confidence)

# Inspect alternatives
for candidate in result.top_k:
    print(candidate.class_name, candidate.confidence)

# Persist the configured classifier
classifier.save("speaker-classifier.json")
# restored = Classifier.load("speaker-classifier.json")
```

Semantra has no hosted API dependency. The default wheel includes the Apache-2.0 ONNX model and tokenizer assets; custom embedding implementations can also be injected through `Classifier(embedding_model=...)`.

## Development

```bash
pip install -e ".[dev]"
pytest
python -m build
```

## License

Apache License 2.0. See `LICENSE`.
