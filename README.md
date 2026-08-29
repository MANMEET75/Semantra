# Semantra

Semantra is an offline, few-shot classification engine for Python. Define a
set of classes with example sentences and classify new text without training a
model, fine-tuning, hosted APIs, or a vector database.

It combines local semantic embeddings from the bundled open-source MiniLM
ONNX model with pure-Python BM25 lexical matching. The scores are fused at
class level and checked with confidence and runner-up margin thresholds, so
ambiguous input can return `Unknown` instead of receiving an unreliable label.

## Installation

```bash
pip install semantra
```

The wheel includes the model and tokenizer assets. After installation,
inference does not require internet access, an API key, a model server, or a
database. The bundled model is English-first and distributed under Apache-2.0.

## Quick start

```python
from semantra import Classifier

classifier = Classifier()
classifier.add_class("account_access", [
    "I cannot sign in to my account",
    "The login page keeps rejecting my password",
    "I am locked out of my profile",
])
classifier.add_class("billing", [
    "I was charged twice for the same order",
    "There is an unexpected charge on my invoice",
    "How can I update my payment details?",
])
classifier.add_class("shipping", [
    "Where is my delivery?",
    "My order has not arrived yet",
    "Can I change the delivery address?",
])

result = classifier.predict("I have been locked out of my profile")
print(result.class_name)   # account_access
print(result.confidence)   # normalized confidence score
```

## Prediction results

Each prediction includes diagnostics for routing, auditing, or downstream
business rules:

```python
print(result.class_name)
print(result.confidence)
print(result.semantic_score)
print(result.lexical_score)
print(result.margin)

for candidate in result.top_k:
    print(candidate.class_name, candidate.confidence)
```

`confidence` is a normalized ranking score, not a calibrated probability.
`semantic_score` measures embedding similarity, `lexical_score` measures BM25
matching, and `margin` is the difference between the top two class scores.

## Unknown and ambiguity handling

By default, Semantra returns `Unknown` when the top score is below the minimum
confidence or is too close to the runner-up:

```python
from semantra import Classifier, ClassifierConfig

classifier = Classifier(config=ClassifierConfig(
    semantic_weight=0.75,
    lexical_weight=0.25,
    min_confidence=0.60,
    min_margin=0.08,
    top_k=3,
))
```

When a prediction is Unknown, `result.top_k` still contains the strongest
candidates so an application can request clarification or route to a human.

## Persistence

Build the classifier during deployment and persist it for fast process startup:

```python
classifier.save("classifier.json")
restored = Classifier.load("classifier.json")
result = restored.predict("Please update the address for my order")
```

Persistence stores the examples, configuration, and precomputed embeddings.
The model remains local and is used only to embed new queries.

## Custom components

The default embedding, lexical, vector, and fusion components are replaceable:

```python
from semantra import Classifier, WeightedScoreFusion

classifier = Classifier(
    embedding_model=my_embedding_model,
    lexical_searcher=my_lexical_searcher,
    vector_searcher=my_vector_searcher,
    score_fusion=WeightedScoreFusion(semantic_weight=0.65, lexical_weight=0.35),
)
```

Custom embedding models must provide `dimension` and an `embed(texts)` method
returning one NumPy vector per input text.

## Google Colab test

Until a PyPI release is published, install directly from GitHub in a fresh
Google Colab notebook:

```python
!pip install -q "git+https://github.com/MANMEET75/Semantra.git"
```

```python
from semantra import Classifier, ClassifierConfig

classifier = Classifier(config=ClassifierConfig(top_k=3))
classifier.add_class("account_access", [
    "I cannot sign in", "My account is locked", "The password reset is not working",
])
classifier.add_class("billing", [
    "I do not recognize this charge", "Please explain my invoice", "I need to change my payment method",
])
classifier.add_class("shipping", [
    "My package is late", "Where is my order?", "I need to update the delivery address",
])

for query in [
    "The password reset link does not work",
    "Why was I charged an extra fee?",
    "My parcel still has not arrived",
    "Tell me something unrelated to these topics",
]:
    result = classifier.predict(query)
    print(f"Query: {query}")
    print(f"Class: {result.class_name or 'Unknown'}")
    print(f"Confidence: {result.confidence:.3f}")
    print("Top candidates:", [(x.class_name, round(x.confidence, 3)) for x in result.top_k])
    print()
```

```python
classifier.save("colab-classifier.json")
restored = Classifier.load("colab-classifier.json")
print(restored.predict("I am unable to access my profile").class_name)
```

## Design and performance

- No training or fine-tuning is required.
- Embeddings run locally with ONNX Runtime on CPU.
- Example embeddings are precomputed when classes are added.
- BM25 operates in memory over the supplied example corpus.
- The default design targets small-to-medium few-shot corpora and low-latency
  application routing.

## Development and releases

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
python -m build
```

Release versions are maintained in `pyproject.toml` and documented in
`CHANGELOG.md`. GitHub Actions tests Python 3.9–3.13 and publishes tagged
releases through PyPI Trusted Publishing, without long-lived API tokens.

## License

Semantra is released under Apache License 2.0. The bundled `all-MiniLM-L6-v2`
ONNX model is also distributed under Apache-2.0; see its `NOTICE` file.
