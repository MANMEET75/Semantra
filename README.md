# Semantra

Semantra is an offline, few-shot classification engine for Python. Define a
set of classes with example sentences and classify new text without training a
model, fine-tuning, hosted APIs, or a vector database.

It combines local semantic embeddings from a bundled open-source ONNX model
with pure-Python BM25 lexical matching. The scores are fused at
class level and checked with confidence and runner-up margin thresholds, so
ambiguous input can return `Unknown` instead of receiving an unreliable label.

## Installation

```bash
pip install semantra
```

The wheel includes the bundled model and tokenizer assets. After installation,
inference does not require internet access, an API key, a model server, or a
database. The default English model is distributed under Apache-2.0. A
multilingual mode is available for mixed-language and Hinglish input.

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

### Multilingual and Hinglish input

Use the bundled multilingual model when examples and queries may contain
Hindi, English, Hinglish, or other supported languages:

```python
from semantra import Classifier

classifier = Classifier(model="multilingual")
classifier.add_class("account_access", [
    "I cannot sign in to my account",
    "Mera account login nahi ho raha",
    "मेरा अकाउंट लॉगिन नहीं हो रहा",
])

result = classifier.predict("Mera account access nahi ho raha")
print(result.class_name, result.confidence)
```

This mode does not run a language detector, translator, or second classifier:
the multilingual embedder handles the normalized input in the same single
embedding call, while the Unicode-aware BM25 matcher uses the same text. This
keeps the pipeline simple and avoids an extra language-processing latency
stage. The multilingual model is larger than the English model, so its model
inference can be slower and use more memory; always benchmark on the target
CPU using `result.inference_time_ms`.

The multilingual model is based on XLM-R and is trained for 100 languages.
Commonly used supported languages include English, Hindi, Urdu, Bengali,
Gujarati, Marathi, Punjabi, Tamil, Telugu, Kannada, Malayalam, Nepali,
Arabic, Persian, Hebrew, Turkish, Russian, Ukrainian, Polish, Czech, Slovak,
Romanian, Hungarian, Bulgarian, Greek, German, Dutch, Danish, Swedish,
Norwegian, Finnish, French, Spanish, Portuguese, Italian, Catalan, Indonesian,
Malay, Vietnamese, Thai, Chinese, Japanese, and Korean. See the
[official multilingual-E5 model card](https://huggingface.co/intfloat/multilingual-e5-small)
for the authoritative language list and model details.

Hinglish is code-switched Hindi and English rather than a separate model
language. It is supported naturally because the same multilingual encoder sees
both scripts and the same Unicode-aware lexical matcher sees words such as
`account`, `login`, `nahi`, and `रहा`. Romanized Hindi spelling is variable, so
include the spellings your users actually write in the examples.

Language support means the model can process text in those languages; it does
not guarantee identical accuracy for every language. Performance may be lower
for low-resource languages, short inputs, slang, transliteration, or domains
that are absent from the examples. For production quality, provide several
representative examples per class in every language and writing style you
expect, then validate with a held-out test set.

## Choosing a model and understanding latency

Semantra performs no language detection, translation, or second inference
stage. Model selection happens when `Classifier` is created:

```python
english = Classifier(model="english")       # default: smaller and faster
multilingual = Classifier(model="multilingual")  # mixed languages/Hinglish
```

Both modes use the same pipeline and expose the same `Prediction` object. The
reported `inference_time_ms` includes query embedding, semantic similarity,
BM25 scoring, score fusion, ranking, and threshold checks. It does not add a
second model call or a separate timing pass; measuring the time uses only two
monotonic clock reads.

The multilingual model has more model data than the English MiniLM model, so
it normally requires more memory and can have higher CPU inference latency.
There is no single fixed latency number: CPU model, thread settings, query
length, number of examples, and warm-up all affect it. Benchmark on the
deployment machine after one warm-up call:

```python
from statistics import median
from semantra import Classifier

def benchmark(model_name):
    c = Classifier(model=model_name)
    c.add_class("support", ["I need help", "Mujhe madad chahiye"])
    c.add_class("billing", ["I have a payment question", "Payment ko lekar sawal hai"])
    c.predict("I need help")  # warm-up: excludes first-load cost
    times = [c.predict("Mujhe help chahiye").inference_time_ms for _ in range(30)]
    return median(times)

print("English median:", benchmark("english"), "ms")
print("Multilingual median:", benchmark("multilingual"), "ms")
```

The first prediction may be slower because ONNX Runtime initializes the local
model. Call `predict()` once during application startup if predictable request
latency matters. Class examples are embedded when added, so they should be
loaded once and reused rather than rebuilding the classifier for every query.

## Practical use cases

Semantra is suitable when a small set of labeled examples is available but
training a dedicated model is unnecessary:

- Customer-support intent routing across teams such as access, billing,
  delivery, returns, or technical support.
- Helpdesk and IT ticket triage for password, device, network, and software
  issues.
- Multilingual chatbot or AI-agent routing before a downstream workflow runs.
- FAQ and knowledge-base category selection using examples written by domain
  experts.
- Contact-center message tagging, escalation detection, and human handoff.
- Form, email, or feedback classification where an `Unknown` result is safer
  than forcing an incorrect class.

Semantra is not a replacement for supervised training when you have a large,
stable labeled dataset, strict regulatory calibration requirements, or highly
specialized language. Use the confidence and margin thresholds, inspect
`top_k`, and route uncertain predictions to clarification or human review.

## Prediction results

Each prediction includes diagnostics for routing, auditing, or downstream
business rules:

```python
print(result.class_name)
print(result.confidence)
print(result.semantic_score)
print(result.lexical_score)
print(result.margin)
print(result.inference_time_ms)  # end-to-end prediction latency in milliseconds

for candidate in result.top_k:
    print(candidate.class_name, candidate.confidence)
```

`confidence` is a normalized ranking score, not a calibrated probability.
`semantic_score` measures embedding similarity, `lexical_score` measures BM25
matching, and `margin` is the difference between the top two class scores.
`inference_time_ms` measures end-to-end `predict()` latency, including query
embedding and ranking, in milliseconds. It uses two monotonic clock reads and
does not perform additional inference or searches, so its measurement overhead
is negligible. For reliable benchmarks, measure multiple calls and report a
median or percentile rather than relying on one prediction.

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

The selected built-in model is saved with the classifier metadata, so loading
a multilingual classifier automatically restores multilingual mode. You can
override it with `Classifier.load("classifier.json", model="english")` when
using a compatible persisted embedding matrix.

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
    print(f"Latency: {result.inference_time_ms:.2f} ms")
    print("Top candidates:", [(x.class_name, round(x.confidence, 3)) for x in result.top_k])
    print()
```

```python
classifier.save("colab-classifier.json")
restored = Classifier.load("colab-classifier.json")
print(restored.predict("I am unable to access my profile").class_name)
```

For a multilingual Colab smoke test, use `Classifier(model="multilingual")`
and add a few equivalent examples in English, Hindi, and Hinglish. The GitHub
repository stores the large ONNX asset with Git LFS; installing from a released
PyPI wheel is the simplest offline path once a release is published.

## Design and performance

- No training or fine-tuning is required.
- Embeddings run locally with ONNX Runtime on CPU.
- Example embeddings are precomputed when classes are added.
- BM25 operates in memory over the supplied example corpus.
- The default design targets small-to-medium few-shot corpora and low-latency
  application routing.
- `model="english"` is the fastest, smallest built-in option; use
  `model="multilingual"` when language coverage is more important than model
  size.

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
