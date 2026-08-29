import numpy as np

from semantra import Classifier


class FakeEmbedding:
    dimension = 2

    def embed(self, texts):
        return np.array(
            [
                [1, 0]
                if "sound" in t or "speaker" in t
                else ([0, 1] if "battery" in t or "charge" in t else [1, 1])
                for t in texts
            ],
            dtype=np.float32,
        )

    def embed_documents(self, texts):
        return self.embed(texts)

    def embed_query(self, text):
        return self.embed([text])[0]


def test_public_api_and_unknown():
    c = Classifier(FakeEmbedding())
    c.add_class("speaker", ["speaker has no sound"])
    c.add_class("battery", ["battery will not charge"])
    assert c.predict("sound stopped").class_name == "speaker"
    assert c.predict("sound stopped").inference_time_ms >= 0
    assert c.predict("something unrelated").is_unknown


def test_save_load(tmp_path):
    c = Classifier(FakeEmbedding()).add_class("speaker", ["speaker sound"])
    path = tmp_path / "classifier.json"
    c.save(str(path))
    loaded = Classifier.load(str(path), FakeEmbedding())
    assert loaded.predict("sound").class_name == "speaker"


def test_multilingual_model_selection_without_language_detection():
    classifier = Classifier(model="multilingual")
    assert classifier.embedding_model.model_name == "multilingual"
    assert classifier.embedding_model.query_prefix == "query: "
    assert classifier.embedding_model.document_prefix == "passage: "
