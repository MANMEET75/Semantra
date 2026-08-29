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


def test_public_api_and_unknown():
    c = Classifier(FakeEmbedding())
    c.add_class("speaker", ["speaker has no sound"])
    c.add_class("battery", ["battery will not charge"])
    assert c.predict("sound stopped").class_name == "speaker"
    assert c.predict("something unrelated").is_unknown


def test_save_load(tmp_path):
    c = Classifier(FakeEmbedding()).add_class("speaker", ["speaker sound"])
    path = tmp_path / "classifier.json"
    c.save(str(path))
    loaded = Classifier.load(str(path), FakeEmbedding())
    assert loaded.predict("sound").class_name == "speaker"
