import numpy as np
import pytest
from familiarity.embedding_models import LabelEmbeddingModel


@pytest.fixture(scope="module")
def dummy_ner_train():
    np.random.seed(42)
    train_labels_set = [
        "person",
        "location",
        "building",
        "eagle",
        "restaurant",
        "util",
    ]
    train_probs = [0.4, 0.1, 0.1, 0.1, 0.1, 0.2]
    train_labels = np.random.choice(train_labels_set, size=30000, p=train_probs).tolist()

    return train_labels


@pytest.fixture(scope="module")
def dummy_ner_test():
    np.random.seed(42)
    test_labels_set = [
        "human",
        "organization",
        "building",
        "review",
        "researcher",
        "car",
    ]
    test_probs = [0.5, 0.2, 0.05, 0.05, 0.1, 0.1]
    test_labels = np.random.choice(test_labels_set, size=30000, p=test_probs).tolist()
    return test_labels


@pytest.fixture(scope="module")
def tiny_glove_file(tmp_path_factory):
    glove_content = "word1 0.1 0.2 0.3\n" "word2 0.4 0.5 0.6"
    glove_path = tmp_path_factory.mktemp("models") / "tiny_glove.txt"
    glove_path.write_text(glove_content)
    return glove_path


@pytest.fixture(scope="module")
def tiny_fasttext_file(tmp_path_factory):
    glove_content = "2 3\n" "word1 0.1 0.2 0.3\n" "word2 0.4 0.5 0.6"
    glove_path = tmp_path_factory.mktemp("models") / "tiny_fasttext.txt"
    glove_path.write_text(glove_content)
    return glove_path


@pytest.fixture(scope="module")
def sample_embedding_model():
    class MockEmbeddingModel(LabelEmbeddingModel):
        def embed(self, batch) -> np.array:
            np.random.seed(42)
            return np.stack([np.random.rand(10) for word in batch])

    return MockEmbeddingModel()
