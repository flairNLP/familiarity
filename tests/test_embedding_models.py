import numpy as np
import pytest

from familarity.embedding_models import (
    FastTextModel,
    GloveModel,
    SentenceTransformerModel,
    TransformerModel,
    infer_embedding_model,
    load_embedding_model,
)


def test_fasttext_model(tiny_fasttext_file):
    model = FastTextModel(model_path=tiny_fasttext_file)
    result = model.embed(["word1", "word2", "unknown_word"])
    assert isinstance(result, np.ndarray)
    assert np.all(result[-1] == np.mean(result[:2], axis=0))
    assert result.shape == (3, 3)


def test_glove_model(tiny_glove_file):
    model = GloveModel(model_path=tiny_glove_file)
    result = model.embed(["word1", "word2", "unknown_word"])
    assert isinstance(result, np.ndarray)
    assert np.all(result[-1] == np.mean(result[:2], axis=0))
    assert result.shape == (3, 3)


def test_transformer_model():
    transformer_model = TransformerModel(model_name_or_path="distilbert-base-uncased")
    result = transformer_model.embed(["This is a test sentence.", "Another sentence."])
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2  # Two sentences


def test_sentence_transformer_model():
    model = SentenceTransformerModel("sentence-transformers/paraphrase-albert-small-v2")
    result = model.embed(["This is a test sentence.", "Another sentence."])
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 768)


@pytest.mark.parametrize(
    "model_path, expected_type",
    [
        ("glove-xyz", "glove"),
        ("sentence-transformers/some-model", "sentence-transformers"),
        ("wiki-news-300d-1M-subword.vec", "fasttext"),
        ("distilbert-base-uncased", "transformers"),
    ],
)
def test_infer_embedding_model(model_path, expected_type):
    assert infer_embedding_model(model_path) == expected_type


@pytest.mark.parametrize("model_path", ["random-gibberish", "unknown-format", "1234!@#$"])
def test_infer_embedding_model_invalid(model_path):
    assert infer_embedding_model(model_path) is None


@pytest.mark.parametrize("model_path", ["random-gibberish", "unknown-format", "1234!@#$"])
def test_load_embedding_model_invalid(model_path):
    with pytest.raises(ValueError, match="Embedding model type can't be inferred."):
        load_embedding_model(model_path)


def test_load_embedding_model_glove(tiny_glove_file):
    model = load_embedding_model(model_name_or_path=tiny_glove_file, embedding_model_type="glove")
    assert isinstance(model, GloveModel)


def test_load_embedding_model_sentence_transformer():
    model = load_embedding_model(
        model_name_or_path="sentence-transformers/paraphrase-albert-small-v2",
        embedding_model_type="sentence-transformers",
    )
    assert isinstance(model, SentenceTransformerModel)


def test_load_embedding_model_fasttext(tiny_fasttext_file):
    model = load_embedding_model(model_name_or_path=tiny_fasttext_file, embedding_model_type="fasttext")
    assert isinstance(model, FastTextModel)


def test_load_embedding_model_transformers():
    model = load_embedding_model(model_name_or_path="distilbert-base-uncased", embedding_model_type="transformers")
    assert isinstance(model, TransformerModel)
