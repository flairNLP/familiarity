from collections import Counter

import pytest
from familiarity.metric import compute_embeddings, compute_familiarity, compute_similarities, weighted_average


def test_compute_embeddings(dummy_ner_train, dummy_ner_test, sample_embedding_model, tmp_path):
    train_counter = Counter(dummy_ner_train)
    test_counter = Counter(dummy_ner_test)
    embedding_df = compute_embeddings(
        train_labels_count=train_counter,
        test_labels_count=test_counter,
        model=sample_embedding_model,
        output_path=tmp_path,
        save_embeddings=True,
    )
    assert "embedding" in embedding_df.columns
    assert len(embedding_df) == 11
    assert embedding_df["count_train"].sum() == 30000
    assert embedding_df["count_test"].sum() == 30000
    assert (tmp_path / "embedding_df.pkl").exists()


def test_compute_similarities(dummy_ner_train, dummy_ner_test, sample_embedding_model, tmp_path):
    train_counter = Counter(dummy_ner_train)
    test_counter = Counter(dummy_ner_test)
    embedding_df = compute_embeddings(
        train_labels_count=train_counter,
        test_labels_count=test_counter,
        model=sample_embedding_model,
    )
    similarity_df = compute_similarities(embedding_df, output_path=tmp_path, save_embeddings=True)
    assert "similarity" in similarity_df.columns
    assert len(similarity_df) == (len(train_counter) * len(test_counter))
    assert similarity_df["similarity"].min() >= 0
    assert similarity_df["similarity"].max() <= 1
    assert (tmp_path / "similarity_df.pkl").exists()


def test_compute_familiarity(dummy_ner_train, dummy_ner_test, sample_embedding_model, tmp_path):
    train_counter = Counter(dummy_ner_train)
    test_counter = Counter(dummy_ner_test)
    embedding_df = compute_embeddings(
        train_labels_count=train_counter,
        test_labels_count=test_counter,
        model=sample_embedding_model,
    )
    similarity_df = compute_similarities(embedding_df)
    familiarity_df = compute_familiarity(
        similarity_df, k=2, weighting="zipf", output_path=tmp_path, save_embeddings=True
    )
    assert "familiarity" in familiarity_df.columns
    assert len(familiarity_df) == len(test_counter)
    assert pytest.approx(familiarity_df[familiarity_df["label"] == "building"]["familiarity"].iloc[0]) == 1
    assert pytest.approx(familiarity_df[familiarity_df["label"] == "car"]["familiarity"].iloc[0]) == 0.907777
    assert pytest.approx(familiarity_df[familiarity_df["label"] == "review"]["familiarity"].iloc[0]) == 0.912969
    assert (tmp_path / "familiarity_df.pkl").exists()


@pytest.mark.parametrize(
    "sims, counts, weighting, k, gold_result",
    [
        ([0.95, 0.8], [200, 200], "zipf", 400, 0.93420),
        ([0.95, 0.8, 0.7], [200, 500, 100], "zipf", 800, 0.91956),
        ([0.95, 0.8], [200, 200], "linear_decay", 400, 0.91240),
        ([0.95, 0.8, 0.7], [200, 500, 100], "linear_decay", 800, 0.86401),
        ([0.95, 0.8], [200, 200], "unweighted", 400, 0.875),
        ([0.95, 0.8, 0.7], [200, 500, 100], "unweighted", 800, 0.825),
    ],
)
def test_weighted_average(sims, counts, weighting, k, gold_result):
    result = weighted_average(sims, counts, k=k, weighting=weighting)
    assert pytest.approx(result, 0.001) == gold_result
