import logging
from collections import Counter
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from familiarity.embedding_models import LabelEmbeddingModel, load_embedding_model
from familiarity.logger import setup_logger
from familiarity.utils import (
    clipped_cosine_similarity,
    combine_counters,
    cumsum_until,
    df_to_prettytable,
    iterate_dict_in_batches,
    make_output_path,
)
from tqdm import tqdm


def compute_embeddings(
    train_labels_count: Counter,
    test_labels_count: Counter,
    model: LabelEmbeddingModel,
    batch_size: int = 32,
    output_path: Path = None,
    save_embeddings: bool = False,
) -> pd.DataFrame:

    embedding_df = pd.DataFrame(columns=["label", "count_train", "count_test", "embedding"])
    iterator = iterate_dict_in_batches(combine_counters(train_labels_count, test_labels_count), batch_size)
    for batch in tqdm(iterator, desc="Embedding Labels..."):
        words, counts = zip(*batch.items())
        train_counts, test_counts = zip(*counts)
        embeddings = model.embed(words)
        embedding_df = pd.concat(
            [
                embedding_df,
                pd.DataFrame(
                    {
                        "label": words,
                        "count_train": train_counts,
                        "count_test": test_counts,
                        "embedding": list(embeddings),
                    }
                ),
            ]
        )

    if save_embeddings and output_path:
        embedding_df.to_pickle(output_path / "embedding_df.pkl")

    return embedding_df


def compute_similarities(
    embedding_df: pd.DataFrame,
    output_path: Path = None,
    save_embeddings: bool = False,
) -> pd.DataFrame:
    train_df = (
        embedding_df.loc[embedding_df["count_train"] > 0]
        .drop(columns=["count_test"])
        .rename(columns={"count_train": "count"})
    )
    test_df = (
        embedding_df.loc[embedding_df["count_test"] > 0]
        .drop(columns=["count_train"])
        .rename(columns={"count_test": "count"})
    )
    similarity_df = pd.merge(train_df, test_df, how="cross", suffixes=["_train", "_test"])
    similarity_df["similarity"] = similarity_df.apply(
        lambda row: clipped_cosine_similarity(row["embedding_train"], row["embedding_test"]),
        axis=1,
    )
    similarity_df.drop(columns=["embedding_train", "embedding_test"], inplace=True)

    if save_embeddings:
        similarity_df.to_pickle(output_path / "similarity_df.pkl")

    return similarity_df


def compute_familiarity(
    similarity_df: pd.DataFrame,
    k: int = 1000,
    weighting: str = "zipf",
    output_path: Path = None,
    save_embeddings: bool = False,
) -> pd.DataFrame:
    familiarity_data = []

    for label_test in similarity_df["label_test"].unique():
        test_label_df = similarity_df[similarity_df["label_test"] == label_test]
        test_label_df = test_label_df.sort_values("similarity", ascending=False)
        counts = cumsum_until(test_label_df["count_train"], k)
        sims = test_label_df["similarity"][: len(counts)]
        familiarity = weighted_average(sims, counts, k, weighting=weighting)
        familiarity_data.append({"label": label_test, "familiarity": familiarity})

    familiarity_df = pd.DataFrame(familiarity_data)

    if save_embeddings:
        familiarity_df.to_pickle(output_path / "familiarity_df.pkl")

    return familiarity_df


def weighted_average(
    similarities: List[float],
    counts: List[int],
    k: int,
    weighting: str = "zipf",
) -> float:
    if weighting not in ["unweighted", "linear_decay", "zipf"]:
        raise ValueError(f"Possible weighting options: unweighted, linear_decay, zipf. {weighting} is not an option.")

    if weighting == "unweighted":
        return np.dot(np.array(similarities), np.array(counts)) / k

    if weighting == "linear_decay":
        linear_decay_weights = np.arange(1, k + 1, 1)[::-1] / k
        return np.dot(linear_decay_weights, np.repeat(similarities, counts)) / np.sum(linear_decay_weights)

    if weighting == "zipf":
        zipf_weights = 1 / np.arange(1, k + 1, 1)
        return np.dot(zipf_weights, np.repeat(similarities, counts)) / np.sum(zipf_weights)


def compute_metric(
    train_labels: List[str],
    test_labels: List[str],
    model_name_or_path: Union[Path, str],
    batch_size: int = 32,
    k: int = 1000,
    weighting: str = "zipf",
    save_results: bool = False,
    save_embeddings: bool = False,
) -> None:

    train_labels_count = Counter(train_labels)
    test_labels_count = Counter(test_labels)
    model = load_embedding_model(model_name_or_path)

    output_path = None
    if save_results or save_embeddings:
        output_path = make_output_path()

    logger = setup_logger(output_path) if output_path else logging.getLogger(__name__)
    logger.info(50 * '-')
    logger.info(f"Train Labels Counter: {train_labels_count}")
    logger.info(f"Test Labels Counter: {test_labels_count}")
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"k-cutoff: {k}")
    logger.info(50 * '-' + "\n")

    embedding_df = compute_embeddings(
        train_labels_count=train_labels_count,
        test_labels_count=test_labels_count,
        model=model,
        batch_size=batch_size,
        output_path=output_path,
        save_embeddings=save_embeddings,
    )

    similarity_df = compute_similarities(embedding_df, output_path=output_path, save_embeddings=save_embeddings)
    familiarity_df = compute_familiarity(
        similarity_df, k=k, weighting=weighting, output_path=output_path, save_embeddings=save_embeddings
    )
    logger.info("Results:\n")
    logger.info(df_to_prettytable(familiarity_df))
