import copy
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
from tqdm import tqdm


def create_splits_for_hf_hub(train_dataset: str):
    # Dataset format should be a list of dictionaries, where each dictionary represents a data point.
    path_to_train_data = f"path/to/train/{train_dataset}.json"
    with open(path_to_train_data, "r") as f:
        data = json.load(f)

    for filter_by in ["entropy", "max"]:
        dataset_dict = DatasetDict()
        for setting in ["easy", "medium", "hard"]:
            new_split = create_splits(
                data,
                train_dataset,
                filter_by=filter_by,
                setting=setting,
            )

            hf_format = [convert_to_hf_format(data_point) for data_point in new_split]

            ds = Dataset.from_pandas(pd.DataFrame(data=hf_format))
            dataset_dict[setting] = ds

        dataset_dict.push_to_hub(f"{train_dataset}_{filter_by}_splits")


def convert_to_hf_format(data_point):
    tags = ["O"] * len(data_point["tokenized_text"])
    spans = []
    for ent in data_point["ner"]:
        start, end, label = ent[0], ent[1], ent[2]
        spans.append({"start": start, "end": end, "label": label})
        if start == end:
            tags[start] = "B-" + label
        else:
            try:
                tags[start] = "B-" + label
                tags[start + 1 : end + 1] = ["I-" + label] * (end - start)
            except IndexError:
                pass
    return {"tokens": data_point["tokenized_text"], "ner_tags": tags, "spans": spans}


def create_splits(
    dataset: List[Dict],
    dataset_name: str,  # The name of the dataset for which the splits should be created
    filter_by: str = "entropy",
    setting: str = "medium",
):
    try:
        df = pd.read_pickle("new_splits.pkl")
    except FileNotFoundError:
        raise FileNotFoundError("Please run the compute_new_splits function first to generate the data.")
    df = df[(df["train_dataset"] == dataset_name)]

    selected_entity_types = []
    for benchmark_name in df["eval_dataset"].unique():
        _df = df[(df["eval_dataset"] == benchmark_name)].copy()

        # The thresholds are dataset specific and may need to be adjusted to account for dataset with different characteristics
        if filter_by == "entropy":
            low_threshold = df[filter_by].quantile(0.01)
            high_threshold = df[filter_by].quantile(0.95)
        elif filter_by == "max":
            low_threshold = df[filter_by].quantile(0.05)
            high_threshold = df[filter_by].quantile(0.99)

        medium_lower_threshold = df[filter_by].quantile(0.495)
        medium_upper_threshold = df[filter_by].quantile(0.505)

        # Define conditions and choices for categorization
        conditions = [
            _df[filter_by] <= low_threshold,  # Bottom
            _df[filter_by].between(medium_lower_threshold, medium_upper_threshold),  # Middle
            _df[filter_by] >= high_threshold,  # Top
        ]
        choices = ["easy", "medium", "hard"] if filter_by == "entropy" else ["hard", "medium", "easy"]

        # Use np.select to create the new column based on the conditions
        _df["difficulty"] = np.select(conditions, choices, default="not relevant")

        selected_entity_types.extend(_df[_df["difficulty"] == setting]["entity"].tolist())

    new_dataset = []
    for dp in tqdm(dataset):
        matched_entities = [x for x in dp["ner"] if x[-1].lower().strip() in selected_entity_types]
        if matched_entities:
            new_np = copy.deepcopy(dp)
            new_np["ner"] = matched_entities
            new_dataset.append(new_np)

    return new_dataset


def compute_new_splits():
    # TODO: you need to load the data into two variables: 'benchmarks' and 'training_datasets'.
    # 'benchmarks' should be a dictionary with the benchmark names as keys and the (list of distinct) entity types as values.
    # 'training_datasets' should be a dictionary with the training dataset names as keys and the (list of distinct) entity types as values.
    # We process multiple benchmarks and training datasets in this example, but you can adjust the code to fit your needs.
    # Further, we stick with the following dataset layout: list of dictionaries, where each dictionary represents a data point.
    # For example: [{'tokenized_text': [...], 'ner': [(start, end, entity_type), ...]}, ...]

    benchmarks = {}
    for benchmark_name in ['path/to/eval/dataset1.json', 'path/to/eval/dataset2.json']:
        # Data loading logic here, e.g.:
        # tokens, entity_types = load_eval_dataset(benchmark_name)
        # benchmarks[benchmark_name] = list(entity_types)
        pass

    training_datasets = {}
    for train_dataset_name in ['path/to/train/dataset1.json', 'path/to/train/dataset2.json']:
        # Data loading logic here, e.g.:
        # tokens, entity_types = load_train_dataset(train_dataset_name)
        # training_datasets[train_dataset_name] = list(entity_types)
        pass

    batch_size = 256
    model = SentenceTransformer("all-mpnet-base-v2").to("cuda")
    eval_encodings = {}
    for benchmark_name, entity_types in benchmarks.items():
        embeddings = model.encode(entity_types, convert_to_tensor=True, device="cuda")
        eval_encodings[benchmark_name] = embeddings

    results = {}
    for dataset_name, entity_types in training_datasets.items():
        for i in tqdm(range(0, len(entity_types), batch_size)):
            dataset_name = dataset_name.split(".")[0]
            batch = entity_types[i : i + batch_size]
            embeddings = model.encode(batch, convert_to_tensor=True, device="cuda")
            for benchmark_name, eval_embeddings in eval_encodings.items():
                similarities = torch.clamp(
                    cosine_similarity(
                        embeddings.unsqueeze(1),
                        eval_embeddings.unsqueeze(0),
                        dim=2,
                    ),
                    min=0.0,
                    max=1.0,
                )
                probabilities = torch.nn.functional.softmax(similarities / 0.01, dim=1)
                entropy_values = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
                max_values, _ = torch.max(similarities, dim=1)

                if dataset_name not in results:
                    results[dataset_name] = {}
                if benchmark_name not in results[dataset_name]:
                    results[dataset_name][benchmark_name] = {}

                for j, entity in enumerate(batch):
                    if entity not in results[dataset_name][benchmark_name]:
                        results[dataset_name][benchmark_name][entity] = {}
                    results[dataset_name][benchmark_name][entity]["entropy"] = entropy_values[j].cpu().numpy().item()
                    results[dataset_name][benchmark_name][entity]["max"] = max_values[j].cpu().numpy().item()

    entries = []
    for dataset_name, eval_comparisons in results.items():
        for benchmark_name, mapping in eval_comparisons.items():
            for entity, values in mapping.items():
                entries.append(
                    {
                        "entity": entity,
                        "entropy": values["entropy"],
                        "max": values["max"],
                        "eval_dataset": benchmark_name,
                        "train_dataset": dataset_name,
                    }
                )
    df = pd.DataFrame.from_dict(entries, orient="columns")
    df.to_pickle("new_splits.pkl")
