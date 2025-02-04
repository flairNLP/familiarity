from collections import Counter
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import rootutils
import torch
from prettytable import PrettyTable


def get_device() -> str:
    """Determine if GPU available."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_output_path(base_path: Path = None) -> Path:
    if not base_path:
        base_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(base_path / f"results/{current_datetime}")
    output_path.mkdir(parents=True)
    return output_path


def df_to_prettytable(df: pd.DataFrame) -> PrettyTable:
    # Create a PrettyTable instance with column headers
    df = df.round(3)
    table = PrettyTable()
    table.field_names = df.columns.tolist()

    # Add each row from the DataFrame to the PrettyTable
    for idx, row in df.iterrows():
        table.add_row(row, divider=True if idx + 1 == len(df) else False)

    table.add_row(["Marco-Avg. Familiarity", round(df["familiarity"].mean().item(), 3)])

    return table


def cumsum_until(counts: List[int], k: int) -> List[int]:
    """Cummulative sum of list of counts until k entries."""
    cumsum = 0
    result = []

    for count in counts:
        if cumsum + count >= k:
            result.append(k - cumsum)
            break
        else:
            cumsum += count
            result.append(count)

    return result


def clipped_cosine_similarity(vec1: np.array, vec2: np.array) -> float:
    """Cosine similarity between two numpy arrays."""
    dot_product = np.dot(vec1, vec2)

    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)

    if norm_a == 0 or norm_b == 0:
        return 0
    cos_sim = dot_product / (norm_a * norm_b)
    return max(cos_sim, 0)


def combine_counters(train_counter: Counter, test_counter: Counter) -> Dict[str, Tuple[int, int]]:
    """Create a combined dictionary where each entry is a tuple (train_count, test_count)"""
    combined = dict(
        sorted(
            {
                word: (train_counter.get(word, 0), test_counter.get(word, 0))
                for word in set(train_counter) | set(test_counter)
            }.items()
        )
    )
    return combined


def iterate_dict_in_batches(d: Dict[Any, Any], batch_size: int) -> Iterator:
    """Iterator over dictionary"""
    it = iter(d.items())
    for _ in range(0, len(d), batch_size):
        batch = dict(islice(it, batch_size))
        yield batch
