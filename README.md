# Label Shift Estimation for Named Entity Recognition using Familarity

This repository computes the label shift for zero-shot NER settings using the Familarity metric. The metric uses semantic similarity between the sets of label seen during training and used for evaluation to indicate how "familiar" the trained model will be with the evaluation labels.

## Installation
```python
conda create -n familarity python=3.11
conda activate familarity
pip install -e .
```

## Usage
```python
import numpy as np
from familarity import compute_metric
train_labels_set = ["person", "location", "building", "eagle", "restaurant", "util"]
train_probs = [0.4, 0.1, 0.1, 0.1, 0.1, 0.2]
train_labels = np.random.choice(train_labels_set, size=30000, p=train_probs).tolist()

test_labels_set = ["human", "organization", "building", "review", "researcher", "car"]
test_probs = [0.5, 0.2, 0.05, 0.05, 0.1, 0.1]
test_labels = np.random.choice(test_labels_set, size=30000, p=test_probs).tolist()

compute_metric(
    train_labels=train_labels,
    test_labels=test_labels,
    model_name_or_path="distilbert-base-cased",
    save_results=True,
    save_embeddings=True,
)
```

## Citation
```
@misc{familarity,
  author = {Golde, Jonas},
  title = {Label Shift Estimation using Familarity},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/whoisjones/familarity}},
}
```
