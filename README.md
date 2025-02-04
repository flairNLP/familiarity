# Label Shift Estimation for Named Entity Recognition using Familiarity

**Our paper got accepted to NAACL 2025 🎉 See our [paper](https://arxiv.org/abs/2412.10121) and find the datasets on the [huggingface hub]()!**

This repository computes the label shift for zero-shot NER settings using the Familiarity metric. The metric uses semantic similarity between the sets of label seen during training and used for evaluation to indicate how "familiar" the trained model will be with the evaluation labels.

## Installation
```python
conda create -n familiarity python=3.11
conda activate familiarity
pip install -e .
```

## Usage
```python
import numpy as np
from familiarity import compute_metric
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
@misc{golde2024familiaritybetterevaluationzeroshot,
      title={Familiarity: Better Evaluation of Zero-Shot Named Entity Recognition by Quantifying Label Shifts in Synthetic Training Data}, 
      author={Jonas Golde and Patrick Haller and Max Ploner and Fabio Barth and Nicolaas Jedema and Alan Akbik},
      year={2024},
      eprint={2412.10121},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.10121}, 
}
```
