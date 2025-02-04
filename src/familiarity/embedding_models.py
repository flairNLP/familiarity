import io
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from huggingface_hub import repo_exists
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from familiarity.utils import get_device


class LabelEmbeddingModel(ABC):
    def __init__(self):
        self.device = get_device()

    @abstractmethod
    def embed(self, batch: List[str]) -> np.array:
        """Abstract method to compute embeddings for a batch of words.

        Args:
            batch: List of strings (words).

        Returns:
            numpy array containing the embeddings.
        """
        pass


class FastTextModel(LabelEmbeddingModel):
    def __init__(self, model_path: Union[Path, str]):
        super().__init__()
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        words, embeddings = self.load_vectors(Path(model_path))
        self.words = words
        self.embeddings = embeddings.to(self.device)

    def load_vectors(self, model_name_or_path: Path) -> Tuple[List[str], torch.nn.Embedding]:
        file = io.open(
            model_name_or_path,
            "r",
            encoding="utf-8",
            newline="\n",
            errors="ignore",
        )
        n, d = map(int, file.readline().split())
        words = []
        embeddings = []
        for line in tqdm(file.readlines(), desc="Loading FastText"):
            tokens = line.strip().split(" ")
            words.append(tokens[0])
            embeddings.append(torch.tensor(list(map(float, tokens[1:]))))

        words = {w: i for i, w in enumerate(words)}
        words[self.unk_token] = len(words)
        words[self.pad_token] = len(words)

        embeddings = torch.stack(embeddings)
        unk_embedding = torch.mean(embeddings, dim=0)
        padding_embedding = torch.zeros(1, embeddings.size(1))
        embeddings = torch.cat([embeddings, unk_embedding.unsqueeze(0), padding_embedding], dim=0)
        embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        return words, embeddings

    def embed(self, batch: List[str]) -> np.array:
        nested_batch = [re.split(r"[-/_ ]", label.lower()) for label in batch]
        max_length = max(len(inner_list) for inner_list in nested_batch)

        input_ids = torch.LongTensor(
            [
                [self.words.get(label, self.words.get(self.unk_token)) for label in labels]
                + [self.words.get(self.pad_token)] * (max_length - len(labels))
                for labels in nested_batch
            ]
        ).to(self.device)

        mask = input_ids != self.words.get(self.pad_token)

        embeddings = torch.sum(self.embeddings(input_ids), dim=1) / mask.sum(dim=1).unsqueeze(1)

        return embeddings.cpu().numpy()


class GloveModel(LabelEmbeddingModel):
    def __init__(self, model_path: Union[Path, str]):
        super().__init__()
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        words, embeddings = self.load_vectors(Path(model_path))
        self.words = words
        self.embeddings = embeddings.to(self.device)

    def load_vectors(self, model_name_or_path: Path) -> Tuple[Dict[str, int], torch.nn.Embedding]:
        word_embedding_pairs = []
        with open(model_name_or_path, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc="Loading GloVe"):
                parts = line.split(" ")
                word = parts[0]
                vector = torch.tensor([float(x) for x in parts[1:]])
                word_embedding_pairs.append((word, vector))

        words, embeddings = zip(*word_embedding_pairs)
        words = {w: i for i, w in enumerate(words)}
        words[self.unk_token] = len(words)
        words[self.pad_token] = len(words)

        embeddings = torch.stack(embeddings)
        unk_embedding = torch.mean(embeddings, dim=0)
        padding_embedding = torch.zeros(1, embeddings.size(1))
        embeddings = torch.cat([embeddings, unk_embedding.unsqueeze(0), padding_embedding], dim=0)
        embeddings = torch.nn.Embedding.from_pretrained(embeddings)
        return words, embeddings

    def embed(self, batch: List[str]) -> np.array:
        nested_batch = [re.split(r"[-/_ ]", label.lower()) for label in batch]
        max_length = max(len(inner_list) for inner_list in nested_batch)

        input_ids = torch.LongTensor(
            [
                [self.words.get(label, self.words.get(self.unk_token)) for label in labels]
                + [self.words.get(self.pad_token)] * (max_length - len(labels))
                for labels in nested_batch
            ]
        ).to(self.device)

        mask = input_ids != self.words.get(self.pad_token)

        embeddings = torch.sum(self.embeddings(input_ids), dim=1) / mask.sum(dim=1).unsqueeze(1)
        return embeddings.cpu().numpy()


class TransformerModel(LabelEmbeddingModel):
    def __init__(self, model_name_or_path: Union[Path, str], pooling: str = "mean"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pooling = pooling

    def embed(self, batch: List[str]) -> np.array:
        inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.pooling == "mean":
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        else:
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()


class SentenceTransformerModel(LabelEmbeddingModel):
    def __init__(self, model_name_or_path: Union[Path, str]):
        super().__init__()
        self.model = SentenceTransformer(model_name_or_path).to(self.device)

    def embed(self, batch: List[str]) -> np.array:
        embedding = self.model.encode(batch, convert_to_tensor=True)
        return embedding.cpu().numpy()


def infer_embedding_model(model_name_or_path: Union[Path, str]):
    try:
        if "glove" in model_name_or_path:
            embedding_model_type = "glove"
        elif "sentence-transformers" in model_name_or_path:
            embedding_model_type = "sentence-transformers"
        elif "wiki-news" in model_name_or_path or "crawl" in model_name_or_path:
            embedding_model_type = "fasttext"
        elif repo_exists(model_name_or_path):
            embedding_model_type = "transformers"
        else:
            embedding_model_type = None
    except Exception as e:
        print(f"Could not infer model type: {e}")
        embedding_model_type = None

    return embedding_model_type


def load_embedding_model(model_name_or_path: str, embedding_model_type: str = None) -> torch.nn.Embedding:
    if not embedding_model_type:
        embedding_model_type = infer_embedding_model(model_name_or_path)
        if not embedding_model_type:
            raise ValueError(
                "Embedding model type can't be inferred. Please provide any of ['glove', 'fasttext', 'sentence-transformers', 'transformers'] as 'embedding_model_type' argument."
            )
    assert embedding_model_type in [
        'glove',
        'fasttext',
        'sentence-transformers',
        'transformers',
    ], f"{embedding_model_type} is not supported. It must be any of ['glove', 'fasttext', 'sentence-transformers', 'transformers']."

    if embedding_model_type == "glove":
        model = GloveModel(model_name_or_path)
    elif embedding_model_type == "sentence-transformers":
        model = SentenceTransformerModel(model_name_or_path)
    elif embedding_model_type == "fasttext":
        model = FastTextModel(model_name_or_path)
    elif embedding_model_type == "transformers":
        model = TransformerModel(model_name_or_path)
    return model
