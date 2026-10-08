"""Embedding adapters; neural dependencies are loaded only on request."""
from typing import Protocol
import re
import zlib
import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return finite float64 embeddings of shape (len(texts), d)."""
        ...


class HashingEmbedder:                                                  #toy embedder for tests
    """Deterministic signed bag-of-words hashing, NOT semantic embeddings."""
    def __init__(self, dim: int = 256):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.dim), dtype=np.float64)     #zero matrix for each sentence
        for row, text in zip(result, texts):                            #iterating matrix lines and sentences themselves
            for token in re.findall(r'\w+', text.lower()):              #test to lowercase and iterate by words/nums
                data = token.encode('utf-8')                            #word to byte
                index = zlib.crc32(data) % self.dim                     #hash func - word to num and num%%256 = index
                sign = 1 if zlib.crc32(b'sign:' + data) & 1 else -1     #sign 
                row[index] += sign
            norm = np.linalg.norm(row)                                  
            if norm:
                row /= norm                                             #normalizing matrix rows
        return result


class SentenceTransformerEmbedder:                                          #real embedder
    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-0.6B',       #default model is Qwen
                 device: str | None = None, batch_size: int = 32):          #device up to library, batch=32
        if not isinstance(batch_size, int) or batch_size <= 0:              #batch size sanity check
            raise ValueError("batch_size must be a positive integer")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError('Install the optional embedder with pip install "promptwork[st]"') from exc
        self.model = SentenceTransformer(model_name, device=device)         #load model 
        self.model.eval()                                                   #'no study' mode
        self.batch_size = batch_size                                        #batch size initiation

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:                                                       #for empty string - zero matrix w right dims
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float64)
        return np.asarray(self.model.encode(texts, batch_size=self.batch_size,  #model.encode - model run
                          convert_to_numpy=True, normalize_embeddings=False,    #to_numpy - storing results, do not normalize
                          show_progress_bar=False), dtype=np.float64)           #no progress bar logs
