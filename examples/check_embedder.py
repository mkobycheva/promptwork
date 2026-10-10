"""Check the embedder before running demo.py or wildchat_run.py."""
import argparse
import numpy as np
from promptwork import HashingEmbedder, SentenceTransformerEmbedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--toy', action='store_true', help='Offline toy embedder check')
    args = parser.parse_args()
    embedder = HashingEmbedder() if args.toy else SentenceTransformerEmbedder(args.model)
    texts = ['Підготуй звіт для ФОП групи 3.', 'Ставка податку 5%.']
    vectors = embedder.embed(texts)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
        raise ValueError('Expected embeddings of shape (2, d), d > 0')
    if not np.all(np.isfinite(vectors)):
        raise ValueError('Embeddings must be finite')
    if args.toy:
        print('WARNING: HashingEmbedder is TOY-ONLY, not semantically meaningful.')
    print(f'Embedding shape: {vectors.shape}; dtype: {vectors.dtype}')
    print(f'Embedding norms: {np.linalg.norm(vectors, axis=1).tolist()}')


if __name__ == '__main__':
    main()
