import json
import os
import subprocess
import sys
import numpy as np
from promptwork import HashingEmbedder


def test_hashing_normalization_empty_case_and_cross_process():
    texts = ['Україна податок invoice', 'УКРАЇНА ПОДАТОК INVOICE', '', '!!!']
    embedder = HashingEmbedder(17)
    vectors = embedder.embed(texts)
    assert vectors.shape == (4, 17) and vectors.dtype == np.float64
    np.testing.assert_array_equal(vectors[0], vectors[1])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1, 1, 0, 0])
    script = 'from promptwork import HashingEmbedder; import json; print(json.dumps(HashingEmbedder(17).embed(["Україна податок invoice"]).tolist()))'
    for seed in ['1', '99']:
        output = subprocess.check_output([sys.executable, '-c', script],
                                          env={**os.environ, 'PYTHONHASHSEED': seed}, text=True)
        np.testing.assert_array_equal(json.loads(output)[0], vectors[0])
