# PROMPTWORK

PROMPTWORK is a Python research library for exploring how much new information
users and assistants contribute during a dialogue. It scores each turn using
Gaussian information gain in embedding space and includes a reproducible,
language-agnostic workflow for local WildChat experiments.

The base metric comes from He, Kasiviswanathan and Janzing, *Measuring Semantic
Progress in Multi-turn Dialogue via Information Gain* (arXiv 2606.12332).
The paper scores assistant answers; our user-turn scoring is an **experimental
extension**, inspired by its chain-rule decomposition.

**The score measures embedding novelty, not correctness or usefulness.** It does
not identify corrections versus agreements or measure what an LLM already knows.

For algorithms, parameters, data preparation, manifests, and the detailed
roadmap, see the [technical guide](docs/TECHNICAL.md).

## What is implemented

- Per-turn information gain, relevance weights, role totals, and user share.
- Three ways to accumulate evidence: both roles, users only, or assistants only.
- Configurable user relevance anchors and sentence/list segmentation.
- A lightweight toy embedder and optional neural embeddings.
- Pinned, selective WildChat downloads followed by local per-language scoring.
- Experiment CSVs, selected conversation hashes, and reproducibility manifests.

The latest verified suite passed **72 offline tests**. Toy runs and compilation
checks passed. Real-data and neural-model runs still need end-to-end validation;
semantic effectiveness has not yet been established.

## Install

Requires Python 3.10 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,data]'
```

For neural embeddings, also install the `st` extra:

```sh
python -m pip install -e '.[st,data,dev]'
```

Without activating the environment, use `.venv/bin/python` instead of `python`.

## Try it offline

Check the embedder first, then run the demos:

```sh
python examples/check_embedder.py --toy
python examples/demo.py
python examples/wildchat_run.py --toy
```

The demos compare scoring modes on fictional dialogues. The WildChat toy run
writes its artifacts into a new directory under `results/`.
**HashingEmbedder is a toy based on word overlap; its scores are not semantic
research results.** These commands require no model or dataset downloads.

## Use the library

```python
from promptwork import HashingEmbedder, IGConfig, score_dialogue

turns = [
    {"role": "user", "content": "Prepare a quarterly report. Revenue is 100000."},
    {"role": "assistant", "content": "The report needs an applicable tax rate."},
    {"role": "user", "content": "Use a tax rate of 5% for this example."},
]

result = score_dialogue(turns, HashingEmbedder(), config=IGConfig())
print(result.total)
print(result.by_role())
for turn in result.turns:
    print(turn.index, turn.role, turn.ig)
```

By default, both users and assistants update one shared state. User-only mode
instead compares each user turn with earlier user evidence. Assistant-only mode
follows the paper's choice of whose evidence updates state.
See [configuration details](docs/TECHNICAL.md#configuration) before interpreting
results or changing these settings.

## Run a WildChat experiment

The workflow has two stages: prepare a local JSONL from pinned dataset shards,
then select and score conversations independently per language. No language
receives special treatment, and summaries do not pool languages.

Once data and the embedding model are cached, scoring runs offline. Follow the
[step-by-step WildChat instructions](docs/TECHNICAL.md#reproducing-the-wildchat-run)
for download confirmation, authentication, language selection, and commands.
Keep public-dataset text local; `--no-text` omits text prefixes from result CSVs.

Dialogue selection is reproducible for the same input and parameters. Numerical
scores can differ slightly across hardware and library versions. Scores for
different scripts should be read per language; the current segmentation and
chunk-length rules do not establish cross-language comparability.

## Project layout

| Directory | Purpose |
| --- | --- |
| `promptwork/` | Library implementation |
| `examples/` | Embedder check, demos, dataset preparation, scoring, and inspection |
| `tests/` | Offline tests |
| `docs/` | Technical reference and detailed roadmap |
| `data/`, `results/` | Local artifacts excluded from Git |

To run the complete test suite after installing `dev` and `data`:

```sh
python -m pytest -q --tb=short
```

## What remains

The next steps are real-data validation, labelled evaluation of user turn types,
comparison of state modes and embedding models, and language-aware segmentation
validation. Engineering work includes stronger model/dependency pinning and
memory-bounded selection for large datasets.

Correction detection and novelty relative to the LLM are not implemented.
The [detailed roadmap](docs/TECHNICAL.md#work-left) separates empirical,
research, and engineering tasks.
