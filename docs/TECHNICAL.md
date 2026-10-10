# PROMPTWORK technical guide

[Back to the project overview](../README.md).

This guide documents the implemented metric, configuration, segmentation,
embedding adapters, reproducible WildChat workflow, validation, and remaining
research and engineering work. All commands run from the repository root.

## Guide contents

- [Metric calculations](#what-the-metric-computes)
- [Configuration and scoring modes](#configuration)
- [Segmentation](#segmentation-specifics)
- [Embedding adapters](#embedders)
- [Reproducing the WildChat run](#reproducing-the-wildchat-run)
- [Output artifacts](#output-artifacts-and-how-to-read-them)
- [Reproducibility boundaries](#reproducibility-boundaries)
- [Tests](#tests)
- [Repository map](#repository-map)
- [Interpretation limits](#interpretation-limits)
- [Work left](#work-left)

PROMPTWORK is a small Python research library for measuring how much new
embedding-space information each user or assistant turn adds to a dialogue.
It also includes a reproducible workflow for preparing and scoring local
WildChat conversations across languages.

The base metric is Gaussian information gain from He, Kasiviswanathan and
Janzing, *Measuring Semantic Progress in Multi-turn Dialogue via Information
Gain* (arXiv 2606.12332). The implementation was developed after reading
Section 4, Algorithm 1, and Appendix H/Table 4 of the supplied PDF.

The paper scores assistant answers. PROMPTWORK extends the same Gaussian
mechanism to user turns and offers different ways to condition their scores.
**Those user-turn extensions are experimental.** They are inspired by the
user contribution term in the paper's Eq. 2; they are not a validated estimator
of that mutual-information term.

## Current status

Implemented:

- A NumPy Gaussian posterior with fast Woodbury updates and a slow test reference.
- Per-turn scores, final relevance weights, role totals, and user share.
- Three state modes and three experimental user relevance modes.
- Sentence/list segmentation, including lowercase starts and wrapped prose.
- A deterministic toy embedder and an optional SentenceTransformers adapter.
- A fictional accounting demo comparing modes.
- Pinned, shard-selective WildChat preparation with checksums and manifests.
- Local, deterministic per-language selection and seven-config scoring.
- Offline tests using random vectors, synthetic dialogues, and synthetic parquet.

The latest verified suite passed **72 tests**. Both WildChat scripts compiled,
and the offline toy runner completed. Real dataset downloads and neural-model
execution have not yet been validated as part of these checks. We have an
implementation and an offline verification suite, not empirical evidence that
the metric identifies useful user contributions.

## Install

Requires Python 3.10 or newer. Run commands from the repository root.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,data]'
```

This installs the library, test tools, and parquet/Hub preparation tools.
For neural embeddings, add the `st` extra:

```sh
python -m pip install -e '.[st,data,dev]'
```

| Dependency group | Purpose |
| --- | --- |
| Required: `numpy` | Core metric and vector operations |
| `st`: `sentence-transformers` | Neural embedding adapter |
| `data`: `huggingface_hub`, `pyarrow` | Dataset preparation |
| `dev`: `pytest` | Tests |

The current pipeline does not require `datasets`. Importing `promptwork` does
not import Torch or SentenceTransformers; neural dependencies are loaded when
the neural adapter is constructed.

If you do not activate the environment, replace `python` with
`.venv/bin/python` in the commands below.

## Start with an offline demo

Check the embedder before running a demo or experiment:

```sh
python examples/check_embedder.py --toy
python examples/demo.py
python examples/wildchat_run.py --toy
```

The first demo prints per-turn chunk counts, mean weights, gains, and cumulative
gains for all three user relevance modes. It also compares user gains under
`both` and `user_only`.

The WildChat toy runner uses three synthetic accounting conversations: a
correction, filler, and repeated information. It writes the same artifact format
as a real run, under a new directory in `results/`.

All toy runs use `HashingEmbedder`. Its vectors represent word-count overlap,
not semantic meaning. **Toy scores are for checking mechanics, not research
conclusions.** Accounting statements in these examples are fictional and include
intentional errors.

## Library quickstart

```python
from promptwork import HashingEmbedder, IGConfig, score_dialogue

turns = [
    {"role": "user", "content": "Prepare a quarterly report. Revenue is 100000."},
    {"role": "assistant", "content": "The report needs revenue and the applicable tax rate."},
    {"role": "user", "content": "Use a tax rate of 5% for this example."},
]

result = score_dialogue(turns, HashingEmbedder(), config=IGConfig())

print(result.total)       # Total Gaussian gain, in nats
print(result.by_role())   # {"user": ..., "assistant": ...}
print(result.user_share)  # User gain / total gain; 0 when total is 0

for turn in result.turns:
    print(turn.index, turn.role, turn.weights, turn.ig, turn.cumulative_ig)
```

`TurnScore` exposes:

| Field | Meaning |
| --- | --- |
| `index` | Position in the original input sequence |
| `role` | `user` or `assistant` |
| `n_chunks` | Number of evidence chunks retained by segmentation |
| `n_active_chunks` | Number of positive-weight chunks enabled to update state |
| `weights` | Final per-chunk relevance weights after exponent and cutoff |
| `ig` | This turn's gain in nats |
| `cumulative_ig` | Sum of gains through this turn |

Roles such as `system` are ignored, including for anchor selection, and produce
no result row. Empty user/assistant turns still produce rows with zero chunks
and zero gain. Each scoring call starts a fresh posterior for that dialogue.

## What the metric computes

A dialogue starts with isotropic uncertainty:

```text
Sigma_0 = sigma0_sq * I
J_0 = inverse(Sigma_0)
```

For each enabled turn, the implementation splits evidence into chunks, embeds
them, and assigns relevance weights. For an anchor vector `q` and chunk `z_i`:

```text
w_i = max(0, cosine(q, z_i)) ** beta
w_i = 0 when w_i < eta
alpha_i = w_i / sigma_sq
J_new = J_old + sum(alpha_i * z_i * z_i.T)
IG = 0.5 * (logdet(J_new) - logdet(J_old))
```

Without an anchor, weights are exactly 1 and bypass the cutoff. A zero-vector
anchor has cosine zero. Embeddings are normalized only for cosine; the posterior
update uses raw embedding vectors.

The production code stores covariance rather than precision. After dropping
zero-weight chunks, it computes:

```text
B = sqrt(A) * Z
M = I + B * Sigma * B.T
IG = 0.5 * logdet(M)
Sigma_new = Sigma - Sigma * B.T * solve(M, B * Sigma)
```

`M` is sized by active chunks, not embedding dimension. The code solves this
system without explicitly computing an inverse, checks the determinant sign,
and symmetrizes the updated covariance. A turn with no active chunks returns
exactly zero and leaves the posterior unchanged.

Storage is still O(d²), and turn cost includes O(md² + m²d + m³), where `d` is
embedding dimension and `m` is active chunks. High-dimensional embeddings and
very long answers can therefore be expensive.

The slow reference accumulates precision and computes its full log determinant.
It exists for tests and is not used by `score_dialogue`. Both posterior APIs take
already-divided `alpha`; scoring performs the division by `sigma_sq` once.
Unlike the paper's optional numerical jitter, the fast implementation raises on
an invalid determinant rather than silently adding jitter.

## Configuration

All current defaults:

| Parameter | Default | Effect |
| --- | --- | --- |
| `sigma0_sq` | `1.0` | Isotropic prior variance |
| `sigma_sq` | `0.25` | Observation variance; smaller values increase precision updates |
| `beta` | `1.0` | Exponent applied to nonnegative cosine relevance |
| `eta` | `0.05` | Weights strictly below this are zeroed |
| `min_chunk_chars` | `12` | Minimum retained chunk length |
| `keep_short_turns` | `True` | Retain a whole nonempty turn if no chunk survives |
| `state_roles` | `"both"` | Whose evidence updates the shared posterior |
| `user_relevance` | `"none"` | User-chunk relevance anchor |
| `join_wrapped_lines` | `True` | Join unfinished prose across single newlines |

The numerical defaults and minimum chunk length follow Table 4. Scoring users,
retaining short turns, and the revised segmentation are explicit extensions.

### State roles: what counts as prior evidence?

| `state_roles` | Posterior receives | Meaning of a user gain |
| --- | --- | --- |
| `both` | User and assistant evidence, in dialogue order | Novelty relative to all previously incorporated evidence |
| `user_only` | User evidence only | Novelty relative to what the user contributed earlier |
| `assistant_only` | Assistant evidence only | Zero; this follows the paper's assistant-only state updates |

In `user_only`, assistant text can still provide a relevance anchor, but it never
updates the posterior. A user agreeing with the assistant may therefore have
higher gain than under `both`, because the assistant evidence has not reduced
uncertainty in that direction.

Disabled turns retain their chunks and report weights of 1, zero gain, and zero
active chunks. Their chunks are not embedded for evidence, although the same
string may be needed as an anchor or enabled evidence elsewhere.

`user_share` is mechanically 1 in positive-total `user_only` runs, and 0 in
`assistant_only` runs. It is 0 for any zero-total run. Compare `user_ig` across
state modes rather than treating those shares as a finding.

### User relevance: what is a chunk relevant to?

All three user relevance modes are experimental:

| `user_relevance` | Anchor for an enabled user turn |
| --- | --- |
| `none` | No anchor; every chunk gets weight 1 |
| `first_user` | Full first nonempty user message; that first turn is unanchored |
| `previous_assistant` | Full most recent preceding nonempty assistant message |

Without a preceding assistant, `previous_assistant` gives weights of 1. Two
consecutive user turns use the same preceding assistant anchor. Assistant turns
always use the full most recent preceding nonempty user message, regardless of
`user_relevance`.

Anchor text is stripped before embedding. Identical strings are deduplicated,
and empty turns do not replace prior anchors. The core scorer makes one batched
embedding call per dialogue when embeddings are needed. Each chunk is encoded
independently, without dialogue context; an anchor is encoded as one full string.

### Paper-style configuration

```python
config = IGConfig(
    state_roles="assistant_only",
    keep_short_turns=False,
    join_wrapped_lines=False,
)
```

This matches assistant-only state updates and drops short evidence, while
restoring newline boundaries. It does not reproduce an original segmenter
exactly: the current abbreviation-aware sentence rules remain in effect.

The runner's config named `assistant_only|paper` sets `assistant_only` and
`keep_short_turns=False`, but leaves `join_wrapped_lines=True`. Its name denotes
paper-style state and short-turn settings, not exact replication of every
preprocessing detail.

## Segmentation specifics

The current splitter:

- Strips bullet markers and one/two-digit numbered-list markers.
- Keeps paragraphs and separate list items apart.
- Joins ordinary hard-wrapped prose when `join_wrapped_lines=True`.
- Splits ordinary periods before lowercase or uppercase letters, subject to
  abbreviation, unit, initial, and bare-number rules.
- Splits question/exclamation endings before lowercase starts as well.
- Splits ellipses and unit abbreviations only before uppercase starts or opening
  quotes/brackets.
- Supports Chinese/Japanese sentence terminators without following whitespace.

Examples that remain intact include `5.5%`, `ст. 293 ПКУ`, and `Dr. Smith`.
The abbreviation rule also works across a newline, such as `Dr.\nSmith`.
`No.` is ambiguous: the splitter treats it as an abbreviation, so a standalone
answer followed by another sentence can remain joined.

Length filtering applies after segmentation. If at least one chunk survives,
shorter chunks are dropped. The whole-turn fallback applies only when no chunk
survives. This preserves short turns such as `ставка 5%`, but it does not preserve
every short sentence inside a longer turn.

This remains a heuristic, not a grammatical parser. Unmarked lines may join an
unfinished list item. The segmenter and 12-character threshold were designed
around spaces and letter case, particularly Latin/Cyrillic text. Supporting CJK
punctuation does not make scores comparable across scripts.

## Embedders

`HashingEmbedder(dim=256)` lowercases Unicode word tokens, applies signed CRC32
feature hashing, and L2-normalizes nonzero vectors. Empty text yields a zero
vector. It is deterministic across Python hash seeds, but collisions and missing
semantic understanding make it suitable only for tests and demonstrations.

`SentenceTransformerEmbedder` defaults to `Qwen/Qwen3-Embedding-0.6B`, with
configurable model name, device, and batch size (default 32). It returns float64
arrays containing raw model embeddings. Alternative model names can be supplied
without changing scoring code; their suitability has not been established here.

Model-specific query prefixes, instructions, and document/query distinctions
are not added automatically. If a model needs them, implement an adapter with
`embed(texts: list[str]) -> numpy.ndarray` returning finite `(n, d)` vectors.

## Reproducing the WildChat run

The dataset pipeline gives no language special treatment. Language filters are
case-insensitive, selection is per language, and summaries do not pool languages.
The toy examples happen to be Ukrainian; they do not define selection defaults.

### 1. Cache and check the embedding model while online

```sh
python examples/check_embedder.py --model Qwen/Qwen3-Embedding-0.6B
python examples/demo.py --model Qwen/Qwen3-Embedding-0.6B
```

The check must run before the demo. Model files are cached separately from the
dataset. Keep that cache available for subsequent offline scoring.

### 2. Prepare a pinned local dataset once

```sh
python examples/prepare_wildchat.py \
  --revision 7d6490e462285cf85d91eabea0f9a954fbddcd1f \
  --shards 0,7 \
  --languages all \
  --sample-fraction 0.05 \
  --yes
```

The dataset defaults to `allenai/WildChat-1M`. The revision above is also the
script's default and is passed explicitly to both Hub listing and download.

Preparation first lists shard names and sizes at that revision. It validates
indices, prints the selected file count and download size, and downloads only
exact selected filenames. No wildcard can fetch unselected shards. Default
shards are 0 and 7; `--shards all` selects every shard.

Omit `--yes` to review the size and confirm interactively. Non-interactive
sessions without `--yes` stop before downloading. Hub downloads use their cache
and resumable behavior; the printed size is an upper bound, not a prediction of
uncached bytes.

If authentication is needed, set `HF_TOKEN`. WildChat-1M may not require login;
WildChat-1M-Full is gated. Depending on the installed CLI version, login is
`hf auth login` or the older `huggingface-cli login`.

Preparation processes parquet files individually. Within each row group it
reads language labels first, counts all scanned labels, and selects language
row indices before converting conversations into Python objects. Parquet can
still decode conversation columns for the whole selected row group internally.

It filters message count, then keeps each hash according to:

```python
int(sha256((salt + conversation_hash).encode()).hexdigest()[:8], 16) / 2**32 < fraction
```

This stage uses no random generator. Selection is independent of row order and
file partitioning. A disk-backed SQLite sort produces output ordered by hash.
The current implementation holds retained rows from one shard in memory, not
the entire dataset.

Defaults and useful options:

| Preparation option | Default | Purpose |
| --- | --- | --- |
| `--repo` | `allenai/WildChat-1M` | Source dataset |
| `--revision` | `7d6490e462285cf85d91eabea0f9a954fbddcd1f` | Pinned dataset snapshot |
| `--shards` | `0,7` | Selected numeric indices; `all` selects every shard |
| `--languages` | `all` | All labels, or a comma-separated case-insensitive list |
| `--min-messages`, `--max-messages` | `4`, `20` | Inclusive conversation message-count limits |
| `--sample-fraction` | `0.05` | Hash-based sampling fraction, in `(0, 1]` |
| `--sample-salt` | `promptwork-v1` | Changes the deterministic hash sample |
| `--out-dir` | `data/wildchat` | JSONL and manifest location |
| `--max-workers` | `4` | Download concurrency |
| `--force` | Off | Overwrite an existing preparation output |

The example writes:

```text
data/wildchat/
  wildchat_all_7d6490e4_s0-7_f0.05.jsonl
  wildchat_all_7d6490e4_s0-7_f0.05.manifest.json
```

Each JSONL line has `conversation_hash`, the original `language` label, and
`turns`; each turn retains only `role` and `content`. The filename includes
languages, revision prefix, shards, and fraction. Other filters and the salt are
recorded in the manifest; conflicting parameters at the same filename require
`--force` or a different output directory.

A matching output/manifest with a valid output checksum is skipped before Hub
access. Preparation prints the 15 most frequent scanned labels with written
counts. The manifest stores parameters, scanned/written totals, counts for the
30 most frequent scanned labels, exact processed filenames, sizes, parquet
SHA-256 checksums, output checksum, versions, and a UTC timestamp.

`--languages all --sample-fraction 1` can produce a large file. Two selected
shards are a partial source sample, not automatically a representative sample of
the entire dataset.

### 3. Score the local file

```sh
python examples/wildchat_run.py \
  --input data/wildchat/wildchat_all_7d6490e4_s0-7_f0.05.jsonl \
  --languages all \
  --top-languages 5 \
  --n-per-language 25 \
  --seed 42 \
  --no-text
```

There is no dataset streaming or download in this step. The runner resolves
neural models using `snapshot_download(..., local_files_only=True)` and loads
the local snapshot. A missing cached snapshot causes an error. A local model
directory can also be passed through `--model`.

With `--languages all`, the runner picks the K most frequent labels in the input
file before applying turn-count limits. Ties are resolved alphabetically.
Explicit comma-separated labels bypass the top-K choice. For each language it
then filters message count, sorts by conversation hash, and samples with:

```python
random.Random(f"{seed}:{language}").sample(candidates, n_per_language)
```

If fewer candidates remain, it takes all and prints that fact. Original labels
are preserved in outputs. The language field is the conversation's most frequent
detected language, not a guarantee that every message uses that language.

| Scoring option | Default | Purpose |
| --- | --- | --- |
| `--input` | Required unless `--toy` | Prepared local JSONL |
| `--languages` | `all` | Top-language selection or explicit labels |
| `--top-languages` | `5` | Number of most frequent input labels when using `all` |
| `--n-per-language` | `25` | Maximum selected conversations per language |
| `--seed` | `42` | Deterministic per-language sampling seed |
| `--min-turns`, `--max-turns` | `4`, `20` | Inclusive number of messages, not user/assistant pairs |
| `--model` | `Qwen/Qwen3-Embedding-0.6B` | Cached neural model name or local directory |
| `--max-seq-length` | `512` | Model token-length cap; long chunks/anchors may be truncated |
| `--run-dir` | New UTC timestamp under `results/` | Destination; must not already exist |
| `--no-text` | Off | Leave CSV text prefixes empty |
| `--toy` | Off | Use built-in synthetic input and hashing embeddings |

The runner evaluates seven configs: `both` and `user_only`, each with `none`,
`first_user`, and `previous_assistant`, plus `assistant_only|paper`. Embeddings
are cached across configs within a dialogue and cleared between dialogues.

## Output artifacts and how to read them

Every run gets its own directory:

```text
results/<UTC timestamp>/
  wildchat_scores.csv
  wildchat_turns.csv
  selected_hashes.txt
  run_manifest.json
  toy_input.jsonl       # Toy runs only
```

`wildchat_scores.csv` contains one row per dialogue/config, with conversation
ID, language, total message count, role counts, total gain, user gain, assistant
gain, and user share. The `conversation_id` CSV column contains the prepared
conversation hash.

`wildchat_turns.csv` contains conversation ID, language, config, original turn
index, role, chunk counts, mean final weight, gain, cumulative gain, and the
first 80 content characters with newlines replaced by spaces. `--no-text`
leaves that last field empty.

Stdout reports mean total and mean user gain separately for each language and
config. For relevance-filtered user configs it also reports the fraction of
user turns with retained chunks but zero active chunks. Empty/dropped turns
are excluded from that denominator. No average across languages is printed.

`selected_hashes.txt` records each selected hash and its language.
`run_manifest.json` records the input path/checksum, language selection, seed,
message filters, model name, maximum sequence length, cached model snapshot
commit when available, every IGConfig field, dependency versions, package
version/git commit when available, norm statistics, and timestamp. A directly
supplied model directory currently has no recorded snapshot commit.

Norm statistics count unique cached strings within each selected dialogue,
across its configs; the same string in different dialogues is counted again.
CSV numeric fields are rounded to four decimals.

Data and result directories are ignored by Git. Public dataset text should stay
local. `--no-text` removes prefixes from CSVs, not from the prepared JSONL or
synthetic input file.

## Reproducibility boundaries

The same prepared input bytes, selection parameters, seed, and Python sampling
implementation produce the same chosen dialogues. Hash subsampling is also
independent of parquet row order. Tests check both properties.

Numeric scores can differ slightly across hardware, model execution backends,
and library versions. The manifest records versions but does not lock them.
The dataset revision is pinned; the model is recorded after local-cache
resolution but is not explicitly revision-pinned by the scoring CLI. A newer
cached model revision could change a later run unless the same local snapshot
is supplied deliberately.

JSONL output bytes are stable for identical records and parameters. Manifest
bytes are not expected to be identical after forced regeneration because they
include a new timestamp. Without `--force`, an idempotent skip preserves both.

The prepared JSONL is loaded into memory for selection. Preparation uses a
bounded source-processing strategy, but the scorer needs further work for very
large local exports. Duplicate selected hashes currently raise an error during
preparation rather than being silently merged.

## Tests

Install both `dev` and `data` extras to run the complete suite:

```sh
python -m pytest -q --tb=short
python -m py_compile examples/prepare_wildchat.py examples/wildchat_run.py
```

The tests require no network or model downloads. They cover:

- Fast/reference agreement, nonnegative gain, telescoping, raw-vector updates,
  decreasing gains for repetition, and unchanged state for zero weights.
- Segmentation, empty/short turns, deterministic hashing, and lazy imports.
- Anchors, relevance cutoffs, state-mode equivalence, disabled turns, batching,
  stripped-anchor deduplication, and nonempty-anchor tracking.
- Synthetic parquet filtering, multilingual counts, exact hash sampling,
  order/partition independence, output sorting, manifests, and idempotence.
- Exact shard patterns, pinned mocked Hub calls, and noninteractive approval.
- Deterministic per-language selection and run-artifact generation.

These checks establish implementation behavior. They do not establish semantic
validity, cross-language comparability, or end-to-end performance on real data.

## Repository map

| Path | Responsibility |
| --- | --- |
| `promptwork/__init__.py` | Public exports |
| `promptwork/core.py` | Fast covariance posterior and slow precision reference |
| `promptwork/scoring.py` | Config/results, anchors, shared-state dialogue scoring |
| `promptwork/segment.py` | Sentence/list evidence segmentation |
| `promptwork/embedders.py` | Protocol, hashing adapter, lazy neural adapter |
| `examples/check_embedder.py` | Shape/finiteness/norm check before demos |
| `examples/demo.py` | Fictional dialogue and mode comparisons |
| `examples/prepare_wildchat.py` | Pinned selective download and local JSONL preparation |
| `examples/wildchat_run.py` | Local per-language selection and seven-config scoring |
| `examples/inspect_turns.py` | Local CSV inspection: weights, chunk-count correlation, and example turns |
| `tests/` | Offline mathematical, API, segmentation, and pipeline tests |
| `pyproject.toml` | Package metadata and optional dependency groups |
| `docs/TECHNICAL.md` | Full implementation, workflow, limitations, and roadmap |

The repository is organized as follows:

```text
promptwork/
├── promptwork/        # Installable library
├── examples/          # Demos, preparation, scoring, and inspection scripts
├── tests/             # Offline test suite
├── docs/              # Technical documentation
├── README.md
├── pyproject.toml
└── .gitignore
```

Local files excluded from commits live separately: downloaded data under
`data/`, runs under `results/`, and obsolete code under `archive/`. The old
root-level streaming runner is preserved at `archive/wildchat_run.py`; earlier
root-level CSVs are preserved under `results/legacy/`. They are not the current
workflow and are not included in Git. Environment, cache, macOS metadata, and
R-session files are ignored as well.

Use the scripts under `examples/`. To inspect an existing run:

```sh
python examples/inspect_turns.py results/<run-directory> \
  --config 'both|previous_assistant' --language English
```

Quote config names in shell commands because `|` is a shell pipe. The inspection
helper summarizes weights and their relationship with chunk count and can show
turn prefixes when available. Supply a language to inspect results separately;
without it the helper combines languages. This is exploratory inspection, not
semantic validation.

## Interpretation limits

The first enabled evidence turn often has a large gain because the prior is
isotropic. Repeated evidence still has positive, diminishing gain; it is not
removed as an exact duplicate from posterior updates.

The score measures novelty relative to incorporated dialogue evidence. It does
not measure factual correctness, usefulness, truth, or novelty relative to what
the LLM already knows. A wrong statement can add a new direction and score well.

Filler does not necessarily have gain near zero. It scores low when its direction
is already well observed or its relevance weight is low. An unanchored filler
turn with a new embedding direction can have positive gain.

Embedding similarity captures topic more reliably than negation. Neither `both`
nor `user_only` distinguishes a correction from an agreement by itself. That
requires turn-type labelling and a model-relative novelty test, neither of which
is implemented.

First-user relevance can suppress legitimate topic changes. Previous-assistant
relevance can suppress corrections expressed with new vocabulary. Model choice,
embedding norms, segmentation, truncation, turn length, and the number of chunks
all affect scores. The result is a Gaussian covariance surrogate, not calibrated
real-world mutual information.

## Work left

### Next empirical steps

1. Validate a small real, pinned shard download and compare its schema, exact
   language labels, counts, hashes, and manifest against expectations.
2. Run the neural embedder check, inspect norms and truncation, then complete a
   small real-data run with preserved artifacts and manifests.
3. Label a sample of user turns as corrections, agreements, added facts,
   clarifications, repeats, filler, and topic changes. Establish annotation rules
   before using these categories as evaluation targets.
4. Compare `both` with `user_only` and all relevance anchors within each language.
   Inspect individual turns as well as distributions; do not interpret toy gains
   or mechanical user shares as evidence.
5. Evaluate models and model-specific input instructions, with explicit model
   snapshots, sensitivity to norms/parameters, and truncation measurements.
6. Examine shard-selection bias and language coverage before making population
   claims. Expand beyond the default two shards when justified.

### Research extensions

- A model-relative novelty test: assess whether a user supplied information the
  assistant would otherwise lack, rather than just a new dialogue direction.
- Explicit correction/negation and turn-type handling, validated against labels.
- Language/script-appropriate segmentation and chunk-length rules; evaluate
  comparability before aggregating across languages or scripts.
- Statistical uncertainty and sensitivity analysis for state modes, anchors,
  prior/noise settings, chunk thresholds, and embedding choices.

### Engineering improvements

- Explicit model revision pinning and a dependency lock for stronger replication.
- Memory-bounded selection for large JSONL files and batch-oriented preparation
  that avoids retaining an entire shard's accepted rows.
- Duplicate-hash policy, clearer missing-language reporting, and recovery after
  interrupted runs; current run directories cannot be reused.
- Stronger source provenance for uncommitted changes/local model directories,
  and machine-readable summaries retaining more than four decimal places.
- Performance measurements on real dimensions and chunk counts, plus stress
  tests for numerical stability over long dialogues.

These are remaining tasks, not features already present in the library.
