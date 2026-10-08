# promptwork

A small NumPy library measuring Gaussian information gain in dialogue, in **nats**.
Based on He, Kasiviswanathan and Janzing, *Measuring Semantic Progress in Multi-turn
Dialogue via Information Gain*, Section 4, Algorithm 1 and Appendix H/Table 4
(the supplied `measuring semantic progress.pdf`). User-turn scoring is an
experimental extension inspired by the user term in Eq. 2, not an estimator
validated by that paper.

## Install and quickstart

```sh
python -m pip install -e '.[dev]'
# Optional neural embeddings:
python -m pip install -e '.[st]'
python examples/demo.py
python examples/demo.py --model Qwen/Qwen3-Embedding-0.6B
python -m pytest
```

```python
from promptwork import score_dialogue, IGConfig, HashingEmbedder

turns = [
    {"role": "user", "content": "Підготуй звіт для ФОП групи 3 без ПДВ."},
    {"role": "assistant", "content": "Для звіту потрібні дохід і період."},
    {"role": "user", "content": "Дохід 100000 гривень за квартал."},
]
result = score_dialogue(turns, HashingEmbedder(), IGConfig())
print(result.total, result.by_role(), result.user_share)
for turn in result.turns:
    print(turn.index, turn.role, turn.weights, turn.ig, turn.cumulative_ig)
```

`HashingEmbedder` is a deterministic, normalized signed bag-of-words toy using
CRC32, **not semantically meaningful**. Use `SentenceTransformerEmbedder` for
research, with a model name supplied directly (e.g. Qwen, bge-m3, or multilingual-e5).
Neural dependencies load only when that adapter is constructed. Model-specific
query prefixes/instructions are not added automatically; configure an external
`Embedder` adapter if your chosen model needs them.

## Design decisions

Defaults match Table 4's numerical settings: prior variance 1, observation
variance 0.25, relevance exponent 1, cutoff 0.05, minimum chunk length 12.
The defaults explicitly extend the paper:

- `state_roles="both"`: user and assistant evidence update one posterior in
  dialogue order, so gains condition on all previously incorporated evidence.
- `keep_short_turns=True`: if no chunk survives the length filter, retain the
  entire nonempty turn, including short corrections such as `ставка 5%`.

Use `IGConfig(state_roles="assistant_only", keep_short_turns=False,
join_wrapped_lines=False)` for assistant-only updates with the original newline
boundaries. The revised abbreviation-aware sentence segmentation remains a
heuristic extension; this does not reproduce the original uppercase-only splitter.
User turns then have gain zero and cannot update state. Their reported weights
are 1 and active chunk count is zero, independent of `user_relevance`.

Experimental user weighting is configurable:

| `user_relevance` | Anchor |
| --- | --- |
| `none` (default) | No anchor; every user chunk has weight 1 |
| `first_user` | Full first user message; first user turn has weight 1 |
| `previous_assistant` | Full most recent preceding assistant message |

Enabled assistant chunks use the full most recent preceding non-empty user message.
Without an anchor, weights are 1 and bypass the cutoff. Otherwise weights are
`max(0, cosine)**beta`, zeroed when strictly below `eta`. Zero-vector cosine is
zero. `beta` must be positive; variances must be positive and finite.
Ignored roles, including system, neither update state nor serve as anchors.
Result rows contain only user/assistant turns; indices preserve input positions.

Evidence segmentation strips bullet and one/two-digit list markers. By default,
`join_wrapped_lines=True` joins unfinished prose across single newlines, while
keeping list items, paragraphs, and completed sentences separate. Set it to
`False` to treat every newline as a boundary. Ordinary periods can split before
lowercase letters; known abbreviations, initials, and decimals remain intact.
Exclamation/question marks split before lowercase text too. Ellipses and units
such as `грн.` split only before uppercase letters or opening quotes/brackets.
CJK terminators can split without whitespace. Abbreviations are ambiguous: for
example, `No.` is treated as an abbreviation even when intended as a standalone
answer. Unmarked continuation lines can join unfinished list items.
Embedding is one deduplicated batch per dialogue, with each evidence chunk encoded
independently and full anchor texts encoded as single strings. Empty dialogues
or dialogues with no enabled evidence need no embedding call.

Raw embedding vectors update covariance; only cosine computation normalizes
vectors. Core updates drop zero-weight chunks, compute log determinants and solves
on the active-chunk matrix, and symmetrize the covariance. Storage remains O(d²);
turn cost includes O(md² + m²d + m³), so many chunks can still be expensive.
Nonpositive determinant signs raise an error rather than silently adding jitter
(the paper permits 1e-12 jitter). The slow precision oracle is used only in tests.

## state_roles

- `"both"` (default): measures each turn's gain conditional on prior user and
  assistant evidence in the shared state.
- `"assistant_only"`: measures assistant progress conditional on earlier assistant
  evidence, using user text as relevance anchors, following the paper's state updates.
- `"user_only"`: measures user novelty relative to earlier user evidence; assistant
  text never updates the state but can still serve as a `previous_assistant` anchor.

Disabled turns retain their chunks and report weights of 1, gain 0, and zero
active chunks. Their chunks are not embedded unless the same text is needed as
an anchor or enabled evidence. In `user_only`, assistant gain is zero and user
share is 1 when total gain is positive (0 when total gain is zero).

Anchor texts are stripped before batching, and empty turns do not replace any
anchor. Empty turns still produce a result row with zero chunks and gain.

Embeddings capture topic, not negation, so neither `both` nor `user_only`
separates a correction from an agreement by itself; that needs turn-type
labelling and a model-relative novelty test, which are not implemented.

## Limitations

The first evidence turn often has a large gain because the prior is isotropic.
The score measures embedding novelty relative to incorporated dialogue evidence,
not correctness, factual truth, or novelty relative to the LLM's knowledge.
Filler does **not** necessarily score near zero: it has low gain only when its
embedding is close to already well-observed directions or receives a low relevance
weight. An unanchored novel filler direction can increase gain. Repeated evidence
still contributes positive, diminishing gains rather than being removed.

Embedding norms, model choice, segmentation, dialogue length, and relevance
anchors affect scores. This is a Gaussian covariance surrogate, not a calibrated
measurement of real-world mutual information or user value. First-user anchors
can suppress topic shifts; previous-assistant anchors can suppress corrections
that use new vocabulary. The demo is fictional and is not accounting guidance.
