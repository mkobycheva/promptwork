from dataclasses import replace
import subprocess
import sys
import numpy as np
import pytest
from promptwork import HashingEmbedder, IGConfig, score_dialogue
from promptwork.core import ReferenceGaussianPosterior
from promptwork.segment import split_evidence


def turn(role, text):
    return {'role': role, 'content': text}


@pytest.fixture
def embedder():
    return HashingEmbedder(256)


def test_novelty_determinism(embedder):
    turns = [turn('user', s) for s in ['apple pear orchard', 'apple pear orchard', 'invoice tax accounting']]
    a = score_dialogue(turns, embedder)
    assert a == score_dialogue(turns, embedder)
    assert a.turns[1].ig < a.turns[2].ig
    assert a.user_share == 1
    assert a.by_role()['assistant'] == 0


@pytest.mark.parametrize('mode', ['none', 'first_user', 'previous_assistant'])
def test_modes(embedder, mode):
    turns = [turn('user', 'apple pear orchard'), turn('assistant', 'invoice tax accounting'),
             turn('user', 'apple pear orchard'), turn('user', 'invoice tax accounting'),
             turn('user', 'Дякую, ок')]
    # Large dimension is unnecessary for posterior state; use 256 here.
    result = score_dialogue(turns, HashingEmbedder(), IGConfig(user_relevance=mode))
    weights = [t.weights[0] for t in result.turns]
    assert weights[0] == 1
    if mode == 'none':
        assert all(weights[i] == 1 for i in [0, 2, 3, 4])
    elif mode == 'first_user':
        assert weights[2] > weights[3]
    else:
        assert weights[3] > weights[4]
        assert weights[4] == 0
        assert result.turns[4].ig == 0
        assert result.turns[4].n_active_chunks == 0


def test_latest_anchor_and_consecutive_users():
    turns = [turn('user', 'apple pear orchard'), turn('assistant', 'apple pear orchard'),
             turn('system', 'ignored anchor'), turn('assistant', 'invoice tax accounting'),
             turn('user', 'invoice tax accounting'), turn('user', 'invoice tax accounting'),
             turn('user', 'apple pear orchard')]
    result = score_dialogue(turns, HashingEmbedder(), IGConfig(user_relevance='previous_assistant'))
    assert [t.index for t in result.turns] == [0, 1, 3, 4, 5, 6]
    assert result.turns[3].weights == result.turns[4].weights == [1.0]
    assert result.turns[5].weights == [0.0]


def test_assistant_weights_independent_and_paper_reference():
    turns = [turn('user', 'apple pear orchard'), turn('assistant', 'apple pear orchard'),
             turn('user', 'invoice tax accounting'), turn('assistant', 'invoice tax accounting')]
    embedder = HashingEmbedder(32)
    results = [score_dialogue(turns, embedder, IGConfig(user_relevance=m))
               for m in ['none', 'first_user', 'previous_assistant']]
    for i in [1, 3]:
        assert results[0].turns[i].weights == results[1].turns[i].weights == results[2].turns[i].weights
    config = IGConfig(state_roles='assistant_only', keep_short_turns=False)
    results = [score_dialogue(turns, embedder, replace(config, user_relevance=m))
               for m in ['none', 'first_user', 'previous_assistant']]
    assert results[0] == results[1] == results[2]
    reference = ReferenceGaussianPosterior(32)
    anchor = None
    for input_turn, scored in zip(turns, results[0].turns):
        if input_turn['role'] == 'user':
            assert scored.ig == 0 and scored.n_active_chunks == 0
            anchor = embedder.embed([input_turn['content']])[0]
        else:
            z = embedder.embed(split_evidence(input_turn['content'], keep_short_turns=False))
            cosine = (z @ anchor) / (np.linalg.norm(z, axis=1) * np.linalg.norm(anchor))
            w = np.maximum(0, cosine)
            w[w < config.eta] = 0
            assert scored.ig == pytest.approx(reference.update(z, w / config.sigma_sq))


def test_batch_dedup_and_full_anchors():
    class Spy:
        calls = []
        def embed(self, texts):
            self.calls.append(texts)
            return HashingEmbedder(8).embed(texts)
    spy = Spy()
    text = 'First evidence sentence. Second evidence sentence.'
    score_dialogue([turn('user', text), turn('assistant', text)], spy)
    assert len(spy.calls) == 1
    assert spy.calls[0] == ['First evidence sentence.', 'Second evidence sentence.', text]


def test_cutoff_beta_and_anchor_bypass():
    class Fixed:
        def embed(self, texts):
            return np.array([{'anchor': [1., 0.], 'weak': [0.1, np.sqrt(.99)],
                              'zero': [0., 0.]}[t] for t in texts])
    config = IGConfig(beta=2, eta=.05, min_chunk_chars=0, state_roles='assistant_only')
    result = score_dialogue([turn('user', 'anchor'), turn('assistant', 'weak')], Fixed(), config)
    assert result.turns[1].weights == [0] and result.turns[1].ig == 0
    result = score_dialogue([turn('assistant', 'anchor')], Fixed(), replace(config, eta=1))
    assert result.turns[0].weights == [1] and result.total > 0
    result = score_dialogue([turn('user', 'zero'), turn('assistant', 'anchor')], Fixed(), config)
    assert result.total == 0


def test_empty_and_lazy_import():
    assert score_dialogue([], HashingEmbedder()).user_share == 0
    code = 'import promptwork, sys; assert "torch" not in sys.modules; assert "sentence_transformers" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('kwargs', [{'sigma_sq': 0}, {'beta': -1}, {'eta': 2},
                                    {'state_roles': 'bad'}, {'user_relevance': 'bad'}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        IGConfig(**kwargs)


def test_user_none_multichunk_and_no_anchor_cutoff():
    turns = [turn('user', 'Apple orchard sentence. Invoice accounting sentence.'),
             turn('user', 'Дякую, ок')]
    result = score_dialogue(turns, HashingEmbedder(16), IGConfig(eta=1))
    assert result.turns[0].weights == [1, 1]
    assert result.turns[1].weights == [1]
    assert all(t.ig > 0 for t in result.turns)


def test_assistant_anchor_is_latest_user_and_ignores_system():
    turns = [turn('user', 'apple pear orchard'), turn('user', 'invoice tax accounting'),
             turn('system', 'apple pear orchard'), turn('assistant', 'invoice tax accounting')]
    result = score_dialogue(turns, HashingEmbedder(16))
    assert result.turns[-1].weights == [1]


class SpyEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return HashingEmbedder(32).embed(texts)


def test_user_only_config():
    assert IGConfig(state_roles='user_only').state_roles == 'user_only'
    assert IGConfig().state_roles == 'both'
    with pytest.raises(ValueError, match='both, assistant_only, or user_only'):
        IGConfig(state_roles='unknown')


def test_user_only_disabled_assistants_and_share():
    turns = [turn('user', 'apple pear orchard'),
             turn('assistant', 'First assistant sentence. Second assistant sentence.'),
             turn('user', 'invoice tax accounting')]
    result = score_dialogue(turns, HashingEmbedder(32), IGConfig(state_roles='user_only'))
    assistant = result.turns[1]
    assert assistant.n_chunks == 2
    assert assistant.ig == 0 and assistant.n_active_chunks == 0
    assert assistant.weights == [1.0, 1.0]
    assert assistant.cumulative_ig == result.turns[0].cumulative_ig
    assert result.total > 0
    assert result.by_role()['assistant'] == 0
    assert result.user_share == 1.0
    spy = SpyEmbedder()
    zero = score_dialogue([turn('assistant', 'Disabled assistant text')], spy,
                          IGConfig(state_roles='user_only'))
    assert zero.total == 0 and zero.user_share == 0.0
    assert spy.calls == []


def test_user_only_equivalent_to_users_without_assistants():
    turns = [turn('user', 'apple pear orchard'), turn('assistant', 'apple pear orchard'),
             turn('user', 'invoice tax accounting'), turn('assistant', 'some other response'),
             turn('user', 'apple pear orchard')]
    users = [t for t in turns if t['role'] == 'user']
    embedder = HashingEmbedder(32)
    actual = score_dialogue(turns, embedder, IGConfig(state_roles='user_only'))
    expected = score_dialogue(users, embedder)
    np.testing.assert_allclose([t.ig for t in actual.turns if t.role == 'user'],
                               [t.ig for t in expected.turns], atol=1e-12, rtol=0)


@pytest.mark.parametrize('relevance', ['none', 'first_user'])
def test_user_only_independent_of_assistant_text(relevance):
    turns = [turn('user', 'apple pear orchard'), turn('assistant', 'apple pear orchard'),
             turn('user', 'apple pear harvest'), turn('assistant', 'invoice tax accounting'),
             turn('user', 'invoice tax accounting')]
    changed = [turn(t['role'], 'Completely different assistant vocabulary.'
                    if t['role'] == 'assistant' else t['content']) for t in turns]
    config = IGConfig(state_roles='user_only', user_relevance=relevance)
    original = score_dialogue(turns, HashingEmbedder(32), config)
    replacement = score_dialogue(changed, HashingEmbedder(32), config)
    assert [(t.ig, t.weights) for t in original.turns if t.role == 'user'] == [
        (t.ig, t.weights) for t in replacement.turns if t.role == 'user']


def test_user_only_previous_assistant_anchor_without_state_update():
    assistant_text = 'apple pear orchard. apple pear harvest.'
    turns = [turn('user', 'apple pear orchard'), turn('assistant', assistant_text),
             turn('user', 'apple pear harvest.')]
    config = IGConfig(user_relevance='previous_assistant')
    both = score_dialogue(turns, HashingEmbedder(32), config)
    spy = SpyEmbedder()
    users = score_dialogue(turns, spy, replace(config, state_roles='user_only'))
    assert [t.weights for t in users.turns if t.role == 'user'] == [
        t.weights for t in both.turns if t.role == 'user']
    assert both.turns[1].ig > 0
    assert both.turns[2].ig < users.turns[2].ig
    assert users.turns[2].weights[0] > 0
    assert len(spy.calls) == 1 and assistant_text in spy.calls[0]
    # Full assistant text is an anchor, but its evidence chunks are excluded.
    assert 'apple pear orchard.' not in spy.calls[0]


def test_user_only_embedding_economy():
    spy = SpyEmbedder()
    turns = [turn('user', 'apple pear orchard'),
             turn('assistant', 'Excluded assistant sentence. Another excluded sentence.'),
             turn('user', 'invoice tax accounting')]
    score_dialogue(turns, spy, IGConfig(state_roles='user_only'))
    assert spy.calls == [['apple pear orchard', 'invoice tax accounting']]


def test_stripped_anchor_deduplication():
    spy = SpyEmbedder()
    score_dialogue([turn('user', 'Склади звіт.\n'), turn('assistant', 'Склади звіт.')], spy)
    assert spy.calls == [['Склади звіт.']]


@pytest.mark.parametrize('state_roles', ['both', 'assistant_only'])
def test_empty_first_user_does_not_anchor_assistant(state_roles):
    result = score_dialogue([turn('user', ''), turn('assistant', 'apple pear orchard')],
                            HashingEmbedder(32), IGConfig(state_roles=state_roles))
    assert result.turns[0].n_chunks == 0 and result.turns[0].ig == 0
    assert result.turns[1].weights == [1.0] and result.turns[1].ig > 0


def test_empty_user_keeps_last_nonempty_anchor():
    spy = SpyEmbedder()
    result = score_dialogue([turn('user', 'apple pear orchard'), turn('user', '   '),
                             turn('assistant', 'apple pear orchard')], spy)
    assert result.turns[1].n_chunks == 0 and result.turns[1].ig == 0
    assert result.turns[2].weights == [1.0]
    assert spy.calls == [['apple pear orchard']]


@pytest.mark.parametrize('relevance', ['first_user', 'previous_assistant'])
def test_empty_turns_do_not_replace_user_relevance_anchors(relevance):
    turns = [turn('user', ''), turn('user', 'apple pear orchard'),
             turn('assistant', 'apple pear orchard'), turn('assistant', ' \n '),
             turn('user', 'apple pear orchard')]
    result = score_dialogue(turns, HashingEmbedder(32),
                            IGConfig(state_roles='user_only', user_relevance=relevance))
    assert result.turns[1].weights == [1.0]
    assert result.turns[-1].weights == [1.0]
