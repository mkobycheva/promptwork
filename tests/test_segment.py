import pytest
from promptwork.segment import split_evidence


@pytest.mark.parametrize('text,expected', [
    ('- First item\n* Second item\n• Third item', ['First item', 'Second item', 'Third item']),
    ('1. First item\n2) Second item', ['First item', 'Second item']),
    ('Перше речення. Інше речення! Їх багато? Є ще… Ґрунтовно.',
     ['Перше речення.', 'Інше речення!', 'Їх багато?', 'Є ще…', 'Ґрунтовно.']),
    ('Див. ст. 293 ПКУ, ставка 5.5% діє.', ['Див. ст. 293 ПКУ, ставка 5.5% діє.']),
    ('First sentence. "Quoted sentence." (Another sentence.)',
     ['First sentence.', '"Quoted sentence."', '(Another sentence.)']),
])
def test_split(text, expected):
    assert split_evidence(text, min_chunk_chars=0) == expected


def test_short():
    assert split_evidence('ставка 5%') == ['ставка 5%']
    assert split_evidence('ставка 5%', keep_short_turns=False) == []
    assert split_evidence(' \n ') == []
    assert split_evidence('коротко\nДостатньо довге речення.', join_wrapped_lines=False) == ['Достатньо довге речення.']


@pytest.mark.parametrize('text,expected', [
    ('the rate is 5%. please correct it. thanks.',
     ['the rate is 5%.', 'please correct it.', 'thanks.']),
    ('ставка 5%. виправ звіт! дякую?', ['ставка 5%.', 'виправ звіт!', 'дякую?']),
    ('The applicable tax rate\nis 5% for this example.',
     ['The applicable tax rate is 5% for this example.']),
    ('Refer to ст.\n293 ПКУ. please verify.', ['Refer to ст. 293 ПКУ.', 'please verify.']),
    ('Ask Dr.\nSmith about this. next sentence.', ['Ask Dr. Smith about this.', 'next sentence.']),
    ('2024. Ставка була 5%. наступне речення.',
     ['2024. Ставка була 5%.', 'наступне речення.']),
    ('А.\nШевченко написав текст.', ['А. Шевченко написав текст.']),
    ('1. first item\n   continued here\n2) second item',
     ['first item continued here', 'second item']),
    ('first paragraph\n\nsecond paragraph', ['first paragraph', 'second paragraph']),
    ('你好。世界！次の文？', ['你好。', '世界！', '次の文？']),
    ('你好。\n世界！', ['你好。', '世界！']),
    ('First sentence.”’ next sentence.', ['First sentence.”’', 'next sentence.']),
    ('See e.g. apples. another sentence.', ['See e.g. apples.', 'another sentence.']),
    ('Ціна 5.5% і 100 грн. Наступне речення.',
     ['Ціна 5.5% і 100 грн.', 'Наступне речення.']),
    ('hello!\n- \nworld', ['hello!', 'world']),
])
def test_altered_segmentation(text, expected):
    assert split_evidence(text, min_chunk_chars=0) == expected


def test_joining_option_and_short_filter():
    text = 'short\nlong enough evidence here'
    assert split_evidence(text) == ['short long enough evidence here']
    assert split_evidence(text, join_wrapped_lines=False) == ['long enough evidence here']


def test_requested_no_example():
    assert split_evidence('no. ст. 293 ПКУ is not correct.') == [
        'no. ст. 293 ПКУ is not correct.']


def test_dialogue_passes_joining_config(monkeypatch):
    import promptwork.scoring as scoring
    from promptwork import HashingEmbedder, IGConfig

    class NoUpdatePosterior:
        def __init__(self, *args):
            pass

        def update(self, z, alpha):
            return 0.0

    # Isolate segmentation configuration from the unrelated current core error.
    monkeypatch.setattr(scoring, 'GaussianPosterior', NoUpdatePosterior)
    turns = [{'role': 'user', 'content': 'first wrapped fragment\nsecond wrapped fragment'}]
    joined = scoring.score_dialogue(turns, HashingEmbedder(8), IGConfig())
    separate = scoring.score_dialogue(turns, HashingEmbedder(8),
                                      IGConfig(join_wrapped_lines=False))
    assert joined.turns[0].n_chunks == 1
    assert separate.turns[0].n_chunks == 2
