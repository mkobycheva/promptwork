"""Offline synthetic parquet and JSONL tests; no model or Hub requests."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import types

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'examples' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script('prepare_wildchat')
runner = load_script('wildchat_run')


def make_rows(n=12):
    return [{'conversation_hash': f'hash-{i:04}', 'language': ['English', 'Ukrainian', 'Chinese'][i % 3],
             'conversation': [{'role': 'user' if j % 2 == 0 else 'assistant', 'content': f'message {i} {j}'}
                              for j in range(3 + i % 4)]} for i in range(n)]


def write_parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=4)
    return path


def parameters(**changes):
    return {'repo': 'test/repo', 'revision': prepare.REVISION, 'languages': 'all',
            'min_messages': 4, 'max_messages': 20, 'sample_fraction': 1.0,
            'sample_salt': 'promptwork-v1', 'shards': [0, 7], **changes}


def test_shard_patterns():
    names = [f'data/train-{i:05}-of-00014.parquet' for i in range(14)] + ['README.md']
    assert prepare.shard_patterns(names, '0,7') == [names[0], names[7]]
    assert prepare.shard_patterns(names[::-1], 'all') == names[:14]
    with pytest.raises(ValueError, match=r'valid range 0..13'):
        prepare.shard_patterns(names, '14')


def test_filter_and_manifest(tmp_path):
    rows = make_rows()
    path = write_parquet(tmp_path / 'shard.parquet', rows[::-1])
    all_rows, scanned = prepare.filter_rows(path, 'all', 0, 20, 1, 'salt')
    assert len(all_rows) == 12 and set(scanned) == {'English', 'Ukrainian', 'Chinese'}
    assert [row['conversation_hash'] for row in all_rows] == sorted(row['conversation_hash'] for row in rows)
    selected, counts = prepare.filter_rows(path, 'english,UKRAINIAN', 4, 5, 1, 'salt')
    assert all(row['language'] in ['English', 'Ukrainian'] and 4 <= len(row['turns']) <= 5 for row in selected)
    assert counts == scanned
    assert all(set(turn) == {'role', 'content'} for row in selected for turn in row['turns'])
    files = [('data/train-00000-of-00014.parquet', path, path.stat().st_size)]
    output, manifest, skipped = prepare.prepare_files(files, parameters(), tmp_path / 'out')
    assert not skipped
    assert '_s0-7_f1.jsonl' in output.name and 'partial' not in output.name
    assert manifest['rows_scanned'] == 12
    assert manifest['dialogues_written'] == 9
    assert manifest['revision'] == prepare.REVISION
    assert manifest['shards'] == [0, 7]
    assert manifest['repo'] == 'test/repo'
    assert manifest['languages'] == 'all'
    assert manifest['sample_fraction'] == 1
    assert manifest['sample_salt'] == 'promptwork-v1'
    assert manifest['min_messages'] == 4 and manifest['max_messages'] == 20
    assert manifest['output_sha256'] == prepare.sha256_file(output)
    assert manifest['downloaded_files'][0] == {'name': files[0][0], 'size_bytes': path.stat().st_size,
                                               'sha256': prepare.sha256_file(path)}
    assert set(manifest['versions']) == {'python', 'pyarrow', 'huggingface_hub'}
    assert manifest['utc_timestamp'].endswith('+00:00')
    written = Counter(row['language'] for row in map(json.loads, output.read_text().splitlines()))
    assert {entry['language']: entry['scanned'] for entry in manifest['per_language_counts']} == scanned
    assert {entry['language']: entry['written'] for entry in manifest['per_language_counts']} == written
    before, manifest_before = output.read_bytes(), output.with_suffix('.manifest.json').read_bytes()
    _, _, skipped = prepare.prepare_files(files, parameters(), tmp_path / 'out')
    assert skipped and output.read_bytes() == before
    assert output.with_suffix('.manifest.json').read_bytes() == manifest_before
    _, _, skipped = prepare.prepare_files(files, parameters(), tmp_path / 'out', force=True)
    assert not skipped and output.read_bytes() == before
    with pytest.raises(FileExistsError):
        prepare.prepare_files(files, parameters(sample_salt='different'), tmp_path / 'out')


def test_hash_sampling_order_and_partition_independence(tmp_path):
    rows = make_rows(2000)
    a = write_parquet(tmp_path / 'a.parquet', rows)
    shuffled = rows.copy()
    random.Random(33).shuffle(shuffled)
    b = write_parquet(tmp_path / 'b.parquet', shuffled)
    c = write_parquet(tmp_path / 'c.parquet', rows[:1000])
    d = write_parquet(tmp_path / 'd.parquet', rows[1000:])
    def hashes(path, salt='salt'):
        return {row['conversation_hash'] for row in prepare.filter_rows(path, 'all', 0, 20, .2, salt)[0]}
    selected = hashes(a)
    assert selected == hashes(b) == hashes(c) | hashes(d)
    assert selected != hashes(a, 'other salt')
    assert 300 <= len(selected) <= 500
    expected = {row['conversation_hash'] for row in rows if
                int(hashlib.sha256(('salt' + row['conversation_hash']).encode()).hexdigest()[:8], 16) / 2**32 < .2}
    assert selected == expected
    assert len(prepare.filter_rows(a, 'all', 0, 20, 1, 'salt')[0]) == 2000


def test_selection(tmp_path):
    rows = [{'conversation_hash': row['conversation_hash'], 'language': row['language'],
             'turns': row['conversation']} for row in make_rows(90)]
    rows += [{'conversation_hash': f'extra-{i}', 'language': 'English', 'turns': rows[0]['turns']} for i in range(10)]
    a, b = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    a.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    b.write_text(''.join(json.dumps(row) + '\n' for row in reversed(rows)))
    selected, used = runner.select_dialogues(a, 'all', 5, 2, 42, 4, 5)
    assert used == ['English', 'Chinese']  # Tie-breaking is alphabetic.
    assert len(selected) == 10 and all(4 <= len(row['turns']) <= 5 for row in selected)
    assert (selected, used) == runner.select_dialogues(a, 'all', 5, 2, 42, 4, 5)
    assert (selected, used) == runner.select_dialogues(b, 'all', 5, 2, 42, 4, 5)
    assert selected != runner.select_dialogues(a, 'all', 5, 2, 99, 4, 5)[0]
    explicit, languages = runner.select_dialogues(a, 'UKRAINIAN,english', 1000, 2, 42, 4, 5)
    assert languages == ['English', 'Ukrainian']
    assert len(explicit) == sum(row['language'] in languages and 4 <= len(row['turns']) <= 5 for row in rows)


def test_download_exact_files_and_revision(monkeypatch, tmp_path):
    rows = make_rows()
    snapshot = tmp_path / 'snapshot'
    (snapshot / 'data').mkdir(parents=True)
    names = [f'data/train-{i:05}-of-00014.parquet' for i in range(14)]
    for index in [0, 7]:
        write_parquet(snapshot / names[index], rows[index:index+1])
    calls = {}
    class FakeApi:
        def list_repo_tree(self, repo, **kwargs):
            calls['listing'] = (repo, kwargs)
            return [types.SimpleNamespace(path=name, size=10) for name in names]
    def download(repo, **kwargs):
        calls['download'] = (repo, kwargs)
        return str(snapshot)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'HfApi', FakeApi)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    monkeypatch.setattr(sys, 'argv', ['prepare', '--out-dir', str(tmp_path / 'out'), '--yes'])
    prepare.main()
    assert calls['listing'][1]['revision'] == prepare.REVISION
    assert calls['download'][1]['revision'] == prepare.REVISION
    assert calls['download'][1]['allow_patterns'] == [names[0], names[7]]
    calls.clear()
    prepare.main()
    assert calls == {}  # Idempotence avoids even the listing network call.


def test_noninteractive_requires_yes(monkeypatch, tmp_path):
    import huggingface_hub
    class FakeApi:
        def list_repo_tree(self, *args, **kwargs):
            return [types.SimpleNamespace(path=f'data/train-{i:05}-of-00014.parquet', size=10) for i in range(14)]
    monkeypatch.setattr(huggingface_hub, 'HfApi', FakeApi)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda *a, **k: pytest.fail('must not download'))
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr(sys, 'argv', ['prepare', '--out-dir', str(tmp_path / 'out')])
    with pytest.raises(SystemExit) as exc:
        prepare.main()
    assert exc.value.code == 2


def test_run_artifacts_and_language_grouping(monkeypatch, tmp_path, capsys):
    source = tmp_path / 'input.jsonl'
    rows = [{'conversation_hash': row['conversation_hash'], 'language': row['language'],
             'turns': row['conversation']} for row in make_rows(12)]
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    model_path = tmp_path / 'model'
    model_path.mkdir()
    from promptwork import HashingEmbedder
    class OfflineEmbedder(HashingEmbedder):
        def __init__(self, path):
            super().__init__(16)
            self.model = types.SimpleNamespace()
    monkeypatch.setattr(runner, 'SentenceTransformerEmbedder', OfflineEmbedder)
    run_dir = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['run', '--input', str(source), '--run-dir', str(run_dir),
                                    '--model', str(model_path), '--n-per-language', '2', '--no-text'])
    runner.main()
    import csv
    with (run_dir / 'wildchat_scores.csv').open() as f:
        scores = list(csv.DictReader(f))
    with (run_dir / 'wildchat_turns.csv').open() as f:
        turns = list(csv.DictReader(f))
    manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    assert len(scores) == 6 * 7 and {row['language'] for row in scores} == {'English', 'Chinese', 'Ukrainian'}
    assert all(row['text_prefix'] == '' for row in turns)
    assert manifest['input_sha256'] == runner.sha256_file(source)
    assert manifest['seed'] == 42 and manifest['n_per_language'] == 2
    assert len(manifest['configs']) == 7
    assert manifest['configs']['both|none']['sigma_sq'] == .25
    assert set(manifest['versions']) == {'numpy', 'sentence_transformers', 'torch', 'promptwork'}
    assert manifest['model_snapshot_commit'] is None
    assert manifest['embedding_norms']['count'] > 0
    assert len((run_dir / 'selected_hashes.txt').read_text().splitlines()) == 6
    stdout = capsys.readouterr().out
    assert 'English | both|none:' in stdout and 'Chinese | both|none:' in stdout
    assert 'Ukrainian | both|none:' in stdout
