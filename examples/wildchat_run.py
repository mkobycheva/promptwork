import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

__doc__ = """Score local WildChat JSONL reproducibly, with equal treatment of languages.

The segmenter and 12-character threshold were designed for scripts with spaces
and letter case (Latin, Cyrillic). Results for other scripts, e.g. Chinese and
Japanese, are not comparable and must be read per language. Public dataset text
should stay local; --no-text omits prefixes. Compare user_ig, not user_share:
user_only has share 1 for positive total, assistant_only has share 0.
Cache the embedding model once before going offline. No dataset Hub access occurs
here, and model snapshot lookup is local-only. --toy requires no model downloads.
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import random
import subprocess

import numpy as np
from promptwork import HashingEmbedder, IGConfig, SentenceTransformerEmbedder, score_dialogue


CONFIGS = {
    f'{state}|{relevance}': IGConfig(state_roles=state, user_relevance=relevance)
    for state in ('both', 'user_only')
    for relevance in ('none', 'first_user', 'previous_assistant')
}
CONFIGS['assistant_only|paper'] = IGConfig(state_roles='assistant_only', keep_short_turns=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class CachingEmbedder:
    def __init__(self, base):
        self.base, self.cache = base, {}

    def embed(self, texts):
        missing = [text for text in dict.fromkeys(texts) if text not in self.cache]
        if missing:
            self.cache.update(zip(missing, self.base.embed(missing)))
        return np.stack([self.cache[text] for text in texts])


def select_rows(rows, languages='all', n_per_language=25, top_languages=5,
                seed=42, min_turns=4, max_turns=20):
    """Select deterministically, counting top languages before message filtering."""
    counts = Counter(row['language'] for row in rows)
    if languages.strip().casefold() == 'all':
        used = sorted(counts, key=lambda label: (-counts[label], label))[:top_languages]
    else:
        requested = {label.strip().casefold() for label in languages.split(',') if label.strip()}
        used = sorted(label for label in counts if label.casefold() in requested)
    selected = []
    for language in used:
        candidates = sorted((row for row in rows if row['language'] == language and
                             min_turns <= len(row['turns']) <= max_turns),
                            key=lambda row: row['conversation_hash'])
        if len(candidates) < n_per_language:
            print(f'{language}: only {len(candidates)} eligible dialogues; taking all (requested {n_per_language})')
            chosen = candidates
        else:
            chosen = random.Random(f'{seed}:{language}').sample(candidates, n_per_language)
        selected.extend(chosen)
    return selected, used


def select_dialogues(path, languages='all', n_per_language=25, top_languages=5,
                     seed=42, min_turns=4, max_turns=20):
    with open(path, encoding='utf-8') as source:
        rows = [json.loads(line) for line in source if line.strip()]
    return select_rows(rows, languages, n_per_language, top_languages, seed, min_turns, max_turns)


def toy_rows():
    """Three fictional Ukrainian accounting conversations, not tax advice."""
    def turn(role, content):
        return {'role': role, 'content': content}

    opening = turn('user', 'Підготуй звіт для ФОП групи 3. Дохід 100000 гривень без ПДВ.')
    return [
        {'conversation_hash': 'toy-correction', 'language': 'Ukrainian', 'toxic': False,
         'conversation': [opening,
            turn('assistant', 'Для ФОП групи 3 ставка 3%. Податок становить 3000 гривень.'),
            turn('user', 'Ставка 5%, а не 3%, згідно зі ст. 293 ПКУ. Виправ розрахунок.'),
            turn('assistant', 'Виправляю ставку на 5%. Податок становить 5000 гривень.')]},
        {'conversation_hash': 'toy-filler', 'language': 'Ukrainian', 'toxic': False,
         'conversation': [opening,
            turn('assistant', 'Для звіту ФОП потрібні дохід і період. Дохід 100000 гривень.'),
            turn('user', 'Дякую, ок'),
            turn('assistant', 'Будь ласка, звертайтеся за потреби.')]},
        {'conversation_hash': 'toy-repeat', 'language': 'Ukrainian', 'toxic': False,
         'conversation': [opening,
            turn('assistant', 'Для звіту ФОП потрібні дохід і період. Дохід 100000 гривень.'),
            opening,
            turn('assistant', 'Дохід 100000 гривень без ПДВ. Звіт для ФОП групи 3.')]},
    ]


def main():
    parser = argparse.ArgumentParser(description='Score local WildChat JSONL by language.')
    parser.add_argument('--input', help='Prepared JSONL; required unless --toy')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--languages', default='all')
    parser.add_argument('--top-languages', type=int, default=5)
    parser.add_argument('--n-per-language', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-turns', type=int, default=4)
    parser.add_argument('--max-turns', type=int, default=20)
    parser.add_argument('--model', default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--max-seq-length', type=int, default=512)
    parser.add_argument('--no-text', action='store_true')
    parser.add_argument('--run-dir', default=None)
    args = parser.parse_args()
    if not args.toy and not args.input:
        parser.error('--input is required unless --toy')
    if args.n_per_language < 1 or args.top_languages < 1 or args.max_seq_length < 1 or \
            not 0 <= args.min_turns <= args.max_turns:
        parser.error('Require positive counts/max-seq-length and 0 <= min-turns <= max-turns')
    run_dir = Path(args.run_dir or ('results/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    run_dir.mkdir(parents=True, exist_ok=False)
    if args.toy:
        print('WARNING: HashingEmbedder results are TOY-ONLY and not semantically meaningful.')
        rows = [{'conversation_hash': row['conversation_hash'], 'language': row['language'],
                 'turns': row['conversation']} for row in toy_rows()]
        input_path = run_dir / 'toy_input.jsonl'
        input_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    else:
        input_path = Path(args.input)
    selected, languages = select_dialogues(input_path, args.languages, args.n_per_language,
                                            args.top_languages, args.seed, args.min_turns, args.max_turns)
    print(f'Selected {len(selected)} dialogues; languages: {languages}')
    snapshot_commit = None
    embedder = None
    if selected:
        if args.toy:
            base = HashingEmbedder()
        else:
            from huggingface_hub import snapshot_download
            # Resolve a cached snapshot to guarantee the scorer itself is offline.
            if Path(args.model).is_dir():
                model_path = args.model
            else:
                model_path = snapshot_download(args.model, local_files_only=True)
                snapshot_commit = Path(model_path).name
            base = SentenceTransformerEmbedder(model_path)
            base.model.max_seq_length = args.max_seq_length
        embedder = CachingEmbedder(base)
    (run_dir / 'selected_hashes.txt').write_text(
        ''.join(f'{row["conversation_hash"]}\t{row["language"]}\n' for row in selected), encoding='utf-8')
    summary = defaultdict(lambda: {'count': 0, 'total': 0.0, 'user_ig': 0.0, 'user_chunks': 0, 'silenced': 0})
    norm_count, norm_sum, norm_min, norm_max = 0, 0.0, None, None
    with (run_dir / 'wildchat_scores.csv').open('w', newline='', encoding='utf-8') as df, \
         (run_dir / 'wildchat_turns.csv').open('w', newline='', encoding='utf-8') as tf:
        dialogue_writer, turn_writer = csv.writer(df), csv.writer(tf)
        dialogue_writer.writerow(['conversation_id', 'language', 'n_turns', 'config', 'total', 'user_ig',
                                   'assistant_ig', 'user_share', 'n_user_turns', 'n_assistant_turns'])
        turn_writer.writerow(['conversation_id', 'language', 'config', 'turn_index', 'role', 'n_chunks',
                              'n_active_chunks', 'mean_weight', 'ig', 'cumulative_ig', 'text_prefix'])
        for row in selected:
            conv_id, language, turns = row['conversation_hash'], row['language'], row['turns']
            counts = Counter(turn['role'] for turn in turns)
            for name, config in CONFIGS.items():
                result = score_dialogue(turns, embedder, config)
                by = result.by_role()
                dialogue_writer.writerow([conv_id, language, len(turns), name, f'{result.total:.4f}',
                                          f'{by["user"]:.4f}', f'{by["assistant"]:.4f}', f'{result.user_share:.4f}',
                                          counts['user'], counts['assistant']])
                stats = summary[language, name]
                stats['count'] += 1
                stats['total'] += result.total
                stats['user_ig'] += by['user']
                for turn in result.turns:
                    prefix = '' if args.no_text else turns[turn.index]['content'][:80].replace('\n', ' ').replace('\r', ' ')
                    weight = sum(turn.weights) / turn.n_chunks if turn.n_chunks else 0.0
                    turn_writer.writerow([conv_id, language, name, turn.index, turn.role, turn.n_chunks,
                                          turn.n_active_chunks, f'{weight:.4f}', f'{turn.ig:.4f}',
                                          f'{turn.cumulative_ig:.4f}', prefix])
                    if turn.role == 'user' and turn.n_chunks:
                        stats['user_chunks'] += 1
                        stats['silenced'] += turn.n_active_chunks == 0
            for vec in embedder.cache.values():
                norm = float(np.linalg.norm(vec))
                norm_count += 1
                norm_sum += norm
                norm_min = norm if norm_min is None else min(norm_min, norm)
                norm_max = norm if norm_max is None else max(norm_max, norm)
            embedder.cache.clear()
    norms = {'count': norm_count, 'min': norm_min, 'mean': norm_sum / norm_count if norm_count else None, 'max': norm_max}
    if norm_count:
        print(f'Embedding norms: min={norm_min:.3f} mean={norms["mean"]:.3f} max={norm_max:.3f}')
    else:
        print('Embedding norms: no evidence embedded')
    print('Summary by language and config (no pooled average):')
    for language in languages:
        for name, config in CONFIGS.items():
            stats = summary[language, name]
            if not stats['count']:
                print(f'{language} | {name}: no dialogues scored')
                continue
            line = (f'{language} | {name}: mean total={stats["total"] / stats["count"]:.4f} '
                    f'mean user_ig={stats["user_ig"] / stats["count"]:.4f}')
            if config.user_relevance != 'none':
                line += (f' silenced user share={stats["silenced"] / stats["user_chunks"]:.4f}'
                         if stats['user_chunks'] else ' silenced user share=n/a')
            print(line)
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {'input_path': str(input_path.resolve()), 'input_sha256': sha256_file(input_path),
                'languages_used': languages, 'n_per_language': args.n_per_language, 'seed': args.seed,
                'filters': {'languages': args.languages, 'top_languages': args.top_languages,
                            'min_turns': args.min_turns, 'max_turns': args.max_turns},
                'model_name': 'HashingEmbedder' if args.toy else args.model,
                'max_seq_length': args.max_seq_length, 'model_snapshot_commit': snapshot_commit,
                'configs': {name: asdict(config) for name, config in CONFIGS.items()},
                'versions': {name: version(name) for name in ['numpy', 'sentence_transformers', 'torch', 'promptwork']},
                'promptwork_git_commit': commit, 'embedding_norms': norms,
                'no_text': args.no_text, 'toy': args.toy,
                'utc_timestamp': datetime.now(timezone.utc).isoformat()}
    (run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote run artifacts to {run_dir}')


if __name__ == '__main__':
    main()
