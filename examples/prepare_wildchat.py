import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

__doc__ = """Prepare pinned WildChat shards as sorted, reproducible local JSONL.

With --languages all --sample-fraction 1 the output can be large. Only exact
selected shards are downloaded; approval is required unless --yes is supplied.
Public dataset text should stay local. Language labels receive equal treatment.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import platform

REVISION = '7d6490e462285cf85d91eabea0f9a954fbddcd1f'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def language_labels(languages):
    if isinstance(languages, str):
        if languages.strip().casefold() == 'all':
            return None
        languages = languages.split(',')
    labels = sorted({label.strip().casefold() for label in languages if label.strip()})
    if not labels:
        raise ValueError('languages must be all or a nonempty list of labels')
    return labels


def shard_patterns(file_names, shard_indices):
    """Return exact shard filenames, validating requested numeric indices."""
    shards = {}
    for name in file_names:
        match = re.fullmatch(r'data/[^/]+-(\d+)-of-\d+\.parquet', name)
        if match:
            index = int(match.group(1))
            if index in shards:
                raise ValueError(f'Ambiguous shard index {index}: multiple files')
            shards[index] = name
    if not shards:
        raise ValueError('No indexed parquet shards found under data/')
    if shard_indices == 'all':
        indices = sorted(shards)
    else:
        try:
            indices = sorted({int(i) for i in (shard_indices.split(',')
                                              if isinstance(shard_indices, str) else shard_indices)})
        except (TypeError, ValueError) as exc:
            raise ValueError('shards must be all or comma-separated integer indices') from exc
        missing = set(indices) - shards.keys()
        if not indices or missing:
            raise ValueError(f'Invalid shard indices {sorted(missing)}; valid range '
                             f'{min(shards)}..{max(shards)}; valid indices: {sorted(shards)}')
    return [shards[index] for index in indices]


def sampled(conversation_hash, fraction, salt):
    return int(hashlib.sha256((salt + conversation_hash).encode()).hexdigest()[:8], 16) / 2**32 < fraction


def filter_rows(parquet_path, languages='all', min_messages=4, max_messages=20,
                fraction=0.05, salt='promptwork-v1'):
    """Read language first; select row indices within one row group at a time.

    Parquet decodes selected groups internally; only selected row indices are
    converted to Python conversation objects. Never accumulate the full dataset.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    labels = language_labels(languages)
    parquet = pq.ParquetFile(parquet_path)
    scanned, kept = Counter(), []
    for group in range(parquet.num_row_groups):
        raw_labels = parquet.read_row_group(group, columns=['language'])['language'].to_pylist()
        scanned.update(raw_labels)
        indices = [i for i, label in enumerate(raw_labels)
                   if labels is None or str(label).casefold() in labels]
        if not indices:
            continue
        table = parquet.read_row_group(group, columns=['conversation_hash', 'language', 'conversation'])
        for row in table.take(pa.array(indices, type=pa.int64())).to_pylist():
            turns = row['conversation']
            conv_hash = row['conversation_hash']
            if min_messages <= len(turns) <= max_messages and sampled(conv_hash, fraction, salt):
                kept.append({'conversation_hash': conv_hash, 'language': row['language'],
                             'turns': [{'role': t['role'], 'content': t['content']} for t in turns]})
    return sorted(kept, key=lambda row: row['conversation_hash']), scanned


def build_manifest(parameters, files, scanned, written, output):
    import pyarrow
    import huggingface_hub
    top = sorted(scanned, key=lambda label: (-scanned[label], str(label)))[:30]
    return {**parameters, 'shards_processed': [file['name'] for file in files],
            'downloaded_files': files, 'rows_scanned': sum(scanned.values()),
            'dialogues_written': sum(written.values()),
            'per_language_counts': [{'language': label, 'scanned': scanned[label],
                                     'written': written[label]} for label in top],
            'output_sha256': sha256_file(output),
            'versions': {'pyarrow': pyarrow.__version__, 'huggingface_hub': huggingface_hub.__version__,
                         'python': platform.python_version()},
            'utc_timestamp': datetime.now(timezone.utc).isoformat()}


def output_path(out_dir, parameters):
    labels = parameters['languages']
    langkey = 'all' if labels == 'all' else '-'.join(labels)
    # Keep labels readable while excluding path separators and unsafe characters.
    langkey = re.sub(r'[^\w.-]', '_', langkey)
    shards = parameters['shards']
    shardkey = 'all' if shards == 'all' else '-'.join(map(str, shards))
    return Path(out_dir) / (f'wildchat_{langkey}_{parameters["revision"][:8]}_s{shardkey}'
                            f'_f{parameters["sample_fraction"]:g}.jsonl')


def existing_output(output, parameters, force=False):
    manifest_path = output.with_suffix('.manifest.json')
    if not output.exists() and not manifest_path.exists():
        return False
    if force:
        return False
    if output.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if all(manifest.get(key) == value for key, value in parameters.items()) and \
                manifest.get('output_sha256') == sha256_file(output):
            print(f'Skipped existing matching output: {output}')
            return True
    raise FileExistsError(f'Output exists with different parameters or invalid checksum: {output}; use --force')


def prepare_files(files, parameters, out_dir, force=False):
    """Prepare local shards without Hub access; return (output, manifest, skipped)."""
    output = output_path(out_dir, parameters)
    if existing_output(output, parameters, force):
        return output, json.loads(output.with_suffix('.manifest.json').read_text()), True
    output.parent.mkdir(parents=True, exist_ok=True)
    scanned, written, file_info = Counter(), Counter(), []
    with tempfile.TemporaryDirectory(dir=output.parent) as temp_dir:
        connection = sqlite3.connect(str(Path(temp_dir) / 'sort.sqlite'))
        try:
            connection.execute('CREATE TABLE rows (hash TEXT PRIMARY KEY, payload TEXT, language TEXT)')
            for name, path, size in files:
                rows, counts = filter_rows(path, parameters['languages'], parameters['min_messages'],
                                           parameters['max_messages'], parameters['sample_fraction'],
                                           parameters['sample_salt'])
                scanned.update(counts)
                for row in rows:
                    connection.execute('INSERT INTO rows VALUES (?, ?, ?)',
                                       (row['conversation_hash'], json.dumps(row, ensure_ascii=False), row['language']))
                    written[row['language']] += 1
                connection.commit()
                file_info.append({'name': name, 'size_bytes': size, 'sha256': sha256_file(path)})
            staged = Path(temp_dir) / 'output.jsonl'
            with staged.open('w', encoding='utf-8', newline='\n') as target:
                for (payload,) in connection.execute('SELECT payload FROM rows ORDER BY hash'):
                    target.write(payload + '\n')
            manifest = build_manifest(parameters, file_info, scanned, written, staged)
            staged.replace(output)
            output.with_suffix('.manifest.json').write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        finally:
            connection.close()
    print('Language labels (top 15): scanned / written')
    for label in sorted(scanned, key=lambda label: (-scanned[label], str(label)))[:15]:
        print(f'{label}: {scanned[label]} / {written[label]}')
    print(f'Wrote {written.total()} dialogues to {output}')
    return output, manifest, False


def main():
    parser = argparse.ArgumentParser(description='Prepare pinned, shard-selective WildChat JSONL.')
    parser.add_argument('--repo', default='allenai/WildChat-1M')
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--languages', default='all')
    parser.add_argument('--min-messages', type=int, default=4)
    parser.add_argument('--max-messages', type=int, default=20)
    parser.add_argument('--sample-fraction', type=float, default=0.05)
    parser.add_argument('--sample-salt', default='promptwork-v1')
    parser.add_argument('--out-dir', default='data/wildchat')
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--shards', default='0,7', help='Comma-separated indices, or all (default: 0,7)')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--yes', action='store_true', help='Approve selected shard download without prompting')
    args = parser.parse_args()
    if not 0 < args.sample_fraction <= 1 or not 0 <= args.min_messages <= args.max_messages or args.max_workers < 1:
        parser.error('Require 0 < fraction <= 1, 0 <= min-messages <= max-messages, max-workers >= 1')
    try:
        labels = language_labels(args.languages)
        shards = 'all' if args.shards == 'all' else sorted({int(i) for i in args.shards.split(',')})
    except ValueError as exc:
        parser.error(str(exc))
    parameters = {'repo': args.repo, 'revision': args.revision, 'languages': labels or 'all',
                  'min_messages': args.min_messages, 'max_messages': args.max_messages,
                  'sample_fraction': args.sample_fraction, 'sample_salt': args.sample_salt, 'shards': shards}
    output = output_path(args.out_dir, parameters)
    if existing_output(output, parameters, args.force):
        return
    if not (os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')):
        print('Hint: set HF_TOKEN if Hub authentication is required (a cached login may also work).')
    from huggingface_hub import HfApi, snapshot_download
    listing = list(HfApi().list_repo_tree(args.repo, repo_type='dataset', revision=args.revision,
                                         path_in_repo='data', expand=True))
    sizes = {entry.path: entry.size for entry in listing if hasattr(entry, 'size')}
    selected = shard_patterns(sizes, shards)
    print(f'Selected {len(selected)} files; download up to {sum(sizes[name] for name in selected) / 1e9:.3f} GB')
    if not args.yes:
        if not sys.stdin.isatty():
            parser.exit(2, 'Non-interactive session: pass --yes to approve the selected download.\n')
        if input('Download selected shards? (y/N) ').strip().lower() != 'y':
            parser.exit(0, 'Download cancelled.\n')
    snapshot = snapshot_download(args.repo, repo_type='dataset', revision=args.revision,
                                 allow_patterns=selected, max_workers=args.max_workers)
    files = [(name, Path(snapshot) / name, sizes[name]) for name in selected]
    prepare_files(files, parameters, args.out_dir, args.force)


if __name__ == '__main__':
    main()
