"""Pinned local ZIP installation, audit and shared read-only SQLite token caches."""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from contextlib import closing
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import zipfile

from .protocol import Encoder, KINDS, canonical

ARCHIVE_SHA = '61e43353485d1d0e3986507b2b4524a2f90afa5f42dc98c67575f8126e61ccf4'
FILES = ('train.jsonl', 'validation.jsonl', 'manifest.json', 'validation_report.json')
PINNED_HASHES = {
    'train.jsonl': '9de0f098edafe174bba811be0a663af96c123031100658854b98b81c6c17338f',
    'validation.jsonl': '7261cdb00bf1e29271d702501d9686f35fe57ba76c99f1270cc530c6959697a5',
    'manifest.json': '7a8016bea2ec8f62acb37ff6ab3a6c96f26adc0d732a4a90b38462cec640c3e0',
    'validation_report.json': 'f60ba48c2437246dab55f10493e7b2d853c090d966f30785f849c8608aa6cd85',
}


def offline():
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def install(archive_path, output, expected_archive=ARCHIVE_SHA):
    archive_path, output = Path(archive_path), Path(output)
    if sha256(archive_path) != expected_archive:
        raise ValueError('Wrong dataset ZIP SHA-256; no installation attempted')
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != 5 or set(names) != set(FILES) | {'bundle_manifest.json'}:
            raise ValueError('Unexpected or duplicated ZIP members')
        manifest = json.loads(archive.read('bundle_manifest.json'))
        expected = {m['name']: m for m in manifest['members']}
        if set(expected) != set(FILES):
            raise ValueError('Missing member fingerprints')
        if expected_archive == ARCHIVE_SHA and any(expected[n]['sha256'] != PINNED_HASHES[n] for n in FILES):
            raise ValueError('Pinned member hashes differ')
        def verify(directory):
            for name, member in expected.items():
                path = directory / name
                if not path.is_file() or path.stat().st_size != member['bytes'] or sha256(path) != member['sha256']:
                    raise ValueError(f'Existing/extracted data is not valid: {path}; preserve it and choose a new directory')
        if output.exists() and any(output.iterdir()):
            verify(output)
            return dict(status='verified_existing', directory=str(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=output.name + '.install-', dir=output.parent))
        try:
            for name in names:
                with archive.open(name) as source, (temporary / name).open('wb') as dest:
                    shutil.copyfileobj(source, dest)
            verify(temporary)
            if output.exists():
                output.rmdir()  # Only the already-checked empty explicit destination.
            temporary.rename(output)
        except Exception:
            # Keep incomplete data recoverable; no recursive cleanup/deletion.
            raise RuntimeError(f'Installation failed; partial files kept at {temporary}') from sys.exc_info()[1]
    return dict(status='installed', directory=str(output), archive_sha256=expected_archive)


def raw_rows(path, limit=None):
    with Path(path).open(encoding='utf-8') as handle:
        for line in itertools.islice(handle, limit):
            yield json.loads(line)


def audit(data_dir):
    data_dir = Path(data_dir)
    for name, expected in PINNED_HASHES.items():
        if sha256(data_dir / name) != expected:
            raise ValueError(f'Pinned data differs: {name}')
    stats, groups, contents = {}, {}, {}
    for split, expected_count in (('train', 274760), ('validation', 1000)):
        counts, group_ids, content_ids, ids = Counter(), set(), set(), set()
        for raw in raw_rows(data_dir / f'{split}.jsonl'):
            row = canonical(raw)
            if row['id'] in ids:
                raise ValueError(f'Duplicate example ID {row["id"]}')
            ids.add(row['id'])
            counts['examples'] += 1
            if row['view'] == 'PR' and row['response'] == '':
                counts['empty_response_examples'] += 1
            counts[f'source:{row["source"]}'] += 1
            counts[f'cell:{row["view"]}:{row["label"]}'] += 1
            counts[f'language:{row["source"]}:{row["language"]}'] += 1
            group_ids.add(row['group'])
            content_ids.add(raw['content_sha256'])
        if counts['examples'] != expected_count:
            raise ValueError(f'Wrong {split} count: {counts["examples"]}')
        stats[split], groups[split], contents[split] = dict(counts), group_ids, content_ids
    if groups['train'] & groups['validation'] or contents['train'] & contents['validation']:
        raise ValueError('Train/validation overlap')
    if stats['train'].get('source:nemotron_v3_9lang') != 192332 or stats['train'].get('source:wildguardtrain_en') != 82428:
        raise ValueError('Unexpected source proportions')
    return dict(valid=True, archive_sha256=ARCHIVE_SHA, splits=stats,
                group_overlap=0, content_overlap=0, hashes=PINNED_HASHES)


def _blob(values):
    data = array('I', values)
    if sys.byteorder != 'little':
        data.byteswap()
    return data.tobytes()


def _unblob(blob):
    data = array('I')
    data.frombytes(blob)
    if sys.byteorder != 'little':
        data.byteswap()
    return data.tolist()


def read_cache_manifest(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        return json.loads(connection.execute('SELECT value FROM meta').fetchone()[0])


class Cache:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.connection = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
        self.manifest = read_cache_manifest(path)
        self.count = self.manifest['examples']

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        item = self.connection.execute('SELECT ids,targets,loss_start,picks,label FROM rows WHERE i=?', (int(index),)).fetchone()
        if item is None:
            raise IndexError(index)
        result = dict(ids=_unblob(item[0]), picks=json.loads(item[3]), label=item[4])
        if item[1] is not None:
            result.update(targets=_unblob(item[1]), loss_start=item[2])
        return result


def create_cache(rows, output, encoder, request, split):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    connections = {}
    for kind in KINDS:
        path = output / f'{split}.{kind}.sqlite'
        if path.exists():
            raise FileExistsError(path)
        db = sqlite3.connect(path)
        db.execute('CREATE TABLE rows(i INTEGER PRIMARY KEY,ids BLOB,targets BLOB,loss_start INTEGER,picks TEXT,label INTEGER,id TEXT,source TEXT,language TEXT,view TEXT,group_id TEXT)')
        db.execute('CREATE TABLE meta(value TEXT)')
        connections[kind] = db
    count, truncated = 0, 0
    digest = hashlib.sha256()
    try:
        for index, raw in enumerate(rows):
            row = canonical(raw)
            content = encoder.content(row)
            digest.update(row['id'].encode() + b'\0' + _blob(content['ids']))
            truncated += int(content['truncated'])
            for kind, connection in connections.items():
                item = encoder.encode(row, kind, content)
                connection.execute('INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    (index, _blob(item['ids']), _blob(item['targets']) if 'targets' in item else None,
                     item.get('loss_start'), json.dumps(item['picks']), item['label'], row['id'],
                     row['source'], row['language'], row['view'], row['group']))
            count += 1
            if count % 1000 == 0:
                for connection in connections.values():
                    connection.commit()
                print(f'prepare {split}: {count} rows', flush=True)
        if count == 0:
            raise ValueError('Empty token cache')
        for kind, connection in connections.items():
            manifest = dict(format='decision_guard_tokens_v1', kind=kind, request=request,
                            examples=count, content_sha256=digest.hexdigest(), truncated=truncated,
                            max_length=encoder.max_length, pad_id=encoder.tokenizer.pad_token_id)
            connection.execute('INSERT INTO meta VALUES(?)', (json.dumps(manifest, sort_keys=True),))
            connection.commit()
    finally:
        for connection in connections.values():
            connection.close()
    return dict(examples=count, truncated=truncated, content_sha256=digest.hexdigest())


def tokenizer_fingerprint(model_dir):
    model_dir = Path(model_dir)
    names = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt')
    hashes = {name: sha256(model_dir / name) for name in names if (model_dir / name).is_file()}
    if 'config.json' not in hashes or 'tokenizer_config.json' not in hashes:
        raise FileNotFoundError('Local model/config/tokenizer files required')
    return hashes


def prepare(data_dir, model_dir, output, max_length=2048, train_limit=None, validation_limit=None):
    offline()
    from transformers import AutoTokenizer
    data_dir, model_dir, output = Path(data_dir), Path(model_dir), Path(output)
    request = dict(data=PINNED_HASHES, tokenizer=tokenizer_fingerprint(model_dir), max_length=max_length,
                   train_limit=train_limit, validation_limit=validation_limit, protocol_sha256=sha256(Path(__file__).with_name('protocol.py')))
    manifest_file = output / 'cache_manifest.json'
    if output.exists() and any(output.iterdir()):
        if not manifest_file.is_file():
            raise ValueError(f'Incomplete cache {output}; choose another cache directory, do not overwrite')
        manifest = json.loads(manifest_file.read_text())
        if manifest['request'] != request:
            raise ValueError('Cache tokenizer/data/protocol/settings changed: choose a new cache directory')
        for name, digest in manifest['files'].items():
            if sha256(output / name) != digest:
                raise ValueError(f'Cache file changed: {name}')
        return dict(status='verified_existing', **manifest)
    # Verify all source bytes before encoding, including for a bounded smoke cache.
    for name, digest in PINNED_HASHES.items():
        if sha256(data_dir / name) != digest:
            raise ValueError(f'Pinned source changed: {name}')
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir.resolve()), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        raise ValueError('Qwen tokenizer must supply its existing padding token; no new vocab')
    for marker in ('<|fim_prefix|>', '<|fim_middle|>', '<|box_start|>', '<|box_end|>', '<|fim_suffix|>'):
        tokens = tokenizer.encode(marker, add_special_tokens=False)
        if len(tokens) != 1 or tokenizer.convert_ids_to_tokens(tokens[0]) != marker:
            raise ValueError(f'Local tokenizer lacks the expected single-token pointer marker {marker}')
    encoder = Encoder(tokenizer, max_length)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + '.prepare-', dir=output.parent))
    splits = {}
    for split, limit in (('train', train_limit), ('validation', validation_limit)):
        splits[split] = create_cache(raw_rows(data_dir / f'{split}.jsonl', limit), temporary, encoder, request, split)
    files = {path.name: sha256(path) for path in sorted(temporary.glob('*.sqlite'))}
    manifest = dict(request=request, splits=splits, files=files, content_budget=encoder.content_budget)
    (temporary / 'cache_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if output.exists():
        output.rmdir()
    temporary.rename(output)
    return dict(status='prepared', **manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('install', 'audit', 'prepare'))
    parser.add_argument('--zip', type=Path)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.phase == 'install':
        result = install(args.zip, args.data_dir)
    elif args.phase == 'audit':
        result = audit(args.data_dir)
    else:
        result = prepare(args.data_dir, args.model, args.output, args.max_length,
                         128 if args.smoke else None, 16 if args.smoke else None)
    if args.phase == 'audit' and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
