import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from decision_guard.data import sha256, install, create_cache, Cache, read_cache_manifest
from decision_guard.protocol import Encoder
from test_protocol import ToyTokenizer, row


class DataTests(unittest.TestCase):
    def test_installer_checks_hash_and_is_idempotent_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'data.zip'
            members = {name: b'{}\n' for name in ('train.jsonl', 'validation.jsonl', 'manifest.json', 'validation_report.json')}
            import hashlib
            manifest = dict(members=[dict(name=n, bytes=len(v), sha256=hashlib.sha256(v).hexdigest())
                                     for n, v in members.items()])
            with zipfile.ZipFile(archive, 'w') as handle:
                for name, content in members.items():
                    handle.writestr(name, content)
                handle.writestr('bundle_manifest.json', json.dumps(manifest))
            output = root / 'installed'
            digest = sha256(archive)
            self.assertEqual(install(archive, output, expected_archive=digest)['status'], 'installed')
            self.assertEqual(install(archive, output, expected_archive=digest)['status'], 'verified_existing')
            (output / 'train.jsonl').write_text('changed')
            with self.assertRaises(ValueError):
                install(archive, output, expected_archive=digest)

    def test_cache_preserves_rows_contract_and_shared_content(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'cache'
            records = [row(label='safe'), dict(row(label='unsafe'), example_id='other')]
            create_cache(iter(records), output, Encoder(ToyTokenizer(), 1400), {'input': 'pinned'}, split='train')
            hashes = []
            for kind in ('json', 'label', 'binary', 'pointer'):
                data = Cache(output / f'train.{kind}.sqlite')
                try:
                    self.assertEqual(len(data), 2)
                    self.assertEqual(data[0]['label'], 0)
                    self.assertEqual(data[1]['label'], 1)
                finally:
                    data.connection.close()
                hashes.append(read_cache_manifest(output / f'train.{kind}.sqlite')['content_sha256'])
            self.assertEqual(len(set(hashes)), 1)


if __name__ == '__main__':
    unittest.main()
