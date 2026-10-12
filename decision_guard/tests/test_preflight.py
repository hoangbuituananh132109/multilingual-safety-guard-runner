import json
import tempfile
import unittest
from pathlib import Path

from decision_guard.preflight import inspect_model, verify_receipt


class PreflightTests(unittest.TestCase):
    def test_local_model_receipt_checks_every_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps(dict(model_type='qwen3', num_hidden_layers=36,
                hidden_size=2560, intermediate_size=9728, num_attention_heads=32,
                num_key_value_heads=8, head_dim=128, vocab_size=151936, tie_word_embeddings=True)))
            (root / 'tokenizer_config.json').write_text('{}')
            (root / 'tokenizer.json').write_text('{}')
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'model-1.safetensors'}}))
            with self.assertRaises(FileNotFoundError):
                inspect_model(root)
            (root / 'model-1.safetensors').write_bytes(b'test-local-shard')
            result = inspect_model(root)
            self.assertIn('model-1.safetensors', result['weights'])
            first = result['model_sha256']
            verify_receipt(root, result)
            (root / 'model-1.safetensors').write_bytes(b'changed-local-shard')
            with self.assertRaises(ValueError):
                verify_receipt(root, result)
            self.assertNotEqual(inspect_model(root)['model_sha256'], first)

    def test_wrong_backbone_cannot_enter_study(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps(dict(model_type='llama', num_hidden_layers=32)))
            with self.assertRaises(ValueError):
                inspect_model(root)


if __name__ == '__main__':
    unittest.main()
