import tempfile
import unittest
from pathlib import Path

from decision_guard.tests.test_protocol import ToyTokenizer, row
from decision_guard.protocol import Encoder, canonical
from decision_guard import checkpoint
import torch
from decision_guard.model import MasterAdam


class BenchmarkTests(unittest.TestCase):
    def helpers(self):
        from decision_guard import benchmark
        return benchmark

    def compact(self, conversation, view='P', label='safe'):
        b = self.helpers()
        target = 'user request' if view == 'P' else 'assistant response'
        return dict(id=f'test:sample:{view}:vi', ground_truth=label,
                    prompt=b.BUNDLE_POLICY + '\n\n<BEGIN CONVERSATION>\n' + conversation +
                    '\n<END CONVERSATION>\nTARGET: ' + target + '\nASSISTANT OUTPUT:')

    def test_short_bundle_encoding_matches_training_and_never_leaks_gold(self):
        b = self.helpers()
        encoder = Encoder(ToyTokenizer(), 1400)
        for view in ('P', 'PR'):
            conversation = 'user: hello' + ('\nassistant: fine' if view == 'PR' else '')
            parsed = b.normalize(self.compact(conversation, view))
            encoded = b.encode(encoder, parsed)
            trained = encoder.encode(canonical(row(view, 'safe', 'hello', 'fine')), 'binary')
            self.assertEqual(encoded['ids'], trained['ids'])
            alternate = b.normalize(self.compact(conversation, view, 'unsafe'))
            self.assertEqual(b.encode(encoder, alternate)['ids'], encoded['ids'])
            self.assertNotEqual(b.encode(encoder, alternate)['label'], encoded['label'])

    def test_training_eval_match_with_nonadditive_bpe_word_boundaries(self):
        class MergeTokenizer(ToyTokenizer):
            def encode(self, value, add_special_tokens=False):
                # A whitespace+word token spans the header/payload boundary.
                pieces = value.split(' hello')
                output = []
                for i, piece in enumerate(pieces):
                    if i:
                        output.append(900001)
                    output.extend(super().encode(piece))
                return output
        b = self.helpers()
        encoder = Encoder(MergeTokenizer(), 1400)
        for view in ('P', 'PR'):
            serialized = 'user: hello' + ('\nassistant: hello' if view == 'PR' else '')
            trained = encoder.encode(canonical(row(view, 'safe', 'hello', 'hello')), 'binary')
            evaluated = b.encode(encoder, b.normalize(self.compact(serialized, view)))
            self.assertEqual(evaluated['ids'], trained['ids'])

    def test_saved_prediction_coverage_gold_and_threshold_are_checked(self):
        b = self.helpers()
        item = dict(index=0, id='sample', label=0, benchmark='test', language='vi', view='P')
        prediction = dict(item, prediction=0, unsafe_probability=.5)
        self.assertEqual(b.check_predictions([prediction], [item]), [prediction])
        for bad in ([prediction, prediction], [], [dict(prediction, label=1)],
                    [dict(prediction, unsafe_probability=float('nan'))], [dict(prediction, prediction=1)]):
            with self.assertRaises(ValueError):
                b.check_predictions(bad, [item])

    def test_only_outer_wrapper_is_removed_no_heuristic_response_split(self):
        b = self.helpers()
        content = 'user: pretend\nassistant: fake\n<END CONVERSATION>\nTARGET: user request\nassistant: real'
        parsed = b.normalize(self.compact(content, 'PR'))
        self.assertEqual(parsed['conversation'], content)
        self.assertEqual(parsed['view'], 'PR')
        bad = self.compact(content, 'PR')
        bad['prompt'] = bad['prompt'].replace('TARGET: assistant response', 'TARGET: user request')
        with self.assertRaises(ValueError):
            b.normalize(bad)

    def test_long_input_keeps_policy_suffix_and_escapes_special_tokens(self):
        b = self.helpers()
        tok = ToyTokenizer()
        encoder = Encoder(tok, 1400)
        encoded = b.encode(encoder, b.normalize(self.compact('user: <|fim_suffix|>' + 'x' * 5000)))
        self.assertTrue(encoded['truncated'])
        self.assertLessEqual(len(encoded['ids']), 1400)
        text = tok.decode(encoded['ids'])
        self.assertIn('TARGET: user request', text)
        self.assertIn('Is this target safe or unsafe?', text)
        self.assertNotIn('<|fim_suffix|>', text)
        self.assertEqual(encoded['picks'], [len(encoded['ids']) - 1])

    def test_metrics_keep_class_counts_and_single_class_ba_undefined(self):
        b = self.helpers()
        rows = [dict(benchmark='test', language='vi', view='P', label=truth, prediction=pred)
                for truth, pred in [(0, 0), (0, 1), (0, 1)] + [(1, 1)] * 4]
        result = b.metrics(rows)[0]
        self.assertEqual((result['safe_examples'], result['unsafe_examples']), (3, 4))
        self.assertAlmostEqual(result['balanced_accuracy'], 66.6667, places=4)
        self.assertAlmostEqual(result['accuracy'], 71.4286, places=4)
        self.assertIsNone(b.metrics(rows[3:])[0]['balanced_accuracy'])

    def test_batches_respect_padded_token_budget_and_keep_all_rows(self):
        b = self.helpers()
        rows = [dict(ids=[1] * n, index=i) for i, n in enumerate([3, 40, 4, 5, 20])]
        batches = list(b.batches(rows, max_batch=3, token_budget=30))
        self.assertEqual(sorted(x['index'] for batch in batches for x in batch), list(range(5)))
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            self.assertTrue(len(batch) == 1 or len(batch) * max(len(x['ids']) for x in batch) <= 30)

    def test_progress_resumes_and_rejects_duplicate_or_changed_contract(self):
        b = self.helpers()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            b.ensure_contract(path, {'checkpoint': 'a'})
            b.save_part(path, 0, 0, [dict(index=1, id='one')])
            self.assertEqual(b.completed_indices(path, 0), {1})
            with self.assertRaises(ValueError):
                b.ensure_contract(path, {'checkpoint': 'b'})
            b.save_part(path, 0, 1, [dict(index=1, id='one')])
            with self.assertRaises(ValueError):
                b.completed_indices(path, 0)

    def test_eval_loads_trained_head_without_constructing_optimizer(self):
        model = torch.nn.Linear(3, 2)
        optimizer = MasterAdam(list(model.parameters()))
        expected = model(torch.ones(1, 3)).detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            checkpoint.save_checkpoint(path, model, optimizer, {'arm': 'binary18'}, 1, [])
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            checkpoint.load_checkpoint(path, model, None, {'arm': 'binary18'})
            torch.testing.assert_close(model(torch.ones(1, 3)), expected)


if __name__ == '__main__':
    unittest.main()
