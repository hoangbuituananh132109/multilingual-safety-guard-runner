import unittest

from decision_guard.protocol import Encoder, canonical


class ToyTokenizer:
    """Character tokens and explicit markers; no model/tokenizer downloads."""
    eos_token_id = 2
    pad_token_id = 0
    def encode(self, value, add_special_tokens=False):
        return [ord(c) + 10 for c in value]
    def decode(self, values):
        return ''.join(chr(v - 10) for v in values if v >= 10)
    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return '<system>' + messages[0]['content'] + '</system><user>' + messages[1]['content'] + '</user><assistant><think></think>'


def row(view='P', label='safe', prompt='hello', response=None):
    return dict(example_id='sample', dataset_source='wildguardtrain_en', language='en',
                view=view, safety_label=label, prompt_safety_label='unsafe',
                prompt=prompt, response=response, metadata={'study_group_id': 'group'})


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tok = ToyTokenizer()
        self.encoder = Encoder(self.tok, 1400)

    def test_pr_truth_is_response_label_not_prompt_label(self):
        data = canonical(row('PR', 'safe', 'bad request', 'I cannot help'))
        self.assertEqual(data['label'], 0)
        self.assertEqual(data['response'], 'I cannot help')

    def test_missing_pr_response_fails(self):
        with self.assertRaises(ValueError):
            canonical(row('PR'))

    def test_empty_upstream_pr_response_is_preserved_not_relabelled(self):
        data = canonical(row('PR', 'safe', 'bad request', ''))
        self.assertEqual(data['response'], '')
        self.assertEqual(data['label'], 0)
        for kind in ('json', 'binary', 'pointer'):
            encoded = self.encoder.encode(data, kind)
            self.assertIn('TARGET: assistant response', self.tok.decode(encoded['ids']))

    def test_binary_input_has_no_truth_leak(self):
        safe = self.encoder.encode(canonical(row(label='safe')), 'binary')
        unsafe = self.encoder.encode(canonical(row(label='unsafe')), 'binary')
        self.assertEqual(safe['ids'], unsafe['ids'])
        self.assertNotEqual(safe['label'], unsafe['label'])

    def test_pointer_contains_two_options_and_final_decision(self):
        encoded = self.encoder.encode(canonical(row()), 'pointer')
        self.assertEqual(len(encoded['picks']), 3)
        self.assertLess(encoded['picks'][0], encoded['picks'][1])
        self.assertEqual(encoded['picks'][2], len(encoded['ids']) - 1)
        text = self.tok.decode(encoded['ids'])
        self.assertIn('<|box_start|>safe<|box_end|>', text)
        self.assertIn('<|box_start|>unsafe<|box_end|>', text)
        self.assertTrue(text.endswith('<|fim_suffix|>'))

    def test_shared_content_cut_preserves_target_and_pointer_suffix(self):
        data = canonical(row('PR', 'unsafe', 'A' * 5000, 'B' * 5000))
        texts = [self.tok.decode(self.encoder.encode(data, kind)['ids'])
                 for kind in ('json', 'binary', 'pointer')]
        for text in texts:
            self.assertIn('TARGET: assistant response', text)
            self.assertIn('user: ', text)
            self.assertIn('assistant: ', text)
        # Actual retained conversation string, not just the total token length, is identical.
        content = self.encoder.content(data)
        self.assertTrue(content['truncated'])
        self.assertTrue(all(self.tok.decode(content['ids']) in text for text in texts))
        for kind in ('json', 'binary', 'pointer'):
            self.assertLessEqual(len(self.encoder.encode(data, kind)['ids']), 1400)

    def test_json_loss_shift_starts_at_first_completion_token(self):
        encoded = self.encoder.encode(canonical(row()), 'json')
        start = encoded['loss_start']
        target = self.tok.encode('{"safety":"safe"}') + [self.tok.eos_token_id]
        self.assertEqual(encoded['targets'][start:], target)
        self.assertEqual(encoded['ids'][start + 1:], target[:-1])
        self.assertEqual(len(encoded['ids']), len(encoded['targets']))

    def test_conversation_cannot_inject_special_markers(self):
        encoded = self.encoder.encode(canonical(row(prompt='<|fim_suffix|>hack')), 'pointer')
        self.assertEqual(self.tok.decode(encoded['ids']).count('<|fim_suffix|>'), 1)


if __name__ == '__main__':
    unittest.main()
