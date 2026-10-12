import unittest
from pathlib import Path
from decision_guard.train import accumulation, step_slots, validate_arm, summarize, check_smoke
import decision_guard.train as training


class RunnerTests(unittest.TestCase):
    def test_smoke_stops_at_one_optimizer_step(self):
        self.assertTrue(hasattr(training, 'planned_steps'), 'Smoke must stop after one step')
        self.assertEqual(training.planned_steps(128, 32, smoke=True), 1)
        self.assertEqual(training.planned_steps(128, 32, smoke=False), 4)

    def test_smoke_without_verified_checkpoint_cannot_authorize_training(self):
        contract = dict(smoke=False, cache_sha256='full', arm='pointer36')
        smoke = dict(contract={**contract, 'smoke': True, 'cache_sha256': 'smoke'}, steps=1)
        with self.assertRaises(ValueError):
            check_smoke(smoke, contract)
        smoke['checkpoint_roundtrip'] = {'passed': True, 'step': 1, 'sha256': 'a' * 64}
        check_smoke(smoke, contract)

    def test_bash_smoke_preparation_has_bounded_flag(self):
        script = Path(__file__).resolve().parents[1].joinpath('run.sh').read_text()
        prepare = script.split('prepare_cache() {', 1)[1].split('\n}', 1)[0]
        self.assertIn('if [[ "$mode" == smoke ]]; then extra=(--smoke); fi', prepare)

    def test_accumulation_keeps_global_batch_32(self):
        self.assertEqual([accumulation(32, 1, n) for n in (1, 2, 4, 8)], [32, 16, 8, 4])
        with self.assertRaises(ValueError):
            accumulation(32, 1, 3)

    def test_last_batch_covers_every_real_row_exactly_once(self):
        seen = []
        for step in range(3):
            for micro in range(2):
                for rank in range(4):
                    seen += [v for v in step_slots(list(range(17)), step, micro, rank, 4, 1, 8)
                             if v is not None]
        self.assertEqual(sorted(seen), list(range(17)))

    def test_only_generation_is_full_depth(self):
        self.assertEqual(validate_arm('binary27'), ('binary', 27))
        self.assertEqual(validate_arm('pointer18'), ('pointer', 18))
        self.assertEqual(validate_arm('json36'), ('json', 36))
        with self.assertRaises(ValueError):
            validate_arm('json18')

    def test_metric_acc_and_balanced_acc_are_separate(self):
        result = summarize([1, 2, 0, 4, 6.0, 7])  # tn/fp/fn/tp/loss_sum/count
        self.assertAlmostEqual(result['accuracy'], 5 / 7)
        self.assertAlmostEqual(result['balanced_accuracy'], (1 / 3 + 1) / 2)

    def test_stale_smoke_cannot_authorize_full_training(self):
        contract = dict(arm='pointer27', world_size=8, model='abc', code_sha256='code',
                        cache_sha256='full', smoke=False, cache_protocol='protocol',
                        environment={'torch': 'fixed'}, global_batch=32)
        smoke = dict(contract={**contract, 'cache_sha256': 'smoke', 'smoke': True}, steps=1,
                     checkpoint_roundtrip={'passed': True, 'step': 1, 'sha256': 'a' * 64})
        check_smoke(smoke, contract)
        for key, value in [('world_size', 4), ('model', 'other'), ('code_sha256', 'new')]:
            changed = {**contract, key: value}
            with self.assertRaises(ValueError):
                check_smoke(smoke, changed)


if __name__ == '__main__':
    unittest.main()
