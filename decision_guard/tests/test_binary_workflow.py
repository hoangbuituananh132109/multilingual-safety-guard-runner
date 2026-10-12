import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class WorkflowTests(unittest.TestCase):
    def test_binary_runner_gates_training_on_smoke_eval_and_runs_no_other_arm(self):
        root = Path(__file__).resolve().parents[2]
        bash = shutil.which('bash')
        if os.name == 'nt':
            bash = r'C:\Program Files\Git\bin\bash.exe'
        if not bash or not Path(bash).is_file():
            self.skipTest('Bash is unavailable')
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / 'decision_guard').mkdir()
            for name in ('run.sh', 'run_binary_b200.sh', 'settings.env.sh'):
                source = root / 'decision_guard' / name
                self.assertTrue(source.is_file(), f'Missing integrated binary workflow: {name}')
                (workspace / 'decision_guard' / name).write_text(source.read_text(), encoding='utf-8', newline='\n')
            fake = workspace / 'fake.py'
            # Only replace expensive CLI boundaries; the actual Bash orchestration is under test.
            fake.write_text('''import json, os, sys
from pathlib import Path
a=sys.argv[1:]
with open('calls.jsonl','a') as f: f.write(json.dumps(a)+'\\n')
def value(name): return a[a.index(name)+1]
if 'decision_guard.preflight' in a:
 p=Path(value('--output')); p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}')
if 'decision_guard.train' in a:
 p=Path(value('--output')); p.mkdir(parents=True,exist_ok=True)
 cert=p/'smoke_pass.json'; resumed=1 if cert.exists() else 0
 cert.write_text(json.dumps({'steps':1,'resumed_from_step':resumed}))
if 'decision_guard.evaluate' in a and '--limit-per-benchmark' in a and os.environ.get('FAIL_EVAL'):
 sys.exit(7)
if a and a[0]=='-c' and 'resumed_from_step' in a[1]:
 assert json.loads(Path(a[-1]).read_text())['resumed_from_step']==1
''', encoding='utf-8')
            # A tiny launcher lets Bash use a path containing spaces to the current Python.
            import sys
            python = Path(sys.executable).as_posix()
            (workspace / 'fake-python').write_text(f'#!/usr/bin/env bash\nexec "{python}" "{fake.as_posix()}" "$@"\n', encoding='utf-8', newline='\n')
            os.chmod(workspace / 'fake-python', 0o755)
            # Git Bash on Windows lacks Linux flock. This test checks phase gates,
            # not OS locking; production still requires the real image's flock.
            (workspace / 'flock').write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8', newline='\n')
            os.chmod(workspace / 'flock', 0o755)
            (workspace / 'fixture.zip').touch()
            (workspace / 'benchmark.jsonl').touch()
            env = dict(os.environ, DG_PYTHON='./fake-python', DG_DATA_ZIP='fixture.zip',
                       DG_BENCHMARK='benchmark.jsonl', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
            command = [bash, '-c', 'PATH="$PWD:$PATH"; exec bash decision_guard/run_binary_b200.sh binary18']
            result = subprocess.run(command, cwd=workspace, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in (workspace / 'calls.jsonl').read_text().splitlines()]
            train = [c for c in calls if 'decision_guard.train' in c]
            evaluation = [c for c in calls if 'decision_guard.evaluate' in c]
            self.assertEqual(len(train), 3)  # one-step save, fresh-process reload, real train
            self.assertTrue(all(c[c.index('--arm')+1] == 'binary18' for c in train + evaluation))
            self.assertTrue(all(c[c.index('--save-every')+1] == '2000' for c in train))
            self.assertEqual(len(evaluation), 2)  # smoke checkpoint, final full benchmark
            self.assertLess(calls.index(evaluation[0]), calls.index(train[-1]))
            self.assertLess(calls.index(train[-1]), calls.index(evaluation[-1]))
            self.assertIn('--limit-per-benchmark', evaluation[0])
            self.assertNotIn('--limit-per-benchmark', evaluation[1])
            (workspace / 'calls.jsonl').unlink()
            result = subprocess.run(command, cwd=workspace, env=dict(env, FAIL_EVAL='1'), capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            calls = [json.loads(line) for line in (workspace / 'calls.jsonl').read_text().splitlines()]
            self.assertFalse(any('decision_guard.train' in c and '--smoke' not in c for c in calls))


if __name__ == '__main__':
    unittest.main()
