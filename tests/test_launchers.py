import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def test_lookaway_launcher_propagates_identity_and_method_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            (data_dir / "trainset_clean.parquet").touch()
            (data_dir / "token_priors_frozen.json").write_text("{}", encoding="utf-8")
            (data_dir / "token_freq_counts.json").write_text("{}", encoding="utf-8")
            (data_dir / "dataset_generation_manifest.json").write_text("[]", encoding="utf-8")

            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            argv_path = tmp_path / "argv.txt"
            fake_python = fake_bin / "python3"
            fake_python.write_text('#!/bin/sh\nprintf \'%s\n\' "$@" > "$ARGV_PATH"\n', encoding="utf-8")
            fake_python.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "ARGV_PATH": str(argv_path),
                    "DATA_DIR": str(data_dir),
                    "WORK_DIR": str(tmp_path / "work"),
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "PYTHON": str(fake_python),
                }
            )
            subprocess.run(["bash", str(ROOT / "scripts/run_lookaway.sh")], env=env, check=True)
            args = argv_path.read_text(encoding="utf-8").splitlines()

        self.assertIn("trainer.project_name=LookAway", args)
        self.assertIn("trainer.group_name=LookAway-Qwen3.5-4B", args)
        self.assertIn("trainer.experiment_name=LookAway-Qwen3.5-4B", args)
        self.assertIn("+actor_rollout_ref.actor.self_distillation.vd_targeted=true", args)
        self.assertIn("+actor_rollout_ref.actor.self_distillation.vd_freq_decay=true", args)
        self.assertIn(
            f"+actor_rollout_ref.actor.self_distillation.vd_freq_file={data_dir / 'token_freq_counts.json'}", args
        )

    def test_lookaway_launcher_disable_freq_decay_needs_no_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            (data_dir / "trainset_clean.parquet").touch()
            (data_dir / "token_priors_frozen.json").write_text("{}", encoding="utf-8")
            (data_dir / "dataset_generation_manifest.json").write_text("[]", encoding="utf-8")
            # NOTE: no token_freq_counts.json -- the disabled path must not require it.

            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            argv_path = tmp_path / "argv.txt"
            fake_python = fake_bin / "python3"
            fake_python.write_text('#!/bin/sh\nprintf \'%s\n\' "$@" > "$ARGV_PATH"\n', encoding="utf-8")
            fake_python.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "ARGV_PATH": str(argv_path),
                    "DATA_DIR": str(data_dir),
                    "WORK_DIR": str(tmp_path / "work"),
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "PYTHON": str(fake_python),
                    "VD_FREQ_DECAY": "false",
                }
            )
            subprocess.run(["bash", str(ROOT / "scripts/run_lookaway.sh")], env=env, check=True)
            args = argv_path.read_text(encoding="utf-8").splitlines()

        self.assertIn("+actor_rollout_ref.actor.self_distillation.vd_targeted=true", args)
        self.assertFalse(any("vd_freq_decay" in arg or "vd_freq_file" in arg for arg in args))

    def test_non_default_remote_model_requires_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env.update(
                {
                    "MODEL_PATH": "another/model",
                    "MODEL_REVISION": "",
                    "PYTHON": sys.executable,
                    "WORK_DIR": tmp,
                }
            )
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/run_lookaway.sh")],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("revision is required", result.stderr)

    def test_training_launcher_uses_python_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            argv_path = tmp_path / "argv.txt"
            fake_python = tmp_path / "custom-python"
            fake_python.write_text(
                chr(35) + "!/bin/sh" + chr(10) + 'printf \'%s\\n\' "$@" > "$ARGV_PATH"' + chr(10),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            for name in ("trainset_clean.parquet", "token_priors_frozen.json",
                         "token_freq_counts.json", "dataset_generation_manifest.json"):
                (data_dir / name).touch()
            env = os.environ.copy()
            env.update({
                "ARGV_PATH": str(argv_path), "PYTHON": str(fake_python),
                "WORK_DIR": str(tmp_path / "work"), "DATA_DIR": str(data_dir),
            })
            subprocess.run(
                ["bash", str(ROOT / "scripts/run_lookaway.sh"), "trainer.total_training_steps=1"],
                env=env,
                check=True,
            )
            args = argv_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(args[:3], ["-m", "verl.trainer.main_ppo", "--config-name"])
        self.assertIn("trainer.total_training_steps=1", args)

