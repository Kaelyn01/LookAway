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
            (data_dir / "train.parquet").touch()
            (data_dir / "token_priors.json").write_text("{}", encoding="utf-8")
            (data_dir / "results.json").write_text("[]", encoding="utf-8")

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
                ["bash", str(ROOT / "scripts/run_vision_opd.sh")],
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
            env = os.environ.copy()
            env.update({"ARGV_PATH": str(argv_path), "PYTHON": str(fake_python), "WORK_DIR": str(tmp_path)})
            subprocess.run(
                ["bash", str(ROOT / "scripts/run_vision_opd.sh"), "trainer.total_training_steps=1"],
                env=env,
                check=True,
            )
            args = argv_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(args[:3], ["-m", "verl.trainer.main_ppo", "--config-name"])
        self.assertIn("trainer.total_training_steps=1", args)

    def test_eval_launcher_keeps_api_keys_out_of_argv(self):
        script = (ROOT / "eval/run_eval.sh").read_text(encoding="utf-8")
        self.assertNotIn("--api_key", script)
        self.assertIn('PYTHON="${PYTHON:-python3}"', script)

    def test_merge_launcher_uses_python_override(self):
        script = (ROOT / "scripts/merge_checkpoint.sh").read_text(encoding="utf-8")
        self.assertIn('PYTHON="${PYTHON:-python3}"', script)
        self.assertIn('"$PYTHON" -m verl.model_merger merge', script)

    def test_failed_checkpoint_merge_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint = tmp_path / "global_step_1"
            (checkpoint / "actor").mkdir(parents=True)
            existing = checkpoint / "config.json"
            existing.write_text("original", encoding="utf-8")

            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            result = subprocess.run(
                ["bash", str(ROOT / "scripts/merge_checkpoint.sh"), str(checkpoint)],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(existing.read_text(encoding="utf-8"), "original")
            self.assertEqual(list(tmp_path.glob("global_step_1.merge.*")), [])
            self.assertEqual(list(tmp_path.glob("global_step_1.backup.*")), [])

    def test_successful_checkpoint_merge_replaces_only_top_level_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint = tmp_path / "global_step_1"
            actor = checkpoint / "actor"
            actor.mkdir(parents=True)
            old_file = checkpoint / "old.bin"
            old_file.write_text("old", encoding="utf-8")

            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\n"
                'while [ "$#" -gt 0 ]; do\n'
                '  if [ "$1" = --target_dir ]; then shift; target=$1; fi\n'
                "  shift\n"
                "done\n"
                'printf merged > "$target/model.safetensors"\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            subprocess.run(
                ["bash", str(ROOT / "scripts/merge_checkpoint.sh"), str(checkpoint)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(actor.is_dir())
            self.assertFalse(old_file.exists())
            self.assertEqual((checkpoint / "model.safetensors").read_text(encoding="utf-8"), "merged")
            self.assertEqual(list(tmp_path.glob("global_step_1.merge.*")), [])
            self.assertEqual(list(tmp_path.glob("global_step_1.backup.*")), [])

    def test_checkpoint_merge_accepts_nested_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint = tmp_path / "global_step_1"
            (checkpoint / "actor").mkdir(parents=True)

            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\n"
                'while [ "$#" -gt 0 ]; do\n'
                '  if [ "$1" = --target_dir ]; then shift; target=$1; fi\n'
                "  shift\n"
                "done\n"
                'mkdir "$target/lora_adapter"\n'
                'printf adapter > "$target/lora_adapter/config.json"\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            subprocess.run(
                ["bash", str(ROOT / "scripts/merge_checkpoint.sh"), str(checkpoint)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual((checkpoint / "lora_adapter/config.json").read_text(), "adapter")

    def test_checkpoint_install_failure_restores_prior_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint = tmp_path / "global_step_1"
            (checkpoint / "actor").mkdir(parents=True)
            existing = checkpoint / "config.json"
            existing.write_text("original", encoding="utf-8")

            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\n"
                'while [ "$#" -gt 0 ]; do\n'
                '  if [ "$1" = --target_dir ]; then shift; target=$1; fi\n'
                "  shift\n"
                "done\n"
                'printf new > "$target/config.json"\n'
                'printf model > "$target/model.safetensors"\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            fake_mv = fake_bin / "mv"
            fake_mv.write_text(
                '#!/bin/sh\ncase "$1" in *.merge.*/model.safetensors) exit 1 ;; esac\nexec /bin/mv "$@"\n',
                encoding="utf-8",
            )
            fake_mv.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            result = subprocess.run(
                ["bash", str(ROOT / "scripts/merge_checkpoint.sh"), str(checkpoint)],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(existing.read_text(encoding="utf-8"), "original")
            self.assertFalse((checkpoint / "model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
