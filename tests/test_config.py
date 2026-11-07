"""Check configuration loading and frozen model settings."""

import tempfile
import unittest
from pathlib import Path

import yaml

from turnguard.config import load_config


class ConfigTests(unittest.TestCase):
    def test_models_and_decoding_settings(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
        config = load_config(path)
        self.assertEqual(
            config["models"],
            {
                "dev": "CohereLabs/tiny-aya-global",
                "headline": "CohereLabs/aya-expanse-8b",
            },
        )
        self.assertEqual(
            config["decoding"],
            {
                "do_sample": False,
                "max_new_tokens": 256,
                "max_steps_per_user_turn": 6,
                "chat_template": "tokenizer",
                "padding_side": "left",
                "tool_result_role": "user",
            },
        )
        self.assertEqual(config["seed"], 1234)

    def test_rejects_non_mapping_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for content in ("", "- item\n", "scalar\n"):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "YAML mapping"):
                        load_config(str(path))

    def test_rejects_unsafe_yaml_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("!!python/object:builtins.object {}", encoding="utf-8")
            with self.assertRaises(yaml.YAMLError):
                load_config(path)
