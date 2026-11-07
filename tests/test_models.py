"""Check scripted generation without models or external services."""

import unittest

from turnguard.models import Generator, GenRequest, StubGenerator


def request(episode_id: str, step: int = 1) -> GenRequest:
    return GenRequest(episode_id=episode_id, step=step, messages=[])


class StubGeneratorTests(unittest.TestCase):
    def test_scripts_advance_independently_in_batch_order(self) -> None:
        generator: Generator = StubGenerator({"a": ["a1", "a2"], "b": ["b1", "b2"]})
        self.assertEqual(generator.generate([request("b"), request("a")]), ["b1", "a1"])
        self.assertEqual(
            generator.generate([request("a", 2), request("b", 2)]), ["a2", "b2"]
        )
        self.assertEqual(generator.name, "stub")

    def test_repeated_episode_in_one_batch_advances_script(self) -> None:
        generator = StubGenerator({"a": ["first", "second"]})
        self.assertEqual(
            generator.generate([request("a"), request("a", 2)]), ["first", "second"]
        )

    def test_exhaustion_identifies_episode_and_step(self) -> None:
        generator = StubGenerator({"a": ["only"]})
        generator.generate([request("a")])
        with self.assertRaisesRegex(RuntimeError, "Script exhausted.*'a'.*step 2"):
            generator.generate([request("a", 2)])

    def test_empty_script_is_exhausted(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Script exhausted.*'a'"):
            StubGenerator({"a": []}).generate([request("a")])

    def test_unknown_episode_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "No script configured.*'missing'"):
            StubGenerator({}).generate([request("missing")])

    def test_empty_batch_does_not_consume_script(self) -> None:
        generator = StubGenerator({"a": ["first"]})
        self.assertEqual(generator.generate([]), [])
        self.assertEqual(generator.generate([request("a")]), ["first"])

    def test_script_is_copied_and_generator_instances_are_independent(self) -> None:
        scripts = {"a": ["first", "second"]}
        first = StubGenerator(scripts)
        second = StubGenerator(scripts)
        scripts["a"][0] = "changed"
        self.assertEqual(first.generate([request("a")]), ["first"])
        self.assertEqual(first.generate([request("a", 2)]), ["second"])
        self.assertEqual(second.generate([request("a")]), ["first"])
