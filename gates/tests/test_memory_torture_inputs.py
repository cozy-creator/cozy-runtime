"""Exercise the shell's actual batch/tag setup without launching Cozy."""

import re
import subprocess
import tempfile
import unittest
from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "memory-torture.sh").read_text()
SETUP = SOURCE[SOURCE.index("COHORT=${"):SOURCE.index('if [ -n "$ON" ]; then')]
PROMPT = re.search(r"prompt_tag\(\) \{.*?\n\}", SOURCE, re.S)[0]


class BatchInputs(unittest.TestCase):
    def test_default_batches_are_unique_even_with_the_same_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            command = 'date() { printf fixed-clock; };\n' + SETUP + '\nprintf "%s" "$D"'
            values = [subprocess.check_output(["bash", "-eu", "-c", command],
                                              cwd=directory, text=True) for _ in range(2)]
            self.assertNotEqual(*values)

    def test_default_cell_tags_differ_and_explicit_control_tag_is_unchanged(self):
        script = PROMPT + '\nTAG=control; INPUT_TAG=""; prompt_tag cold; echo; prompt_tag warm; echo; '
        script += 'INPUT_TAG=control; prompt_tag cold; echo; prompt_tag warm'
        lines = subprocess.check_output(["bash", "-eu", "-c", script], text=True).splitlines()
        self.assertEqual(lines, ["control cell=cold", "control cell=warm", "control", "control"])


if __name__ == "__main__":
    unittest.main()
