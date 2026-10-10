"""Read-only metadata and retained-work tests; no installation or network."""

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "startup", Path(__file__).with_name("startup_update.py")
)
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


class StartupMetadata(unittest.TestCase):
    def test_only_runtime_owned_software_state_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            self.assertTrue(startup.software_state(store)["quiescent"])
            path = store / ".cozy-workspace/journal.sqlite3"
            path.parent.mkdir()
            with sqlite3.connect(path) as db:
                db.executescript(
                    "CREATE TABLE state(quiescent INT, accepted_work INT, workspace_id TEXT);"
                    "CREATE VIEW machine_software_update_v1 AS SELECT * FROM state;"
                )
                db.execute("INSERT INTO state VALUES(1,1,'original')")
                db.commit()
                self.assertEqual(
                    startup.software_state(store),
                    {"quiescent": True, "accepted_work": True, "workspace_id": "original"},
                )
                db.execute("DROP VIEW machine_software_update_v1")
                db.commit()
                with self.assertRaises(sqlite3.OperationalError):
                    startup.software_state(store)

    def test_zero_major_stays_within_minor_line(self):
        with patch.object(
            startup, "read_json", return_value={"releases": {"0.4.0": [], "1.0.0": []}}
        ):
            self.assertEqual(list(startup.candidates("tensorfs", "0.3.78")), [])

    def test_explicit_development_versions(self):
        for version in ("0.18.86.dev1", "0.18.86+local", "0.18.86rc1"):
            self.assertFalse(startup.stable(version))
        self.assertTrue(startup.stable("0.18.86"))

    def test_cpu_does_not_invent_torch_extra_and_never_upgrades_it(self):
        requirements = {
            "cozy-runtime": ["tensorfs>=0.3.78", 'torch>=2.14; extra == "model-execution"']
        }
        installed = {"cozy-runtime": "0.18.86", "tensorfs": "0.3.78"}
        self.assertTrue(startup.compatible({}, installed, requirements))
        self.assertFalse(startup.compatible({}, installed | {"torch": "2.13"}, requirements))
        self.assertTrue(startup.compatible({}, installed | {"torch": "2.14"}, requirements))
        requirements["consumer"] = ["cozy-runtime<0.18.86"]
        self.assertFalse(startup.compatible({}, installed, requirements))

    def test_newest_compatible_pair_with_dependency_fallback(self):
        installed = {"cozy-runtime": "0.18.85", "tensorfs": "0.3.78"}
        requirements = {"cozy-runtime": ["tensorfs>=0.3.78"], "tensorfs": []}
        candidates = {
            "cozy-runtime": [
                {"version": "0.18.87", "requires": ["tensorfs>=0.4"]},
                {"version": "0.18.86", "requires": ["tensorfs>=0.3.78"]},
            ],
            "tensorfs": [{"version": "0.3.78", "requires": []}],
        }
        with (
            patch.object(startup, "installed_metadata", return_value=(installed, requirements)),
            patch.object(startup, "candidates", side_effect=lambda name, _: candidates[name]),
        ):
            selected = startup.plan()
            self.assertEqual(selected["cozy-runtime"]["version"], "0.18.86")
            self.assertEqual(selected["tensorfs"]["version"], "0.3.78")


class AdditiveDependencies(unittest.TestCase):
    def pair(self, requirements):
        return {
            "cozy-runtime": {"version": "0.18.88", "requires": requirements},
            "tensorfs": {"version": "0.3.78", "requires": []},
        }

    def test_missing_transitive_closure_never_changes_installed_framework(self):
        installed = {"cozy-runtime": "0.18.87", "tensorfs": "0.3.78", "framework": "8.0+local"}
        requirements = {"cozy-runtime": [], "tensorfs": [], "framework": ["new-dependency<2"]}
        choices = {
            "new-dependency": [
                {"name": "new-dependency", "version": "2.0", "requires": []},
                {
                    "name": "new-dependency",
                    "version": "1.0",
                    "requires": ["leaf>=3", "framework>=8"],
                },
            ],
            "leaf": [{"name": "leaf", "version": "3.0", "requires": []}],
        }
        with patch.object(
            startup, "dependency_candidates", side_effect=lambda name, _: choices[name]
        ):
            selected = startup.resolve_additive(
                self.pair(["new-dependency>=1"]), installed, requirements
            )
        self.assertEqual(
            {row["name"]: row["version"] for row in selected},
            {"new-dependency": "1.0", "leaf": "3.0"},
        )
        self.assertEqual(installed["framework"], "8.0+local")

    def test_conflicting_existing_version_defers_before_any_download(self):
        with patch.object(startup, "dependency_candidates") as fetch:
            with self.assertRaisesRegex(ValueError, "base upgrades"):
                startup.resolve_additive(
                    self.pair(["framework>=9"]),
                    {"framework": "8", "tensorfs": "0.3.78"},
                    {"framework": []},
                )
            fetch.assert_not_called()

    def test_existing_distribution_extras_are_traversed(self):
        installed = {"library": "1", "tensorfs": "0.3.78"}
        requirements = {"library": ['addon; extra == "feature"']}
        with patch.object(
            startup,
            "dependency_candidates",
            return_value=[{"name": "addon", "version": "1", "requires": []}],
        ):
            selected = startup.resolve_additive(
                self.pair(["library[feature]"]), installed, requirements
            )
        self.assertEqual([row["name"] for row in selected], ["addon"])
        with patch.object(startup, "dependency_candidates") as fetch:
            self.assertEqual(
                startup.resolve_additive(self.pair(["library"]), installed, requirements), []
            )
            fetch.assert_not_called()

    def test_actual_overlay_metadata_must_have_complete_closure(self):
        with patch.object(startup, "dependency_candidates") as fetch:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                startup.resolve_additive(
                    self.pair(["missing"]), {"tensorfs": "0.3.78"}, {}, allow_fetch=False
                )
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
