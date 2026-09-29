"""Read-only metadata and retained-work tests; no installation or network."""
import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("startup", Path(__file__).with_name("startup_update.py"))
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


class StartupMetadata(unittest.TestCase):
    def test_all_owners_and_unfinished_states(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            self.assertTrue(startup.safe_journal(store))
            path = store / ".cozy-workspace/journal.sqlite3"
            path.parent.mkdir()
            with sqlite3.connect(path) as db:
                db.executescript("CREATE TABLE runs(owner TEXT,state TEXT,collected INT); CREATE VIEW machine_runs_v1 AS SELECT * FROM runs;")
                for state in ("queued", "running", "paused", "unknown"):
                    db.execute("DELETE FROM runs")
                    db.execute("INSERT INTO runs VALUES('other-owner',?,1)", (state,))
                    db.commit()
                    self.assertFalse(startup.safe_journal(store), state)
                for state in ("succeeded", "failed", "canceled"):
                    for collected in (0, 1):
                        db.execute("DELETE FROM runs")
                        db.execute("INSERT INTO runs VALUES('other-owner',?,?)", (state, collected))
                        db.commit()
                        self.assertEqual(startup.safe_journal(store), bool(collected))
                db.execute("DROP VIEW machine_runs_v1")
                db.commit()
                with self.assertRaises(sqlite3.OperationalError):
                    startup.safe_journal(store)

    def test_explicit_development_versions(self):
        for version in ("0.18.86.dev1", "0.18.86+local", "0.18.86rc1"):
            self.assertFalse(startup.stable(version))
        self.assertTrue(startup.stable("0.18.86"))

    def test_cpu_does_not_invent_torch_extra_and_never_upgrades_it(self):
        requirements = {"cozy-runtime": ['tensorfs>=0.3.78', 'torch>=2.14; extra == "model-execution"']}
        installed = {"cozy-runtime": "0.18.86", "tensorfs": "0.3.78"}
        self.assertTrue(startup.compatible({}, installed, requirements))
        self.assertFalse(startup.compatible({}, installed | {"torch": "2.13"}, requirements))
        self.assertTrue(startup.compatible({}, installed | {"torch": "2.14"}, requirements))
        requirements["consumer"] = ["cozy-runtime<0.18.86"]
        self.assertFalse(startup.compatible({}, installed, requirements))

    def test_newest_compatible_pair_with_dependency_fallback(self):
        installed = {"cozy-runtime": "0.18.85", "tensorfs": "0.3.78"}
        requirements = {"cozy-runtime": ["tensorfs>=0.3.78"], "tensorfs": []}
        candidates = {"cozy-runtime": [{"version": "0.18.87", "requires": ["tensorfs>=0.4"]}, {"version": "0.18.86", "requires": ["tensorfs>=0.3.78"]}], "tensorfs": [{"version": "0.3.78", "requires": []}]}
        with patch.object(startup, "installed_metadata", return_value=(installed, requirements)), patch.object(startup, "candidates", side_effect=lambda name, _: candidates[name]):
            selected = startup.plan()
            self.assertEqual(selected["cozy-runtime"]["version"], "0.18.86")
            self.assertEqual(selected["tensorfs"]["version"], "0.3.78")


if __name__ == "__main__":
    unittest.main()
