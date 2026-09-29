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
    def test_only_runtime_owned_software_state_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            self.assertTrue(startup.software_state(store)["quiescent"])
            path = store / ".cozy-workspace/journal.sqlite3"
            path.parent.mkdir()
            with sqlite3.connect(path) as db:
                db.executescript("CREATE TABLE state(quiescent INT, accepted_work INT, workspace_id TEXT); CREATE VIEW machine_software_update_v1 AS SELECT * FROM state;")
                db.execute("INSERT INTO state VALUES(1,1,'original')")
                db.commit()
                self.assertEqual(startup.software_state(store), {"quiescent": True, "accepted_work": True, "workspace_id": "original"})
                db.execute("DROP VIEW machine_software_update_v1")
                db.commit()
                with self.assertRaises(sqlite3.OperationalError):
                    startup.software_state(store)

    def test_zero_major_stays_within_minor_line(self):
        with patch.object(startup, "read_json", return_value={"releases": {"0.4.0": [], "1.0.0": []}}):
            self.assertEqual(list(startup.candidates("tensorfs", "0.3.78")), [])

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
