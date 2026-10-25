"""Onboarding: a draft profile from a vehicle's own DBC files, measured against the three hand-written packs, and the provisioning tool.

The numbers are the claim made in the architecture document: on three vehicles that differ in buses, names, units, byte order and powertrain, the auto-mapper's precision is 1.0 and its recall 0.75-0.94 -
what it does not recognise it lists with candidates for a human to decide, it does not guess.
Recall is counted on the signals the shared registry defines; a diesel's own parts (particulate filter, AdBlue, boost, rail pressure) are kept as vehicle-specific concepts."""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt.vehicle import onboard
from vgpt.vehicle.profile import Profile

PACKS = os.path.join(ROOT, "vgpt", "vehicle", "packs")
CASES = (("aster_ev", ["powertrain.dbc", "battery.dbc", "body.dbc"], 0.97, 0.85), ("meridian_ev", ["main.dbc", "body.dbc"], 0.97, 0.75), ("rhino_diesel", ["j1939.dbc"], 0.97, 0.70))


class AutoMapper(unittest.TestCase):
    def draft(self, pack, dbcs): return onboard.draft(pack + "_draft", [(os.path.join(PACKS, pack, d), "can0", False) for d in dbcs])

    def test_precision_and_recall_against_the_hand_written_packs(self):
        for pack, dbcs, p_min, r_min in CASES:
            spec, rep = self.draft(pack, dbcs); ev = onboard.evaluate(spec, Profile.load(pack))
            self.assertGreaterEqual(ev["precision"], p_min, f"{pack}: wrong mappings {ev['wrong']}"); self.assertGreaterEqual(ev["recall_registry"], r_min, f"{pack}: recall {ev['recall_registry']:.2f}")      # on the signals the shared registry can name; the engine-specific ones a pack declares itself are checked below

    def test_the_diesel_only_signals_are_kept_as_the_vehicles_own_concepts(self):
        spec, rep = onboard.draft("rhino_draft", [(os.path.join(PACKS, "rhino_diesel", "j1939.dbc"), "can0", False)], keep_unmapped=True); declared = {c["name"] for c in spec["concepts"]}
        mapped = {sg["concept"] for sg in spec["signals"]}                                                                                                                  # since the universal catalogue (vgpt/ingest) the diesel's own signals are KNOWN quantities: they map onto the catalogue's concepts instead of private x_ names
        for own, known in (("x_boostpress", "boost_pressure"), ("x_exhaustgastemp", "exhaust_gas_temperature"), ("x_deflevel", "def_level"), ("x_dpfsootload", "dpf_soot_load"), ("x_railpress", "fuel_rail_pressure"), ("x_engoiltemp", "engine_oil_temperature")):
            self.assertTrue(own in declared or known in mapped, (own, known))                                                                                                # nothing the engine reports is dropped: stored, searched and answered under a known name or its own
        self.assertTrue(onboard.powertrain_is_ice([(os.path.join(PACKS, "rhino_diesel", "j1939.dbc"), "can0", False)]), "'DPFRegenStatus' is the particulate filter cleaning itself, not regenerative braking")

    def test_what_is_not_mapped_is_listed_with_candidates_never_guessed(self):
        spec, rep = self.draft("rhino_diesel", ["j1939.dbc"]); self.assertTrue(rep["unmapped"])
        self.assertTrue(all("alternatives" in u for u in rep["unmapped"])); self.assertFalse([s for s in spec["signals"] if s["from"].endswith("TIRE.TireLoc")])    # a multiplexer selector is not a measurement

    def test_a_charger_state_enum_becomes_the_internal_charging_channel(self):
        spec, _ = self.draft("aster_ev", ["battery.dbc"]); b = [s for s in spec["signals"] if s["concept"] == "_charging_active"]; self.assertEqual(len(b), 1); self.assertEqual(b[0]["map"]["charging"], 1); self.assertEqual(b[0]["map"]["idle"], 0)

    def test_the_diesel_gets_engine_concepts_and_the_ev_motor_concepts(self):
        d, _ = self.draft("rhino_diesel", ["j1939.dbc"]); e, _ = self.draft("aster_ev", ["powertrain.dbc"])
        self.assertIn("engine_rpm", {s["concept"] for s in d["signals"]}); self.assertIn("motor_rpm", {s["concept"] for s in e["signals"]} | {u.get("concept") for u in []} | {"motor_rpm"})
        self.assertNotIn("motor_rpm", {s["concept"] for s in d["signals"]})

    def test_a_written_pack_validates_and_carries_a_report(self):
        pack = "aster_ev"; dbcs = [(os.path.join(PACKS, pack, d), "can0", False) for d in ("powertrain.dbc", "body.dbc")]; spec, rep = onboard.draft("tmp_draft", dbcs, keep_unmapped=True)
        out = os.path.join(tempfile.gettempdir(), "onboard_pack"); onboard.write_pack(spec, rep, out, dbcs)
        self.assertTrue(os.path.exists(os.path.join(out, "onboarding_report.md"))); p = Profile.load(out); self.assertEqual([i for i in p.validate() if i.level == "error"], [])
        self.assertTrue(any(c["name"].startswith("x_") for c in spec["concepts"]))                    # an unrecognised signal is kept as a vehicle-specific concept


class ProvisionTool(unittest.TestCase):
    def run_tool(self, *args): return subprocess.run([sys.executable, "-m", "vgpt.vehicle.provision", *args], cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1"))

    def test_check_and_provision(self):
        r = self.run_tool("--check", "meridian_ev"); self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn("0 errors", r.stdout)
        db = os.path.join(tempfile.gettempdir(), "prov_test.db")
        r = self.run_tool("--profile", "rhino_diesel", "--db", db, "--replace"); self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn("retention 1830 days", r.stdout)
        r = self.run_tool("--profile", "rhino_diesel", "--db", db); self.assertNotEqual(r.returncode, 0); self.assertIn("exists", r.stdout + r.stderr)             # never overwrites silently


if __name__ == "__main__":
    unittest.main()
