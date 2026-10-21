"""The full run starts only after the follow-up pilot is auditable."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/start_full_after_pilot.py"
SPEC = importlib.util.spec_from_file_location("start_full_after_pilot", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PilotGateTests(unittest.TestCase):
    def test_complete_auditable_pilot_passes(self):
        status = {"stage": "complete", "rows": 10, "traces": 10, "judged": 10}
        audit = {"summary": {"errors": 0, "missing_full_tool_outputs": 0}}
        self.assertTrue(module.pilot_is_ready(status, audit)[0])

    def test_missing_tool_text_or_missing_row_blocks_full_run(self):
        status = {"stage": "complete", "rows": 10, "traces": 10, "judged": 10}
        audit = {"summary": {"errors": 0, "missing_full_tool_outputs": 1}}
        self.assertFalse(module.pilot_is_ready(status, audit)[0])
        status["rows"] = 9
        audit["summary"]["missing_full_tool_outputs"] = 0
        self.assertFalse(module.pilot_is_ready(status, audit)[0])


if __name__ == "__main__":
    unittest.main()
