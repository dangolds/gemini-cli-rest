"""Per-run baseline file (TestPRD FR-8): what the container did on a given day.

One JSON file per LIVE run under logs/baseline/ (host side; override with
BRIDGE_BASELINE_DIR): date/time, run stamp, container image id, per story its
nodeid, agent/port, outcome, duration, group, declared units, whether its call
phase ran (`ran`) and the units it actually spent (`units_spent`: the declared
units, counted only when the call phase ran - pass, fail or skip inside the
call - and zero when setup failed or skipped), and totals. A run in which no
live story actually ran (hermetic-only, or every live story skipped because no
bridge was up) writes nothing - there is no baseline to record.
stories/conftest.py wires the pytest hooks to `BaselineRecorder`.

The REFERENCE (baseline/reference.json, tracked in git; BRIDGE_BASELINE_REFERENCE
to move it) is the run every later run is compared to by story nodeid
(`compare`: changed outcomes, missing and new stories). BRIDGE_BASELINE_APPROVE=1
makes a fully green run the new reference (`approve`); a run with any failure,
an unresolved teardown or an unclean pytest exit is refused.

Conventions the recorder reads from a story:
  @pytest.mark.units(n)     live spawn-and-turn units the story spends (0 if absent)
  @pytest.mark.group("B")   TestPRD group; module-level GROUP = "B" is the fallback
  @pytest.mark.hermetic     the story is hermetic (its units never count as live)
  bridge / fake_bridge      the parametrized fixture id gives the agent
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from bridgetests import live, names

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "logs" / "baseline"
REFERENCE = REPO_ROOT / "baseline" / "reference.json"
APPROVE_FLAG = "BRIDGE_BASELINE_APPROVE"


def baseline_dir() -> Path:
    return Path(os.environ.get("BRIDGE_BASELINE_DIR") or DEFAULT_DIR)


def reference_path() -> Path:
    return Path(os.environ.get("BRIDGE_BASELINE_REFERENCE") or REFERENCE)


def approve_requested() -> bool:
    return os.environ.get(APPROVE_FLAG, "").strip() == "1"


def compare(run: dict[str, Any], reference: dict[str, Any]) -> dict[str, list]:
    """Run vs reference by story nodeid. `changed`: (nodeid, reference outcome,
    run outcome); `missing`: in the reference but not in the run; `new`: in the
    run but not in the reference. A story skipped in the run says nothing (it
    is neither changed nor, when it exists in the reference, missing)."""
    ref = {s["nodeid"]: s.get("outcome") for s in reference.get("stories", [])}
    now = {s["nodeid"]: s.get("outcome") for s in run.get("stories", [])}
    changed = [(n, ref[n], now[n]) for n in sorted(ref)
               if n in now and now[n] != "skipped" and now[n] != ref[n]]
    missing = [n for n in sorted(ref) if n not in now]
    new = [n for n in sorted(now) if n not in ref and now[n] != "skipped"]
    return {"changed": changed, "missing": missing, "new": new}


def approve(run_path: Path, *, exit_ok: bool = True) -> Path | None:
    """Copy the run file to the reference path, only when every story that ran
    passed (skipped stories are fine), no teardown failed or stayed unresolved
    and pytest finished cleanly (*exit_ok*); None, with the reason printed,
    otherwise."""
    if not exit_ok:
        print("[baseline] not approved: pytest did not finish cleanly", flush=True)
        return None
    data = json.loads(Path(run_path).read_text(encoding="utf-8"))
    bad = [(s["nodeid"], s.get("outcome")) for s in data.get("stories", [])
           if s.get("outcome") not in ("passed", "skipped")]
    if bad:
        print(f"[baseline] not approved: {len(bad)} story(ies) did not pass, e.g. {bad[0]}",
              flush=True)
        return None
    torn = [s["nodeid"] for s in data.get("stories", []) if s.get("teardown_error")]
    if torn:
        print(f"[baseline] not approved: {len(torn)} story(ies) failed at teardown, e.g. {torn[0]}",
              flush=True)
        return None
    unresolved = data.get("unresolved_teardowns") or []
    if unresolved:
        print(f"[baseline] not approved: {len(unresolved)} teardown(s) unresolved, e.g. {unresolved[0]}",
              flush=True)
        return None
    if not any(s.get("ran") and not s.get("hermetic") for s in data.get("stories", [])):
        print("[baseline] not approved: no live story ran", flush=True)
        return None
    dest = reference_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_path, dest)
    return dest


def item_agent(item) -> str | None:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    for fixture in ("bridge", "fake_bridge"):
        if fixture in callspec.params:
            return str(callspec.params[fixture])
    return None


def item_group(item) -> str:
    m = item.get_closest_marker("group")
    if m and m.args:
        return str(m.args[0])
    return str(getattr(item.module, "GROUP", "?"))


def item_units(item) -> int:
    m = item.get_closest_marker("units")
    if m and m.args:
        try:
            return int(m.args[0])
        except (TypeError, ValueError):
            return 0
    return 0


def is_hermetic(item) -> bool:
    return item.get_closest_marker("hermetic") is not None


class BaselineRecorder:
    def __init__(self, ports: dict[str, int], container: str, image_id_fn=None):
        self.ports = ports
        self.container = container
        self._image_id_fn = image_id_fn
        self.started_at = datetime.now()
        self._t0 = time.monotonic()
        self.records: dict[str, dict[str, Any]] = {}

    # --- collection ---------------------------------------------------------------

    def record(self, item, report) -> None:
        """Fold one phase report (setup/call/teardown) into the story's record."""
        rec = self.records.setdefault(report.nodeid, {
            "nodeid": report.nodeid,
            "agent": item_agent(item),
            "port": self.ports.get(item_agent(item) or "", None),
            "group": item_group(item),
            "units": item_units(item),
            "ran": False,          # the call phase executed
            "units_spent": 0,      # declared units, once the call phase ran
            "hermetic": is_hermetic(item),
            "outcome": None,
            "duration": 0.0,
            "phases": {},
        })
        rec["duration"] += float(getattr(report, "duration", 0.0) or 0.0)
        rec["phases"][report.when] = report.outcome
        if report.when == "setup":
            if report.outcome == "skipped":
                rec["outcome"] = "skipped"
            elif report.outcome == "failed":
                rec["outcome"] = "error"
        elif report.when == "call":
            rec["outcome"] = report.outcome  # passed | failed | skipped
            rec["ran"] = True
            rec["units_spent"] = rec["units"]
        elif report.when == "teardown" and report.outcome == "failed":
            rec["teardown_error"] = True
            if rec["outcome"] in (None, "passed", "skipped"):
                rec["outcome"] = "error"

    # --- output -------------------------------------------------------------------

    def live_ran(self) -> bool:
        return any(not r["hermetic"] and r["ran"] for r in self.records.values())

    def build(self) -> dict[str, Any]:
        stories = sorted(self.records.values(), key=lambda r: r["nodeid"])
        per_group: Counter = Counter()
        units_per_agent: dict[str, int] = defaultdict(int)
        outcomes: Counter = Counter()
        ran = 0
        for r in stories:
            per_group[r["group"]] += 1
            outcomes[r["outcome"] or "unknown"] += 1
            if not r["hermetic"] and r["ran"]:
                ran += 1
                units_per_agent[r["agent"] or "both"] += r["units_spent"]
        return {
            "date": self.started_at.strftime("%Y-%m-%d"),
            "time": self.started_at.strftime("%H:%M:%S"),
            "run_stamp": names.STAMP,
            "repo_prefix": names.REPO_PREFIX,
            "container": self.container,
            "image_id": self._image_id_fn() if self._image_id_fn else "unknown",
            "wall_time_s": round(time.monotonic() - self._t0, 1),
            "totals": {
                "stories": len(stories),
                "stories_per_group": dict(sorted(per_group.items())),
                "live_stories_ran": ran,
                "live_units_per_agent": dict(sorted(units_per_agent.items())),
                "outcomes": dict(sorted(outcomes.items())),
            },
            # read at write time: every fixture teardown has run by then
            "unresolved_teardowns": [list(t) for t in live.UNRESOLVED],
            "stories": [
                {k: v for k, v in r.items() if k != "phases"} | {"duration": round(r["duration"], 2)}
                for r in stories
            ],
        }

    def write(self) -> Path | None:
        if not self.live_ran():
            return None
        out = baseline_dir()
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.started_at.strftime('%Y%m%d-%H%M%S')}-{names.STAMP}.json"
        path.write_text(json.dumps(self.build(), indent=2) + "\n", encoding="utf-8")
        return path
