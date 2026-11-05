"""
tests/test_diagnostics.py
─────────────────────────
Tests for the read-only diagnose_system tool.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.tools import diagnose_system
from app.tools.diagnose_system import (
    DIAGNOSE_SYSTEM_TOOL,
    MAX_RESULT_CHARS,
    _build_findings,
    _collect_facts,
    _diagnose_system,
    _protected_pids,
    _short_executable,
    _top_processes,
)
from app.tools.registry import build_default_registry
from app.tools.windows_facts import FileEntry, RecycleBinInfo, ScanResult


def _base_facts():
    return {
        "cpu_percent": 20.0,
        "logical_cores": 8,
        "ram_percent": 40.0,
        "ram_used_gb": 6.4,
        "ram_total_gb": 16.0,
        "swap_percent": 10.0,
        "swap_used_gb": 0.5,
        "top_cpu": [],
        "top_ram": [],
        "drives": [],
        "temp_bytes": 0,
        "temp_files": 0,
        "temp_truncated": False,
        "recycle_count": 0,
        "recycle_bytes": 0,
        "startup_count": 0,
        "uptime_days": 1.0,
        "battery_percent": 80,
        "plugged_in": True,
        "battery_saver": False,
    }


class FakeProcess:
    def __init__(
        self,
        name: str,
        cpu: float,
        rss: int,
        *,
        pid: int = 100,
        exe: str | None = None,
    ):
        self.pid = pid
        self.info = {
            "pid": pid,
            "name": name,
            "memory_info": SimpleNamespace(rss=rss),
            "exe": exe if exe is not None else f"C:\\Apps\\{name}",
        }
        self._cpu = cpu

    def cpu_percent(self, interval=None):
        return self._cpu


def _fake_processes():
    return [
        FakeProcess(
            "chrome.exe",
            30.0,
            500 * 1024 ** 2,
            pid=101,
            exe=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        ),
        FakeProcess(
            "chrome.exe",
            20.0,
            500 * 1024 ** 2,
            pid=102,
            exe=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        ),
        FakeProcess(
            "code.exe",
            10.0,
            1_500 * 1024 ** 2,
            pid=103,
            exe=r"C:\Users\alice\AppData\Local\Programs\Microsoft VS Code\Code.exe",
        ),
        FakeProcess(
            "spotify.exe",
            5.0,
            300 * 1024 ** 2,
            pid=104,
            exe=r"C:\Users\alice\AppData\Roaming\Spotify\Spotify.exe",
        ),
        FakeProcess("discord.exe", 8.0, 400 * 1024 ** 2, pid=105),
        FakeProcess("notepad.exe", 1.0, 50 * 1024 ** 2, pid=106),
        FakeProcess("explorer.exe", 2.0, 200 * 1024 ** 2, pid=107),
    ]


def _patch_process_sampling(
    monkeypatch,
    processes,
    *,
    cores: int = 8,
    protected=None,
):
    monkeypatch.setattr(diagnose_system.psutil, "cpu_count", lambda logical=True: cores)
    monkeypatch.setattr(
        diagnose_system.psutil, "process_iter", lambda *args, **kwargs: processes
    )
    monkeypatch.setattr(diagnose_system.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        diagnose_system,
        "_protected_pids",
        lambda root_pid=None: set(protected or set()),
    )


def _patch_collectors(monkeypatch, tmp_path, *, power=None):
    monkeypatch.setattr(diagnose_system.psutil, "cpu_percent", lambda interval=None: 22.0)
    monkeypatch.setattr(
        diagnose_system.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=45.0, used=7 * 1024 ** 3, total=16 * 1024 ** 3),
    )
    monkeypatch.setattr(
        diagnose_system.psutil,
        "swap_memory",
        lambda: SimpleNamespace(percent=8.0, used=256 * 1024 ** 2),
    )
    monkeypatch.setattr(diagnose_system.psutil, "boot_time", lambda: time.time() - 86400)
    _patch_process_sampling(monkeypatch, _fake_processes())
    monkeypatch.setattr(diagnose_system.psutil, "sensors_battery", lambda: None)
    monkeypatch.setattr(diagnose_system, "fixed_drives", lambda: [])
    monkeypatch.setattr(
        diagnose_system,
        "query_recycle_bin",
        lambda: RecycleBinInfo(count=0, bytes=0),
    )
    monkeypatch.setattr(diagnose_system, "count_startup_programs", lambda: 3)
    monkeypatch.setattr(
        diagnose_system,
        "get_power_status",
        lambda: power
        or {"plugged_in": True, "battery_percent": 80, "battery_saver": False},
    )
    monkeypatch.setattr(diagnose_system, "resolve_temp_root", lambda: tmp_path)
    monkeypatch.setattr(
        diagnose_system,
        "scan_files",
        lambda *args, **kwargs: ScanResult(
            files=[FileEntry(tmp_path / "old.tmp", 1024, time.time() - 90000)],
            skipped=0,
            truncated=False,
        ),
    )


def test_descriptor_and_registry():
    assert DIAGNOSE_SYSTEM_TOOL.name == "diagnose_system"
    assert DIAGNOSE_SYSTEM_TOOL.parameters["properties"] == {}
    assert "slow" in DIAGNOSE_SYSTEM_TOOL.description.lower()
    assert "diagnose_system" in build_default_registry().list_tools()


def test_collects_required_facts(tmp_path, monkeypatch):
    _patch_collectors(monkeypatch, tmp_path)

    facts = _collect_facts()

    assert facts["cpu_percent"] == 22.0
    assert facts["logical_cores"] == 8
    assert facts["ram_percent"] == 45.0
    assert facts["swap_percent"] == 8.0
    assert facts["temp_files"] == 1
    assert facts["temp_bytes"] == 1024
    assert facts["startup_count"] == 3
    assert facts["battery_percent"] == 80
    assert facts["top_cpu"]
    assert facts["top_ram"]


def test_top_processes_are_grouped_by_name_and_capped(monkeypatch):
    _patch_process_sampling(monkeypatch, _fake_processes())

    top_cpu, top_ram = _top_processes()

    assert len(top_cpu) == 5
    assert len(top_ram) == 5
    chrome = next(item for item in top_cpu if item["name"] == "Google Chrome")
    assert chrome["process_count"] == 2
    assert chrome["cpu_percent"] == pytest.approx(6.25, abs=0.1)
    assert chrome["pid"] == 101
    assert chrome["exe"] == r"Chrome\Application\chrome.exe"
    assert chrome["protected"] is False
    assert top_ram[0]["name"] == "Visual Studio Code"
    assert top_ram[0]["exe"] == r"Programs\Microsoft VS Code\Code.exe"


def test_process_cpu_is_normalized_by_logical_cores(monkeypatch):
    processes = [
        FakeProcess("busy.exe", 80.0, 100 * 1024 ** 2, pid=201),
        FakeProcess("light.exe", 16.0, 100 * 1024 ** 2, pid=202),
    ]
    _patch_process_sampling(monkeypatch, processes, cores=8)

    top_cpu, _ = _top_processes()

    busy = next(item for item in top_cpu if item["name"] == "busy.exe")
    light = next(item for item in top_cpu if item["name"] == "light.exe")
    assert busy["cpu_percent"] == 10.0
    assert light["cpu_percent"] == 2.0


def test_grouped_process_uses_heaviest_pid_for_each_metric(monkeypatch):
    processes = [
        FakeProcess("gamma.exe", 30.0, 100 * 1024 ** 2, pid=301),
        FakeProcess("gamma.exe", 10.0, 900 * 1024 ** 2, pid=302),
    ]
    _patch_process_sampling(monkeypatch, processes, cores=8)

    top_cpu, top_ram = _top_processes()

    assert top_cpu[0]["name"] == "gamma.exe"
    assert top_cpu[0]["pid"] == 301
    assert top_ram[0]["name"] == "gamma.exe"
    assert top_ram[0]["pid"] == 302


def test_short_executable_hides_user_profile():
    short = _short_executable(
        r"C:\Users\alice\AppData\Local\Programs\Python\Python311\python.exe"
    )

    assert short == r"Python\Python311\python.exe"
    assert "alice" not in short.lower()


class FakeChainProcess:
    def __init__(self, pid, name, *, parent=None, children=None):
        self.pid = pid
        self._name = name
        self._parent = parent
        self._children = list(children or [])

    def name(self):
        return self._name

    def parent(self):
        return self._parent

    def children(self, recursive=False):
        return list(self._children)


def test_protected_pids_stops_at_non_python_parent(monkeypatch):
    terminal = FakeChainProcess(300, "terminal.exe")
    powershell = FakeChainProcess(400, "powershell.exe", parent=terminal)
    child = FakeChainProcess(501, "python.exe")
    current = FakeChainProcess(500, "python.exe", parent=powershell, children=[child])
    mapping = {300: terminal, 400: powershell, 500: current, 501: child}
    monkeypatch.setattr(diagnose_system.psutil, "Process", lambda pid: mapping[pid])

    assert _protected_pids(500) == {500, 501}


def test_protected_pids_allows_one_python_parent(monkeypatch):
    powershell = FakeChainProcess(600, "powershell.exe")
    python_parent = FakeChainProcess(601, "python.exe", parent=powershell)
    current = FakeChainProcess(602, "python.exe", parent=python_parent)
    mapping = {600: powershell, 601: python_parent, 602: current}
    monkeypatch.setattr(diagnose_system.psutil, "Process", lambda pid: mapping[pid])

    assert _protected_pids(602) == {602, 601}


def test_bob_pid_is_marked_protected(monkeypatch):
    processes = [
        FakeProcess(
            "python.exe",
            40.0,
            500 * 1024 ** 2,
            pid=100,
            exe=r"C:\Users\alice\venv\Scripts\python.exe",
        )
    ]
    _patch_process_sampling(monkeypatch, processes, protected={100})

    top_cpu, _ = _top_processes()

    assert top_cpu[0]["name"] == "BOB / Ollama (this assistant)"
    assert top_cpu[0]["protected"] is True
    assert top_cpu[0]["pid"] == 100
    assert top_cpu[0]["exe"] == r"venv\Scripts\python.exe"


def test_protected_python_does_not_merge_with_normal_python(monkeypatch):
    processes = [
        FakeProcess(
            "python.exe",
            30.0,
            500 * 1024 ** 2,
            pid=100,
            exe=r"C:\Users\alice\venv\Scripts\python.exe",
        ),
        FakeProcess(
            "python.exe",
            10.0,
            300 * 1024 ** 2,
            pid=200,
            exe=r"C:\Users\alice\AppData\Local\Programs\Python\Python311\python.exe",
        ),
    ]
    _patch_process_sampling(monkeypatch, processes, protected={100})

    top_cpu, _ = _top_processes()
    bob = next(item for item in top_cpu if item["protected"])
    normal = next(item for item in top_cpu if not item["protected"])

    assert bob["name"] == "BOB / Ollama (this assistant)"
    assert bob["process_count"] == 1
    assert bob["pid"] == 100
    assert normal["name"] == "python.exe"
    assert normal["process_count"] == 1
    assert normal["pid"] == 200


def test_ollama_is_protected_by_name(monkeypatch):
    processes = [
        FakeProcess(
            "ollama.exe",
            20.0,
            500 * 1024 ** 2,
            pid=700,
            exe=r"C:\Program Files\Ollama\ollama.exe",
        )
    ]
    _patch_process_sampling(monkeypatch, processes)

    top_cpu, top_ram = _top_processes()

    assert top_cpu[0]["name"] == "BOB / Ollama (this assistant)"
    assert top_cpu[0]["protected"] is True
    assert top_cpu[0]["pid"] == 700
    assert top_ram[0]["name"] == "BOB / Ollama (this assistant)"


def test_findings_are_sorted_by_severity():
    facts = _base_facts()
    facts.update(
        {
            "ram_percent": 96.0,
            "cpu_percent": 90.0,
            "uptime_days": 8.0,
            "drives": [
                {"mountpoint": "C:\\", "percent": 95.0, "free": 1024 ** 3},
            ],
        }
    )

    findings = _build_findings(facts)
    ranks = [{"critical": 0, "warning": 1, "info": 2}[item["severity"]] for item in findings]

    assert ranks == sorted(ranks)
    assert findings[0]["severity"] == "critical"


def test_threshold_boundaries():
    facts = _base_facts()
    facts["ram_percent"] = 85.0
    assert any("RAM usage is 85%" in item["message"] for item in _build_findings(facts))

    facts["ram_percent"] = 95.0
    assert any("RAM usage is 95%" in item["message"] for item in _build_findings(facts))

    facts = _base_facts()
    facts["drives"] = [{"mountpoint": "D:\\", "percent": 91.0, "free": 50 * 1024 ** 3}]
    assert any(item["severity"] == "critical" for item in _build_findings(facts))

    facts = _base_facts()
    facts["uptime_days"] = 7.5
    assert any("PC uptime is 7.5 days" in item["message"] for item in _build_findings(facts))


def test_battery_and_power_findings():
    facts = _base_facts()
    facts["battery_percent"] = 10
    facts["plugged_in"] = False
    facts["battery_saver"] = True

    findings = _build_findings(facts)
    messages = " ".join(item["message"] for item in findings)

    assert "Battery level is 10%" in messages
    assert "Battery saver is on" in messages


def test_findings_label_every_number():
    facts = _base_facts()
    facts.update(
        {
            "ram_percent": 96.0,
            "cpu_percent": 96.0,
            "swap_percent": 96.0,
            "drives": [
                {"mountpoint": "C:\\", "percent": 92.0, "free": 12 * 1024 ** 3}
            ],
            "temp_bytes": 6 * 1024 ** 3,
            "recycle_bytes": 6 * 1024 ** 3,
            "startup_count": 30,
            "uptime_days": 10.0,
            "battery_percent": 10,
            "plugged_in": False,
            "top_cpu": [
                {
                    "name": "Google Chrome",
                    "process_count": 2,
                    "pid": 101,
                    "exe": r"Chrome\Application\chrome.exe",
                    "cpu_percent": 90.0,
                    "ram_mb": 100.0,
                    "protected": False,
                }
            ],
            "top_ram": [
                {
                    "name": "Visual Studio Code",
                    "process_count": 1,
                    "pid": 103,
                    "exe": r"Programs\Microsoft VS Code\Code.exe",
                    "cpu_percent": 1.0,
                    "ram_mb": 4096.0,
                    "protected": False,
                }
            ],
        }
    )

    messages = " ".join(item["message"] for item in _build_findings(facts))

    assert "RAM usage is 96%" in messages
    assert "CPU usage is 96%" in messages
    assert "Swap usage is 96%" in messages
    assert r"Drive C:\ free space is 8% (12.0 GB free)" in messages
    assert "Temp folder size is 6.0 GB" in messages
    assert "Recycle Bin size is 6.0 GB" in messages
    assert "Startup programs: 30 start with Windows" in messages
    assert "PC uptime is 10.0 days" in messages
    assert "Battery level is 10%" in messages
    assert (
        r"Google Chrome (2 processes) [PID 101, Chrome\Application\chrome.exe] "
        "is using 90.0% CPU (system-wide)"
    ) in messages
    assert (
        r"Visual Studio Code [PID 103, Programs\Microsoft VS Code\Code.exe] "
        "is using 4.0 GB RAM"
    ) in messages


def test_protected_findings_never_suggest_closing():
    facts = _base_facts()
    bob = {
        "name": "BOB / Ollama (this assistant)",
        "process_count": 3,
        "pid": 500,
        "exe": r"venv\Scripts\python.exe",
        "cpu_percent": 90.0,
        "ram_mb": 4096.0,
        "protected": True,
    }
    facts.update({"cpu_percent": 96.0, "ram_percent": 96.0})
    facts["top_cpu"] = [dict(bob)]
    facts["top_ram"] = [dict(bob)]

    findings = _build_findings(facts)
    bob_findings = [item for item in findings if "BOB / Ollama" in item["message"]]
    cpu_finding = next(
        item for item in findings if item["message"].startswith("CPU usage is")
    )

    assert bob_findings
    assert "do not close" in cpu_finding["fix"].lower()
    for item in bob_findings:
        text = item["fix"].lower()
        assert "do not close" in text
        assert "close or restart" not in text


def test_normal_process_finding_still_suggests_closing():
    facts = _base_facts()
    facts["top_cpu"] = [
        {
            "name": "Google Chrome",
            "process_count": 1,
            "pid": 101,
            "exe": r"Chrome\Application\chrome.exe",
            "cpu_percent": 90.0,
            "ram_mb": 100.0,
            "protected": False,
        }
    ]

    findings = _build_findings(facts)
    process_finding = next(
        item for item in findings if "Google Chrome" in item["message"]
    )

    assert "close or restart" in process_finding["fix"].lower()


def test_metrics_include_logical_cores(tmp_path, monkeypatch):
    _patch_collectors(monkeypatch, tmp_path)

    result = _diagnose_system()

    assert result.success is True
    assert result.data["metrics"]["logical_cores"] == 8


def test_output_length_is_capped(monkeypatch):
    facts = _base_facts()
    facts["ram_percent"] = 99.0
    facts["cpu_percent"] = 99.0
    facts["swap_percent"] = 99.0
    facts["temp_bytes"] = 9 * 1024 ** 3
    facts["recycle_bytes"] = 9 * 1024 ** 3
    facts["startup_count"] = 100
    facts["uptime_days"] = 30.0
    facts["top_cpu"] = [
        {
            "name": f"process-{index}",
            "process_count": 1,
            "pid": 1000 + index,
            "exe": f"Apps\\process-{index}.exe",
            "cpu_percent": 99.0,
            "ram_mb": 100.0,
            "protected": False,
        }
        for index in range(8)
    ]
    facts["top_ram"] = [
        {
            "name": f"process-{index}",
            "process_count": 1,
            "pid": 2000 + index,
            "exe": f"Apps\\process-{index}.exe",
            "cpu_percent": 1.0,
            "ram_mb": 9000.0,
            "protected": False,
        }
        for index in range(8)
    ]
    facts["drives"] = [
        {"mountpoint": f"{letter}:\\", "percent": 99.0, "free": 1024 ** 3}
        for letter in "CDEFGH"
    ]
    monkeypatch.setattr(diagnose_system, "_collect_facts", lambda: facts)

    result = _diagnose_system()
    encoded = json.dumps(result.to_dict(), default=str)

    assert result.success is True
    assert len(encoded) <= MAX_RESULT_CHARS
    assert any(item["severity"] == "critical" for item in result.data["findings"])


def test_result_findings_have_message_and_fix(tmp_path, monkeypatch):
    _patch_collectors(monkeypatch, tmp_path)

    result = _diagnose_system()

    assert result.success is True
    for finding in result.data["findings"]:
        assert finding["severity"] in {"critical", "warning", "info"}
        assert finding["message"]
        assert finding["fix"]


def test_diagnostics_has_no_side_effects(tmp_path, monkeypatch):
    _patch_collectors(monkeypatch, tmp_path)

    def _forbidden(*args, **kwargs):
        raise AssertionError("diagnostics attempted a side effect")

    monkeypatch.setattr(os, "remove", _forbidden)
    monkeypatch.setattr(os, "rmdir", _forbidden)

    result = _diagnose_system()

    assert result.success is True


def test_diagnostics_source_has_no_shell_or_eval():
    source = Path(__file__).resolve().parents[1] / "app" / "tools" / "diagnose_system.py"
    text = source.read_text(encoding="utf-8")
    assert "subprocess" not in text
    assert "shell=True" not in text
    assert "eval(" not in text
