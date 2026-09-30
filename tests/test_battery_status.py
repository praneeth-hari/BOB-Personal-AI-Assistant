"""
tests/test_battery_status.py
────────────────────────────
Unit and integration tests for the get_battery_status tool.

Covers:
  • Normal battery information (discharging, time remaining)
  • Charging state (plugged in, under 100%)
  • Fully charged state (plugged in, 100%)
  • Unavailable battery information (desktop PC, sensors_battery returning None)
  • Unknown / calculating remaining time
  • Malformed / unexpected psutil returns (invalid strings, None attributes)
  • Exception handling when psutil raises an error
  • Integration with ToolRegistry and live hardware sanity check
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.tools.get_battery_status import (
    BATTERY_STATUS_TOOL,
    _format_time_remaining,
    _get_battery_status,
)
from app.tools.registry import ToolRegistry, build_default_registry
from app.tools.tool import ToolResult

# Namedtuple mimicking psutil._common.sbattery
SBattery = namedtuple("sbattery", ["percent", "secsleft", "power_plugged"])


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: _format_time_remaining
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatTimeRemaining:
    def test_unlimited_constant(self):
        secs, text = _format_time_remaining(-2)
        assert secs is None
        assert text == "Unlimited (Plugged In)"

    def test_unknown_constant(self):
        secs, text = _format_time_remaining(-1)
        assert secs is None
        assert text == "Calculating / Unknown"

    def test_negative_unknown(self):
        secs, text = _format_time_remaining(-99)
        assert secs is None
        assert text == "Calculating / Unknown"

    def test_none_value(self):
        secs, text = _format_time_remaining(None)
        assert secs is None
        assert text == "Unknown"

    def test_invalid_string(self):
        secs, text = _format_time_remaining("invalid")  # type: ignore[arg-type]
        assert secs is None
        assert text == "Unknown"

    def test_hours_and_minutes_plural(self):
        secs, text = _format_time_remaining(7200 + 180)  # 2 hours 3 minutes
        assert secs == 7380
        assert text == "2 hours 3 minutes"

    def test_hours_and_minutes_singular(self):
        secs, text = _format_time_remaining(3600 + 60)  # 1 hour 1 minute
        assert secs == 3660
        assert text == "1 hour 1 minute"

    def test_minutes_only(self):
        secs, text = _format_time_remaining(1800)  # 30 minutes
        assert secs == 1800
        assert text == "30 minutes"

    def test_minutes_only_singular(self):
        secs, text = _format_time_remaining(60)  # 1 minute
        assert secs == 60
        assert text == "1 minute"


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: _get_battery_status
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetBatteryStatus:
    @patch("psutil.sensors_battery")
    def test_normal_discharging_battery(self, mock_battery: MagicMock):
        mock_battery.return_value = SBattery(percent=75.5, secsleft=9000, power_plugged=False)
        result = _get_battery_status()

        assert result.success is True
        assert result.error is None
        data = result.data
        assert data["available"] is True
        assert data["percent"] == 75.5
        assert data["power_plugged"] is False
        assert data["is_plugged_in"] is False
        assert data["charging_status"] == "discharging"
        assert "75.5%" in data["status"]
        assert "Discharging" in data["status"]
        assert data["time_remaining_seconds"] == 9000
        assert data["time_remaining_formatted"] == "2 hours 30 minutes"

    @patch("psutil.sensors_battery")
    def test_charging_state_plugged_in(self, mock_battery: MagicMock):
        mock_battery.return_value = SBattery(percent=50.0, secsleft=-2, power_plugged=True)
        result = _get_battery_status()

        assert result.success is True
        data = result.data
        assert data["available"] is True
        assert data["percent"] == 50.0
        assert data["power_plugged"] is True
        assert data["is_plugged_in"] is True
        assert data["charging_status"] == "charging"
        assert "Charging (Plugged In)" in data["status"]
        assert data["time_remaining_seconds"] is None
        assert data["time_remaining_formatted"] == "Unlimited (Plugged In)"

    @patch("psutil.sensors_battery")
    def test_fully_charged_state(self, mock_battery: MagicMock):
        mock_battery.return_value = SBattery(percent=100.0, secsleft=-2, power_plugged=True)
        result = _get_battery_status()

        assert result.success is True
        data = result.data
        assert data["available"] is True
        assert data["percent"] == 100.0
        assert data["power_plugged"] is True
        assert data["charging_status"] == "fully_charged"
        assert "Fully Charged" in data["status"]

    @patch("psutil.sensors_battery")
    def test_unavailable_battery_desktop(self, mock_battery: MagicMock):
        mock_battery.return_value = None
        result = _get_battery_status()

        assert result.success is True
        data = result.data
        assert data["available"] is False
        assert data["percent"] is None
        assert data["power_plugged"] is None
        assert data["is_plugged_in"] is None
        assert data["charging_status"] == "unavailable"
        assert "No battery detected" in data["status"]
        assert "No battery detected" in data["message"]
        assert data["time_remaining_seconds"] is None

    @patch("psutil.sensors_battery")
    def test_unknown_power_state(self, mock_battery: MagicMock):
        # When power_plugged is None
        mock_battery.return_value = SBattery(percent=80.0, secsleft=-1, power_plugged=None)
        result = _get_battery_status()

        assert result.success is True
        data = result.data
        assert data["available"] is True
        assert data["percent"] == 80.0
        assert data["power_plugged"] is None
        assert data["charging_status"] == "unknown"
        assert "Unknown Power State" in data["status"]
        assert data["time_remaining_formatted"] == "Calculating / Unknown"

    @patch("psutil.sensors_battery")
    def test_malformed_battery_values(self, mock_battery: MagicMock):
        # Non-numeric percent and invalid secsleft
        mock_battery.return_value = SBattery(percent="corrupt", secsleft="invalid", power_plugged=True)
        result = _get_battery_status()

        assert result.success is True
        data = result.data
        assert data["available"] is True
        assert data["percent"] is None
        assert data["power_plugged"] is True
        assert data["charging_status"] == "charging"
        assert data["time_remaining_seconds"] is None
        assert data["time_remaining_formatted"] == "Unknown"

    @patch("psutil.sensors_battery")
    def test_psutil_exception_returns_error(self, mock_battery: MagicMock):
        mock_battery.side_effect = RuntimeError("Battery hardware sensor bus error")
        result = _get_battery_status()

        assert result.success is False
        assert "Battery hardware sensor bus error" in str(result.error)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests: Tool Descriptor and Registry
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatteryToolIntegration:
    def test_tool_descriptor(self):
        assert BATTERY_STATUS_TOOL.name == "get_battery_status"
        assert "battery" in BATTERY_STATUS_TOOL.description.lower()
        assert "charging" in BATTERY_STATUS_TOOL.description.lower()
        assert BATTERY_STATUS_TOOL.parameters.get("type") == "object"

    def test_registry_execution(self):
        reg = build_default_registry()
        assert "get_battery_status" in reg.list_tools()

        res_dict = reg.execute("get_battery_status", {})
        assert "success" in res_dict
        assert res_dict["success"] is True
        assert "data" in res_dict

    def test_live_battery_query(self):
        # Real query against this machine's actual hardware
        result = _get_battery_status()
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "available" in result.data
        if result.data["available"]:
            assert isinstance(result.data["percent"], (int, float))
            assert isinstance(result.data["power_plugged"], bool)
            assert result.data["charging_status"] in ("charging", "fully_charged", "discharging")
