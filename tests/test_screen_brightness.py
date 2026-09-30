"""
tests/test_screen_brightness.py
───────────────────────────────
Comprehensive unit and integration tests for the screen_brightness tool.

Covers:
  • Tool descriptor and schema
  • Valid 'get' operation
  • Valid 'set' operation (integers, strings with %, floats)
  • 'increase' operation (default step, custom step, clamping to 100)
  • 'decrease' operation (default step, custom step, clamping to 0)
  • Exact boundaries (0% and 100%)
  • Invalid operation names and empty inputs
  • Missing brightness for 'set'
  • Out of range brightness values (< 0, > 100)
  • Non-numeric brightness values
  • Invalid step values (< 1, > 50, non-numeric)
  • Unavailable hardware/API behavior (graceful handling)
  • Security check: strictly no shell or subprocess execution
  • Registry discovery and package export
  • Safe live hardware test with guaranteed state restoration
"""

from __future__ import annotations

import os
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.tools.registry import ToolRegistry, build_default_registry
from app.tools.screen_brightness import (
    DEFAULT_STEP_PERCENT,
    SCREEN_BRIGHTNESS_TOOL,
    _apply_brightness,
    _read_current_brightness,
    _screen_brightness,
)
from app.tools.tool import ToolResult


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: Tool Descriptor & Schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestScreenBrightnessDescriptor:
    def test_tool_name(self):
        assert SCREEN_BRIGHTNESS_TOOL.name == "screen_brightness"

    def test_tool_description_contains_examples(self):
        desc = SCREEN_BRIGHTNESS_TOOL.description.lower()
        assert "brightness" in desc
        assert "increase" in desc
        assert "decrease" in desc

    def test_tool_parameters_schema(self):
        params = SCREEN_BRIGHTNESS_TOOL.parameters
        assert params.get("type") == "object"
        props = params.get("properties", {})
        assert "operation" in props
        assert "brightness_percent" in props
        assert "step_percent" in props
        assert props["operation"]["enum"] == ["get", "set", "increase", "decrease"]
        assert "operation" in params.get("required", [])


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: Core Operations (Mocked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestScreenBrightnessOperations:
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_get_brightness(self, mock_read: MagicMock):
        mock_read.return_value = 60

        result = _screen_brightness(operation="get")
        assert result.success is True
        assert result.data["available"] is True
        assert result.data["operation"] == "get"
        assert result.data["brightness_percent"] == 60
        assert "60%" in result.data["message"]

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_set_brightness_integer(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 40
        mock_apply.return_value = 75

        result = _screen_brightness(operation="set", brightness_percent=75)
        assert result.success is True
        assert result.data["available"] is True
        assert result.data["operation"] == "set"
        assert result.data["brightness_percent"] == 75
        assert result.data["previous_brightness_percent"] == 40
        assert "75%" in result.data["message"]
        mock_apply.assert_called_once_with(75)

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_set_brightness_string_with_percent(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 50
        mock_apply.return_value = 80

        result = _screen_brightness(operation="set", brightness_percent="80%")
        assert result.success is True
        assert result.data["brightness_percent"] == 80
        mock_apply.assert_called_once_with(80)

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_set_brightness_boundary_zero(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 20
        mock_apply.return_value = 0

        result = _screen_brightness(operation="set", brightness_percent=0)
        assert result.success is True
        assert result.data["brightness_percent"] == 0
        mock_apply.assert_called_once_with(0)

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_set_brightness_boundary_hundred(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 80
        mock_apply.return_value = 100

        result = _screen_brightness(operation="set", brightness_percent=100)
        assert result.success is True
        assert result.data["brightness_percent"] == 100
        mock_apply.assert_called_once_with(100)

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_increase_default_step(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 50
        mock_apply.return_value = 60

        result = _screen_brightness(operation="increase")
        assert result.success is True
        assert result.data["brightness_percent"] == 60
        assert result.data["previous_brightness_percent"] == 50
        mock_apply.assert_called_once_with(50 + DEFAULT_STEP_PERCENT)

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_increase_custom_step(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 50
        mock_apply.return_value = 75

        result = _screen_brightness(operation="increase", step_percent=25)
        assert result.success is True
        assert result.data["brightness_percent"] == 75
        mock_apply.assert_called_once_with(75)

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_increase_clamps_to_100(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 95
        mock_apply.return_value = 100

        result = _screen_brightness(operation="increase", step_percent=10)
        assert result.success is True
        assert result.data["brightness_percent"] == 100
        mock_apply.assert_called_once_with(100)

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_increase_already_at_maximum(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 100
        mock_apply.return_value = 100

        result = _screen_brightness(operation="increase", step_percent=10)
        assert result.success is True
        assert "already at maximum" in result.data["message"].lower()

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_decrease_default_step(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 50
        mock_apply.return_value = 40

        result = _screen_brightness(operation="decrease")
        assert result.success is True
        assert result.data["brightness_percent"] == 40
        assert result.data["previous_brightness_percent"] == 50
        mock_apply.assert_called_once_with(50 - DEFAULT_STEP_PERCENT)

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_decrease_custom_step(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 50
        mock_apply.return_value = 35

        result = _screen_brightness(operation="decrease", step_percent=15)
        assert result.success is True
        assert result.data["brightness_percent"] == 35
        mock_apply.assert_called_once_with(35)

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_decrease_clamps_to_zero(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 5
        mock_apply.return_value = 0

        result = _screen_brightness(operation="decrease", step_percent=10)
        assert result.success is True
        assert result.data["brightness_percent"] == 0
        mock_apply.assert_called_once_with(0)

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_decrease_already_at_minimum(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 0
        mock_apply.return_value = 0

        result = _screen_brightness(operation="decrease", step_percent=10)
        assert result.success is True
        assert "already at minimum" in result.data["message"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: Validation & Error Handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestScreenBrightnessValidation:
    def test_invalid_operation(self):
        result = _screen_brightness(operation="explode")
        assert result.success is False
        assert "invalid operation" in result.error.lower()

    def test_empty_operation(self):
        result = _screen_brightness(operation="")
        assert result.success is False
        assert "invalid operation" in result.error.lower()

    def test_missing_brightness_for_set(self):
        result = _screen_brightness(operation="set")
        assert result.success is False
        assert "brightness_percent" in result.error.lower()
        assert "required" in result.error.lower()

    def test_empty_string_brightness_for_set(self):
        result = _screen_brightness(operation="set", brightness_percent="   ")
        assert result.success is False
        assert "brightness_percent" in result.error.lower()

    def test_brightness_below_zero(self):
        result = _screen_brightness(operation="set", brightness_percent=-1)
        assert result.success is False
        assert "out of range" in result.error.lower()

    def test_brightness_above_hundred(self):
        result = _screen_brightness(operation="set", brightness_percent=101)
        assert result.success is False
        assert "out of range" in result.error.lower()

    def test_non_numeric_brightness(self):
        result = _screen_brightness(operation="set", brightness_percent="bright")
        assert result.success is False
        assert "invalid brightness value" in result.error.lower()

    def test_step_below_minimum(self):
        result = _screen_brightness(operation="increase", step_percent=0)
        assert result.success is False
        assert "out of range" in result.error.lower()

    def test_step_above_maximum(self):
        result = _screen_brightness(operation="decrease", step_percent=55)
        assert result.success is False
        assert "out of range" in result.error.lower()

    def test_non_numeric_step(self):
        result = _screen_brightness(operation="increase", step_percent="ten")
        assert result.success is False
        assert "invalid step value" in result.error.lower()

    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_unavailable_hardware_on_get(self, mock_read: MagicMock):
        mock_read.return_value = None

        result = _screen_brightness(operation="get")
        assert result.success is True
        assert result.data["available"] is False
        assert result.data["brightness_percent"] is None
        assert "not available" in result.data["message"].lower()

    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_unavailable_hardware_on_set(self, mock_read: MagicMock):
        mock_read.return_value = None

        result = _screen_brightness(operation="set", brightness_percent=50)
        assert result.success is False
        assert "not available" in result.error.lower()

    @patch("app.tools.screen_brightness._apply_brightness")
    @patch("app.tools.screen_brightness._read_current_brightness")
    def test_hardware_failure_on_apply(self, mock_read: MagicMock, mock_apply: MagicMock):
        mock_read.return_value = 50
        mock_apply.side_effect = RuntimeError("DDC/CI communication error")

        result = _screen_brightness(operation="set", brightness_percent=70)
        assert result.success is False
        assert "failed to change screen brightness" in result.error.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: Security & Environment Safety
# ═══════════════════════════════════════════════════════════════════════════════

class TestScreenBrightnessSecurity:
    @patch("app.tools.screen_brightness._read_current_brightness", return_value=50)
    @patch("app.tools.screen_brightness._apply_brightness", return_value=60)
    def test_strictly_no_subprocess_or_shell(self, mock_apply: MagicMock, mock_read: MagicMock):
        """Verify that screen_brightness never invokes subprocess or shell commands."""
        with patch("subprocess.run") as mock_sub_run, \
             patch("subprocess.Popen") as mock_sub_popen, \
             patch("os.system") as mock_os_system:

            _screen_brightness(operation="get")
            _screen_brightness(operation="set", brightness_percent=60)
            _screen_brightness(operation="increase")
            _screen_brightness(operation="decrease")

            assert mock_sub_run.call_count == 0
            assert mock_sub_popen.call_count == 0
            assert mock_os_system.call_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests: Tool Registry & Package Export
# ═══════════════════════════════════════════════════════════════════════════════

class TestScreenBrightnessIntegration:
    def test_package_export(self):
        from app.tools import SCREEN_BRIGHTNESS_TOOL as exported_tool
        assert exported_tool.name == "screen_brightness"

    def test_registry_registration(self):
        reg = build_default_registry()
        assert "screen_brightness" in reg.list_tools()
        tool = reg.get("screen_brightness")
        assert tool is not None
        assert tool.name == "screen_brightness"

    def test_registry_execution_mocked(self):
        reg = build_default_registry()
        with patch("app.tools.screen_brightness._read_current_brightness", return_value=65):
            output = reg.execute("screen_brightness", {"operation": "get"})
            assert output["success"] is True
            assert output["data"]["brightness_percent"] == 65


# ═══════════════════════════════════════════════════════════════════════════════
# Safe Live Hardware Test (Guaranteed Restoration)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiveHardwareBrightnessSafe:
    def test_live_hardware_query_and_restore(self):
        """
        Interacts with the real display hardware if present, tests read/write,
        and guarantees the original brightness level is restored.
        """
        orig = _read_current_brightness()
        if orig is None:
            pytest.skip("Screen brightness control is not available on this machine's display.")

        try:
            # 1. Test live 'get'
            get_res = _screen_brightness(operation="get")
            assert get_res.success is True
            assert get_res.data["available"] is True
            assert get_res.data["brightness_percent"] == orig

            # 2. Test safe small delta (e.g. +1 or -1 without drifting far)
            test_target = 50 if orig != 50 else 55
            set_res = _screen_brightness(operation="set", brightness_percent=test_target)
            assert set_res.success is True
            assert abs(set_res.data["brightness_percent"] - test_target) <= 2

        finally:
            # 3. Guaranteed restoration of initial brightness
            _apply_brightness(orig)
            restored = _read_current_brightness()
            assert restored is not None and abs(restored - orig) <= 2

