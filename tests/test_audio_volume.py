"""
tests/test_audio_volume.py
──────────────────────────
Comprehensive unit and integration tests for the audio_volume tool.

Covers:
  • Tool descriptor and schema
  • 'get' operation (unmuted and muted)
  • 'set' operation (integer, float, string with %)
  • 'mute' operation
  • 'unmute' operation
  • Volume boundaries (0 and 100)
  • Invalid volume values (negative, >100, non-numeric)
  • Missing volume for 'set'
  • Invalid / unrecognized operations
  • Mocked audio endpoint errors / device failures
  • Registry registration and execution
  • Package export
  • Safe live hardware test with automatic state restoration
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from app.tools.audio_volume import (
    AUDIO_VOLUME_TOOL,
    SUPPORTED_OPERATIONS,
    _audio_volume,
    _get_volume_endpoint,
)
from app.tools.registry import ToolRegistry, build_default_registry
from app.tools.tool import ToolResult


# ─── Mock Endpoint for Isolated Unit Testing ──────────────────────────────────

class MockVolumeEndpoint:
    """Mock Windows Core Audio IAudioEndpointVolume interface."""

    def __init__(self, scalar: float = 0.5, mute: int = 0) -> None:
        self.scalar = scalar
        self.mute = mute

    def GetMasterVolumeLevelScalar(self) -> float:
        return self.scalar

    def SetMasterVolumeLevelScalar(self, level: float, context: Any = None) -> None:
        self.scalar = level

    def GetMute(self) -> int:
        return self.mute

    def SetMute(self, mute_val: int, context: Any = None) -> None:
        self.mute = mute_val


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: Tool Descriptor & Schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioVolumeDescriptor:
    def test_tool_name(self):
        assert AUDIO_VOLUME_TOOL.name == "audio_volume"

    def test_tool_description_contains_examples(self):
        desc = AUDIO_VOLUME_TOOL.description.lower()
        assert "volume" in desc
        assert "mute" in desc
        assert "unmute" in desc

    def test_tool_parameters_schema(self):
        params = AUDIO_VOLUME_TOOL.parameters
        assert params.get("type") == "object"
        props = params.get("properties", {})
        assert "operation" in props
        assert "volume" in props
        assert props["operation"]["enum"] == ["get", "set", "mute", "unmute"]
        assert "operation" in params.get("required", [])


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: Core Operations (Mocked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioVolumeOperations:
    @patch("app.tools.audio_volume._get_volume_endpoint")
    def test_get_unmuted(self, mock_endpoint_fn: MagicMock):
        mock_endpoint_fn.return_value = MockVolumeEndpoint(scalar=0.75, mute=0)

        result = _audio_volume(operation="get")
        assert result.success is True
        assert result.data["operation"] == "get"
        assert result.data["volume_percent"] == 75
        assert result.data["muted"] is False
        assert "75%" in result.data["message"]
        assert "MUTED" not in result.data["message"]

    @patch("app.tools.audio_volume._get_volume_endpoint")
    def test_get_muted(self, mock_endpoint_fn: MagicMock):
        mock_endpoint_fn.return_value = MockVolumeEndpoint(scalar=0.40, mute=1)

        result = _audio_volume(operation="get")
        assert result.success is True
        assert result.data["operation"] == "get"
        assert result.data["volume_percent"] == 40
        assert result.data["muted"] is True
        assert "MUTED" in result.data["message"]

    @patch("app.tools.audio_volume._get_volume_endpoint")
    def test_set_volume_integer(self, mock_endpoint_fn: MagicMock):
        mock = MockVolumeEndpoint(scalar=0.20, mute=0)
        mock_endpoint_fn.return_value = mock

        result = _audio_volume(operation="set", volume=50)
        assert result.success is True
        assert result.data["operation"] == "set"
        assert result.data["volume_percent"] == 50
        assert mock.scalar == 0.50
        assert "50%" in result.data["message"]

    @patch("app.tools.audio_volume._get_volume_endpoint")
    def test_set_volume_string_with_percent(self, mock_endpoint_fn: MagicMock):
        mock = MockVolumeEndpoint(scalar=0.20, mute=0)
        mock_endpoint_fn.return_value = mock

        result = _audio_volume(operation="set", volume="65%")
        assert result.success is True
        assert result.data["volume_percent"] == 65
        assert mock.scalar == 0.65

    @patch("app.tools.audio_volume._get_volume_endpoint")
    def test_set_volume_boundaries(self, mock_endpoint_fn: MagicMock):
        mock = MockVolumeEndpoint()
        mock_endpoint_fn.return_value = mock

        # 0% minimum
        res0 = _audio_volume(operation="set", volume=0)
        assert res0.success is True
        assert res0.data["volume_percent"] == 0
        assert mock.scalar == 0.0

        # 100% maximum
        res100 = _audio_volume(operation="set", volume=100)
        assert res100.success is True
        assert res100.data["volume_percent"] == 100
        assert mock.scalar == 1.0

    @patch("app.tools.audio_volume._get_volume_endpoint")
    def test_set_volume_while_muted_message(self, mock_endpoint_fn: MagicMock):
        mock = MockVolumeEndpoint(scalar=0.10, mute=1)
        mock_endpoint_fn.return_value = mock

        result = _audio_volume(operation="set", volume=30)
        assert result.success is True
        assert result.data["muted"] is True
        assert "muted" in result.data["message"]

    @patch("app.tools.audio_volume._get_volume_endpoint")
    def test_mute_operation(self, mock_endpoint_fn: MagicMock):
        mock = MockVolumeEndpoint(scalar=0.60, mute=0)
        mock_endpoint_fn.return_value = mock

        result = _audio_volume(operation="mute")
        assert result.success is True
        assert result.data["operation"] == "mute"
        assert result.data["muted"] is True
        assert mock.mute == 1
        assert "muted" in result.data["message"].lower()

    @patch("app.tools.audio_volume._get_volume_endpoint")
    def test_unmute_operation(self, mock_endpoint_fn: MagicMock):
        mock = MockVolumeEndpoint(scalar=0.60, mute=1)
        mock_endpoint_fn.return_value = mock

        result = _audio_volume(operation="unmute")
        assert result.success is True
        assert result.data["operation"] == "unmute"
        assert result.data["muted"] is False
        assert mock.mute == 0
        assert "unmuted" in result.data["message"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: Input Validation & Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioVolumeValidation:
    def test_invalid_operation(self):
        result = _audio_volume(operation="invalid_action")
        assert result.success is False
        assert "invalid operation" in result.error.lower()
        assert "supported operations" in result.error.lower()

    def test_empty_operation(self):
        result = _audio_volume(operation="")
        assert result.success is False
        assert "invalid operation" in result.error.lower()

    def test_missing_volume_for_set(self):
        result = _audio_volume(operation="set")
        assert result.success is False
        assert "volume" in result.error.lower()
        assert "required" in result.error.lower()

    def test_empty_string_volume_for_set(self):
        result = _audio_volume(operation="set", volume="   ")
        assert result.success is False
        assert "volume" in result.error.lower()

    def test_non_numeric_volume(self):
        result = _audio_volume(operation="set", volume="fifty")
        assert result.success is False
        assert "invalid volume value" in result.error.lower()

    def test_negative_volume(self):
        result = _audio_volume(operation="set", volume=-1)
        assert result.success is False
        assert "out of range" in result.error.lower()

    def test_volume_exceeds_100(self):
        result = _audio_volume(operation="set", volume=101)
        assert result.success is False
        assert "out of range" in result.error.lower()

    def test_case_insensitive_operation(self):
        with patch("app.tools.audio_volume._get_volume_endpoint") as mock_fn:
            mock = MockVolumeEndpoint(scalar=0.5, mute=0)
            mock_fn.return_value = mock
            result = _audio_volume(operation="  GeT  ")
            assert result.success is True
            assert result.data["operation"] == "get"


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: Error Handling (Device / API Failures)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioVolumeErrorHandling:
    @patch("app.tools.audio_volume._get_volume_endpoint")
    def test_device_not_found_error(self, mock_endpoint_fn: MagicMock):
        mock_endpoint_fn.side_effect = RuntimeError("No active speaker device found")

        result = _audio_volume(operation="get")
        assert result.success is False
        assert "windows audio device error" in result.error.lower()
        assert "no active speaker" in result.error.lower()

    @patch("app.tools.audio_volume._get_volume_endpoint")
    def test_endpoint_method_failure(self, mock_endpoint_fn: MagicMock):
        mock = MagicMock()
        mock.GetMasterVolumeLevelScalar.side_effect = OSError("COM dispatch failed")
        mock_endpoint_fn.return_value = mock

        result = _audio_volume(operation="get")
        assert result.success is False
        assert "failed to execute audio operation" in result.error.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests: Package Export & Tool Registry
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioVolumeIntegration:
    def test_package_export(self):
        from app.tools import AUDIO_VOLUME_TOOL as exported_tool
        assert exported_tool.name == "audio_volume"

    def test_registry_registration(self):
        reg = build_default_registry()
        assert "audio_volume" in reg.list_tools()
        tool = reg.get("audio_volume")
        assert tool is not None
        assert tool.name == "audio_volume"

    def test_registry_execution_mocked(self):
        reg = build_default_registry()
        with patch("app.tools.audio_volume._get_volume_endpoint") as mock_fn:
            mock_fn.return_value = MockVolumeEndpoint(scalar=0.45, mute=0)
            output = reg.execute("audio_volume", {"operation": "get"})
            assert output["success"] is True
            assert output["data"]["volume_percent"] == 45


# ═══════════════════════════════════════════════════════════════════════════════
# Safe Live Hardware Test (Saves and Restores Master Volume)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiveHardwareVolumeSafe:
    def test_live_hardware_query_and_restore(self):
        """
        Interacts with the real system audio endpoint, tests get/set/restore,
        and guarantees the original volume and mute state are restored.
        """
        try:
            endpoint = _get_volume_endpoint()
        except RuntimeError:
            pytest.skip("No real Windows audio output device available on this machine.")

        # 1. Capture original live state
        original_scalar = float(endpoint.GetMasterVolumeLevelScalar())
        original_mute = int(endpoint.GetMute())

        try:
            # 2. Test live 'get'
            get_res = _audio_volume(operation="get")
            assert get_res.success is True
            assert isinstance(get_res.data["volume_percent"], int)
            assert isinstance(get_res.data["muted"], bool)

            # 3. Test controlled small change (e.g. current scalar preserved or slightly adjusted)
            test_target = 42
            set_res = _audio_volume(operation="set", volume=test_target)
            assert set_res.success is True
            assert set_res.data["volume_percent"] == test_target

            # 4. Verify endpoint updated
            read_scalar = float(endpoint.GetMasterVolumeLevelScalar())
            assert round(read_scalar * 100) == test_target

        finally:
            # 5. Guaranteed restoration of original master volume and mute state
            endpoint.SetMasterVolumeLevelScalar(original_scalar, None)
            endpoint.SetMute(original_mute, None)

            # Confirm restoration
            restored_scalar = float(endpoint.GetMasterVolumeLevelScalar())
            restored_mute = int(endpoint.GetMute())
            assert abs(restored_scalar - original_scalar) < 0.02
            assert restored_mute == original_mute
