# BOB tools package
from app.tools.tool import Tool, ToolResult
from app.tools.registry import ToolRegistry, build_default_registry
from app.tools.close_application import CLOSE_APPLICATION_TOOL
from app.tools.search_computer import SEARCH_COMPUTER_TOOL
from app.tools.get_battery_status import BATTERY_STATUS_TOOL
from app.tools.get_running_applications import RUNNING_APPLICATIONS_TOOL
from app.tools.audio_volume import AUDIO_VOLUME_TOOL
from app.tools.screen_brightness import SCREEN_BRIGHTNESS_TOOL
from app.tools.window_control import WINDOW_CONTROL_TOOL
from app.tools.web_search import WEB_SEARCH_TOOL

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "build_default_registry",
    "CLOSE_APPLICATION_TOOL",
    "SEARCH_COMPUTER_TOOL",
    "BATTERY_STATUS_TOOL",
    "RUNNING_APPLICATIONS_TOOL",
    "AUDIO_VOLUME_TOOL",
    "SCREEN_BRIGHTNESS_TOOL",
    "WINDOW_CONTROL_TOOL",
    "WEB_SEARCH_TOOL",
]


