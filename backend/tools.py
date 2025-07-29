# This module provides an abstraction layer for file system tools.
# It currently uses a mock implementation for development purposes,
# allowing for easy swapping to a real API later.

from backend.yandex_disk_mock import get_file_content, list_files

# This dictionary maps tool names to their corresponding functions.
# The AI agent's core logic will use this to dispatch tool calls.
available_tools = {
    "list_files": list_files,
    "get_file_content": get_file_content,
}
