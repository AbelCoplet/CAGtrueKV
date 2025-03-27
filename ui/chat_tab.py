#!/usr/bin/env python3
"""
Chat tab for LlamaCag UI

Provides a chat interface for interacting with the model.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QCheckBox, QSlider, QSpinBox,
    QComboBox, QFileDialog, QSplitter, QFrame, QApplication,
    QGroupBox, QStyle, QToolTip # Added QGroupBox, QStyle, QToolTip
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor, QColor, QPalette, QPixmap # Added QPixmap

from core.chat_engine import ChatEngine
from core.model_manager import ModelManager
from core.cache_manager import CacheManager
from utils.config import ConfigManager


class ChatTab(QWidget):
    """Chat interface tab for interacting with the model"""

    def __init__(self, chat_engine: ChatEngine, model_manager: ModelManager,
                 cache_manager: CacheManager, config_manager: ConfigManager):
        """Initialize chat tab"""
        super().__init__()

        self.chat_engine = chat_engine
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.config_manager = config_manager
        self.config = config_manager.get_config()

        # Initialize UI
        self.setup_ui()

        # Connect signals
        self.connect_signals()

        # Initialize with current settings
        self.initialize_state()

    def setup_ui(self):
        """Set up the user interface"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # --- Cache Status Section ---
        cache_status_group = QGroupBox("KV Cache Status")
        cache_status_layout = QHBoxLayout(cache_status_group)

        # Icon (Temporarily disabled due to path issues)
        # self.cache_status_icon = QLabel()
        # self.cache_status_icon.setFixedSize(16, 16)
        # self.icon_active = QPixmap("resources/icons/cache_active.png").scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # self.icon_inactive = QPixmap("resources/icons/cache_inactive.png").scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # self.icon_error = QPixmap("resources/icons/cache_error.png").scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # cache_status_layout.addWidget(self.cache_status_icon)

        # Cache Name/Status Text
        self.cache_name_label = QLabel("Cache: None")
        self.cache_name_label.setWordWrap(True) # Allow wrapping
        cache_status_layout.addWidget(self.cache_name_label)

        # Descriptive Status Text (Renamed and simplified)
        self.cache_effective_status_label = QLabel("(Status: Unknown)")
        self.cache_effective_status_label.setStyleSheet("color: gray;")
        self.cache_effective_status_label.setWordWrap(True) # Allow wrapping
        cache_status_layout.addWidget(self.cache_effective_status_label)

        # Help Icon for Status
        self.cache_status_help_icon = QLabel()
        help_icon = QApplication.style().standardIcon(QStyle.SP_MessageBoxQuestion)
        pixmap = help_icon.pixmap(QSize(16, 16))
        # Attempt to style the label - might not affect the standard icon color
        self.cache_status_help_icon.setStyleSheet("color: white;") 
        self.cache_status_help_icon.setPixmap(pixmap)
        self.cache_status_help_icon.setFixedSize(16, 16)
        self.cache_status_help_icon.setToolTip(
            """<b>KV Cache Status Explanations:</b><br>
            - <font color='green'><b>(Using TRUE KV Cache):</b></font> A specific document's KV cache is selected and actively being used for faster responses.<br>
            - <font color='orange'><b>(Fallback: Using Master Cache):</b></font> 'Use KV Cache' is enabled, but no specific document cache is selected. The general 'master' cache (if available) is used.<br>
            - <font color='red'><b>(Fallback: Cache Missing/Error):</b></font> A specific cache was selected, but the file is missing or cannot be read. Falling back to generation without cache.<br>
            - <font color='gray'><b>(Disabled - Fallback):</b></font> 'Use KV Cache' is disabled. Generation proceeds without using any KV cache."""
        )
        cache_status_layout.addWidget(self.cache_status_help_icon)


        cache_status_layout.addStretch()

        # Toggle Checkbox
        self.cache_toggle = QCheckBox("Use KV Cache")
        self.cache_toggle.setChecked(self.chat_engine.use_kv_cache) # Initialize from engine state
        cache_status_layout.addWidget(self.cache_toggle)

        layout.addWidget(cache_status_group)
        # --- End Cache Status Section ---


        # Chat history display
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Monospace", 10)) # Use monospace for better formatting
        layout.addWidget(self.chat_history)

        # Input area layout
        input_layout = QHBoxLayout()

        # User input field
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter your message here...")
        input_layout.addWidget(self.user_input)

        # Send button
        self.send_button = QPushButton("Send")
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

        # Status label (Removed - redundant, main window status bar shows engine state)
        # self.status_label = QLabel("Ready")
        # layout.addWidget(self.status_label)

    def connect_signals(self):
        """Connect signals between components"""
        # Input signals
        self.send_button.clicked.connect(self.send_message)
        self.user_input.returnPressed.connect(self.send_message) # Send on Enter key

        # Chat engine signals
        self.chat_engine.response_complete.connect(self.on_response_complete) # Renamed slot
        self.chat_engine.response_chunk.connect(self.append_response_chunk) # Connect chunk signal
        # self.chat_engine.status_updated.connect(self.update_status) # Removed - Slot was removed as label is gone
        self.chat_engine.error_occurred.connect(self.display_error)

        # Cache toggle signal
        self.cache_toggle.stateChanged.connect(self.on_cache_toggle_changed)

        # Cache Manager signal (to detect external deletions)
        self.cache_manager.cache_list_updated.connect(self.update_cache_status_display)


    def initialize_state(self):
        """Initialize UI state from current settings"""
        self.update_cache_status_display() # Update cache status on init
        # self.update_status("Idle") # Removed call to deleted method

    def send_message(self):
        """Send the user's message to the chat engine"""
        message = self.user_input.text().strip()
        if not message:
            return # Don't send empty messages

        # Display user message immediately
        self.append_message("You", message)

        # Clear input field
        self.user_input.clear()

        # Send to chat engine
        try:
            self.chat_engine.send_message(message)
            # Status update is now handled by ChatEngine signal
            # self.update_status("Sending message...")
            self.send_button.setEnabled(False) # Disable button while processing
            # self.user_input.setEnabled(False) # Keep input enabled
        except Exception as e:
            self.display_error(f"Failed to send message: {e}")

    # Slot for response chunks
    @pyqtSlot(str)
    def append_response_chunk(self, chunk: str):
        """Append a chunk of the model's response"""
        # Append chunk without sender prefix or extra newlines
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_history.setTextCursor(cursor)
        cursor.insertText(chunk)
        self.chat_history.ensureCursorVisible()


    # Slot for final response completion
    @pyqtSlot(str, bool)
    def on_response_complete(self, response: str, success: bool):
        """Handle the completion of a model response"""
        if success:
            # Add final newline formatting if needed (chunk handling might miss last one)
            cursor = self.chat_history.textCursor()
            cursor.movePosition(QTextCursor.End)
            # Check if the last character is not a newline
            self.chat_history.setTextCursor(cursor)
            # A bit complex, maybe just add newlines after the whole response?
            # Let's assume ChatEngine sends the full response including final formatting.
            # We already displayed chunks, so just add the final formatting.
            self.append_message("", "\n") # Add spacing after response
            # Status is updated by ChatEngine signal ("Idle")
        else:
            # Error message is handled by display_error
            pass # Error already displayed by display_error signal

        self.send_button.setEnabled(True) # Re-enable button
        # self.user_input.setEnabled(True) # Keep input enabled
        self.user_input.setFocus() # Set focus back to input

    # Removed update_status as the label is removed
    # @pyqtSlot(str)
    # def update_status(self, status: str):
    #     """Update the status label"""
    #     # self.status_label.setText(status) # Label removed
    #     logging.info(f"Chat Status: {status}") # Keep logging

    @pyqtSlot(str)
    def display_error(self, error_message: str):
        """Display an error message in the chat history and status"""
        self.append_message("Error", error_message, color=QColor("red"))
        # Status is updated by ChatEngine signal ("Error")
        # self.update_status(f"Error: {error_message[:50]}...") # Show truncated error in status
        logging.error(f"Chat Error: {error_message}")
        self.send_button.setEnabled(True) # Re-enable button on error
        # self.user_input.setEnabled(True) # Keep input enabled
        self.user_input.setFocus()

    def append_message(self, sender: str, message: str, color: QColor = None):
        """Append a formatted message (sender + content) to the chat history."""
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_history.setTextCursor(cursor)

        # Set color if provided
        if color:
            format = cursor.charFormat()
            format.setForeground(color)
            cursor.setCharFormat(format)

        # Append sender (if provided)
        if sender:
            cursor.insertText(f"{sender}: ", cursor.charFormat()) # Keep color for sender

            # Reset color for message content (if color was set)
            if color:
                 format.setForeground(self.chat_history.palette().color(QPalette.Text)) # Default text color
                 cursor.setCharFormat(format)

        # Append message content
        cursor.insertText(message)

        # Add spacing (handle potential double newlines if message ends with one)
        if not message.endswith('\n'):
            cursor.insertText("\n")
        cursor.insertText("\n")


        # Ensure the view scrolls to the bottom
        self.chat_history.ensureCursorVisible()

    @pyqtSlot(int)
    def on_cache_toggle_changed(self, state):
        """Handle the 'Use KV Cache' checkbox state change."""
        enabled = state == Qt.Checked
        self.chat_engine.toggle_kv_cache(enabled)
        self.update_cache_status_display() # Update UI immediately

    def on_model_changed(self, model_id: str):
        """Handle model change (placeholder)"""
        # Might clear chat history or update status
        # self.update_status(f"Model changed to {model_id}. Chat context might be reset.") # Removed call to deleted method
        logging.info(f"ChatTab: Model changed to {model_id}. Updating cache status display.") # Keep logging info
        # self.chat_history.clear() # Optional: Clear history on model change
        self.update_cache_status_display() # Model change might affect cache compatibility/choice

    def on_cache_selected(self, cache_path: str):
        """Handle KV cache selection from CacheTab."""
        # Inform chat engine about the selected cache
        if not self.chat_engine.set_kv_cache(cache_path):
             # Error signal should be emitted by chat_engine if set_kv_cache fails
             # self.display_error(f"Failed to set KV cache: {Path(cache_path).name}")
             pass
        # Update UI regardless of success/failure, as chat_engine state changed
        self.update_cache_status_display()

    def update_cache_status_display(self):
        """Update the KV cache status indicators in the UI."""
        cache_path_str = self.chat_engine.current_kv_cache_path
        use_cache = self.chat_engine.use_kv_cache
        logging.debug(f"Updating cache status display: use_cache={use_cache}, cache_path_str='{cache_path_str}'")
        cache_exists = False
        cache_name = "None"
        status_text = "(Status: Unknown)"
        status_color = "gray"

        if cache_path_str:
            cache_path = Path(cache_path_str)
            cache_name = cache_path.name
            try:
                if cache_path.exists():
                    cache_exists = True
                    logging.debug(f"Cache file exists: {cache_path_str}")
                else:
                    cache_name = f"{cache_name} (Not Found!)"
                    logging.warning(f"Cache file path set ('{cache_path_str}') but file does not exist.")
            except OSError as e:
                 logging.error(f"Error checking if cache file exists '{cache_path_str}': {e}")
                 cache_name = f"{cache_name} (Error Checking!)"
                 cache_exists = False

        # Determine status text and color based on state
        if use_cache:
            if cache_exists:
                status_text = "(Using TRUE KV Cache)" # Updated text
                status_color = "green"
            elif cache_path_str: # Cache selected but not found or error checking
                status_text = "(Fallback: Cache Missing/Error)"
                status_color = "red"
            else: # No cache selected
                # No specific cache selected, but 'Use KV Cache' is ticked.
                # Assume the engine falls back to the master cache if available.
                status_text = "(Fallback: Using Master Cache)" # More informative text
                status_color = "orange" # Keep orange to indicate fallback status
                cache_name = "Master (Default)" # Update name label for clarity
        else: # Cache usage disabled
            status_text = "(Disabled - Fallback)"
            status_color = "gray"
            # Keep showing cache name if one was selected, otherwise show None
            cache_name = cache_name if cache_path_str else "None"

        # Update UI elements
        self.cache_name_label.setText(f"Cache: {cache_name}")
        self.cache_effective_status_label.setText(status_text)
        self.cache_effective_status_label.setStyleSheet(f"color: {status_color};")

        logging.debug(f"Cache status display updated. Status: {status_text}")

        # Ensure checkbox reflects engine state
        # Block signals temporarily to prevent recursion if setChecked triggers stateChanged
        self.cache_toggle.blockSignals(True)
        self.cache_toggle.setChecked(use_cache)
        self.cache_toggle.blockSignals(False)
