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
    QGroupBox, QStyle, QToolTip, QFormLayout, QRadioButton, # Added QRadioButton
    QButtonGroup # Added QButtonGroup
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor, QColor, QPalette, QPixmap

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

        # Add Warm Up Button
        self.warmup_button = QPushButton("Warm Up Cache")
        self.warmup_button.setToolTip("Load the selected KV cache into the model for faster responses.")
        self.warmup_button.setEnabled(False) # Disabled initially
        cache_status_layout.addWidget(self.warmup_button)

        # --- Cache Behavior Mode ---
        self.cache_mode_group = QGroupBox("Cache Behavior (When Warmed Up)")
        cache_mode_layout = QVBoxLayout(self.cache_mode_group) # Use QVBoxLayout for vertical stacking

        self.mode_standard_radio = QRadioButton("Standard (State Persists)")
        self.mode_standard_radio.setToolTip("Cache state evolves with conversation (default).")
        self.mode_standard_radio.setChecked(True) # Default mode

        self.mode_fresh_before_radio = QRadioButton("Fresh Context (Reload Before Query)")
        self.mode_fresh_before_radio.setToolTip("Reloads clean cache state before each query.\nGuarantees stateless response generation.")

        self.mode_fresh_after_radio = QRadioButton("Fresh Context (Reload After Query)")
        self.mode_fresh_after_radio.setToolTip("Reloads clean cache state after each query.\nMay feel faster, response uses previous state.")

        cache_mode_layout.addWidget(self.mode_standard_radio)
        cache_mode_layout.addWidget(self.mode_fresh_before_radio)
        cache_mode_layout.addWidget(self.mode_fresh_after_radio)

        # Button group to manage radio buttons
        self.cache_mode_button_group = QButtonGroup(self)
        self.cache_mode_button_group.addButton(self.mode_standard_radio, 1) # Assign IDs
        self.cache_mode_button_group.addButton(self.mode_fresh_before_radio, 2)
        self.cache_mode_button_group.addButton(self.mode_fresh_after_radio, 3)

        cache_status_layout.addWidget(self.cache_mode_group) # Add the group box to the main status layout

        # Add Reset Button
        self.reset_button = QPushButton("Reset Engine")
        self.reset_button.setToolTip("Reset the chat engine to its initial state (unloads model).")
        cache_status_layout.addWidget(self.reset_button)


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

        # --- Chat Controls ---
        chat_controls_layout = QHBoxLayout()
        chat_controls_layout.addStretch() # Push button to the right

        self.clear_chat_button = QPushButton("Clear Chat")
        self.clear_chat_button.setToolTip("Clear the chat history display (does not affect model memory).")
        chat_controls_layout.addWidget(self.clear_chat_button)

        layout.addLayout(chat_controls_layout)
        # --- End Chat Controls ---

        # --- Cache Performance Section ---
        perf_group = QGroupBox("Cache Performance")
        perf_layout = QFormLayout(perf_group)

        self.load_time_label = QLabel("N/A")
        self.tokens_label = QLabel("N/A")
        self.file_size_label = QLabel("N/A")

        perf_layout.addRow("Load Time:", self.load_time_label)
        perf_layout.addRow("Tokens:", self.tokens_label)
        perf_layout.addRow("File Size:", self.file_size_label)

        layout.addWidget(perf_group)
        # --- End Cache Performance Section ---

        # --- Generation Settings ---
        gen_settings_group = QGroupBox("Generation Settings")
        gen_settings_layout = QHBoxLayout(gen_settings_group)

        gen_settings_layout.addWidget(QLabel("Max Response Tokens:"))
        self.max_tokens_spinbox = QSpinBox()
        self.max_tokens_spinbox.setRange(64, 8192) # Set a reasonable range
        self.max_tokens_spinbox.setValue(1024) # Default value
        self.max_tokens_spinbox.setSingleStep(64)
        self.max_tokens_spinbox.setToolTip("Maximum number of tokens the model should generate for a response.")
        gen_settings_layout.addWidget(self.max_tokens_spinbox)

        # Add Temperature Slider
        gen_settings_layout.addWidget(QLabel("Temperature:"))
        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setRange(0, 100) # Represents 0.0 to 1.0
        self.temperature_slider.setValue(70) # Default 0.7
        self.temperature_slider.setTickInterval(10)
        self.temperature_slider.setTickPosition(QSlider.TicksBelow)
        self.temperature_slider.setToolTip("Controls randomness. Lower values (e.g., 0.1) make output more focused/deterministic,\nhigher values (e.g., 0.9) make it more creative/random. Default: 0.7")
        gen_settings_layout.addWidget(self.temperature_slider)

        self.temperature_label = QLabel("0.7")
        self.temperature_label.setFixedWidth(30) # Fixed width for consistent layout
        gen_settings_layout.addWidget(self.temperature_label)

        gen_settings_layout.addStretch()

        layout.addWidget(gen_settings_group)
        # --- End Generation Settings ---


    def connect_signals(self):
        """Connect signals between components"""
        # Input signals
        self.send_button.clicked.connect(self.send_message)
        self.user_input.returnPressed.connect(self.send_message) # Send on Enter key

        # Chat engine signals
        self.chat_engine.response_complete.connect(self.on_response_complete)
        self.chat_engine.response_chunk.connect(self.append_response_chunk)
        self.chat_engine.error_occurred.connect(self.display_error)
        # Connect new ChatEngine signals for warm-up
        self.chat_engine.cache_warming_started.connect(self.on_cache_warming_started)
        self.chat_engine.cache_warmed_up.connect(self.on_cache_warmed_up)
        self.chat_engine.cache_unloaded.connect(self.on_cache_unloaded)
        self.chat_engine.cache_status_changed.connect(self.on_cache_status_changed) # Connect specific status signal
        self.chat_engine.response_started.connect(self.on_response_started) # Connect start signal

        # Cache toggle signal
        self.cache_toggle.stateChanged.connect(self.on_cache_toggle_changed)

        # Warmup button signal
        self.warmup_button.clicked.connect(self.on_warmup_button_clicked)

        # Cache Manager signal (to detect external deletions or updates)
        self.cache_manager.cache_list_updated.connect(self.update_cache_status_display)

        # Chat controls
        self.clear_chat_button.clicked.connect(self.clear_chat_display)

        # Cache Behavior Mode change - Connect to the engine's setter method
        self.cache_mode_button_group.buttonClicked[int].connect(self.on_cache_mode_changed) # Keep this signal

        # Reset button
        self.reset_button.clicked.connect(self.reset_engine)

        # Temperature slider signal
        self.temperature_slider.valueChanged.connect(self.update_temperature_label)


    def initialize_state(self):
        """Initialize UI state from current settings"""
        self.update_cache_status_display() # Update cache status on init
        self.on_cache_status_changed("Idle") # Initialize specific status
        # Initialize Cache Behavior Mode state from engine
        self.set_cache_mode_ui(self.chat_engine.cache_behavior_mode)

    def send_message(self):
        """Send the user's message to the chat engine"""
        message = self.user_input.text().strip()
        if not message:
            return # Don't send empty messages

        # Display user message immediately
        self.append_message("You", message)

        # Clear input field
        self.user_input.clear()

        # Get generation parameters from UI
        max_tokens = self.max_tokens_spinbox.value()
        temperature = self.temperature_slider.value() / 100.0

        # Send to chat engine
        try:
            # Disable input immediately (response_started signal will also do this, but belt-and-suspenders)
            self.set_input_enabled(False)
            # Pass max_tokens and temperature from UI to the engine
            # Note: chat_engine.send_message needs to be updated to accept temperature
            if not self.chat_engine.send_message(message, max_tokens=max_tokens, temperature=temperature):
                 # If send_message returns False (e.g., safety check failed), re-enable input
                 self.set_input_enabled(True)
            # Otherwise, input remains disabled until response_complete or error_occurred
        except Exception as e:
            self.display_error(f"Failed to send message: {e}")
            self.set_input_enabled(True) # Re-enable on exception

    # Slot for response started signal
    @pyqtSlot()
    def on_response_started(self):
        """Disable input when response generation starts."""
        self.set_input_enabled(False)

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

        self.set_input_enabled(True) # Re-enable input on completion

    @pyqtSlot(str)
    def display_error(self, error_message: str):
        """Display an error message in the chat history and update status"""
        self.append_message("Error", error_message, color=QColor("red"))
        # Status is updated by ChatEngine signal ("Error") -> cache_status_changed("Error")
        logging.error(f"Chat Error: {error_message}")
        self.set_input_enabled(True) # Re-enable input on error
        self.warmup_button.setEnabled(self._can_warmup()) # Re-evaluate warmup button state

    def set_input_enabled(self, enabled: bool):
        """Enable or disable the user input field and send button."""
        self.user_input.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        if enabled:
            self.user_input.setFocus() # Set focus back when enabled

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


        # Force scrolling to the bottom
        scrollbar = self.chat_history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        self.chat_history.ensureCursorVisible() # Keep this as well

    @pyqtSlot()
    def clear_chat_display(self):
        """Clear the chat display area."""
        self.chat_history.clear()
        # Optionally add a system message indicating clearance
        self.append_message("System", "Chat display cleared.", color=QColor("blue"))
        logging.info("Chat display cleared by user.")

    @pyqtSlot()
    def reset_engine(self):
        """Reset the chat engine to its initial state via button click."""
        logging.info("Reset Engine button clicked.")
        if self.chat_engine.reset_state():
            self.append_message("System", "Chat engine has been reset.", color=QColor("blue"))
            # Update UI elements to reflect the reset
            self.update_cache_status_display()
            # Clear performance labels explicitly as update_cache_status_display might not if cache name is still set
            self.load_time_label.setText("N/A")
            self.tokens_label.setText("N/A")
            self.file_size_label.setText("N/A")
        else:
            self.append_message("Error", "Failed to reset chat engine.", color=QColor("red"))

    @pyqtSlot(int)
    def on_cache_mode_changed(self, mode_id: int):
        """Handle changes in the selected cache behavior mode via UI click."""
        # Map ID back to a mode identifier
        mode_map = {
            1: "standard",
            2: "fresh_before",
            3: "fresh_after"
        }
        selected_mode = mode_map.get(mode_id, "standard")
        logging.info(f"UI requested cache behavior mode change to: {selected_mode}")
        # Call the engine's method to actually change the mode
        self.chat_engine.set_cache_behavior_mode(selected_mode)
        # The engine's signal `cache_status_changed` might update the UI,
        # but call update_cache_status_display here too for robustness.
        self.update_cache_status_display()

    @pyqtSlot(int)
    def on_cache_toggle_changed(self, state):
        """Handle the 'Use KV Cache' checkbox state change."""
        enabled = state == Qt.Checked
        self.chat_engine.toggle_kv_cache(enabled)
        self.update_cache_status_display() # Update UI immediately
        self.warmup_button.setEnabled(self._can_warmup()) # Update button state

    def on_model_changed(self, model_id: str):
        """Handle model change."""
        logging.info(f"ChatTab: Model changed to {model_id}. Updating cache status display.")
        # Check if the current warmed cache is compatible with the new model
        if self.chat_engine.warmed_cache_path:
            cache_info = self.cache_manager.get_cache_info(self.chat_engine.warmed_cache_path)
            if not cache_info or cache_info.get('model_id') != model_id:
                logging.warning(f"Model changed to {model_id}, unloading incompatible warmed cache.")
                self.chat_engine.unload_cache() # Unload if incompatible
            else:
                logging.info("Warmed cache is compatible with the new model.")
        self.update_cache_status_display() # Update display based on potential unload

    def on_cache_selected(self, cache_path: str):
        """Handle KV cache selection from CacheTab."""
        # Unload previous cache if different one is selected
        if self.chat_engine.warmed_cache_path and self.chat_engine.warmed_cache_path != cache_path:
            logging.info("New cache selected, unloading previously warmed cache.")
            self.chat_engine.unload_cache()

        # Inform chat engine about the selected cache
        if not self.chat_engine.set_kv_cache(cache_path):
             # Error signal should be emitted by chat_engine if set_kv_cache fails
             pass
        # Update UI regardless of success/failure, as chat_engine state changed
        self.update_cache_status_display()
        self.warmup_button.setEnabled(self._can_warmup()) # Update button state

    def _can_warmup(self) -> bool:
        """Check if conditions are met to enable the warm-up button."""
        return (self.chat_engine.use_kv_cache and
                self.chat_engine.current_kv_cache_path is not None and
                Path(self.chat_engine.current_kv_cache_path).exists())

    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human-readable string"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    # --- Slots for Warm-up Signals ---
    @pyqtSlot()
    def on_cache_warming_started(self):
        """Handle cache warming start."""
        self.warmup_button.setEnabled(False)
        self.warmup_button.setText("Warming Up...")
        # Status label updated by on_cache_status_changed

    @pyqtSlot(float, int, int)
    def on_cache_warmed_up(self, load_time: float, token_count: int, file_size: int):
        """Handle cache warming completion."""
        self.warmup_button.setText("Unload Cache")
        self.warmup_button.setEnabled(True)
        self.load_time_label.setText(f"{load_time:.2f} s")
        self.tokens_label.setText(f"{token_count:,}")
        self.file_size_label.setText(self._format_size(file_size))
        # Status label updated by on_cache_status_changed

    @pyqtSlot()
    def on_cache_unloaded(self):
        """Handle cache unloading completion."""
        self.warmup_button.setText("Warm Up Cache")
        self.warmup_button.setEnabled(self._can_warmup()) # Re-evaluate if button should be enabled
        self.load_time_label.setText("N/A")
        self.tokens_label.setText("N/A")
        self.file_size_label.setText("N/A")
        # Status label updated by on_cache_status_changed

    @pyqtSlot(str)
    def on_cache_status_changed(self, status: str):
        """Update the specific cache status label in the chat tab."""
        logging.info(f"ChatTab Cache Status Update: {status}")
        status_color = "gray" # Default
        if status == "Warming Up" or status == "Warming Up (Loading State)..." or status == "Unloading":
            status_color = "orange"
        elif status == "Warmed Up" or status == "Warmed Up (Generating)":
            status_color = "green"
        elif status == "Error":
            status_color = "red"
            # Reset button state on error during warm-up/unload
            self.warmup_button.setText("Warm Up Cache")
            self.warmup_button.setEnabled(self._can_warmup())
        elif status == "Using TRUE KV Cache" or status == "Using TRUE KV Cache (Generating)":
             # This status is for temporary loads, not persistent warm-up
             status_color = "blue" # Use a different color? Or just green? Let's use blue for distinction.
        elif status == "Fallback (Generating)":
             status_color = "orange"
        # Handle Fresh Context status messages
        elif "Fresh Context" in status:
             if "Reset OK" in status or "Enabled" in status:
                 status_color = "blue"
             elif "Resetting" in status:
                 status_color = "orange"
             elif "Reset Failed" in status or "Disabled" in status:
                 status_color = "gray" # Or maybe orange for failed? Let's stick with gray for disabled/failed reset.


        self.cache_effective_status_label.setText(f"({status})")
        self.cache_effective_status_label.setStyleSheet(f"color: {status_color};")

    @pyqtSlot()
    def on_warmup_button_clicked(self):
        """Handle clicks on the warm-up/unload button."""
        if self.chat_engine.warmed_cache_path:
            # Currently warmed up, so unload
            self.chat_engine.unload_cache()
        elif self._can_warmup():
            # Not warmed up, conditions met, so warm up
            self.chat_engine.warm_up_cache(self.chat_engine.current_kv_cache_path)
        else:
            logging.warning("Warmup button clicked but conditions not met.")


    def update_cache_status_display(self):
        """Update the KV cache status indicators in the UI, including warm-up button state."""
        # --- Update Cache Name Label ---
        cache_path_str = self.chat_engine.current_kv_cache_path
        display_text = "Cache: None"
        cache_exists = False
        cache_name = "None" # Initialize cache_name with a default value
        if cache_path_str:
            cache_path = Path(cache_path_str)
            cache_name = cache_path.name
            model_id_str = "(Unknown Model)" # Default
            try:
                if cache_path.exists():
                    cache_exists = True
                    # Get model ID from cache info
                    cache_info = self.cache_manager.get_cache_info(cache_path_str)
                    if cache_info:
                        model_id = cache_info.get('model_id')
                        if model_id:
                            model_id_str = f"({model_id})"
                    display_text = f"Cache: {cache_name} {model_id_str}"
                else:
                    display_text = f"Cache: {cache_name} (Not Found!)"
            except OSError as e:
                 logging.error(f"Error checking cache file existence '{cache_path_str}': {e}")
                 display_text = f"Cache: {cache_name} (Error Checking!)"
        self.cache_name_label.setText(display_text)

        # --- Update Warmup Button State ---
        # Use cache_exists determined above
        can_warmup_now = (self.chat_engine.use_kv_cache and
                          cache_path_str is not None and
                          cache_exists) # Use the existence check result
        is_currently_warming = "Warming Up" in self.cache_effective_status_label.text() # Check current status text

        if self.chat_engine.warmed_cache_path == cache_path_str and cache_exists:
             # Correct cache is warmed up
             self.warmup_button.setText("Unload Cache")
             self.warmup_button.setEnabled(True)
        elif is_currently_warming:
             # Operation in progress
             self.warmup_button.setText("Warming Up...")
             self.warmup_button.setEnabled(False)
        else:
             # Not warmed up or wrong cache warmed up
             self.warmup_button.setText("Warm Up Cache")
             self.warmup_button.setEnabled(can_warmup_now) # Enable only if possible

        # --- Update Status Label (Handled by on_cache_status_changed) ---
        # The specific status label (Idle, Warming Up, Warmed Up, etc.)
        # is now updated primarily by the on_cache_status_changed slot.
        # We might call it here just to ensure consistency if needed,
        # but it might cause redundant updates. Let's rely on the signal for now.
        # self.on_cache_status_changed(self.chat_engine.get_current_cache_status()) # Needs engine method

        # --- Update Performance Labels (If not warmed up, clear them) ---
        if not (self.chat_engine.warmed_cache_path == cache_path_str and cache_exists):
             self.load_time_label.setText("N/A")
             self.tokens_label.setText("N/A")
             self.file_size_label.setText("N/A")

        # --- Ensure Checkbox Reflects Engine State ---
        use_cache = self.chat_engine.use_kv_cache
        self.cache_toggle.blockSignals(True)
        self.cache_toggle.setChecked(use_cache)
        self.cache_toggle.blockSignals(False)

        # --- Ensure Cache Behavior Mode Radio Buttons Reflect Engine State ---
        self.set_cache_mode_ui(self.chat_engine.cache_behavior_mode)

        # Initialize temperature label
        self.update_temperature_label(self.temperature_slider.value())


        logging.debug(f"Cache status display updated. Selected: '{cache_name}', Warmed: '{Path(self.chat_engine.warmed_cache_path).name if self.chat_engine.warmed_cache_path else 'None'}'")

    @pyqtSlot(int)
    def update_temperature_label(self, value):
        """Update the label displaying the current temperature."""
        temp = value / 100.0
        self.temperature_label.setText(f"{temp:.1f}")

    def set_cache_mode_ui(self, mode: str):
        """Updates the radio buttons to reflect the given mode."""
        self.cache_mode_button_group.blockSignals(True)
        if mode == "fresh_before":
            self.mode_fresh_before_radio.setChecked(True)
        elif mode == "fresh_after":
            self.mode_fresh_after_radio.setChecked(True)
        else: # Default to standard
            self.mode_standard_radio.setChecked(True)
        self.cache_mode_button_group.blockSignals(False)
