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
    QComboBox, QFileDialog, QSplitter, QFrame, QApplication
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor, QColor, QPalette

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

        # Header label
        header = QLabel("Chat Interface")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)

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

        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

    def connect_signals(self):
        """Connect signals between components"""
        # Input signals
        self.send_button.clicked.connect(self.send_message)
        self.user_input.returnPressed.connect(self.send_message) # Send on Enter key

        # Chat engine signals
        # Connect to the correct signal: response_complete(str, bool)
        self.chat_engine.response_complete.connect(self.display_response)
        # self.chat_engine.status_updated.connect(self.update_status) # Signal doesn't exist in ChatEngine
        self.chat_engine.error_occurred.connect(self.display_error)

    def initialize_state(self):
        """Initialize UI state from current settings"""
        # Could load initial prompt or settings here if needed
        self.update_status("Initialized. Ready for input.")

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
            self.update_status("Sending message...")
            self.send_button.setEnabled(False) # Disable button while processing
            self.user_input.setEnabled(False)
        except Exception as e:
            self.display_error(f"Failed to send message: {e}")

    # Update slot to accept (str, bool) from response_complete signal
    @pyqtSlot(str, bool)
    def display_response(self, response: str, success: bool):
        """Display the model's response in the chat history"""
        if success:
            self.append_message("Model", response)
            self.update_status("Ready")
        else:
            # If response_complete signals failure, show an error
            self.append_message("Error", f"Response generation failed. Details: {response}", color=QColor("orange"))
            self.update_status("Response failed")

        self.send_button.setEnabled(True) # Re-enable button
        self.user_input.setEnabled(True)
        self.user_input.setFocus() # Set focus back to input

    @pyqtSlot(str)
    def update_status(self, status: str):
        """Update the status label"""
        self.status_label.setText(status)
        logging.info(f"Chat Status: {status}")

    @pyqtSlot(str)
    def display_error(self, error_message: str):
        """Display an error message in the chat history and status"""
        self.append_message("Error", error_message, color=QColor("red"))
        self.update_status(f"Error: {error_message[:50]}...") # Show truncated error in status
        logging.error(f"Chat Error: {error_message}")
        self.send_button.setEnabled(True) # Re-enable button on error
        self.user_input.setEnabled(True)
        self.user_input.setFocus()

    def append_message(self, sender: str, message: str, color: QColor = None):
        """Append a message to the chat history QTextEdit"""
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_history.setTextCursor(cursor)

        # Set color if provided
        if color:
            format = cursor.charFormat()
            format.setForeground(color)
            cursor.setCharFormat(format)

        # Append sender and message
        cursor.insertText(f"{sender}: ", cursor.charFormat()) # Keep color for sender

        # Reset color for message content (if color was set)
        if color:
             format.setForeground(self.chat_history.palette().color(QPalette.Text)) # Default text color
             cursor.setCharFormat(format)

        cursor.insertText(message + "\n\n") # Add message and extra newline for spacing

        # Ensure the view scrolls to the bottom
        self.chat_history.ensureCursorVisible()

    def on_model_changed(self, model_id: str):
        """Handle model change (placeholder)"""
        # Might clear chat history or update status
        self.update_status(f"Model changed to {model_id}. Chat context might be reset.")
        # self.chat_history.clear() # Optional: Clear history on model change

    def on_cache_selected(self, cache_path: str):
        """Handle KV cache selection from CacheTab (placeholder)"""
        # Inform chat engine about the selected cache
        if self.chat_engine.set_kv_cache(cache_path):
             self.update_status(f"Using KV cache: {Path(cache_path).name}")
        else:
             self.display_error(f"Failed to set KV cache: {Path(cache_path).name}")
