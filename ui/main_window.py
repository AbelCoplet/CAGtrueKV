#!/usr/bin/env python3
"""
Main window for LlamaCag UI
The main application window with tabbed interface for all functionality.
"""
import os
import sys
import logging
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QStatusBar, QPushButton, QMessageBox, QSizePolicy # Added QSizePolicy
)
from PyQt5.QtCore import Qt, QSize, QSettings, QTimer, pyqtSlot # Added pyqtSlot import
from core.llama_manager import LlamaManager
from core.model_manager import ModelManager
from core.document_processor import DocumentProcessor
from core.cache_manager import CacheManager
from core.chat_engine import ChatEngine
from core.n8n_interface import N8nInterface
from ui.model_tab import ModelTab
from ui.document_tab import DocumentTab
from ui.chat_tab import ChatTab
from ui.cache_tab import CacheTab
from ui.settings_tab import SettingsTab
from ui.welcome_dialog import WelcomeDialog # Added import
from utils.config import ConfigManager

class MainWindow(QMainWindow):
    """Main application window for LlamaCag UI"""
    def __init__(self, config_manager, llama_manager, model_manager, cache_manager,
                 document_processor, chat_engine, n8n_interface):
        """Initialize main window"""
        super().__init__()
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.llama_manager = llama_manager
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.document_processor = document_processor
        # Pass the config_manager instance, not the dict
        self.chat_engine = ChatEngine(self.config_manager, self.llama_manager, self.model_manager, self.cache_manager)
        self.n8n_interface = n8n_interface

        # Set up UI
        self.setup_ui()

        # Connect signals
        self.connect_signals()

        # Update status
        self.update_status()

        # Restore window state
        self.restore_settings()

        # Check for updates
        QTimer.singleShot(5000, self.check_updates)

        # Show welcome dialog if needed (after UI setup and settings restore)
        self.maybe_show_welcome_dialog()

        # Set initial tab to Models (index 0)
        self.tabs.setCurrentIndex(0)

    def setup_ui(self):
        """Set up the user interface"""
        # Set window properties
        self.setWindowTitle("LlamaCag UI - Context-Augmented Generation")
        self.setMinimumSize(1000, 700)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Create tabs
        self.model_tab = ModelTab(self.model_manager, self.config_manager)
        # Pass cache_manager to DocumentTab constructor
        self.document_tab = DocumentTab(self.document_processor, self.model_manager, self.cache_manager, self.config_manager)
        self.chat_tab = ChatTab(self.chat_engine, self.model_manager, self.cache_manager, self.config_manager)
        self.cache_tab = CacheTab(self.cache_manager, self.document_processor, self.config_manager)
        self.settings_tab = SettingsTab(self.config_manager, self.llama_manager, self.n8n_interface, self.model_manager)

        # Add tabs to tab widget
        self.tabs.addTab(self.model_tab, "Models")
        self.tabs.addTab(self.document_tab, "Documents")
        self.tabs.addTab(self.chat_tab, "Chat")
        self.tabs.addTab(self.cache_tab, "KV Cache Monitor")
        self.tabs.addTab(self.settings_tab, "Settings")

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status bar components
        self.status_llama = QLabel("llama.cpp: Checking...")
        self.status_model = QLabel("Model: None")
        self.status_kv_cache = QLabel("KV Cache: None") # Restored
        self.status_n8n = QLabel("n8n: Checking...")
        self.status_chat_engine = QLabel("Chat Engine: Idle") # Label for chat status

        # --- Status Bar Layout Modification ---
        # Container for status labels
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(5, 0, 5, 0) # Adjust margins as needed
        status_layout.setSpacing(10) # Spacing between items

        # Add status labels to the layout
        status_layout.addWidget(self.status_llama)
        status_layout.addWidget(self.status_model)
        status_layout.addWidget(self.status_kv_cache)
        status_layout.addWidget(self.status_n8n)
        status_layout.addWidget(self.status_chat_engine)
        status_layout.addStretch(1) # Push labels to the left

        # Add the container widget to the status bar
        self.status_bar.addWidget(status_widget, 1) # Add with stretch factor 1
        # --- End Status Bar Layout Modification ---

        # Overall status indicator
        self.status_indicator = QPushButton("System Status")
        self.status_indicator.setObjectName("statusIndicator") # Set object name for QSS
        self.status_indicator.setFlat(True)
        # Removed setFixedWidth and setStyleSheet - handled by QSS in style.qss
        self.status_bar.addPermanentWidget(self.status_indicator) # Add as permanent widget

    def connect_signals(self):
        """Connect signals between components"""
        # Model tab signals
        self.model_tab.model_changed.connect(self.on_model_changed)

        # Document tab signals
        self.document_tab.kv_cache_created.connect(self.update_status)
        self.document_tab.kv_cache_created.connect(self.cache_tab.refresh_caches)
        # Connect the new signal from the processor (via the tab)
        self.document_processor.cache_ready_for_use.connect(self.on_cache_ready_for_use)

        # Cache tab signals
        self.cache_tab.cache_selected.connect(self.chat_tab.on_cache_selected)
        self.cache_tab.cache_purged.connect(self.update_status)

        # Chat signals
        self.model_tab.model_changed.connect(self.chat_tab.on_model_changed)
        self.model_tab.model_changed.connect(self.document_tab.on_model_changed)

        # N8n interface signals
        self.n8n_interface.status_changed.connect(self.update_n8n_status)

        # llama.cpp installation signals
        self.llama_manager.installation_progress.connect(self.on_installation_progress)
        self.llama_manager.installation_complete.connect(self.on_installation_complete)

        # Chat engine status signal
        self.chat_engine.status_updated.connect(self.on_chat_status_updated)

        # Settings signals
        self.settings_tab.settings_changed.connect(self.on_settings_changed)

    @pyqtSlot(str) # Slot for chat engine status updates
    def on_chat_status_updated(self, status: str):
        """Update the chat engine status label."""
        self.status_chat_engine.setText(f"Chat Engine: {status}")

    def update_status(self):
        """Update status bar information"""
        # llama.cpp status
        if self.llama_manager.is_installed():
            version = self.llama_manager.get_version()
            self.status_llama.setText(f"llama.cpp: Installed (v{version})")
        else:
            self.status_llama.setText("llama.cpp: Not installed")

        # Current model
        model_id = self.config.get('CURRENT_MODEL_ID')
        if model_id:
            model_info = self.model_manager.get_model_info(model_id)
            if model_info:
                self.status_model.setText(f"Model: {model_info.get('name', model_id)}")
            else:
                self.status_model.setText(f"Model: {model_id}")
        else:
            self.status_model.setText("Model: None")

        # Calculate cache_count for overall status check
        cache_count = len(self.cache_manager.get_cache_list())

        # KV cache status - Restored general status here
        if cache_count > 0:
            cache_size = self.cache_manager.get_total_cache_size()
            size_str = self.format_size(cache_size)
            self.status_kv_cache.setText(f"KV Cache: {cache_count} documents ({size_str})") # Restored
        else:
            self.status_kv_cache.setText("KV Cache: None") # Restored

        # N8n status will be updated by the signal handler

        # Overall status
        if not self.llama_manager.is_installed():
            self.set_status_indicator("error", "llama.cpp not installed")
        elif not model_id:
            self.set_status_indicator("warning", "No model selected")
        elif cache_count == 0:
            self.set_status_indicator("warning", "No KV caches available")
        else:
            self.set_status_indicator("ok", "All systems go")

    def update_n8n_status(self, is_running: bool):
        """Update n8n status in status bar"""
        if is_running:
            self.status_n8n.setText("n8n: Running")
        else:
            self.status_n8n.setText("n8n: Stopped")

    def set_status_indicator(self, status: str, tooltip: str):
        """Set the overall status indicator"""
        # --- Status Indicator Styling Modification ---
        if status == "ok":
            self.status_indicator.setText("All Systems Go")
            # Set dynamic property for QSS styling
            self.status_indicator.setProperty("status", "ok")
        elif status == "warning":
            self.status_indicator.setText("Warning")
            # Set dynamic property for QSS styling
            self.status_indicator.setProperty("status", "warning")
        elif status == "error":
            self.status_indicator.setText("Error")
            # Set dynamic property for QSS styling
            self.status_indicator.setProperty("status", "error")
        else: # Default or unknown status
             self.status_indicator.setProperty("status", "unknown") # Or handle default differently

        # Re-polish the widget to apply property-based styles
        self.style().unpolish(self.status_indicator)
        self.style().polish(self.status_indicator)
        # --- End Status Indicator Styling Modification ---

        self.status_indicator.setToolTip(tooltip)

    def restore_settings(self):
        """Restore window state and settings"""
        settings = QSettings("LlamaCag", "LlamaCagUI")

        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        state = settings.value("windowState")
        if state:
            self.restoreState(state)

        # Restore selected tab - Keep this for subsequent launches
        # The initial tab is set after setup_ui now
        tab_index = settings.value("selectedTab", 0, type=int)
        # Only restore if it's not the very first launch (where geometry might be None)
        if geometry:
            self.tabs.setCurrentIndex(tab_index)

    def save_settings(self):
        """Save window state and settings"""
        settings = QSettings("LlamaCag", "LlamaCagUI")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("selectedTab", self.tabs.currentIndex())

    def check_updates(self):
        """Check for updates to llama.cpp and models"""
        # llama.cpp update check
        try:
            if self.model_manager.check_for_llama_cpp_updates():
                self.show_update_notification()
        except Exception as e:
            logging.error(f"Failed to check for updates: {str(e)}")

    def show_update_notification(self):
        """Show notification about available updates"""
        QMessageBox.information(
            self,
            "Update Available",
            "An update to llama.cpp is available. You can update it in the Settings tab."
        )

    def on_model_changed(self, model_id: str):
        """Handle model change"""
        # Update config
        self.config['CURRENT_MODEL_ID'] = model_id
        self.config_manager.save_config()

        # Update status
        self.update_status()

    def on_settings_changed(self):
        """Handle settings changes"""
        # Reload config
        self.config = self.config_manager.get_config()

        # Update components with new config
        self.llama_manager.update_config(self.config)
        self.model_manager.update_config(self.config)
        self.cache_manager.update_config(self.config_manager) # Pass instance
        self.document_processor.update_config(self.config_manager) if hasattr(self.document_processor, 'update_config') else None # Pass instance
        self.chat_engine.update_config(self.config_manager) if hasattr(self.chat_engine, 'update_config') else None # Pass instance
        self.n8n_interface.update_config(self.config_manager) if hasattr(self.n8n_interface, 'update_config') else None # Pass instance

        # Update status
        self.update_status()

    def on_installation_progress(self, progress: int, message: str):
        """Handle llama.cpp installation progress"""
        self.status_bar.showMessage(f"Installing llama.cpp: {progress}% - {message}")

    def on_installation_complete(self, success: bool, message: str):
        """Handle llama.cpp installation completion"""
        if success:
            self.status_bar.showMessage("llama.cpp installed successfully!")

            QMessageBox.information(
                self,
                "Installation Complete",
                "llama.cpp has been installed successfully. You can now download models and process documents."
            )
        else:
            self.status_bar.showMessage(f"llama.cpp installation failed: {message}")

            QMessageBox.warning(
                self,
                "Installation Failed",
                f"Failed to install llama.cpp: {message}\n\nPlease try again or install manually."
            )

        # Update status after installation
        self.update_status()

    def closeEvent(self, event):
        """Handle window close event"""
        # Save settings
        self.save_settings()

        # Save config
        self.config_manager.save_config()

        # Accept event
        event.accept()

    def format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human-readable string"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def maybe_show_welcome_dialog(self):
        """Checks settings and shows the welcome dialog if required."""
        if WelcomeDialog.should_show():
            # Store reference to prevent garbage collection
            # Store reference to prevent garbage collection
            self.welcome_dialog_instance = WelcomeDialog(self)
            # Show non-modally
            self.welcome_dialog_instance.show()
            # Use QTimer.singleShot to ensure activation happens after event loop processing
            QTimer.singleShot(0, self.welcome_dialog_instance.raise_)
            QTimer.singleShot(0, self.welcome_dialog_instance.activateWindow)

    @pyqtSlot(str)
    def on_cache_ready_for_use(self, cache_path: str):
        """Slot to handle when a cache is created and 'use now' was checked."""
        logging.info(f"Main window received signal to use cache now: {cache_path}")
        # Tell the chat tab to select this cache
        self.chat_tab.on_cache_selected(cache_path)
        # Switch focus to the chat tab (index 2)
        self.tabs.setCurrentIndex(2)
        # Optionally, trigger warm-up automatically? For now, just select it.
        # self.chat_tab.on_warmup_button_clicked()
