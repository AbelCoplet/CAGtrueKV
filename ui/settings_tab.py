#!/usr/bin/env python3
"""
Settings tab for LlamaCag UI
Provides an interface for configuring the application.
"""
import os
import sys
import logging # Added logging
import threading # Added threading
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QFormLayout, QFileDialog,
    QCheckBox, QSpinBox, QMessageBox, QProgressDialog, QComboBox, # Added QComboBox
    QSpacerItem, QSizePolicy # Added QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, QObject, pyqtSlot # Added pyqtSlot
from core.llama_manager import LlamaManager
from core.n8n_interface import N8nInterface
from core.model_manager import ModelManager
from core.cache_manager import CacheManager # Added CacheManager
from core.chat_engine import ChatEngine # Added ChatEngine
from utils.config import ConfigManager


class LlamaCppUpdateWorker(QObject):
    """Worker for updating llama.cpp in a separate thread"""
    update_complete = pyqtSignal(bool, str)  # success, message

    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager

    def run(self):
        """Run the update process"""
        success, message = self.model_manager.update_llama_cpp()
        self.update_complete.emit(success, message)

# Worker for pre-loading model and cache
class PreloadWorker(QObject):
    """Worker for pre-loading model and cache in a separate thread"""
    preload_status = pyqtSignal(str) # Status message
    preload_complete = pyqtSignal(bool, str) # success, message/error

    def __init__(self, chat_engine, model_id, cache_path):
        super().__init__()
        self.chat_engine = chat_engine
        self.model_id = model_id
        self.cache_path = cache_path

    def run(self):
        """Run the pre-load process"""
        try:
            self.preload_status.emit("Loading model...")
            success, message = self.chat_engine.preload_model_and_cache(self.model_id, self.cache_path)
            self.preload_complete.emit(success, message)
        except Exception as e:
            logging.exception("Error during pre-loading worker execution")
            self.preload_complete.emit(False, f"Unexpected error: {str(e)}")


class SettingsTab(QWidget):
    """Settings tab for configuration"""
    # Signals
    settings_changed = pyqtSignal()
    request_preload = pyqtSignal(str, str) # model_id, cache_path
    request_unload = pyqtSignal()

    def __init__(self, config_manager: ConfigManager,
                 llama_manager: LlamaManager, n8n_interface: N8nInterface,
                 model_manager: ModelManager, cache_manager: CacheManager, # Added CacheManager
                 chat_engine: ChatEngine): # Added ChatEngine
        """Initialize settings tab"""
        super().__init__()
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.llama_manager = llama_manager
        self.n8n_interface = n8n_interface
        self.model_manager = model_manager
        self.cache_manager = cache_manager # Store cache manager
        self.chat_engine = chat_engine # Store chat engine

        # Set up UI
        self.setup_ui()

        # Connect signals
        self.connect_signals()

        # Load settings
        self.load_settings()

        # Schedule update check
        QTimer.singleShot(5000, self.check_for_updates)

        # Populate combos initially
        self._populate_preload_combos()

    def setup_ui(self):
        """Set up the user interface"""
        # Main layout
        layout = QVBoxLayout(self)

        # Header label
        header = QLabel("Settings")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)

        # llama.cpp group
        llama_group = QGroupBox("llama.cpp")
        llama_layout = QVBoxLayout(llama_group)

        # Version info
        self.llama_version_label = QLabel("Current Version: Checking...")
        llama_layout.addWidget(self.llama_version_label)

        # Update status
        self.update_status_label = QLabel("Update Status: Checking...")
        llama_layout.addWidget(self.update_status_label)

        # Update button
        update_button_layout = QHBoxLayout()
        self.check_updates_button = QPushButton("Check for Updates")
        update_button_layout.addWidget(self.check_updates_button)

        self.update_button = QPushButton("Update llama.cpp")
        self.update_button.setEnabled(False)
        update_button_layout.addWidget(self.update_button)

        llama_layout.addLayout(update_button_layout)

        layout.addWidget(llama_group)

        # Paths group
        paths_group = QGroupBox("Paths")
        paths_layout = QFormLayout(paths_group)

        # llamacpp_path
        self.llamacpp_path_edit = QLineEdit()
        self.llamacpp_path_button = QPushButton("Browse...")
        path_layout_1 = QHBoxLayout() # Renamed to avoid conflict
        path_layout_1.addWidget(self.llamacpp_path_edit)
        path_layout_1.addWidget(self.llamacpp_path_button)
        paths_layout.addRow("llama.cpp Path:", path_layout_1)

        # models_path
        self.models_path_edit = QLineEdit()
        self.models_path_button = QPushButton("Browse...")
        path_layout_2 = QHBoxLayout() # Renamed to avoid conflict
        path_layout_2.addWidget(self.models_path_edit)
        path_layout_2.addWidget(self.models_path_button)
        paths_layout.addRow("Models Path:", path_layout_2)

        # kv_cache_path
        self.kv_cache_path_edit = QLineEdit()
        self.kv_cache_path_button = QPushButton("Browse...")
        path_layout_3 = QHBoxLayout() # Renamed to avoid conflict
        path_layout_3.addWidget(self.kv_cache_path_edit)
        path_layout_3.addWidget(self.kv_cache_path_button)
        paths_layout.addRow("KV Cache Path:", path_layout_3)

        # temp_path
        self.temp_path_edit = QLineEdit()
        self.temp_path_button = QPushButton("Browse...")
        path_layout_4 = QHBoxLayout() # Renamed to avoid conflict
        path_layout_4.addWidget(self.temp_path_edit)
        path_layout_4.addWidget(self.temp_path_button)
        paths_layout.addRow("Temp Path:", path_layout_4)

        # documents_path
        self.documents_path_edit = QLineEdit()
        self.documents_path_button = QPushButton("Browse...")
        path_layout_5 = QHBoxLayout() # Renamed to avoid conflict
        path_layout_5.addWidget(self.documents_path_edit)
        path_layout_5.addWidget(self.documents_path_button)
        paths_layout.addRow("Documents Path:", path_layout_5)

        layout.addWidget(paths_group)

        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout(model_group)

        # threads
        self.threads_spin = QSpinBox()
        self.threads_spin.setMinimum(1)
        self.threads_spin.setMaximum(os.cpu_count() or 64) # Use actual CPU count if available
        model_layout.addRow("Threads:", self.threads_spin)

        # batch_size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setMinimum(1)
        self.batch_size_spin.setMaximum(4096)
        model_layout.addRow("Batch Size:", self.batch_size_spin)

        # GPU Layers
        gpu_layout = QHBoxLayout()
        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setMinimum(0)
        self.gpu_layers_spin.setMaximum(128) # Increased max
        gpu_layout.addWidget(self.gpu_layers_spin)
        # Placeholder for auto-detect button if needed later
        # self.gpu_detect_button = QPushButton("Auto-detect")
        # gpu_layout.addWidget(self.gpu_detect_button)
        model_layout.addRow("GPU Layers:", gpu_layout)

        layout.addWidget(model_group)

        # --- Pre-load Model & Cache Group ---
        preload_group = QGroupBox("Persistent Model Loading (Experimental)")
        preload_layout = QVBoxLayout(preload_group)

        preload_info_label = QLabel(
            "Enable this to keep a selected model and KV cache loaded in RAM constantly.\n"
            "This significantly speeds up chat responses and external API calls after the initial load,\n"
            "but uses a large amount of RAM continuously (potentially several GB)."
        )
        preload_info_label.setWordWrap(True)
        preload_layout.addWidget(preload_info_label)

        preload_controls_layout = QFormLayout()

        # Enable Checkbox
        self.preload_enabled_checkbox = QCheckBox("Enable Persistent Loading")
        preload_controls_layout.addRow(self.preload_enabled_checkbox)

        # Model Selection
        self.preload_model_combo = QComboBox()
        preload_controls_layout.addRow("Model to Pre-load:", self.preload_model_combo)

        # Cache Selection
        self.preload_cache_combo = QComboBox()
        preload_controls_layout.addRow("KV Cache to Pre-load:", self.preload_cache_combo)

        # Status Label
        self.preload_status_label = QLabel("Status: Idle")
        preload_controls_layout.addRow("Status:", self.preload_status_label)

        # Apply Button
        self.preload_apply_button = QPushButton("Apply & Load / Unload")
        preload_controls_layout.addRow(self.preload_apply_button)

        preload_layout.addLayout(preload_controls_layout)
        layout.addWidget(preload_group)
        # --- End Pre-load Group ---


        # n8n settings group
        n8n_group = QGroupBox("n8n Integration")
        n8n_layout = QFormLayout(n8n_group)

        # n8n_host
        self.n8n_host_edit = QLineEdit()
        n8n_layout.addRow("n8n Host:", self.n8n_host_edit)

        # n8n_port
        self.n8n_port_spin = QSpinBox()
        self.n8n_port_spin.setMinimum(1)
        self.n8n_port_spin.setMaximum(65535)
        n8n_layout.addRow("n8n Port:", self.n8n_port_spin)

        # n8n controls
        n8n_buttons = QHBoxLayout()
        self.n8n_start_button = QPushButton("Start n8n")
        self.n8n_stop_button = QPushButton("Stop n8n")
        self.n8n_status_label = QLabel("n8n Status: Unknown")
        n8n_buttons.addWidget(self.n8n_start_button)
        n8n_buttons.addWidget(self.n8n_stop_button)
        n8n_buttons.addWidget(self.n8n_status_label)
        n8n_layout.addRow("n8n Controls:", n8n_buttons)

        layout.addWidget(n8n_group)

        # Spacer to push buttons down
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Button layout
        button_layout = QHBoxLayout()

        # Save button
        self.save_button = QPushButton("Save Settings")
        button_layout.addWidget(self.save_button)

        # Reset button
        self.reset_button = QPushButton("Reset to Defaults")
        button_layout.addWidget(self.reset_button)

        layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

    def connect_signals(self):
        """Connect signals between components"""
        # Path browser buttons
        self.llamacpp_path_button.clicked.connect(lambda: self.browse_path(self.llamacpp_path_edit, "llama.cpp Path"))
        self.models_path_button.clicked.connect(lambda: self.browse_path(self.models_path_edit, "Models Path"))
        self.kv_cache_path_button.clicked.connect(lambda: self.browse_path(self.kv_cache_path_edit, "KV Cache Path"))
        self.temp_path_button.clicked.connect(lambda: self.browse_path(self.temp_path_edit, "Temp Path"))
        self.documents_path_button.clicked.connect(lambda: self.browse_path(self.documents_path_edit, "Documents Path"))

        # Save/reset buttons
        self.save_button.clicked.connect(self.save_settings)
        self.reset_button.clicked.connect(self.reset_settings)

        # n8n controls
        self.n8n_start_button.clicked.connect(self.start_n8n)
        self.n8n_stop_button.clicked.connect(self.stop_n8n)

        # n8n interface signals
        self.n8n_interface.status_changed.connect(self.update_n8n_status)

        # Update buttons
        self.check_updates_button.clicked.connect(self.check_for_updates)
        self.update_button.clicked.connect(self.update_llama_cpp)

        # Pre-load controls
        self.preload_enabled_checkbox.stateChanged.connect(self._update_preload_controls_state)
        self.preload_apply_button.clicked.connect(self._apply_preload_settings)

        # Connect to ChatEngine preload status updates (assuming ChatEngine will have these signals)
        if hasattr(self.chat_engine, 'preload_status_update'):
             self.chat_engine.preload_status_update.connect(self.on_preload_status_update)
        if hasattr(self.chat_engine, 'preload_finished'):
             self.chat_engine.preload_finished.connect(self.on_preload_finished)

        # Connect internal signals for preload worker
        # self.request_preload.connect(self.chat_engine.handle_preload_request) # Connect in main window or controller
        # self.request_unload.connect(self.chat_engine.handle_unload_request) # Connect in main window or controller


    def load_settings(self):
        """Load settings from config"""
        # Paths
        self.llamacpp_path_edit.setText(os.path.expanduser(self.config.get('LLAMACPP_PATH', '~/Documents/llama.cpp')))
        self.models_path_edit.setText(os.path.expanduser(self.config.get('LLAMACPP_MODEL_DIR', '~/Documents/llama.cpp/models'))) # Corrected key
        self.kv_cache_path_edit.setText(os.path.expanduser(self.config.get('KV_CACHE_DIR', '~/cag_project/kv_caches'))) # Corrected key
        self.temp_path_edit.setText(os.path.expanduser(self.config.get('TEMP_DIR', '~/cag_project/temp_chunks'))) # Corrected key
        # self.documents_path_edit.setText(os.path.expanduser(self.config.get('DOCUMENTS_FOLDER', '~/Documents/cag_documents'))) # This seems less relevant now

        # Model settings
        self.threads_spin.setValue(int(self.config.get('LLAMACPP_THREADS', os.cpu_count() or 4))) # Use CPU count default
        self.batch_size_spin.setValue(int(self.config.get('LLAMACPP_BATCH_SIZE', '512'))) # Default 512
        self.gpu_layers_spin.setValue(int(self.config.get('LLAMACPP_GPU_LAYERS', '0'))) # Load GPU layers

        # n8n settings
        self.n8n_host_edit.setText(self.config.get('N8N_HOST', 'localhost'))
        self.n8n_port_spin.setValue(int(self.config.get('N8N_PORT', '5678')))

        # Pre-load settings
        preload_enabled = self.config.get('PRELOAD_ENABLED', False)
        self.preload_enabled_checkbox.setChecked(preload_enabled)
        self._update_preload_controls_state() # Update enabled state of combos/button

        # Set combo values AFTER populating them
        QTimer.singleShot(100, self._set_initial_preload_combos) # Delay slightly

        # Update n8n status
        self.update_n8n_status(self.n8n_interface.is_running())

        # Update llama.cpp version
        self.update_llama_version()

    def _set_initial_preload_combos(self):
        """Set combo box values after they have been populated."""
        preload_model_id = self.config.get('PRELOAD_MODEL_ID')
        preload_cache_path = self.config.get('PRELOAD_CACHE_PATH')

        if preload_model_id:
            index = self.preload_model_combo.findData(preload_model_id)
            if index >= 0:
                self.preload_model_combo.setCurrentIndex(index)

        if preload_cache_path:
            index = self.preload_cache_combo.findData(preload_cache_path)
            if index >= 0:
                self.preload_cache_combo.setCurrentIndex(index)

        # Update status based on initial config (might need signal from chat engine later)
        if self.preload_enabled_checkbox.isChecked():
             # Assume idle until chat engine confirms loading status
             self.preload_status_label.setText("Status: Enabled (Load on Apply/Startup)")
        else:
             self.preload_status_label.setText("Status: Disabled")


    def browse_path(self, line_edit, title):
        """Browse for a directory path"""
        current_path = os.path.expanduser(line_edit.text())
        path = QFileDialog.getExistingDirectory(
            self, f"Select {title}", current_path
        )
        if path:
            line_edit.setText(path)

    def save_settings(self):
        """Save settings to config"""
        # Paths
        self.config['LLAMACPP_PATH'] = self.llamacpp_path_edit.text()
        self.config['LLAMACPP_MODEL_DIR'] = self.models_path_edit.text() # Corrected key
        self.config['KV_CACHE_DIR'] = self.kv_cache_path_edit.text() # Corrected key
        self.config['TEMP_DIR'] = self.temp_path_edit.text() # Corrected key
        # self.config['DOCUMENTS_FOLDER'] = self.documents_path_edit.text()

        # Model settings
        self.config['LLAMACPP_THREADS'] = str(self.threads_spin.value())
        self.config['LLAMACPP_BATCH_SIZE'] = str(self.batch_size_spin.value())
        self.config['LLAMACPP_GPU_LAYERS'] = str(self.gpu_layers_spin.value()) # Save GPU layers

        # n8n settings
        self.config['N8N_HOST'] = self.n8n_host_edit.text()
        self.config['N8N_PORT'] = str(self.n8n_port_spin.value())

        # Pre-load settings
        self.config['PRELOAD_ENABLED'] = self.preload_enabled_checkbox.isChecked()
        self.config['PRELOAD_MODEL_ID'] = self.preload_model_combo.currentData()
        self.config['PRELOAD_CACHE_PATH'] = self.preload_cache_combo.currentData()

        # Save config
        self.config_manager.save_config()

        # Update status
        self.status_label.setText("Settings saved")

        # Emit signal
        self.settings_changed.emit()

    def reset_settings(self):
        """Reset settings to defaults"""
        # Confirm
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        # Set defaults (Consider defining defaults centrally)
        self.llamacpp_path_edit.setText('~/Documents/llama.cpp')
        self.models_path_edit.setText('~/Documents/llama.cpp/models')
        self.kv_cache_path_edit.setText('~/cag_project/kv_caches')
        self.temp_path_edit.setText('~/cag_project/temp_chunks')
        # self.documents_path_edit.setText('~/Documents/cag_documents')
        self.threads_spin.setValue(os.cpu_count() or 4)
        self.batch_size_spin.setValue(512)
        self.gpu_layers_spin.setValue(0)
        self.n8n_host_edit.setText('localhost')
        self.n8n_port_spin.setValue(5678)
        self.preload_enabled_checkbox.setChecked(False)
        self.preload_model_combo.setCurrentIndex(0)
        self.preload_cache_combo.setCurrentIndex(0)


        # Update status
        self.status_label.setText("Settings reset to defaults (not saved)")

    def start_n8n(self):
        """Start n8n services"""
        success = self.n8n_interface.start_services()
        if success:
            self.status_label.setText("n8n services started")
        else:
            self.status_label.setText("Failed to start n8n services")
            QMessageBox.warning(
                self,
                "n8n Start Failed",
                "Failed to start n8n services. Check the logs for details."
            )

    def stop_n8n(self):
        """Stop n8n services"""
        success = self.n8n_interface.stop_services()
        if success:
            self.status_label.setText("n8n services stopped")
        else:
            self.status_label.setText("Failed to stop n8n services")
            QMessageBox.warning(
                self,
                "n8n Stop Failed",
                "Failed to stop n8n services. Check the logs for details."
            )

    def update_n8n_status(self, is_running: bool):
        """Update n8n status display"""
        if is_running:
            self.n8n_status_label.setText("n8n Status: Running")
            self.n8n_status_label.setStyleSheet("color: green;")
            self.n8n_start_button.setEnabled(False)
            self.n8n_stop_button.setEnabled(True)
        else:
            self.n8n_status_label.setText("n8n Status: Stopped")
            self.n8n_status_label.setStyleSheet("color: red;")
            self.n8n_start_button.setEnabled(True)
            self.n8n_stop_button.setEnabled(False)

    def update_llama_version(self):
        """Update llama.cpp version display"""
        if self.llama_manager.is_installed():
            version = self.llama_manager.get_version()
            self.llama_version_label.setText(f"Current Version: {version}")
        else:
            self.llama_version_label.setText("Current Version: Not installed")

    def check_for_updates(self):
        """Check for updates to llama.cpp"""
        self.update_status_label.setText("Update Status: Checking...")

        if not self.llama_manager.is_installed():
            self.update_status_label.setText("Update Status: llama.cpp not installed")
            self.update_button.setEnabled(False)
            return

        try:
            update_available = self.model_manager.check_for_llama_cpp_updates()
            if update_available:
                self.update_status_label.setText("Update Status: Update available")
                self.update_status_label.setStyleSheet("color: green; font-weight: bold;")
                self.update_button.setEnabled(True)
            else:
                self.update_status_label.setText("Update Status: Up to date")
                self.update_status_label.setStyleSheet("") # Reset style
                self.update_button.setEnabled(False)
        except Exception as e:
             logging.error(f"Failed to check llama.cpp updates: {e}")
             self.update_status_label.setText("Update Status: Check failed")
             self.update_status_label.setStyleSheet("color: orange;")
             self.update_button.setEnabled(False)


    def update_llama_cpp(self):
        """Update llama.cpp to the latest version"""
        # Confirm
        reply = QMessageBox.question(
            self,
            "Update llama.cpp",
            "Are you sure you want to update llama.cpp to the latest version?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        # Create progress dialog
        progress = QProgressDialog("Updating llama.cpp...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Update in Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        # Create worker thread
        self.update_worker = LlamaCppUpdateWorker(self.model_manager) # Renamed worker instance
        self.update_thread = QThread() # Renamed thread instance
        self.update_worker.moveToThread(self.update_thread)

        # Connect signals
        self.update_thread.started.connect(self.update_worker.run)
        self.update_worker.update_complete.connect(self.on_update_complete)
        self.update_worker.update_complete.connect(self.update_thread.quit)
        self.update_worker.update_complete.connect(self.update_worker.deleteLater)
        self.update_thread.finished.connect(self.update_thread.deleteLater)

        # Connect to close progress dialog
        self.update_worker.update_complete.connect(progress.close)

        # Start the thread
        self.update_thread.start()

    def on_update_complete(self, success: bool, message: str):
        """Handle update completion"""
        if success:
            self.status_label.setText("llama.cpp updated successfully")
            QMessageBox.information(
                self,
                "Update Complete",
                "llama.cpp has been updated successfully."
            )
        else:
            self.status_label.setText(f"Failed to update llama.cpp: {message}")
            QMessageBox.warning(
                self,
                "Update Failed",
                f"Failed to update llama.cpp: {message}"
            )

        # Update version display
        self.update_llama_version()

        # Check for updates again
        self.check_for_updates()

    # --- Pre-load Methods ---
    def _populate_preload_combos(self):
        """Populate model and cache combo boxes."""
        # Models
        self.preload_model_combo.clear()
        self.preload_model_combo.addItem("Select Model...", None)
        models = self.model_manager.get_available_models()
        if not models:
             self.preload_model_combo.setEnabled(False)
        else:
             self.preload_model_combo.setEnabled(True)
             for model in sorted(models, key=lambda m: m.get('name', '')):
                 self.preload_model_combo.addItem(f"{model['name']} ({model['filename']})", model['id'])

        # Caches
        self.preload_cache_combo.clear()
        self.preload_cache_combo.addItem("Select KV Cache...", None)
        caches = self.cache_manager.get_cache_list()
        if not caches:
             self.preload_cache_combo.setEnabled(False)
        else:
             self.preload_cache_combo.setEnabled(True)
             for cache in sorted(caches, key=lambda c: c.get('name', '')):
                 # Display name and maybe original document filename
                 display_name = cache.get('name', Path(cache['path']).stem)
                 orig_doc = cache.get('original_document')
                 if orig_doc and orig_doc != "Unknown":
                      display_name += f" ({Path(orig_doc).name})"
                 self.preload_cache_combo.addItem(display_name, cache['path'])

        self._update_preload_controls_state() # Re-evaluate apply button state

    def _update_preload_controls_state(self):
        """Enable/disable preload controls based on checkbox."""
        enabled = self.preload_enabled_checkbox.isChecked()
        self.preload_model_combo.setEnabled(enabled and self.preload_model_combo.count() > 1) # Check if models exist
        self.preload_cache_combo.setEnabled(enabled and self.preload_cache_combo.count() > 1) # Check if caches exist
        # Apply button enabled only if checkbox is checked AND a model AND cache are selected
        can_apply = (enabled and
                     self.preload_model_combo.currentIndex() > 0 and
                     self.preload_cache_combo.currentIndex() > 0) or \
                    (not enabled and self.chat_engine.is_preloaded()) # Allow unload if enabled=False and model is loaded

        self.preload_apply_button.setEnabled(can_apply)

        # Also connect state change to combo boxes
        self.preload_model_combo.currentIndexChanged.connect(self._update_preload_controls_state)
        self.preload_cache_combo.currentIndexChanged.connect(self._update_preload_controls_state)


    def _apply_preload_settings(self):
        """Handle Apply & Load / Unload button click."""
        is_enabled = self.preload_enabled_checkbox.isChecked()

        if is_enabled:
            model_id = self.preload_model_combo.currentData()
            cache_path = self.preload_cache_combo.currentData()

            if not model_id or not cache_path:
                QMessageBox.warning(self, "Selection Missing", "Please select both a model and a KV cache to pre-load.")
                return

            logging.info(f"Requesting pre-load: Model={model_id}, Cache={cache_path}")
            self.preload_status_label.setText("Status: Loading...")
            self.preload_apply_button.setEnabled(False) # Disable while loading
            self.preload_enabled_checkbox.setEnabled(False)
            self.preload_model_combo.setEnabled(False)
            self.preload_cache_combo.setEnabled(False)

            # Use worker thread
            self.preload_worker = PreloadWorker(self.chat_engine, model_id, cache_path)
            self.preload_thread = QThread()
            self.preload_worker.moveToThread(self.preload_thread)

            self.preload_thread.started.connect(self.preload_worker.run)
            self.preload_worker.preload_status.connect(self.on_preload_status_update)
            self.preload_worker.preload_complete.connect(self.on_preload_finished)
            self.preload_worker.preload_complete.connect(self.preload_thread.quit)
            self.preload_worker.preload_complete.connect(self.preload_worker.deleteLater)
            self.preload_thread.finished.connect(self.preload_thread.deleteLater)

            self.preload_thread.start()

        else:
            # Request unload if currently loaded
            if self.chat_engine.is_preloaded():
                 logging.info("Requesting unload of persistent model.")
                 self.preload_status_label.setText("Status: Unloading...")
                 self.preload_apply_button.setEnabled(False) # Disable while unloading
                 self.preload_enabled_checkbox.setEnabled(False)
                 # No worker needed for unload? Assume it's fast enough for now.
                 self.chat_engine.unload_persistent_model()
                 self.on_preload_finished(True, "Model unloaded.") # Manually trigger finish state
            else:
                 # Just saving the disabled state
                 self.save_settings()
                 self.preload_status_label.setText("Status: Disabled")
                 self._update_preload_controls_state()


    @pyqtSlot(str)
    def on_preload_status_update(self, status_message: str):
        """Update the status label during pre-loading."""
        self.preload_status_label.setText(f"Status: {status_message}")

    @pyqtSlot(bool, str)
    def on_preload_finished(self, success: bool, message: str):
        """Handle completion of pre-loading or unloading."""
        if success:
            if "unloaded" in message.lower():
                 self.preload_status_label.setText("Status: Disabled (Unloaded)")
            else:
                 model_name = self.preload_model_combo.currentText().split('(')[0].strip()
                 cache_name = self.preload_cache_combo.currentText().split('(')[0].strip()
                 self.preload_status_label.setText(f"Status: Loaded ({model_name} / {cache_name})")
            self.preload_status_label.setStyleSheet("color: green;")
            # Save successful state
            self.save_settings()
        else:
            self.preload_status_label.setText(f"Status: Error - {message}")
            self.preload_status_label.setStyleSheet("color: red;")
            QMessageBox.warning(self, "Pre-load Failed", f"Failed to load model/cache: {message}")
            # Revert checkbox if loading failed? Or just show error? Show error for now.
            # self.preload_enabled_checkbox.setChecked(False)

        # Re-enable controls
        self.preload_apply_button.setEnabled(True)
        self.preload_enabled_checkbox.setEnabled(True)
        self._update_preload_controls_state() # Re-evaluate enabled state based on checkbox
