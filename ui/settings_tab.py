#!/usr/bin/env python3
"""
Settings tab for LlamaCag UI
Provides an interface for configuring the application.
"""
import os
import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QFormLayout, QFileDialog,
    QCheckBox, QSpinBox, QMessageBox, QProgressDialog,
    QSlider, QComboBox # Added for Metal settings
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, QObject
from core.llama_manager import LlamaManager
from core.n8n_interface import N8nInterface
from core.model_manager import ModelManager
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


class SettingsTab(QWidget):
    """Settings tab for configuration"""
    # Signals
    settings_changed = pyqtSignal()
    
    def __init__(self, config_manager: ConfigManager,
                 llama_manager: LlamaManager, n8n_interface: N8nInterface,
                 model_manager: ModelManager):
        """Initialize settings tab"""
        super().__init__()
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.llama_manager = llama_manager
        self.n8n_interface = n8n_interface
        self.model_manager = model_manager
        
        # Set up UI
        self.setup_ui()
        
        # Connect signals
        self.connect_signals()
        
        # Load settings
        self.load_settings()
        
        # Schedule update check
        QTimer.singleShot(5000, self.check_for_updates)
        
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
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.llamacpp_path_edit)
        path_layout.addWidget(self.llamacpp_path_button)
        paths_layout.addRow("llama.cpp Path:", path_layout)
        
        # models_path
        self.models_path_edit = QLineEdit()
        self.models_path_button = QPushButton("Browse...")
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.models_path_edit)
        path_layout.addWidget(self.models_path_button)
        paths_layout.addRow("Models Path:", path_layout)
        
        # kv_cache_path
        self.kv_cache_path_edit = QLineEdit()
        self.kv_cache_path_button = QPushButton("Browse...")
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.kv_cache_path_edit)
        path_layout.addWidget(self.kv_cache_path_button)
        paths_layout.addRow("KV Cache Path:", path_layout)
        
        # temp_path
        self.temp_path_edit = QLineEdit()
        self.temp_path_button = QPushButton("Browse...")
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.temp_path_edit)
        path_layout.addWidget(self.temp_path_button)
        paths_layout.addRow("Temp Path:", path_layout)
        
        # documents_path
        self.documents_path_edit = QLineEdit()
        self.documents_path_button = QPushButton("Browse...")
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.documents_path_edit)
        path_layout.addWidget(self.documents_path_button)
        paths_layout.addRow("Documents Path:", path_layout)
        
        layout.addWidget(paths_group)
        
        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout(model_group)
        
        # threads
        self.threads_spin = QSpinBox()
        self.threads_spin.setMinimum(1)
        self.threads_spin.setMaximum(64)
        model_layout.addRow("Threads:", self.threads_spin)
        
        # batch_size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setMinimum(1)
        self.batch_size_spin.setMaximum(4096)
        self.batch_size_spin.setToolTip("Number of tokens processed in parallel during KV cache creation. Higher values can be faster but use more RAM.")
        model_layout.addRow("Batch Size:", self.batch_size_spin)

        # Add GPU Layers setting
        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setMinimum(0) # 0 means CPU only
        self.gpu_layers_spin.setMaximum(100) # Allow up to 100 layers, user needs to check memory
        self.gpu_layers_spin.setToolTip(
            "Number of model layers to offload to the GPU (Metal on macOS).\n"
            "Increases speed significantly but uses more RAM/VRAM.\n"
            "Start with 15-20 for 4B models, 10-15 for 8B models on M4 Pro 24GB.\n"
            "Increase cautiously while monitoring memory usage in Activity Monitor.\n"
            "Set to 0 to use CPU only."
        )
        model_layout.addRow("GPU Layers:", self.gpu_layers_spin)

        # Add Troubleshooting section with Analyze button
        self.analyze_button = QPushButton("Analyze Model Token Patterns")
        self.analyze_button.setToolTip("Run a test generation to analyze token output patterns (useful for debugging).")
        model_layout.addRow("Troubleshooting:", self.analyze_button)

        layout.addWidget(model_group)

        # GPU-specific settings for Apple Silicon (Metal)
        if sys.platform == 'darwin':
            metal_group = QGroupBox("Metal Acceleration (Apple Silicon)")
            metal_layout = QFormLayout(metal_group)

            self.metal_enabled_checkbox = QCheckBox("Enable Metal Acceleration")
            self.metal_enabled_checkbox.setChecked(True) # Default to enabled
            metal_layout.addRow("Metal:", self.metal_enabled_checkbox)

            # Memory allocation slider
            self.metal_memory_slider = QSlider(Qt.Horizontal)
            self.metal_memory_slider.setRange(1024, 16384)  # 1GB to 16GB (adjust max based on typical systems)
            self.metal_memory_slider.setValue(4096)        # Default 4GB
            self.metal_memory_slider.setTickPosition(QSlider.TicksBelow)
            self.metal_memory_slider.setTickInterval(1024)
            self.metal_memory_slider.setSingleStep(512) # Step by 512MB

            memory_layout = QHBoxLayout()
            memory_layout.addWidget(self.metal_memory_slider)
            self.memory_label = QLabel("4096 MB")
            memory_layout.addWidget(self.memory_label)

            metal_layout.addRow("Metal Memory:", memory_layout)

            # Add performance profile selection
            self.metal_profile_combo = QComboBox()
            self.metal_profile_combo.addItem("Balanced", "balanced")
            self.metal_profile_combo.addItem("Performance", "performance")
            self.metal_profile_combo.addItem("Efficiency", "efficiency")
            metal_layout.addRow("Performance Profile:", self.metal_profile_combo)

            # Add detect button
            self.detect_metal_button = QPushButton("Detect Optimal Settings")
            metal_layout.addRow("Auto-Configure:", self.detect_metal_button)

            layout.addWidget(metal_group)

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

        # Connect Analyze button
        self.analyze_button.clicked.connect(self.analyze_model_output_patterns)

        # Connect Metal UI elements if they exist (macOS only)
        if sys.platform == 'darwin' and hasattr(self, 'metal_memory_slider'):
            # Connect slider value change to update label
            self.metal_memory_slider.valueChanged.connect(
                lambda v: self.memory_label.setText(f"{v} MB")
            )
            # Connect detect button
            self.detect_metal_button.clicked.connect(self.detect_optimal_metal_settings)

    def load_settings(self):
        """Load settings from config"""
        # Paths
        self.llamacpp_path_edit.setText(os.path.expanduser(self.config.get('LLAMACPP_PATH', '~/Documents/llama.cpp')))
        self.models_path_edit.setText(os.path.expanduser(self.config.get('LLAMACPP_MODEL_PATH', '~/Documents/llama.cpp/models')))
        self.kv_cache_path_edit.setText(os.path.expanduser(self.config.get('LLAMACPP_KV_CACHE_DIR', '~/cag_project/kv_caches')))
        self.temp_path_edit.setText(os.path.expanduser(self.config.get('LLAMACPP_TEMP_DIR', '~/cag_project/temp_chunks')))
        self.documents_path_edit.setText(os.path.expanduser(self.config.get('DOCUMENTS_FOLDER', '~/Documents/cag_documents')))
        
        # Model settings
        self.threads_spin.setValue(int(self.config.get('LLAMACPP_THREADS', os.cpu_count() // 2 or 4))) # Default to half cores or 4
        self.batch_size_spin.setValue(int(self.config.get('LLAMACPP_BATCH_SIZE', '512'))) # Default 512
        self.gpu_layers_spin.setValue(int(self.config.get('LLAMACPP_GPU_LAYERS', '0'))) # Default 0 (CPU)
        
        # n8n settings
        self.n8n_host_edit.setText(self.config.get('N8N_HOST', 'localhost'))
        self.n8n_port_spin.setValue(int(self.config.get('N8N_PORT', '5678')))

        # Load Metal settings if on macOS
        if sys.platform == 'darwin' and hasattr(self, 'metal_enabled_checkbox'):
            self.metal_enabled_checkbox.setChecked(self.config.get('METAL_ENABLED', True))
            self.metal_memory_slider.setValue(int(self.config.get('METAL_MEMORY_MB', 4096)))
            profile_data = self.config.get('METAL_PROFILE', 'balanced')
            profile_index = self.metal_profile_combo.findData(profile_data)
            if profile_index >= 0:
                self.metal_profile_combo.setCurrentIndex(profile_index)
            else:
                self.metal_profile_combo.setCurrentIndex(0) # Default to Balanced if not found
            # Update memory label based on loaded slider value
            self.memory_label.setText(f"{self.metal_memory_slider.value()} MB")

        # Update n8n status
        self.update_n8n_status(self.n8n_interface.is_running())
        
        # Update llama.cpp version
        self.update_llama_version()
        
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
        self.config['LLAMACPP_MODEL_PATH'] = self.models_path_edit.text()
        self.config['LLAMACPP_KV_CACHE_DIR'] = self.kv_cache_path_edit.text()
        self.config['LLAMACPP_TEMP_DIR'] = self.temp_path_edit.text()
        self.config['DOCUMENTS_FOLDER'] = self.documents_path_edit.text()
        
        # Model settings
        self.config['LLAMACPP_THREADS'] = str(self.threads_spin.value())
        self.config['LLAMACPP_BATCH_SIZE'] = str(self.batch_size_spin.value())
        self.config['LLAMACPP_GPU_LAYERS'] = str(self.gpu_layers_spin.value()) # Save GPU layers
        
        # n8n settings
        self.config['N8N_HOST'] = self.n8n_host_edit.text()
        self.config['N8N_PORT'] = str(self.n8n_port_spin.value())

        # Save Metal settings if on macOS
        if sys.platform == 'darwin' and hasattr(self, 'metal_enabled_checkbox'):
            self.config['METAL_ENABLED'] = self.metal_enabled_checkbox.isChecked()
            self.config['METAL_MEMORY_MB'] = self.metal_memory_slider.value()
            self.config['METAL_PROFILE'] = self.metal_profile_combo.currentData()
            # Mark that Metal config has been initialized/saved at least once
            self.config['METAL_CONFIG_INITIALIZED'] = True

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
            
        # Set defaults
        self.llamacpp_path_edit.setText('~/Documents/llama.cpp')
        self.models_path_edit.setText('~/Documents/llama.cpp/models')
        self.kv_cache_path_edit.setText('~/cag_project/kv_caches')
        self.temp_path_edit.setText('~/cag_project/temp_chunks')
        self.documents_path_edit.setText('~/Documents/cag_documents')
        self.threads_spin.setValue(os.cpu_count() // 2 or 4) # Default to half cores or 4
        self.batch_size_spin.setValue(512) # Default 512
        self.gpu_layers_spin.setValue(0) # Default 0 (CPU)
        self.n8n_host_edit.setText('localhost')
        self.n8n_port_spin.setValue(5678)

        # Reset Metal settings if on macOS
        if sys.platform == 'darwin' and hasattr(self, 'metal_enabled_checkbox'):
            self.metal_enabled_checkbox.setChecked(True)
            self.metal_memory_slider.setValue(4096)
            self.metal_profile_combo.setCurrentIndex(0) # Balanced
            self.memory_label.setText("4096 MB")

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
            
        if self.model_manager.check_for_llama_cpp_updates():
            self.update_status_label.setText("Update Status: Update available")
            self.update_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.update_button.setEnabled(True)
        else:
            self.update_status_label.setText("Update Status: Up to date")
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
        self.worker = LlamaCppUpdateWorker(self.model_manager)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.update_complete.connect(self.on_update_complete)
        self.worker.update_complete.connect(self.thread.quit)
        self.worker.update_complete.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        # Connect to close progress dialog
        self.worker.update_complete.connect(progress.close)
        
        # Start the thread
        self.thread.start()
        
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

    def analyze_model_output_patterns(self):
        """Placeholder: Run a token pattern analysis to identify potential issues"""
        # TODO: Implement actual analysis logic, potentially in a separate thread
        # For now, just show a message box
        QMessageBox.information(
            self,
            "Analyze Model Output",
            "This feature is not yet fully implemented.\n"
            "It will eventually run a test generation to analyze token patterns."
        )
        # Example of how to start analysis in a thread (when implemented):
        # progress = QProgressDialog("Analyzing model output patterns...", "Cancel", 0, 0, self)
        # progress.setWindowTitle("Model Analysis")
        # progress.setWindowModality(Qt.WindowModal)
        # progress.show()
        # # Create and start worker thread...

    # --- Metal Auto-Configuration ---
    def detect_optimal_metal_settings(self):
        """Detect optimal Metal settings based on hardware (macOS only)"""
        if sys.platform != 'darwin':
            QMessageBox.information(self, "Metal Detection", "Metal detection is only available on macOS.")
            return

        try:
            # Get Metal capabilities from llama_manager
            capabilities = self.llama_manager.detect_metal_capabilities()

            if not capabilities.get("supported", False):
                QMessageBox.information(
                    self, "Metal Detection",
                    f"Metal acceleration not supported or detection failed: {capabilities.get('reason', 'Unknown reason')}"
                )
                return

            # --- Calculate optimal values ---
            gpu_cores = capabilities.get("gpu_cores", 0)
            gpu_model = capabilities.get("gpu_model", "Unknown")
            feature_set = capabilities.get("feature_set", "Unknown")
            get_rec_layers_func = capabilities.get("get_recommended_layers")

            # Set GPU layers based on cores and model (if function available)
            optimal_layers = 0
            current_model_id = self.config_manager.get('CURRENT_MODEL_ID', '') # Get current model ID
            if get_rec_layers_func and current_model_id:
                 optimal_layers = get_rec_layers_func(current_model_id, gpu_cores)
                 logging.info(f"Recommended GPU layers for model '{current_model_id}' with {gpu_cores} cores: {optimal_layers}")
            elif gpu_cores > 0:
                 # Fallback based only on cores if function or model ID missing
                 optimal_layers = min(gpu_cores, 12) # Conservative fallback
                 logging.warning(f"Using fallback GPU layer calculation based on cores: {optimal_layers}")
            else:
                 optimal_layers = 0 # Default to 0 if no cores detected
                 logging.warning("Could not detect GPU cores, defaulting GPU layers to 0.")

            self.gpu_layers_spin.setValue(optimal_layers)

            # Set memory allocation based on system memory
            metal_memory_mb = 4096 # Default
            try:
                import subprocess
                # Get total physical memory using sysctl
                memory_info = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True, stderr=subprocess.PIPE).strip()
                total_memory_bytes = int(memory_info)
                total_memory_gb = total_memory_bytes / (1024**3)
                logging.info(f"Detected total system memory: {total_memory_gb:.2f} GB")

                # Allocate ~1/3 of system memory to Metal, capped between 2GB and 12GB
                calculated_mem_mb = int(total_memory_bytes / (1024**2) / 3)
                metal_memory_mb = max(2048, min(calculated_mem_mb, 12288)) # Clamp between 2GB and 12GB
                logging.info(f"Calculated optimal Metal memory: {metal_memory_mb} MB")

            except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e_mem:
                logging.error(f"Failed to detect system memory via sysctl: {e_mem}. Using default Metal memory (4096 MB).")
                metal_memory_mb = 4096 # Fallback default

            self.metal_memory_slider.setValue(metal_memory_mb)
            self.memory_label.setText(f"{metal_memory_mb} MB") # Update label immediately

            # Update profile based on device type (heuristic)
            # Assume 'mini' might benefit from 'Performance' due to active cooling
            # Assume MacBooks might prefer 'Balanced' or 'Efficiency'
            # This is a rough guess and might need refinement
            profile_index = 0 # Default to Balanced
            if 'mini' in gpu_model.lower():
                 profile_index = 1 # Performance
            elif 'macbook' in gpu_model.lower():
                 profile_index = 0 # Balanced
            # Add more heuristics if needed

            self.metal_profile_combo.setCurrentIndex(profile_index)

            # --- Show results ---
            QMessageBox.information(
                self, "Metal Detection Complete",
                f"Applied suggested Metal settings based on detected hardware:\n\n"
                f"- GPU Model: {gpu_model}\n"
                f"- GPU Cores: {gpu_cores if gpu_cores > 0 else 'Not Detected'}\n"
                f"- Metal Feature Set: {feature_set}\n\n"
                f"- Suggested GPU Layers: {optimal_layers} (for current model: {current_model_id or 'None Selected'})\n"
                f"- Suggested Metal Memory: {metal_memory_mb} MB\n"
                f"- Suggested Profile: {self.metal_profile_combo.currentText()}\n\n"
                f"Recommended model formats: {', '.join(capabilities.get('recommended_formats', ['N/A']))}\n\n"
                f"Please review and save these settings if they look correct."
            )
        except Exception as e:
            logging.error("Error during optimal Metal settings detection", exc_info=True)
            QMessageBox.warning(
                self, "Metal Detection Error",
                f"Failed to detect optimal Metal settings: {str(e)}"
            )
