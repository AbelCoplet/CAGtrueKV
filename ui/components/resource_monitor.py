#!/usr/bin/env python3
"""
Resource Monitor Widget for LlamaCag UI status bar.
Displays current application RAM usage.
"""
import os
import logging
import psutil
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt

class ResourceMonitor(QWidget):
    """A widget to display application RAM usage."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = psutil.Process(os.getpid())
        self.setup_ui()
        self.start_timer()

    def setup_ui(self):
        """Set up the UI elements."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 5, 0) # Add some right margin

        self.ram_label = QLabel("App RAM: --- MB")
        self.ram_label.setToolTip(
            "Total RAM used by the LlamaCag UI process.\n"
            "Includes Python, UI, and any loaded model/cache."
        )
        layout.addWidget(self.ram_label)

        # Initial update
        self.update_ram_usage()

    def start_timer(self):
        """Start the timer for periodic updates."""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ram_usage)
        self.timer.start(3000) # Update every 3 seconds

    def update_ram_usage(self):
        """Fetch and display the current RAM usage."""
        try:
            # Get Resident Set Size (RSS) which is a good measure of actual RAM usage
            memory_info = self.process.memory_info()
            rss_bytes = memory_info.rss
            rss_mb = rss_bytes / (1024 * 1024)
            self.ram_label.setText(f"App RAM: {rss_mb:.0f} MB")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logging.warning(f"Could not get process memory info: {e}")
            self.ram_label.setText("App RAM: Error")
            self.timer.stop() # Stop timer if process is gone
        except Exception as e:
            logging.error(f"Unexpected error updating RAM usage: {e}")
            self.ram_label.setText("App RAM: Error")

    def stop_monitor(self):
        """Stop the update timer."""
        if hasattr(self, 'timer'):
            self.timer.stop()

# Example usage (optional)
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QStatusBar
    app = QApplication(sys.argv)
    window = QMainWindow()
    status_bar = QStatusBar()
    window.setStatusBar(status_bar)
    monitor = ResourceMonitor()
    status_bar.addPermanentWidget(monitor)
    window.setWindowTitle("Resource Monitor Test")
    window.setGeometry(100, 100, 300, 100)
    window.show()
    sys.exit(app.exec_())
