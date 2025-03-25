#!/usr/bin/env python3
"""
Cache tab for LlamaCag UI
Provides an interface for managing KV caches.
"""
import os
import sys
import logging # <-- Add this import
from pathlib import Path
import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
    QSplitter, QFrame, QGridLayout, QGroupBox, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal
from core.cache_manager import CacheManager
from core.document_processor import DocumentProcessor
from utils.config import ConfigManager
class CacheTab(QWidget):
    """KV cache management tab"""
    # Signals
    cache_selected = pyqtSignal(str)  # cache_path
    cache_purged = pyqtSignal()
    def __init__(self, cache_manager: CacheManager, document_processor: DocumentProcessor,
                 config_manager: ConfigManager):
        """Initialize cache tab"""
        super().__init__()
        self.cache_manager = cache_manager
        self.document_processor = document_processor
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        # Set up UI
        self.setup_ui()
        # Connect signals
        self.connect_signals()
        # Load caches
        self.refresh_caches()
    def setup_ui(self):
        """Set up the user interface"""
        # Main layout
        layout = QVBoxLayout(self)
        # Header label
        header = QLabel("KV Cache Management")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)
        # Info label
        info_label = QLabel("Manage your KV caches for large context window models.")
        layout.addWidget(info_label)
        # Cache table
        self.cache_table = QTableWidget()
        self.cache_table.setColumnCount(6)
        self.cache_table.setHorizontalHeaderLabels([
            "Cache Name", "Size", "Document", "Model", "Last Used", "Usage Count"
        ])
        self.cache_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.cache_table)
        # Button layout
        button_layout = QHBoxLayout()
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        button_layout.addWidget(self.refresh_button)
        # Purge button
        self.purge_button = QPushButton("Purge Selected")
        button_layout.addWidget(self.purge_button)
        # Purge all button
        self.purge_all_button = QPushButton("Purge All")
        button_layout.addWidget(self.purge_all_button)
        # Use as master button
        self.use_button = QPushButton("Use Selected")
        button_layout.addWidget(self.use_button)
        layout.addLayout(button_layout)
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    def connect_signals(self):
        """Connect signals between components"""
        # Button signals
        self.refresh_button.clicked.connect(self.refresh_caches)
        self.purge_button.clicked.connect(self.purge_selected_cache)
        self.purge_all_button.clicked.connect(self.purge_all_caches)
        self.use_button.clicked.connect(self.use_selected_cache)
        # Table signals
        self.cache_table.itemSelectionChanged.connect(self.on_cache_selected)
        # Cache manager signals
        self.cache_manager.cache_list_updated.connect(self.refresh_caches)
        self.cache_manager.cache_purged.connect(self.on_cache_purged)

    def refresh_caches(self):
        """Refresh the cache list by rescanning the directory and updating the table"""
        try:
            # Explicitly tell CacheManager to rescan the directory and update its internal state
            self.cache_manager.refresh_cache_list()
        except Exception as e:
             logging.error(f"Error refreshing cache list from directory: {e}")
             # Optionally show an error to the user
             QMessageBox.warning(self, "Refresh Error", f"Could not fully refresh cache list:\n{e}")
             # Proceed to display whatever is in the registry anyway

        self.cache_table.setRowCount(0)
        # Get the potentially updated cache list from the manager's registry
        caches = self.cache_manager.get_cache_list()
        # Sort by name
        caches.sort(key=lambda x: x.get('filename', ''))
        # Add to table
        for i, cache in enumerate(caches):
            self.cache_table.insertRow(i)
            # Cache name
            item = QTableWidgetItem(cache.get('filename', 'Unknown'))
            item.setData(Qt.UserRole, cache.get('path', ''))
            self.cache_table.setItem(i, 0, item)
            # Size
            size_bytes = cache.get('size', 0)
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            self.cache_table.setItem(i, 1, QTableWidgetItem(size_str))
            # Document
            self.cache_table.setItem(i, 2, QTableWidgetItem(cache.get('document_id', 'Unknown')))
            # Model
            self.cache_table.setItem(i, 3, QTableWidgetItem(cache.get('model_id', 'Unknown')))
            # Last used
            last_used = cache.get('last_used')
            if last_used:
                last_used_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_used))
            else:
                last_used_str = "Never"
            self.cache_table.setItem(i, 4, QTableWidgetItem(last_used_str))
            # Usage count
            self.cache_table.setItem(i, 5, QTableWidgetItem(str(cache.get('usage_count', 0))))
        # Update status
        total_size = self.cache_manager.get_total_cache_size()
        if total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        else:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        self.status_label.setText(f"{len(caches)} caches, total size: {size_str}")
    def on_cache_selected(self):
        """Handle cache selection change"""
        selected_items = self.cache_table.selectedItems()
        if not selected_items:
            return
        # Get selected row
        row = selected_items[0].row()
        # Get cache path
        cache_path = self.cache_table.item(row, 0).data(Qt.UserRole)
        # Update status
        self.status_label.setText(f"Selected: {cache_path}")
    def purge_selected_cache(self):
        """Purge the selected cache"""
        selected_items = self.cache_table.selectedItems()
        if not selected_items:
            return
        # Get selected row
        row = selected_items[0].row()
        # Get cache path
        cache_path = self.cache_table.item(row, 0).data(Qt.UserRole)
        cache_name = self.cache_table.item(row, 0).text()
        # Confirm
        reply = QMessageBox.question(
            self,
            "Purge Cache",
            f"Are you sure you want to purge the cache for {cache_name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return
        # Purge cache
        success = self.cache_manager.purge_cache(cache_path)
        if success:
            self.status_label.setText(f"Purged cache: {cache_name}")
        else:
            self.status_label.setText(f"Failed to purge cache: {cache_name}")
            QMessageBox.warning(
                self,
                "Purge Failed",
                f"Failed to purge cache {cache_name}."
            )
    def purge_all_caches(self):
        """Purge all caches"""
        # Confirm
        reply = QMessageBox.question(
            self,
            "Purge All Caches",
            "Are you sure you want to purge ALL caches? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.No:
            return
        # Purge all caches
        success = self.cache_manager.purge_all_caches()
        if success:
            self.status_label.setText("Purged all caches")
        else:
            self.status_label.setText("Failed to purge all caches")
            QMessageBox.warning(
                self,
                "Purge Failed",
                "Failed to purge all caches."
            )
    def use_selected_cache(self):
        """Use the selected cache"""
        selected_items = self.cache_table.selectedItems()
        if not selected_items:
            return
        # Get selected row
        row = selected_items[0].row()
        # Get cache path
        cache_path = self.cache_table.item(row, 0).data(Qt.UserRole)
        cache_name = self.cache_table.item(row, 0).text()
        # Emit signal
        self.cache_selected.emit(cache_path)
        # Update status
        self.status_label.setText(f"Using cache: {cache_name}")
    def on_cache_purged(self, cache_path: str, success: bool):
        """Handle cache purged signal"""
        if success:
            self.refresh_caches()
            self.cache_purged.emit()
