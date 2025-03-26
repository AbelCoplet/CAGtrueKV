#!/usr/bin/env python3
"""
Simplest possible cache_tab.py for LlamaCag UI
"""
import os
import sys
import time
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QMessageBox, QTableWidget,
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
        
        # Cache table
        self.cache_table = QTableWidget()
        self.cache_table.setColumnCount(3)
        self.cache_table.setHorizontalHeaderLabels([
            "Cache Name", "Size", "Document"
        ])
        layout.addWidget(self.cache_table)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        button_layout.addWidget(self.refresh_button)
        
        # Purge button
        self.purge_button = QPushButton("Purge Selected")
        button_layout.addWidget(self.purge_button)
        
        # Use button
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
        self.use_button.clicked.connect(self.use_selected_cache)
        
        # Table signals
        self.cache_table.itemSelectionChanged.connect(self.on_cache_selected)
        
        # Cache manager signals
        self.cache_manager.cache_list_updated.connect(self.refresh_caches)
        self.cache_manager.cache_purged.connect(self.on_cache_purged)

    def refresh_caches(self):
        """Refresh the cache list"""
        try:
            # Clear the table
            self.cache_table.setRowCount(0)
            
            # Refresh cache list in manager
            try:
                self.cache_manager.refresh_cache_list()
            except Exception as e:
                QMessageBox.warning(self, "Refresh Error", "Could not refresh cache list")
            
            # Get the cache list
            caches = self.cache_manager.get_cache_list()
            
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
                    size_str = str(size_bytes) + " B"
                elif size_bytes < 1024 * 1024:
                    size_str = str(int(size_bytes / 1024)) + " KB"
                else:
                    size_str = str(int(size_bytes / (1024 * 1024))) + " MB"
                self.cache_table.setItem(i, 1, QTableWidgetItem(size_str))
                
                # Document
                self.cache_table.setItem(i, 2, QTableWidgetItem(cache.get('document_id', 'Unknown')))
            
            # Update status
            self.status_label.setText(str(len(caches)) + " caches")
            
        except Exception as e:
            self.status_label.setText("Error refreshing caches")
    
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
        self.status_label.setText("Selected: " + str(cache_path))
    
    def purge_selected_cache(self):
        """Purge the selected cache"""
        selected_items = self.cache_table.selectedItems()
        if not selected_items:
            return
        
        # Get selected row
        row = selected_items[0].row()
        
        # Get cache path
        cache_path = self.cache_table.item(row, 0).data(Qt.UserRole)
        
        # Purge cache
        success = self.cache_manager.purge_cache(cache_path)
        if success:
            self.status_label.setText("Cache purged")
        else:
            self.status_label.setText("Failed to purge cache")
    
    def use_selected_cache(self):
        """Use the selected cache"""
        selected_items = self.cache_table.selectedItems()
        if not selected_items:
            return
        
        # Get selected row
        row = selected_items[0].row()
        
        # Get cache path
        cache_path = self.cache_table.item(row, 0).data(Qt.UserRole)
        
        # Emit signal
        self.cache_selected.emit(cache_path)
        
        # Update status
        self.status_label.setText("Using selected cache")
    
    def on_cache_purged(self, cache_path, success):
        """Handle cache purged signal"""
        if success:
            self.refresh_caches()
            self.cache_purged.emit()
