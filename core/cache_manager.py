#!/usr/bin/env python3
"""
Cache Manager for LlamaCag UI

Manages KV cache files (.llama_cache) associated with processed documents.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from PyQt5.QtCore import QObject, pyqtSignal

class CacheManager(QObject):
    # Signals
    cache_list_updated = pyqtSignal()
    cache_purged = pyqtSignal(str, bool)  # cache_path, success

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._cache_registry = {} # Stores info about known cache files {cache_path_str: info_dict}
        self._document_registry_path = None # Path to the document registry JSON
        self.kv_cache_dir = None # Path object for the cache directory
        self._last_scan_results = set() # Keep track of files found in last scan

        self.update_config(config) # Initialize paths based on config

    def update_config(self, config):
        """Update cache directory based on config."""
        self.config = config
        cache_dir_str = config.get('LLAMACPP_KV_CACHE_DIR', '')
        if not cache_dir_str:
            cache_dir_str = os.path.join(os.path.expanduser('~'), 'cag_project', 'kv_caches')
        
        new_cache_dir = Path(os.path.expanduser(cache_dir_str)).resolve()
        
        if self.kv_cache_dir != new_cache_dir:
            self.kv_cache_dir = new_cache_dir
            self._document_registry_path = self.kv_cache_dir / 'document_registry.json'
            logging.info(f"Cache directory set to: {self.kv_cache_dir}")
            os.makedirs(self.kv_cache_dir, exist_ok=True)
            self.refresh_cache_list(scan_now=True) # Rescan if directory changed

    def _load_document_registry(self) -> Dict:
        """Load the document registry JSON file."""
        if self._document_registry_path and self._document_registry_path.exists():
            try:
                with open(self._document_registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load document registry {self._document_registry_path}: {e}")
        return {}

    def refresh_cache_list(self, scan_now=True):
        """
        Scans the cache directory for .llama_cache files and updates the registry.
        Emits cache_list_updated only if changes are detected compared to the last scan.
        """
        if not self.kv_cache_dir:
            logging.error("Cache directory not set. Cannot refresh cache list.")
            return

        if not scan_now:
             print("Cache list refresh requested (NO SCANNING)")
             # Optionally emit here if UI needs update even without scanning,
             # but be careful about recursion if connected back to this function.
             # self.cache_list_updated.emit()
             return

        logging.info(f"Scanning cache directory: {self.kv_cache_dir}")
        found_files = set()
        current_registry = {}
        doc_registry = self._load_document_registry() # Load mapping from doc_id to info

        try:
            for item in self.kv_cache_dir.glob('*.llama_cache'):
                if item.is_file():
                    file_path_str = str(item)
                    found_files.add(file_path_str)
                    try:
                        stat_result = item.stat()
                        size_bytes = stat_result.st_size
                        
                        # Try to find associated document_id from filename or doc registry
                        doc_id = item.stem # e.g., "my_document" from "my_document.llama_cache"
                        original_doc_path = "Unknown"
                        if doc_id in doc_registry:
                            original_doc_path = doc_registry[doc_id].get('original_file_path', 'Unknown')
                        
                        current_registry[file_path_str] = {
                            'path': file_path_str,
                            'filename': item.name,
                            'size': size_bytes,
                            'last_modified': stat_result.st_mtime,
                            'document_id': doc_id, # Best guess from filename
                            'original_document': original_doc_path # From registry if found
                        }
                    except OSError as e:
                        logging.warning(f"Could not stat cache file {item}: {e}")
                    except Exception as e:
                         logging.error(f"Unexpected error processing cache file {item}: {e}")

            # Check if the found files differ from the last scan
            if found_files != self._last_scan_results:
                logging.info(f"Cache list changed. Found {len(found_files)} caches.")
                self._cache_registry = current_registry
                self._last_scan_results = found_files
                self.cache_list_updated.emit() # Emit signal ONLY if changes detected
            else:
                logging.info("Cache list unchanged.")
                # Update registry anyway in case file metadata changed, but don't emit signal
                self._cache_registry = current_registry


        except Exception as e:
            logging.error(f"Failed to scan cache directory {self.kv_cache_dir}: {e}")
            # Decide if we should clear the registry or leave it stale
            # self._cache_registry = {}
            # self._last_scan_results = set()
            # self.cache_list_updated.emit() # Emit on error?

    def get_cache_list(self) -> List[Dict]:
        """Returns a list of dictionaries, each describing a cache file."""
        # Return values from the registry
        return list(self._cache_registry.values())

    def get_cache_info(self, cache_path: str) -> Optional[Dict]:
        """Get information about a specific cache file."""
        return self._cache_registry.get(str(Path(cache_path).resolve()))

    def register_cache(self, document_id, cache_path, context_size,
                      token_count=0, original_file_path="", model_id="",
                      is_master=False):
        """
        Explicitly register or update a cache file in the registry.
        This is typically called by DocumentProcessor after creating a cache.
        """
        cache_path_obj = Path(cache_path).resolve()
        cache_path_str = str(cache_path_obj)
        logging.info(f"Registering cache: {document_id} at {cache_path_str}")

        if not cache_path_obj.exists():
             logging.warning(f"Attempted to register non-existent cache file: {cache_path_str}")
             # Don't add non-existent files, maybe trigger a rescan?
             self.refresh_cache_list(scan_now=True)
             return False

        try:
            stat_result = cache_path_obj.stat()
            new_info = {
                'path': cache_path_str,
                'filename': cache_path_obj.name,
                'size': stat_result.st_size,
                'last_modified': stat_result.st_mtime,
                'document_id': document_id,
                'original_document': original_file_path,
                'context_size': context_size, # Store context size if provided
                'token_count': token_count,   # Store token count if provided
                'model_id': model_id,         # Store model id if provided
                'is_master': is_master        # Store master status
            }

            # Check if registry needs updating
            needs_update = True
            if cache_path_str in self._cache_registry:
                 # Simple check: if size changed, assume update needed
                 if self._cache_registry[cache_path_str].get('size') == new_info['size']:
                     needs_update = False # Avoid unnecessary signal if only metadata changed

            self._cache_registry[cache_path_str] = new_info
            self._last_scan_results.add(cache_path_str) # Ensure it's in the scan results

            if needs_update:
                self.cache_list_updated.emit() # Emit signal as cache was added/updated

            return True
        except Exception as e:
             logging.error(f"Failed to register cache {cache_path_str}: {e}")
             return False

    def update_usage_by_path(self, cache_path):
        """Updates usage timestamp for a given cache path (Not fully implemented)."""
        # In a real implementation, you might update 'last_used' or 'usage_count'
        # in the registry and potentially save it.
        cache_path_str = str(Path(cache_path).resolve())
        if cache_path_str in self._cache_registry:
             # self._cache_registry[cache_path_str]['last_used'] = time.time()
             # self._cache_registry[cache_path_str]['usage_count'] = self._cache_registry[cache_path_str].get('usage_count', 0) + 1
             # Need persistence mechanism if usage stats should survive restarts
             pass
        return True

    def purge_cache(self, cache_path: str) -> bool:
        """Deletes a cache file and removes it from the registry."""
        cache_path_obj = Path(cache_path).resolve()
        cache_path_str = str(cache_path_obj)
        logging.info(f"Attempting to purge cache: {cache_path_str}")
        success = False
        try:
            if cache_path_obj.exists():
                cache_path_obj.unlink() # Use unlink for Path objects
                logging.info(f"Successfully deleted cache file: {cache_path_str}")
                success = True
            else:
                logging.warning(f"Cache file not found for purging: {cache_path_str}")
                success = True # Consider it success if file is already gone

            # Remove from registry and scan results
            if cache_path_str in self._cache_registry:
                del self._cache_registry[cache_path_str]
            if cache_path_str in self._last_scan_results:
                self._last_scan_results.remove(cache_path_str)

            self.cache_purged.emit(cache_path_str, True)
            self.cache_list_updated.emit() # List has changed
            return True

        except Exception as e:
            logging.error(f"Failed to purge cache {cache_path_str}: {e}")
            self.cache_purged.emit(cache_path_str, False)
            return False

    def purge_all_caches(self) -> bool:
        """Deletes all .llama_cache files in the cache directory."""
        logging.info(f"Attempting to purge all caches in: {self.kv_cache_dir}")
        all_purged = True
        if not self.kv_cache_dir:
            logging.error("Cache directory not set. Cannot purge all caches.")
            return False
            
        # Create a list of paths to avoid modifying the iterator during deletion
        caches_to_purge = list(self.kv_cache_dir.glob('*.llama_cache'))
        
        for cache_path_obj in caches_to_purge:
            if not self.purge_cache(str(cache_path_obj)):
                 all_purged = False # Keep track if any individual purge fails

        # Final update after purging
        self.refresh_cache_list(scan_now=False) # Update internal state
        self.cache_list_updated.emit() # Emit signal once after all purging
        logging.info("Finished purging all caches.")
        return all_purged

    def get_total_cache_size(self) -> int:
        """Calculates the total size of all managed cache files."""
        return sum(info.get('size', 0) for info in self._cache_registry.values())

    def check_cache_compatibility(self, model_context_size):
        """Checks cache files for compatibility (placeholder)."""
        # In a real implementation, this might check metadata stored within the cache
        # or in the registry against the provided context size or model hash.
        compatible_caches = []
        for path, info in self._cache_registry.items():
             # Example check (needs actual metadata):
             # if info.get('context_size') == model_context_size:
             #    compatible_caches.append(path)
             pass # Placeholder - currently returns empty list
        return compatible_caches
