#!/usr/bin/env python3
"""
Cache Manager for LlamaCag UI

Manages KV cache files (.llama_cache) associated with processed documents.
Provides support for document recitation and Fresh Context mode.
"""

import os
import sys
import json
import time
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PyQt5.QtCore import QObject, pyqtSignal

class CacheManager(QObject):
    # Signals
    cache_list_updated = pyqtSignal()
    cache_purged = pyqtSignal(str, bool)  # cache_path, success
    cache_verified = pyqtSignal(str, bool, str)  # cache_path, success, message

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._cache_registry = {} # Stores info about known cache files {cache_path_str: info_dict}
        self._document_registry_path = None # Path to the document registry JSON
        self.kv_cache_dir = None # Path object for the cache directory
        self._last_scan_results = set() # Keep track of files found in last scan
        self._compatibility_cache = {} # Cache the results of compatibility checks

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
            # Clear compatibility cache when config changes
            self._compatibility_cache = {}
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

    def _save_document_registry(self, registry_data: Dict) -> bool:
        """Save updated document registry data."""
        if not self._document_registry_path:
            logging.error("Document registry path not set. Cannot save registry.")
            return False
            
        try:
            # Ensure directory exists
            self._document_registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self._document_registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
            logging.info(f"Document registry saved: {len(registry_data)} entries")
            return True
        except Exception as e:
            logging.error(f"Failed to save document registry: {e}")
            return False

    def refresh_cache_list(self, scan_now=True):
        """
        Scans the cache directory for .llama_cache files and updates the registry.
        Preserves explicitly registered info (like original_document).
        Emits cache_list_updated only if entries are added or removed.
        """
        if not self.kv_cache_dir:
            logging.error("Cache directory not set. Cannot refresh cache list.")
            return

        if not scan_now:
            logging.debug("Cache list refresh requested (NO SCANNING)")
            return

        logging.info(f"Scanning cache directory: {self.kv_cache_dir}")
        found_paths = set()
        new_entries = {}
        updated_existing = False
        doc_registry = self._load_document_registry() # Load mapping from doc_id to info

        try:
            for item in self.kv_cache_dir.glob('*.llama_cache'):
                if item.is_file():
                    file_path_str = str(item.resolve()) # Use resolved path as key
                    found_paths.add(file_path_str)
                    try:
                        stat_result = item.stat()
                        size_bytes = stat_result.st_size
                        last_modified = stat_result.st_mtime

                        if file_path_str in self._cache_registry:
                            # Existing entry: Update only size and modified time
                            if (self._cache_registry[file_path_str].get('size') != size_bytes or
                                self._cache_registry[file_path_str].get('last_modified') != last_modified):
                                self._cache_registry[file_path_str]['size'] = size_bytes
                                self._cache_registry[file_path_str]['last_modified'] = last_modified
                                # Clear compatibility cache for this file
                                model_id = self._cache_registry[file_path_str].get('model_id', '')
                                if model_id and (file_path_str, model_id) in self._compatibility_cache:
                                    del self._compatibility_cache[(file_path_str, model_id)]
                                updated_existing = True # Mark potential change for signal emission
                                logging.debug(f"Updated metadata for existing cache: {item.name}")
                        else:
                            # New entry: Add to temporary dict using doc_registry lookup
                            doc_id = item.stem
                            doc_info_from_registry = doc_registry.get(doc_id, {}) # Get info dict or empty dict

                            original_doc_path = doc_info_from_registry.get('original_file_path', 'Unknown')
                            token_count = doc_info_from_registry.get('token_count', 0)
                            context_size = doc_info_from_registry.get('context_size', 0)
                            model_id = doc_info_from_registry.get('model_id', '') # Load model_id
                            is_master = doc_info_from_registry.get('is_master', False) # Load master status
                            last_used = doc_info_from_registry.get('last_used', None)
                            usage_count = doc_info_from_registry.get('usage_count', 0)

                            new_entries[file_path_str] = {
                                'path': file_path_str,
                                'filename': item.name,
                                'size': size_bytes,
                                'last_modified': last_modified,
                                'document_id': doc_id,
                                'original_document': original_doc_path,
                                'token_count': token_count,   # Store from registry
                                'context_size': context_size, # Store from registry
                                'model_id': model_id,         # Store from registry
                                'is_master': is_master,       # Store from registry
                                'last_used': last_used,       # Track last used timestamp
                                'usage_count': usage_count,   # Track usage count
                                'verified': False,            # Initially not verified
                                'compatible_with_current': None # Unknown compatibility
                            }
                            logging.debug(f"Found new cache file to add (from scan): {item.name}")

                    except OSError as e:
                        logging.warning(f"Could not stat cache file {item}: {e}")
                    except Exception as e:
                        logging.error(f"Unexpected error processing cache file {item}: {e}")

            # Add newly found entries
            added_new = bool(new_entries)
            self._cache_registry.update(new_entries)

            # Remove entries for files that no longer exist
            removed_paths = set(self._cache_registry.keys()) - found_paths
            removed_any = bool(removed_paths)
            if removed_any:
                for path_to_remove in removed_paths:
                    logging.info(f"Removing missing cache file from registry: {Path(path_to_remove).name}")
                    # Also remove from compatibility cache
                    keys_to_remove = []
                    for key in self._compatibility_cache:
                        if isinstance(key, tuple) and len(key) == 2 and key[0] == path_to_remove:
                            keys_to_remove.append(key)
                    for key in keys_to_remove:
                        del self._compatibility_cache[key]
                    # Remove from main registry
                    del self._cache_registry[path_to_remove]

            # Update last scan results
            self._last_scan_results = found_paths

            # Emit signal only if entries were added or removed
            if added_new or removed_any:
                logging.info(f"Cache list updated: {len(self._cache_registry)} entries total.")
                self.cache_list_updated.emit()
            elif updated_existing:
                 logging.info("Cache metadata updated, but list structure unchanged.")
                 # Optionally emit signal even if only metadata changed, if UI needs it
                 self.cache_list_updated.emit()
            else:
                 logging.info("Cache list scan found no changes.")

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
        try:
            # Normalize path to ensure consistent lookups
            cache_path_str = str(Path(cache_path).resolve())
            return self._cache_registry.get(cache_path_str)
        except Exception as e:
            logging.error(f"Error retrieving cache info for {cache_path}: {e}")
            return None

    def verify_cache_integrity(self, cache_path: str) -> Tuple[bool, str]:
        """
        Verifies that a KV cache file can be loaded successfully.
        Returns (success, message)
        """
        cache_path_obj = Path(cache_path).resolve()
        cache_path_str = str(cache_path_obj)
        
        if not cache_path_obj.exists():
            msg = f"Cache file does not exist: {cache_path_obj.name}"
            logging.warning(msg)
            return False, msg
            
        try:
            # Try to load the cache file using pickle
            with open(cache_path_obj, 'rb') as f:
                state_data = pickle.load(f)
                
            # Update registry to mark as verified
            if cache_path_str in self._cache_registry:
                self._cache_registry[cache_path_str]['verified'] = True
                
            msg = f"Cache file verified successfully: {cache_path_obj.name}"
            logging.info(msg)
            self.cache_verified.emit(cache_path_str, True, msg)
            return True, msg
            
        except Exception as e:
            msg = f"Cache verification failed: {str(e)}"
            logging.error(f"{msg} for {cache_path_obj.name}")
            self.cache_verified.emit(cache_path_str, False, msg)
            return False, msg

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
            
            # Check if this cache already exists in the registry
            if cache_path_str in self._cache_registry:
                # Update existing entry
                existing_info = self._cache_registry[cache_path_str]
                # Preserve usage stats if they exist
                last_used = existing_info.get('last_used')
                usage_count = existing_info.get('usage_count', 0)
            else:
                # Default values for new entry
                last_used = None
                usage_count = 0
                
            # Load document registry to check for additional metadata
            doc_registry = self._load_document_registry()
            if document_id in doc_registry:
                # If we have additional info in the document registry, use it
                doc_info = doc_registry[document_id]
                # Only override if values are not provided or empty
                if not token_count and 'token_count' in doc_info:
                    token_count = doc_info['token_count']
                if not context_size and 'context_size' in doc_info:
                    context_size = doc_info['context_size']
                if not model_id and 'model_id' in doc_info:
                    model_id = doc_info['model_id']
                if not original_file_path and 'original_file_path' in doc_info:
                    original_file_path = doc_info['original_file_path']
            
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
                'is_master': is_master,       # Store master status
                'last_used': last_used,       # Track last used timestamp
                'usage_count': usage_count,   # Track usage count
                'verified': False,            # Will be verified later
                'compatible_with_current': None # Unknown compatibility
            }

            # Check if registry needs updating
            needs_update = True
            if cache_path_str in self._cache_registry:
                 # Check if only metadata changed
                 old_info = self._cache_registry[cache_path_str]
                 if (old_info.get('size') == new_info['size'] and
                     old_info.get('document_id') == new_info['document_id'] and
                     old_info.get('model_id') == new_info['model_id']):
                     needs_update = False

            # Update cache registry
            self._cache_registry[cache_path_str] = new_info
            self._last_scan_results.add(cache_path_str) # Ensure it's in the scan results
            
            # Also update document registry for persistence
            doc_registry = self._load_document_registry()
            if document_id not in doc_registry:
                doc_registry[document_id] = {}
            
            # Update document registry with relevant fields
            doc_registry[document_id].update({
                'document_id': document_id,
                'original_file_path': original_file_path,
                'kv_cache_path': cache_path_str,
                'token_count': token_count,
                'context_size': context_size,
                'model_id': model_id,
                'is_master': is_master,
                'created_at': doc_registry[document_id].get('created_at', time.time()),
                'last_used': last_used,
                'usage_count': usage_count
            })
            
            # Save updated document registry
            self._save_document_registry(doc_registry)

            # Clear compatibility cache for this file
            if model_id:
                if (cache_path_str, model_id) in self._compatibility_cache:
                    del self._compatibility_cache[(cache_path_str, model_id)]

            if needs_update:
                self.cache_list_updated.emit() # Emit signal as cache was added/updated

            return True
        except Exception as e:
             logging.error(f"Failed to register cache {cache_path_str}: {e}")
             return False

    def update_usage_by_path(self, cache_path: str) -> bool:
        """
        Updates usage timestamp and count for a given cache path.
        Also updates the document registry for persistence.
        """
        try:
            cache_path_obj = Path(cache_path).resolve()
            cache_path_str = str(cache_path_obj)
            
            if cache_path_str in self._cache_registry:
                # Update in-memory registry
                self._cache_registry[cache_path_str]['last_used'] = time.time()
                self._cache_registry[cache_path_str]['usage_count'] = self._cache_registry[cache_path_str].get('usage_count', 0) + 1
                
                # Update document registry for persistence
                doc_id = self._cache_registry[cache_path_str].get('document_id')
                if doc_id:
                    doc_registry = self._load_document_registry()
                    if doc_id in doc_registry:
                        doc_registry[doc_id]['last_used'] = time.time()
                        doc_registry[doc_id]['usage_count'] = doc_registry[doc_id].get('usage_count', 0) + 1
                        self._save_document_registry(doc_registry)
                
                logging.debug(f"Updated usage stats for cache: {cache_path_obj.name}")
                return True
            else:
                logging.warning(f"Cannot update usage: Cache not found in registry: {cache_path_obj.name}")
                return False
        except Exception as e:
            logging.error(f"Error updating usage stats: {e}")
            return False

    def purge_cache(self, cache_path: str) -> bool:
        """Deletes a cache file and removes it from the registry."""
        cache_path_obj = Path(cache_path).resolve()
        cache_path_str = str(cache_path_obj)
        logging.info(f"Attempting to purge cache: {cache_path_str}")
        success = False
        try:
            # Determine document_id before deletion (for registry cleanup)
            document_id = None
            if cache_path_str in self._cache_registry:
                document_id = self._cache_registry[cache_path_str].get('document_id')
            
            # Delete the file
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
                
            # Clear from compatibility cache
            keys_to_remove = []
            for key in self._compatibility_cache:
                if isinstance(key, tuple) and len(key) == 2 and key[0] == cache_path_str:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self._compatibility_cache[key]
                
            # Update document registry if we have a document_id
            if document_id:
                doc_registry = self._load_document_registry()
                if document_id in doc_registry:
                    # Check if this is the registered cache for this document
                    if doc_registry[document_id].get('kv_cache_path') == cache_path_str:
                        # Clear the kv_cache_path but keep the document entry
                        doc_registry[document_id]['kv_cache_path'] = ""
                        # Reset master status if this was the master
                        if doc_registry[document_id].get('is_master', False):
                            doc_registry[document_id]['is_master'] = False
                        self._save_document_registry(doc_registry)

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
        self._compatibility_cache = {} # Clear all compatibility cache entries
        self.refresh_cache_list(scan_now=True) # Update internal state
        self.cache_list_updated.emit() # Emit signal once after all purging
        logging.info("Finished purging all caches.")
        return all_purged

    def get_total_cache_size(self) -> int:
        """Calculates the total size of all managed cache files."""
        return sum(info.get('size', 0) for info in self._cache_registry.values())

    def check_cache_compatibility(self, cache_path: str, model_id: str) -> Tuple[bool, str]:
        """
        Checks if a cache file is compatible with the specified model.
        Returns (is_compatible, reason)
        """
        cache_path_obj = Path(cache_path).resolve()
        cache_path_str = str(cache_path_obj)
        
        # Check if we've already determined compatibility for this combination
        cache_key = (cache_path_str, model_id)
        if cache_key in self._compatibility_cache:
            return self._compatibility_cache[cache_key]
            
        # Get cache info
        cache_info = self.get_cache_info(cache_path_str)
        if not cache_info:
            result = (False, "Cache info not found")
            self._compatibility_cache[cache_key] = result
            return result
            
        # Check model compatibility
        cache_model_id = cache_info.get('model_id')
        if not cache_model_id:
            result = (True, "Cache has no model ID recorded, assuming compatible")
            self._compatibility_cache[cache_key] = result
            return result
            
        if cache_model_id != model_id:
            result = (False, f"Cache created with model '{cache_model_id}', but current model is '{model_id}'")
            self._compatibility_cache[cache_key] = result
            return result
            
        # Check file exists
        if not cache_path_obj.exists():
            result = (False, "Cache file does not exist")
            self._compatibility_cache[cache_key] = result
            return result
            
        # If we get here, the cache is compatible
        result = (True, "Compatible")
        self._compatibility_cache[cache_key] = result
        
        # Update the registry entry if it exists
        if cache_path_str in self._cache_registry:
            self._cache_registry[cache_path_str]['compatible_with_current'] = True
            
        return result

    def get_master_cache_path(self) -> Optional[str]:
        """
        Returns the path to the current master cache file.
        """
        # Check config first for explicit setting
        master_cache_path = self.config.get('MASTER_KV_CACHE_PATH')
        if master_cache_path and Path(master_cache_path).exists():
            return master_cache_path
            
        # Otherwise scan registry for master cache
        for path, info in self._cache_registry.items():
            if info.get('is_master', False) and Path(path).exists():
                return path
                
        # Check document registry as fallback
        doc_registry = self._load_document_registry()
        for doc_id, info in doc_registry.items():
            if info.get('is_master', False) and info.get('kv_cache_path'):
                master_path = info.get('kv_cache_path')
                if Path(master_path).exists():
                    return master_path
                    
        return None

    def backup_state(self, cache_path: str, backup_suffix: str = ".backup") -> Tuple[bool, str]:
        """
        Creates a backup copy of a cache file to support Fresh Context Mode.
        Returns (success, backup_path_or_error_message)
        """
        import shutil
        
        try:
            cache_path_obj = Path(cache_path)
            if not cache_path_obj.exists():
                return False, f"Cache file does not exist: {cache_path}"
                
            backup_path = str(cache_path_obj) + backup_suffix
            backup_path_obj = Path(backup_path)
            
            # Create backup
            shutil.copy2(cache_path_obj, backup_path_obj)
            logging.info(f"Created backup of {cache_path_obj.name} at {backup_path_obj.name}")
            
            # Verify backup
            if not backup_path_obj.exists():
                return False, "Backup file was not created"
                
            if backup_path_obj.stat().st_size != cache_path_obj.stat().st_size:
                return False, "Backup file has different size than original"
                
            return True, backup_path
            
        except Exception as e:
            error_msg = f"Failed to create backup: {str(e)}"
            logging.error(error_msg)
            return False, error_msg

    def restore_state(self, backup_path: str, target_path: Optional[str] = None) -> bool:
        """
        Restores a cache file from a backup for Fresh Context Mode.
        If target_path is None, will use backup_path without suffix.
        Returns success boolean.
        """
        import shutil
        
        try:
            backup_path_obj = Path(backup_path)
            if not backup_path_obj.exists():
                logging.error(f"Backup file does not exist: {backup_path}")
                return False
                
            # Determine target path if not specified
            if not target_path:
                # Assume backup has a suffix like .backup
                for suffix in [".backup", ".bak", ".fresh"]:
                    if str(backup_path_obj).endswith(suffix):
                        target_path = str(backup_path_obj)[:-len(suffix)]
                        break
                
                if not target_path:
                    logging.error(f"Could not determine target path from backup: {backup_path}")
                    return False
            
            target_path_obj = Path(target_path)
            
            # If target exists, remove it
            if target_path_obj.exists():
                target_path_obj.unlink()
                
            # Copy backup to target
            shutil.copy2(backup_path_obj, target_path_obj)
            logging.info(f"Restored cache from {backup_path_obj.name} to {target_path_obj.name}")
            
            # Verify restore
            if not target_path_obj.exists():
                logging.error(f"Target file was not created after restore: {target_path}")
                return False
                
            if target_path_obj.stat().st_size != backup_path_obj.stat().st_size:
                logging.warning(f"Restored file has different size than backup")
                
            return True
            
        except Exception as e:
            logging.error(f"Failed to restore from backup: {str(e)}")
            return False
