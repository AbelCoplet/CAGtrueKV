#!/usr/bin/env python3
"""
KV cache management for LlamaCag UI
Manages KV caches for large context window models.
"""
import os
import sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PyQt5.QtCore import QObject, pyqtSignal

class CacheManager(QObject):
    """Manages KV caches for large context window models"""
    # Signals
    cache_list_updated = pyqtSignal()
    cache_purged = pyqtSignal(str, bool)  # cache_path, success

    def __init__(self, config):
        """Initialize cache manager"""
        super().__init__()
        self.config = config
        self.kv_cache_dir = Path(os.path.expanduser(config.get('LLAMACPP_KV_CACHE_DIR', '~/cag_project/kv_caches')))
        # Ensure directory exists
        self.kv_cache_dir.mkdir(parents=True, exist_ok=True)
        # Cache registry
        self._cache_registry = {} # Maps cache_path -> {metadata}
        self._usage_registry = {} # Maps cache_path -> {usage_stats}
        # Load registries
        self._load_registries()

    def _load_registries(self):
        """Load cache and usage registries from disk"""
        registry_file = self.kv_cache_dir / 'cache_registry.json'
        usage_file = self.kv_cache_dir / 'usage_registry.json'

        if registry_file.exists():
            try:
                 with open(registry_file, 'r') as f:
                     self._cache_registry = json.load(f)
                 logging.info(f"Loaded cache registry with {len(self._cache_registry)} entries")
            except Exception as e:
                 logging.error(f"Failed to load cache registry: {str(e)}")
                 self._cache_registry = {} # Reset on error

        if usage_file.exists():
            try:
                 with open(usage_file, 'r') as f:
                     self._usage_registry = json.load(f)
                 logging.info(f"Loaded usage registry with {len(self._usage_registry)} entries")
            except Exception as e:
                 logging.error(f"Failed to load usage registry: {str(e)}")
                 self._usage_registry = {} # Reset on error

    def _save_registries(self):
        """Save cache and usage registries to disk"""
        registry_file = self.kv_cache_dir / 'cache_registry.json'
        usage_file = self.kv_cache_dir / 'usage_registry.json'

        try:
             with open(registry_file, 'w') as f:
                 json.dump(self._cache_registry, f, indent=2)
        except Exception as e:
             logging.error(f"Failed to save cache registry: {str(e)}")

        try:
             with open(usage_file, 'w') as f:
                 json.dump(self._usage_registry, f, indent=2)
        except Exception as e:
             logging.error(f"Failed to save usage registry: {str(e)}")

    def refresh_cache_list(self):
        """Scan directory using os.walk and update registry"""
        found_paths = set()
        try:
            # Use os.walk as a more robust alternative
            logging.debug(f"Scanning directory using os.walk: {self.kv_cache_dir}")
            for root, dirs, files in os.walk(self.kv_cache_dir, followlinks=False): # Avoid following symlinks
                # Skip hidden directories (optional, but good practice)
                dirs[:] = [d for d in dirs if not d.startswith('.')]

                for filename in files:
                    if filename.endswith('.llama_cache') and not filename.startswith('.'):
                        full_path_str = os.path.join(root, filename)
                        found_paths.add(full_path_str)
                        if full_path_str not in self._cache_registry:
                            logging.warning(f"Found untracked cache file: {full_path_str}. Adding with minimal info.")
                            try:
                                file_stat = os.stat(full_path_str)
                                self._cache_registry[full_path_str] = {
                                    'document_id': Path(filename).stem, # Best guess
                                    'original_file_path': 'Unknown',
                                    'context_size': 0, # Unknown
                                    'token_count': 0, # Unknown
                                    'model_id': 'Unknown',
                                    'created_at': file_stat.st_mtime # Use file mod time
                                }
                                if full_path_str not in self._usage_registry:
                                    self._usage_registry[full_path_str] = {'last_used': None, 'usage_count': 0}
                            except Exception as stat_error:
                                 logging.error(f"Could not stat untracked file {full_path_str}: {stat_error}")

        except RecursionError:
             # Handle RecursionError specifically with a simple message
             logging.error(f"Recursion error while scanning cache directory {self.kv_cache_dir}. Check for symlink loops or excessive depth.")
        except Exception as e:
             # Log other errors more simply to avoid recursion in logging itself
             logging.error(f"Error scanning cache directory {self.kv_cache_dir}. Type: {type(e).__name__}")
             # Proceed with potentially incomplete found_paths or just use registry?
             # Let's proceed with what we found, if anything.

        # Remove registry entries for files that no longer exist
        removed_caches = False
        for registered_path in list(self._cache_registry.keys()):
            if registered_path not in found_paths:
                # Check if the file actually exists before logging warning (it might have been purged)
                if Path(registered_path).exists():
                     logging.warning(f"Cache file missing for registered path: {registered_path}. Removing entry.")
                else:
                     logging.debug(f"Removing registry entry for already deleted file: {registered_path}")
                del self._cache_registry[registered_path]
                if registered_path in self._usage_registry:
                    del self._usage_registry[registered_path]
                removed_caches = True

        if removed_caches:
            self._save_registries()
        self.cache_list_updated.emit()


    def get_cache_list(self) -> List[Dict]:
        """Get list of available KV caches from the registry"""
        cache_list = []
        # Iterate through registered caches
        for cache_path_str, registry_info in self._cache_registry.items():
            file_path = Path(cache_path_str)
            # Ensure file still exists before adding to list
            if not file_path.exists():
                 logging.warning(f"Registry contains path for non-existent file: {cache_path_str}. Skipping.")
                 continue

            usage_info = self._usage_registry.get(cache_path_str, {})
            try:
                file_stat = file_path.stat()
                cache_info = {
                    'id': registry_info.get('document_id', file_path.stem), # Use document_id from registry
                    'path': cache_path_str,
                    'filename': file_path.name, # Keep filename for display
                    'size': file_stat.st_size,
                    'last_modified': file_stat.st_mtime, # File mod time
                    'document_id': registry_info.get('document_id', file_path.stem),
                    'original_file_path': registry_info.get('original_file_path', ''), # Use original path
                    'context_size': registry_info.get('context_size', 0),
                    'token_count': registry_info.get('token_count', 0),
                    'model_id': registry_info.get('model_id', ''),
                    'created_at': registry_info.get('created_at', None), # Add creation time
                    'last_used': usage_info.get('last_used', None),
                    'usage_count': usage_info.get('usage_count', 0),
                    'is_master': registry_info.get('is_master', False) # Add master status if available
                }
                cache_list.append(cache_info)
            except Exception as e:
                 logging.error(f"Could not get info for cache file {cache_path_str}: {e}")


        # Sort list, e.g., by last used or creation date
        cache_list.sort(key=lambda x: x.get('last_used') or x.get('created_at') or 0, reverse=True)

        return cache_list

    def get_cache_info(self, cache_path_str: str) -> Optional[Dict]:
        """Get detailed information about a KV cache from the registry"""
        if cache_path_str not in self._cache_registry:
             # Try refreshing in case it's untracked
             self.refresh_cache_list()
             if cache_path_str not in self._cache_registry:
                 logging.warning(f"Cache path not found in registry: {cache_path_str}")
                 return None

        registry_info = self._cache_registry[cache_path_str]
        usage_info = self._usage_registry.get(cache_path_str, {})
        file_path = Path(cache_path_str)

        if not file_path.exists():
             logging.warning(f"File missing for registered cache: {cache_path_str}")
             return None # Or maybe return registry info with a 'missing' flag?

        try:
            file_stat = file_path.stat()
            cache_info = {
                'id': registry_info.get('document_id', file_path.stem),
                'path': cache_path_str,
                'filename': file_path.name,
                'size': file_stat.st_size,
                'last_modified': file_stat.st_mtime,
                'document_id': registry_info.get('document_id', file_path.stem),
                'original_file_path': registry_info.get('original_file_path', ''),
                'context_size': registry_info.get('context_size', 0),
                'token_count': registry_info.get('token_count', 0),
                'model_id': registry_info.get('model_id', ''),
                'created_at': registry_info.get('created_at', None),
                'last_used': usage_info.get('last_used', None),
                'usage_count': usage_info.get('usage_count', 0),
                'is_master': registry_info.get('is_master', False)
            }
            return cache_info
        except Exception as e:
             logging.error(f"Could not get info for cache file {cache_path_str}: {e}")
             return None


    def register_cache(self, document_id: str, cache_path: str, context_size: int,
                       token_count: int, original_file_path: str, model_id: str, is_master: bool = False) -> bool:
        """Register a KV cache in the registry with detailed info"""
        path_obj = Path(cache_path)
        if not path_obj.exists() or path_obj.suffix != '.llama_cache':
            logging.error(f"Cannot register invalid or non-existent cache: {cache_path}")
            return False

        cache_path_str = str(path_obj)
        self._cache_registry[cache_path_str] = {
            'document_id': document_id,
            'original_file_path': original_file_path,
            'context_size': context_size,
            'token_count': token_count,
            'model_id': model_id,
            'created_at': time.time(),
            'is_master': is_master # Store master status
        }
        # Ensure usage entry exists
        if cache_path_str not in self._usage_registry:
             self._usage_registry[cache_path_str] = {'last_used': None, 'usage_count': 0}

        self._save_registries()
        self.cache_list_updated.emit()
        return True

    def update_usage_by_path(self, cache_path_str: str) -> bool:
        """Update usage statistics for a KV cache using its path"""
        if cache_path_str not in self._cache_registry:
            logging.warning(f"Attempted to update usage for untracked cache: {cache_path_str}")
            # Optionally, try to refresh and see if it appears
            self.refresh_cache_list()
            if cache_path_str not in self._cache_registry:
                 logging.error(f"Cannot update usage, cache path not found: {cache_path_str}")
                 return False

        # Update usage registry
        usage_info = self._usage_registry.get(cache_path_str, {'usage_count': 0})
        usage_info['last_used'] = time.time()
        usage_info['usage_count'] = usage_info.get('usage_count', 0) + 1
        self._usage_registry[cache_path_str] = usage_info

        self._save_registries()
        self.cache_list_updated.emit() # Update UI to show new usage stats
        return True

    def purge_cache(self, cache_path_str: str) -> bool:
        """Purge a KV cache file and its registry entries"""
        path_obj = Path(cache_path_str)
        file_existed = path_obj.exists()

        try:
            if file_existed:
                path_obj.unlink()
                logging.info(f"Deleted cache file: {cache_path_str}")

            # Remove from registries regardless of whether file existed (cleans up stale entries)
            registry_updated = False
            if cache_path_str in self._cache_registry:
                del self._cache_registry[cache_path_str]
                registry_updated = True
            if cache_path_str in self._usage_registry:
                del self._usage_registry[cache_path_str]
                registry_updated = True

            if registry_updated:
                self._save_registries()

            self.cache_purged.emit(cache_path_str, True)
            self.cache_list_updated.emit() # Refresh UI list
            return True
        except Exception as e:
            logging.error(f"Failed to purge cache {cache_path_str}: {str(e)}")
            self.cache_purged.emit(cache_path_str, False)
            # Don't emit cache_list_updated on failure? Or do to reflect potential partial state?
            # Let's emit it to be safe, UI should handle missing files.
            self.cache_list_updated.emit()
            return False

    def purge_all_caches(self) -> bool:
        """Purge all *.llama_cache files and clear registries"""
        logging.info("Purging all KV caches...")
        all_purged = True
        try:
            # Iterate through registered paths first to ensure cleanup
            registered_paths = list(self._cache_registry.keys())
            for path_str in registered_paths:
                 if not self.purge_cache(path_str):
                      all_purged = False # Logged in purge_cache

            # Additionally scan directory for any untracked .llama_cache files using os.walk
            for root, dirs, files in os.walk(self.kv_cache_dir, followlinks=False):
                 dirs[:] = [d for d in dirs if not d.startswith('.')] # Skip hidden dirs
                 for filename in files:
                     if filename.endswith('.llama_cache') and not filename.startswith('.'):
                         full_path_str = os.path.join(root, filename)
                         if full_path_str not in registered_paths: # Only purge if not already handled
                             logging.info(f"Purging untracked cache file: {full_path_str}")
                             try:
                                 os.unlink(full_path_str)
                             except Exception as e:
                                 logging.error(f"Failed to remove untracked cache {full_path_str}: {str(e)}")
                                 all_purged = False

            # Explicitly clear registries after attempting deletion
            self._cache_registry = {}
            self._usage_registry = {}
            self._save_registries()

            logging.info("Finished purging all caches.")
            self.cache_list_updated.emit() # Update UI
            return all_purged
        except Exception as e:
            logging.error(f"Failed during purge all caches: {str(e)}")
            self.cache_list_updated.emit() # Update UI even on error
            return False

    def get_total_cache_size(self) -> int:
        """Get the total size of all registered KV caches in bytes"""
        total_size = 0
        for cache_path_str in self._cache_registry.keys():
             path_obj = Path(cache_path_str)
             if path_obj.exists():
                 try:
                     total_size += path_obj.stat().st_size
                 except Exception as e:
                      logging.warning(f"Could not get size for {path_obj}: {e}")
        return total_size

    # This method might need adjustment based on how llama-cpp handles context size mismatches
    # For now, assume a cache is incompatible if its creation context > model context
    def check_cache_compatibility(self, model_context_size: int) -> List[str]:
        """Check which caches might be incompatible with the given model context size"""
        incompatible_caches = []
        for cache_path_str, registry_info in self._cache_registry.items():
             cache_creation_context = registry_info.get('context_size', 0)
             # If cache was created with a larger context than the current model supports, flag it
             if cache_creation_context > model_context_size:
                 incompatible_caches.append(cache_path_str)
        return incompatible_caches

    def update_config(self, config):
        """Update configuration and reload registries if path changed"""
        self.config = config
        new_kv_cache_dir = Path(os.path.expanduser(config.get('LLAMACPP_KV_CACHE_DIR', '~/cag_project/kv_caches')))
        if new_kv_cache_dir != self.kv_cache_dir:
             logging.info(f"KV Cache directory changed to: {new_kv_cache_dir}. Reloading registries.")
             self.kv_cache_dir = new_kv_cache_dir
             self.kv_cache_dir.mkdir(parents=True, exist_ok=True)
             self._load_registries()
             self.refresh_cache_list() # Refresh list after path change
        else:
             logging.info("CacheManager configuration updated (path unchanged).")
