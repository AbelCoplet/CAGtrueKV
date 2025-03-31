#!/usr/bin/env python3
"""
Chat functionality for LlamaCag UI

Handles interaction with the model using KV caches.
Includes implementation for true KV cache loading and fallback.
"""

import os
import sys
import tempfile
import logging
import json
import time
import threading
import re
import pickle # Import pickle
import threading # Added for locking and background tasks
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PyQt5.QtCore import QObject, pyqtSignal, QCoreApplication
from llama_cpp import Llama, LlamaCache

class ChatEngine(QObject):
    """Chat functionality using large context window models with KV caches"""

    # Signals
    response_started = pyqtSignal()
    response_chunk = pyqtSignal(str)  # Text chunk
    response_complete = pyqtSignal(str, bool)  # Full response, success
    error_occurred = pyqtSignal(str)  # Error message
    status_updated = pyqtSignal(str) # General status updates for status bar

    # New signals for warm-up feature
    cache_warming_started = pyqtSignal()
    cache_warmed_up = pyqtSignal(float, int, int) # load_time, token_count, file_size
    cache_unloaded = pyqtSignal()
    cache_status_changed = pyqtSignal(str) # Specific status for chat tab: Idle, Warming Up, Warmed Up, Unloading, Error

    def __init__(self, config_manager, llama_manager, model_manager, cache_manager):
        """Initialize chat engine"""
        super().__init__()
        self.config_manager = config_manager
        self.llama_manager = llama_manager
        self.model_manager = model_manager
        self.cache_manager = cache_manager

        # Chat history
        self.history = []

        # Current KV cache selection
        self.current_kv_cache_path = None # Store the path of the *selected* cache
        self.use_kv_cache = True # Whether the user wants to use *a* cache

        # Persistent model instance for warm-up
        self.persistent_llm: Optional[Llama] = None
        self.loaded_model_path: Optional[str] = None # Model loaded in persistent_llm
        self.warmed_cache_path: Optional[str] = None # Cache loaded in persistent_llm
        self._lock = threading.Lock() # Protect access to persistent_llm and related state

        # Config setting for true KV cache logic
        self.use_true_kv_cache_logic = self.config_manager.get('USE_TRUE_KV_CACHE', True)
        # Cache Behavior Mode setting (replaces old fresh_context_mode)
        # Modes: "standard", "fresh_before", "fresh_after"
        self.cache_behavior_mode = self.config_manager.get('CACHE_BEHAVIOR_MODE', 'standard')
        self.debug_token_generation = False # Toggle for detailed token debugging
        logging.info(f"ChatEngine initialized. True KV Cache Logic: {self.use_true_kv_cache_logic}, Cache Behavior Mode: {self.cache_behavior_mode}")

    # Add a method to set the cache behavior mode
    def set_cache_behavior_mode(self, mode: str):
        """Sets the cache behavior mode ('standard', 'fresh_before', 'fresh_after')."""
        valid_modes = ["standard", "fresh_before", "fresh_after"]
        if mode not in valid_modes:
            logging.warning(f"Invalid cache behavior mode '{mode}'. Defaulting to 'standard'.")
            mode = "standard"

        with self._lock: # Ensure thread safety when changing mode
            self.config_manager.set('CACHE_BEHAVIOR_MODE', mode) # Update runtime config
            self.cache_behavior_mode = mode
            logging.info(f"Cache behavior mode set to: {self.cache_behavior_mode}")
            # Emit status change to update UI elements potentially
            status_msg = f"Cache Mode: {mode.replace('_', ' ').title()}"
            self.cache_status_changed.emit(status_msg)
            self.status_updated.emit(status_msg) # Also update main status bar

    # Add a method to toggle debugging
    def toggle_token_debugging(self, enabled: bool):
        """Enable or disable detailed token generation debugging"""
        self.debug_token_generation = enabled
        logging.info(f"Token generation debugging {'enabled' if enabled else 'disabled'}")

    def set_kv_cache(self, kv_cache_path: Optional[Union[str, Path]]):
        """Set the current KV cache path to use"""
        if kv_cache_path:
            cache_path = Path(kv_cache_path)
            # Expecting .llama_cache files now
            if not cache_path.exists() or cache_path.suffix != '.llama_cache':
                error_msg = f"KV cache not found or invalid: {cache_path}"
                logging.error(error_msg)
                self.error_occurred.emit(error_msg)
                return False

            # --- Strict Model Compatibility Check ---
            current_model_id = self.config_manager.get('CURRENT_MODEL_ID')
            cache_info = self.cache_manager.get_cache_info(str(cache_path))
            cache_model_id = cache_info.get('model_id') if cache_info else None

            if cache_model_id and current_model_id and cache_model_id != current_model_id:
                error_msg = f"Cache '{cache_path.name}' created with model '{cache_model_id}' is incompatible with current model '{current_model_id}'."
                logging.error(error_msg)
                self.error_occurred.emit(error_msg)
                # Do not set the cache path if incompatible
                return False
            elif not cache_model_id:
                logging.warning(f"Cache '{cache_path.name}' has no model ID metadata. Assuming compatible, but use with caution.")
            # --- End Check ---

            self.current_kv_cache_path = str(cache_path)
            logging.info(f"Set current KV cache path to {self.current_kv_cache_path}")
            return True
        else:
            # If clearing selection, unload any warmed cache
            if self.warmed_cache_path:
                self.unload_cache() # Trigger unload if selection is cleared
            self.current_kv_cache_path = None
            logging.info("Cleared current KV cache path")
            return True

    def toggle_kv_cache(self, enabled: bool):
        """Toggle KV cache usage"""
        # If disabling cache usage, unload any warmed cache
        if not enabled and self.warmed_cache_path:
            self.unload_cache()
        self.use_kv_cache = enabled
        logging.info(f"KV cache usage toggled: {enabled}")
        # Status bar update is handled by MainWindow based on overall state
        # Emit specific status for chat tab display
        self.cache_status_changed.emit("Idle" if not self.warmed_cache_path else "Warmed Up")

    # --- Detect recitation command ---
    def _is_recitation_command(self, message: str) -> Tuple[bool, str]:
        """
        Detect if the message is a document recitation command.
        Returns (is_recitation, modified_prompt)
        """
        # Convert to lowercase and strip for consistent comparison
        msg = message.lower().strip()

        # --- Check for ANY Recitation Command (Paragraph or Generic) ---
        # Simplify: If any recitation pattern matches, treat it as a generic request
        # to recite from the beginning. Paragraph-specific recitation is too unreliable.
        recitation_patterns = [
            r"(?:recite|read|show)\s+(?:the\s+)?document",
            r"(?:recite|read|show)\s+from\s+(?:the\s+)?beginning",
            r"start\s+from\s+(?:the\s+)?beginning",
            r"(?:recite|read|show)\s+(?:the\s+)?(first|second|third|fourth|fifth|\d+)\s+paragraph", # Detect but ignore specificity
            r"^recite$" # Exact command
        ]
        is_recitation = False
        for pattern in recitation_patterns:
            try:
                if re.search(pattern, msg):
                    is_recitation = True
                    break
            except re.error as re_err:
                logging.error(f"Regex error during recitation detection (pattern: {pattern}): {re_err}")
                continue # Try next pattern

        # Check simple prefix/exact matches as fallback
        if not is_recitation:
            # Check simple prefixes
            simple_prefixes = ["recite", "read", "show"]
            for prefix in simple_prefixes:
                 if msg.startswith(prefix):
                      # Check if it's likely a QA question despite the prefix
                      if "what" in msg or "who" in msg or "when" in msg or "where" in msg or "why" in msg or "how" in msg or "?" in msg:
                           continue # Likely QA, not recitation
                      is_recitation = True
                      break

        if is_recitation:
            # Always use the generic prompt now
            modified_prompt = "Please recite the document from the very beginning."
            logging.info(f"Detected recitation command: '{msg}'. Using generic prompt: '{modified_prompt}'")
            return True, modified_prompt
        else:
            # Not a recitation command
            return False, message

    # --- Warm-up and Unload Methods ---
    def warm_up_cache(self, cache_path: str):
        """Loads the model and specified cache state into the persistent instance."""
        if not cache_path or not Path(cache_path).exists():
            self.error_occurred.emit(f"Cannot warm up: Cache path invalid or file missing: {cache_path}")
            self.cache_status_changed.emit("Error")
            return

        # Run in background thread
        thread = threading.Thread(target=self._warm_up_cache_thread, args=(cache_path,), daemon=True)
        thread.start()

    def _warm_up_cache_thread(self, cache_path: str):
        """Background thread logic for warming up the cache."""
        with self._lock:
            # Check if already warmed up with the same cache
            if self.persistent_llm and self.warmed_cache_path == cache_path:
                logging.info(f"Cache '{Path(cache_path).name}' is already warmed up.")
                # Ensure status is correct
                self.cache_status_changed.emit("Warmed Up")
                return

            # --- Strict Model Compatibility Check ---
            current_model_id = self.config_manager.get('CURRENT_MODEL_ID')
            cache_info = self.cache_manager.get_cache_info(cache_path)
            cache_model_id = cache_info.get('model_id') if cache_info else None

            if cache_model_id and current_model_id and cache_model_id != current_model_id:
                error_msg = f"Cannot warm up: Cache '{Path(cache_path).name}' (model: {cache_model_id}) is incompatible with current model '{current_model_id}'."
                logging.error(error_msg)
                self.error_occurred.emit(error_msg)
                self.cache_status_changed.emit("Error")
                return # Stop warm-up
            elif not cache_model_id:
                 logging.warning(f"Cache '{Path(cache_path).name}' has no model ID metadata. Assuming compatible for warm-up, but use with caution.")
            # --- End Check ---

            # Get required model info from cache metadata (use cache_info already fetched)
            # cache_info = self.cache_manager.get_cache_info(cache_path) # Already fetched above
            if not cache_info:
                logging.error(f"Failed to get cache info for warming up: {cache_path}")
                self.error_occurred.emit(f"Failed to get cache info for: {Path(cache_path).name}")
                self.cache_status_changed.emit("Error")
                return

            required_model_id = cache_info.get('model_id')
            if not required_model_id:
                # If cache_model_id was None above, we assume compatibility, but need a model to load.
                # Use the current model as the best guess.
                required_model_id = current_model_id
                if not required_model_id:
                    logging.error(f"Cache info for {cache_path} is missing 'model_id' and no current model selected. Cannot warm up.")
                    self.error_occurred.emit(f"Cache '{Path(cache_path).name}' is missing model information and no model selected.")
                    self.cache_status_changed.emit("Error")
                    return
                logging.warning(f"Cache '{Path(cache_path).name}' missing model ID, attempting warm-up with current model '{required_model_id}'.")


            model_info = self.model_manager.get_model_info(required_model_id)
            if not model_info or not model_info.get('path'):
                logging.error(f"Model '{required_model_id}' required by cache '{cache_path}' not found.")
                self.error_occurred.emit(f"Model '{required_model_id}' needed for cache not found.")
                self.cache_status_changed.emit("Error")
                return
            required_model_path = str(Path(model_info['path']).resolve())
            context_window = model_info.get('context_window', 4096) # Get context window for model loading

            # --- Start Warming Process ---
            self.cache_warming_started.emit()
            self.cache_status_changed.emit("Warming Up")
            logging.info(f"Starting warm-up for cache: {cache_path} (Model: {required_model_path})")

            try:
                # Unload existing persistent model if it's different or cache was loaded
                if self.persistent_llm and (self.loaded_model_path != required_model_path or self.warmed_cache_path):
                    logging.info(f"Unloading previous model/cache ({self.loaded_model_path} / {self.warmed_cache_path}) before warming up.")
                    self.persistent_llm = None # Allow garbage collection
                    self.loaded_model_path = None
                    self.warmed_cache_path = None

                # Load model if not already loaded
                if not self.persistent_llm:
                    logging.info(f"Loading model for warm-up: {required_model_path}")
                    self.status_updated.emit("Loading model...") # Update main status bar
                    threads = int(self.config_manager.get('LLAMACPP_THREADS', os.cpu_count() or 4))
                    batch_size = int(self.config_manager.get('LLAMACPP_BATCH_SIZE', 512))
                    gpu_layers = int(self.config_manager.get('LLAMACPP_GPU_LAYERS', 0))

                    self.persistent_llm = Llama(
                        model_path=required_model_path,
                        n_ctx=context_window,
                        n_threads=threads,
                        n_batch=batch_size,
                        n_gpu_layers=gpu_layers,
                        verbose=False
                    )
                    self.loaded_model_path = required_model_path
                    logging.info("Model loaded into persistent instance.")
                    self.status_updated.emit("Idle") # Reset main status bar

                # Load cache state
                logging.info(f"Loading KV cache state for warm-up: {cache_path}")
                self.cache_status_changed.emit("Warming Up (Loading State)...")
                start_time = time.perf_counter()
                with open(cache_path, 'rb') as f_pickle:
                    state_data = pickle.load(f_pickle)
                self.persistent_llm.load_state(state_data)
                load_time = time.perf_counter() - start_time
                self.warmed_cache_path = cache_path
                logging.info(f"KV cache state loaded successfully in {load_time:.2f}s.")

                # Get metrics
                token_count = cache_info.get('token_count', 0)
                file_size = cache_info.get('size', 0)

                # Emit success signals
                self.cache_warmed_up.emit(load_time, token_count, file_size)
                self.cache_status_changed.emit("Warmed Up")

            except Exception as e:
                logging.exception(f"Error during cache warm-up for {cache_path}: {e}")
                self.error_occurred.emit(f"Error warming up cache: {e}")
                self.cache_status_changed.emit("Error")
                # Clean up potentially partially loaded state
                self.persistent_llm = None
                self.loaded_model_path = None
                self.warmed_cache_path = None
            finally:
                 self.status_updated.emit("Idle") # Ensure main status bar is reset

    def unload_cache(self):
        """Unloads the persistent model instance and cache state."""
        # Run in background thread
        thread = threading.Thread(target=self._unload_cache_thread, daemon=True)
        thread.start()

    def _unload_cache_thread(self):
        """Background thread logic for unloading the cache."""
        with self._lock:
            if not self.persistent_llm:
                logging.info("Unload called, but no persistent model/cache is loaded.")
                self.cache_status_changed.emit("Idle") # Ensure status is Idle
                return

            logging.info(f"Unloading persistent model/cache: {self.loaded_model_path} / {self.warmed_cache_path}")
            self.cache_status_changed.emit("Unloading")
            try:
                # Simply discard the reference, Python's GC will handle it
                self.persistent_llm = None
                self.loaded_model_path = None
                self.warmed_cache_path = None
                logging.info("Persistent model/cache unloaded.")
                self.cache_unloaded.emit()
                self.cache_status_changed.emit("Idle")
            except Exception as e:
                 logging.exception(f"Error during cache unload: {e}")
                 self.error_occurred.emit(f"Error unloading cache: {e}")
                 self.cache_status_changed.emit("Error") # Indicate error state

    # --- Reset State ---
    def reset_state(self) -> bool:
        """Reset the engine state, releasing any loaded models."""
        with self._lock:
            logging.info("Resetting ChatEngine state...")
            try:
                # Unload any persistent model
                self.persistent_llm = None
                self.loaded_model_path = None
                self.warmed_cache_path = None

                # Reset other relevant state
                # self._post_recitation = False # Removed

                # Update UI status
                self.cache_status_changed.emit("Idle (Reset)")
                self.cache_unloaded.emit() # Signal that cache is definitely unloaded
                self.status_updated.emit("Chat engine reset.")
                logging.info("ChatEngine state reset successfully.")
                return True
            except Exception as e:
                logging.exception(f"Error during ChatEngine reset: {e}")
                self.error_occurred.emit(f"Error resetting engine: {e}")
                self.cache_status_changed.emit("Error")
                return False

    # --- Context Safety Check ---
    def check_context_safety(self, message: str, max_tokens_for_response: int) -> Tuple[bool, Optional[str]]:
        """
        Check if adding a message might exceed context limits based on current mode.
        Returns (is_safe, warning_or_error_message).
        """
        # --- Get Context Window Size ---
        context_size = 4096 # Default fallback
        model_id = self.config_manager.get('CURRENT_MODEL_ID')
        if model_id:
            model_info = self.model_manager.get_model_info(model_id)
            if model_info:
                context_size = model_info.get('context_window', 4096)
        logging.debug(f"Context safety check using context size: {context_size}")

        # --- Get Document Token Count (from selected cache) ---
        document_tokens = 0
        # Use the currently selected cache path for the check, even if not warmed up yet
        cache_path_to_check = self.current_kv_cache_path
        if cache_path_to_check:
            cache_info = self.cache_manager.get_cache_info(cache_path_to_check)
            if cache_info:
                document_tokens = cache_info.get('token_count', 0)
        logging.debug(f"Context safety check using document tokens: {document_tokens}")

        # --- Estimate Message Tokens (Approximation) ---
        # TODO: Use a more accurate tokenizer from utils/token_counter.py if available
        message_tokens = len(message) // 4 # Rough approximation
        logging.debug(f"Context safety check using estimated message tokens: {message_tokens}")

        # --- Apply Mode-Specific Logic ---
        # Check if using any "fresh" mode
        is_fresh_mode = self.cache_behavior_mode in ["fresh_before", "fresh_after"]

        if is_fresh_mode:
            # In Fresh Context Modes, only document + new message + response matters
            buffer = 256 # Small buffer for prompts/structure
            estimated_total = document_tokens + message_tokens + max_tokens_for_response + buffer
            logging.debug(f"Fresh Context ({self.cache_behavior_mode}) safety check: Estimated total = {estimated_total}")

            if estimated_total > context_size * 0.98: # Use a slightly higher threshold for hard limit
                error_msg = (f"Potential context overflow in Fresh Context Mode ({self.cache_behavior_mode}). "
                             f"Estimated tokens ({estimated_total}) exceed 98% of limit ({context_size}). "
                             f"Try reducing 'Max Tokens' or using a smaller document.")
                logging.error(error_msg)
                return False, error_msg
            elif estimated_total > context_size * 0.90: # Warning threshold
                warning_msg = (f"High context usage warning (Fresh Context Mode: {self.cache_behavior_mode}). "
                               f"Estimated tokens ({estimated_total}) > 90% of limit ({context_size}).")
                logging.warning(warning_msg)
                return True, warning_msg # Safe, but warn
            else:
                return True, None # Safe

        else: # Standard Mode
            # In Standard Warmed-up Mode
            # The primary risk is the initial document size. Precise tracking is hard.
            # Simplified check: Warn if document itself is large.
            if document_tokens > context_size * 0.80:
                warning_msg = (f"High context usage warning (Standard Mode). "
                               f"Document uses {document_tokens} tokens (>80% of limit {context_size}). "
                               f"Conversation may lead to errors. Consider using a 'Fresh Context' mode for stability.")
                logging.warning(warning_msg)
                # Emit warning to UI as well
                self.status_updated.emit(warning_msg) # Use main status bar for this persistent warning
                return True, warning_msg # Technically safe to *start*, but warn strongly
            else:
                 # If document is small, assume okay for now in warmed-up mode
                 # (Acknowledging this doesn't prevent eventual overflow from conversation)
                 return True, None

    # --- Diagnostics for KV caches ---
    def diagnose_kv_cache(self, cache_path: str) -> Dict:
        """
        Run diagnostics on a KV cache file and return information about it.
        """
        results = {
            "exists": False,
            "size": 0,
            "compatible": False,
            "original_document_exists": False,
            "model_id": None,
            "document_id": None,
            "errors": []
        }
        
        try:
            cache_path_obj = Path(cache_path)
            
            # Check if file exists
            if not cache_path_obj.exists():
                results["errors"].append(f"KV cache file not found at {cache_path}")
                return results
                
            results["exists"] = True
            results["size"] = cache_path_obj.stat().st_size
            
            # Get cache info from manager
            cache_info = self.cache_manager.get_cache_info(cache_path)
            if not cache_info:
                results["errors"].append("No metadata found for this cache")
                return results
                
            # Check model compatibility
            results["model_id"] = cache_info.get("model_id")
            current_model_id = self.config_manager.get('CURRENT_MODEL_ID')
            
            if results["model_id"] == current_model_id:
                results["compatible"] = True
            else:
                results["errors"].append(f"Cache created with model {results['model_id']}, but current model is {current_model_id}")
                
            # Check original document
            results["document_id"] = cache_info.get("document_id")
            doc_path = cache_info.get("original_document")
            if doc_path and Path(doc_path).exists():
                results["original_document_exists"] = True
            elif doc_path:
                results["errors"].append(f"Original document not found at {doc_path}")
            else:
                results["errors"].append("No original document path in cache metadata")
                
            # Try to open the pickle file
            try:
                with open(cache_path, 'rb') as f:
                    _ = pickle.load(f)  # Just try to load it
            except Exception as e:
                results["errors"].append(f"Failed to load cache file: {str(e)}")
                
            return results
        except Exception as e:
            results["errors"].append(f"Diagnostic error: {str(e)}")
            return results

    # --- Send Message Implementation ---
    # Updated signature to accept temperature from UI
    def send_message(self, message: str, max_tokens: int = 1024, temperature: float = 0.7):
        """Send a message to the model and get a response with true KV caching support"""

        # --- REMOVED Post-Recitation State Refresh Logic ---

        # --- Context Safety Check ---
        # Use the max_tokens value passed from the UI for the safety check
        is_safe, warning_or_error = self.check_context_safety(message, max_tokens)
        if not is_safe:
            self.error_occurred.emit(warning_or_error) # Emit the specific error message
            return False # Stop processing
        elif warning_or_error:
            # Log the warning, potentially show it briefly in UI status?
            # The check_context_safety method already logs and emits for persistent warnings.
            # Maybe a short status update here?
            self.status_updated.emit("Note: High context usage.") # Brief note
            # Proceed with sending the message despite the warning

        # --- Determine if using persistent warmed-up cache ---
        # This block now ONLY determines *if* the persistent instance *should* be used
        # and gathers necessary info. The actual instance is passed to the thread.
        # The lock is acquired *inside* the thread if needed.
        use_persistent_instance = False
        llm_instance_to_use = None # Will hold either persistent or temporary llm
        model_path = None
        context_window = 4096 # Default
        model_id = self.config_manager.get('CURRENT_MODEL_ID')

        # Check conditions for using persistent instance (outside lock initially)
        should_try_persistent = (self.use_kv_cache and
                                 self.persistent_llm and
                                 self.warmed_cache_path and
                                 self.warmed_cache_path == self.current_kv_cache_path)

        if should_try_persistent:
             # If we intend to use persistent instance
             # (No pre-emptive reset needed here for Fresh Context, thread handles it)
             if should_try_persistent:
                 use_persistent_instance = True
                 llm_instance_to_use = self.persistent_llm # Pass the reference
                 model_path = self.loaded_model_path
                 try:
                     # Get context window from potentially reset instance (needs lock?)
                     # Reading n_ctx might be safe without lock, but let's be cautious if needed.
                     # For now, assume reading n_ctx is safe concurrently.
                     context_window = llm_instance_to_use.n_ctx()
                 except Exception as e_ctx:
                     logging.warning(f"Could not get n_ctx from persistent instance: {e_ctx}. Using config value.")
                     # model_id already fetched from config_manager
                     model_info = self.model_manager.get_model_info(model_id) if model_id else None
                     context_window = model_info.get('context_window', 4096) if model_info else 4096
                 logging.info(f"Will use persistent instance (Mode: {self.cache_behavior_mode}). Model: {model_path}, Cache: {self.warmed_cache_path}")

        # If not using persistent, get info for temporary load
        if not use_persistent_instance:
            logging.info("Will use temporary instance or fallback.")
            # model_id already fetched from config_manager
            if not model_id:
                self.error_occurred.emit("No model selected in configuration.")
                return False
            model_info = self.model_manager.get_model_info(model_id)
            if not model_info:
                self.error_occurred.emit(f"Model '{model_id}' not found.")
                return False
            model_path = model_info.get('path')
            if not model_path or not Path(model_path).exists():
                self.error_occurred.emit(f"Model file not found for '{model_id}': {model_path}")
                return False
            context_window = model_info.get('context_window', 4096)

        # --- Determine KV Cache Path for this specific inference ---
        actual_kv_cache_path_for_inference = None
        if self.use_kv_cache:
            if self.current_kv_cache_path and Path(self.current_kv_cache_path).exists():
                 actual_kv_cache_path_for_inference = self.current_kv_cache_path
                 logging.info(f"Target cache for inference: {actual_kv_cache_path_for_inference}")
            else:
                 # Try master cache if specific one is missing/not selected but toggle is on
                 master_cache_path_str = self.config_manager.get('MASTER_KV_CACHE_PATH')
                 if master_cache_path_str and Path(master_cache_path_str).exists():
                     actual_kv_cache_path_for_inference = str(master_cache_path_str)
                     logging.info(f"Using master KV cache for inference: {actual_kv_cache_path_for_inference}")
                 else:
                     logging.warning("KV cache enabled, but selected cache invalid and master cache invalid/missing.")
                     # Proceed without cache (will use fallback without context prepending)

        # Check for document recitation request
        is_recitation_request, modified_message = self._is_recitation_command(message)
        if is_recitation_request:
            logging.info(f"Detected document recitation request. Original: '{message}', Modified: '{modified_message}'")
            
            # For recitation, increase max_tokens if it's set too low
            if max_tokens < 2048:
                logging.info(f"Increasing max_tokens from {max_tokens} to 2048 for document recitation")
                max_tokens = 2048
                
        # Add user message to history (use original message, not modified)
        self.history.append({"role": "user", "content": message})

        # --- Start Inference Thread ---
        target_thread_func = self._inference_thread_fallback # Default to fallback

        # Use true KV cache logic if:
        # 1. A cache path is determined for this inference AND
        # 2. The global setting use_true_kv_cache_logic is enabled
        if actual_kv_cache_path_for_inference and self.use_true_kv_cache_logic:
             target_thread_func = self._inference_thread_with_true_kv_cache
             logging.info("Dispatching to TRUE KV Cache inference thread.")
        else:
             logging.info("Dispatching to FALLBACK (manual context or no context) inference thread.")

        # Pass the determined llm instance *reference* if using persistent, otherwise None
        llm_arg = llm_instance_to_use if use_persistent_instance else None

        # --- Use the temperature passed from the UI ---
        # The fixed_low_temp variable is removed. Temperature is passed directly.
        logging.info(f"Using temperature from UI: {temperature}")

        # Pass the max_tokens value and temperature from the UI, and the recitation flag
        inference_thread = threading.Thread(
            target=target_thread_func,
            args=(message, model_path, model_id, context_window,
                  actual_kv_cache_path_for_inference, max_tokens, temperature, # Pass UI temperature
                  llm_arg, is_recitation_request, modified_message), # Removed was_post_recitation_refresh flag
            daemon=True,
        )
        # Emit signal *before* starting thread so UI can disable input
        self.response_started.emit()
        inference_thread.start()

        return True

    # --- Inference thread with true KV cache logic ---
    # Modified to accept optional pre-loaded llm instance, model_id, and recitation flags
    def _inference_thread_with_true_kv_cache(self, message: str, model_path: str, model_id: str, context_window: int,
                         kv_cache_path: Optional[str], max_tokens: int, temperature: float,
                         llm: Optional[Llama] = None, is_recitation_request: bool = False,
                         modified_message: str = None): # Removed was_post_recitation_refresh flag
        """
        Thread function for model inference using true KV cache loading.
        Can use a pre-loaded persistent llm instance or load temporarily.
        Acquires lock if using the persistent instance.
        Uses the passed max_tokens value.
        """
        is_using_persistent_llm = llm is not None # Check if we received a persistent instance
        temp_llm = None # To hold temporarily loaded instance if needed
        error_message = "" # Initialize error_message
        acquired_lock = False # Track if we acquired the lock
        response_text = "" # Initialize response_text
        generated_eos = False # Flag to track if EOS was naturally generated

        try:
            # --- Get Model-Specific Configuration ---
            # Use the passed model_id, fallback to config_manager if None (shouldn't happen ideally)
            current_model_id = model_id if model_id else self.config_manager.get('CURRENT_MODEL_ID', 'unknown')
            model_config = self.config_manager.get_model_specific_config(current_model_id)
            logging.info(f"Using model-specific config for '{current_model_id}': {model_config}")

            # Extract config values for easier use
            eos_detection_method = model_config.get('eos_detection_method', 'default')
            additional_stop_tokens = set(model_config.get('additional_stop_tokens', [])) # Use a set for faster lookup
            stop_on_repetition = model_config.get('stop_on_repetition', True)
            max_no_output = model_config.get('max_empty_tokens', 50)
            repeat_threshold = model_config.get('repetition_threshold', 2)
            
            # --- Modify thresholds for recitation ---
            if is_recitation_request:
                # For document recitation, we need different thresholds:
                # 1. Higher repeat threshold (documents may have legitimate repetition)
                # 2. More tokens allowed without visible output
                # Increased from original values
                repeat_threshold = max(repeat_threshold * 6, 12)  # Significantly higher threshold for recitation
                max_no_output = max(max_no_output * 3, 150)  # Allow much more processing without visible output
                logging.info(f"Using enhanced recitation thresholds: repeat={repeat_threshold}, max_no_output={max_no_output}")

                # Also increase the temperature slightly for recitation to avoid repetition loops
                temperature = 0.2  # Slight increase from default 0.1
                logging.info(f"Using adjusted temperature for recitation: {temperature}")
            else:
                # For regular QA, use the configured values
                logging.info(f"Using standard QA thresholds: repeat={repeat_threshold}, max_no_output={max_no_output}")

            # --- Acquire lock and Handle Fresh Context Reload (if using persistent instance) ---
            if is_using_persistent_llm:
                logging.debug("Attempting to acquire lock for persistent LLM.")
                self._lock.acquire()
                acquired_lock = True
                logging.debug("Lock acquired for persistent LLM.")

                # Verify persistent_llm still exists after acquiring lock
                if not self.persistent_llm:
                     raise RuntimeError("Persistent LLM instance disappeared before inference.")
                llm = self.persistent_llm # Use the instance variable now we have the lock

                # --- Handle Cache Reload Logic (Before Query) ---
                if self.cache_behavior_mode == "fresh_before":
                    logging.info(f"Cache Mode '{self.cache_behavior_mode}': Reloading KV cache state before query.")
                    self.cache_status_changed.emit("Cache Mode: Reloading Before...")
                    try:
                        if self.warmed_cache_path and Path(self.warmed_cache_path).exists():
                            with open(self.warmed_cache_path, 'rb') as f_pickle:
                                state_data = pickle.load(f_pickle)
                            llm.load_state(state_data)
                            logging.info(f"Cache Mode '{self.cache_behavior_mode}': State reloaded successfully.")
                            self.cache_status_changed.emit("Cache Mode: Reload OK")
                        else:
                            raise FileNotFoundError(f"Warmed cache path invalid for '{self.cache_behavior_mode}' reload: {self.warmed_cache_path}")
                    except Exception as e_reload:
                        logging.error(f"Cache Mode '{self.cache_behavior_mode}': Failed to reload state: {e_reload}")
                        self.error_occurred.emit(f"Cache Reload Failed ({self.cache_behavior_mode}): {e_reload}")
                        # Critical error if reload fails in fresh context, raise to stop processing
                        raise RuntimeError(f"Cache state reload failed ({self.cache_behavior_mode}): {e_reload}") from e_reload
                else:
                    logging.debug(f"Cache Mode '{self.cache_behavior_mode}': Not reloading state before query.")

            # self.response_started.emit() # Moved to before thread start
            self.status_updated.emit("Processing...") # General status update

            if is_using_persistent_llm:
                logging.info(f"True KV cache thread using PERSISTENT instance (Mode: {self.cache_behavior_mode}). Cache: {kv_cache_path}")
                self.cache_status_changed.emit("Warmed Up (Generating)") # Update chat tab status
            else:
                # --- Load Model Temporarily ---
                 logging.info(f"True KV cache thread loading TEMPORARILY. Model: {model_path}, Cache: {kv_cache_path}")
                 self.status_updated.emit("Loading model...")
                 abs_model_path = str(Path(model_path).resolve())
                 if not Path(abs_model_path).exists():
                     raise FileNotFoundError(f"Model file not found: {abs_model_path}")

                 # Explicitly get settings from config_manager for temporary load
                 threads = int(self.config_manager.get('LLAMACPP_THREADS', os.cpu_count() or 4)) 
                 batch_size = int(self.config_manager.get('LLAMACPP_BATCH_SIZE', 512)) 
                 gpu_layers = int(self.config_manager.get('LLAMACPP_GPU_LAYERS', 0)) 
                 logging.info(f"Temporary Load Params: threads={threads}, batch_size={batch_size}, gpu_layers={gpu_layers}")

                 temp_llm = Llama(
                    model_path=abs_model_path, n_ctx=context_window, n_threads=threads,
                    n_batch=batch_size, n_gpu_layers=gpu_layers, verbose=False
                )
                 llm = temp_llm # Use the temporary instance for this inference
                 logging.info("Temporary model loaded.")
                 self.status_updated.emit("Loading KV cache state...")

                 # --- Load KV Cache Temporarily ---
                 if kv_cache_path and Path(kv_cache_path).exists():
                     logging.info(f"Loading KV cache state temporarily from: {kv_cache_path}")
                     # --- Strict Model Compatibility Check (Temporary Load) ---
                     cache_info = self.cache_manager.get_cache_info(kv_cache_path)
                     cache_model_id = cache_info.get('model_id') if cache_info else None
                     # current_model_id was already determined for the temporary model load
                     if cache_model_id and current_model_id and cache_model_id != current_model_id:
                         logging.error(f"Cache '{Path(kv_cache_path).name}' (model: {cache_model_id}) is incompatible with current model '{current_model_id}'. Skipping temporary load_state.")
                         self.error_occurred.emit(f"Cache incompatible with current model ({current_model_id}). Cannot use.")
                         # Do NOT proceed without loading state if incompatible
                         raise RuntimeError(f"Incompatible cache detected during temporary load.")
                     elif not cache_model_id:
                         logging.warning(f"Cache '{Path(kv_cache_path).name}' has no model ID metadata. Assuming compatible for temporary load.")
                     # --- End Check ---

                     # Proceed with loading state if compatible or compatibility unknown/assumed
                     try:
                         with open(kv_cache_path, 'rb') as f_pickle:
                             state_data = pickle.load(f_pickle)
                         llm.load_state(state_data)
                         logging.info("Temporary KV cache state loaded successfully.")
                         self.cache_status_changed.emit("Using TRUE KV Cache") # Update chat tab status
                     except Exception as e_load:
                         logging.error(f"Error loading temporary KV cache state: {e_load}. Proceeding without cache state.")
                         # Don't raise, just proceed without the loaded state
                 else:
                      logging.warning("KV cache path invalid or missing for temporary load. Proceeding without cache state.")

            # --- Common Logic: Tokenize, Evaluate, Generate ---
            self.status_updated.emit("Generating response...")
            self.cache_status_changed.emit("Warmed Up (Generating)" if is_using_persistent_llm else "Using TRUE KV Cache (Generating)")

            # ---> WARM-UP STEP: Evaluate BOS token after state load <---
            # This is crucial to ensure the model's attention mechanism
            # correctly incorporates the loaded KV cache state, especially
            # when using modes like 'standard' or 'fresh_after' where the
            # state might have been loaded earlier (during warm-up).
            # Also beneficial after 'fresh_before' reload.
            try:
                bos_token = llm.token_bos()
                if bos_token is not None: # Ensure BOS token exists
                    logging.info(f"Evaluating BOS token ({bos_token}) to prime attention after state load/reload...")
                    llm.eval([bos_token])
                    logging.info("BOS token evaluated successfully.")
                else:
                    logging.warning("Model does not have a BOS token defined, skipping BOS evaluation.")
            except Exception as e_bos:
                # Log error but don't necessarily stop inference, as it might still work partially
                logging.error(f"Error evaluating BOS token after state load: {e_bos}")
                self.error_occurred.emit(f"Warning: Error priming model state with BOS token: {e_bos}")
            # ---> END WARM-UP STEP <---


            # --- Use different prompts for recitation vs QA ---
            # is_paragraph_recitation = False # Flag for post-processing - REMOVED
            if is_recitation_request:
                # Always use generic full document recitation prompt now
                instruction_prefix = "\n\nYou are a precise document recitation system. Your task is to accurately recite the content of the loaded document, starting from the VERY BEGINNING of the document. Do not skip any content. Start with the first sentence and continue in order.\n\n"
                recitation_prompt = "Document recitation request: Output the document text in order starting from the beginning:\n\n"
                logging.info("Using generic full document recitation prompt.")
                full_input_text = instruction_prefix + recitation_prompt
            else:
                # Refined QA prompt for True KV Cache, emphasizing structured context
                instruction_prefix = "\n\nYou are an assistant answering questions based on a structured document (formatted using Markdown with headings, lists, tables, etc.) that has been loaded into your memory via a KV cache. Carefully consult this entire loaded context to answer the question accurately and comprehensively. Pay close attention to details like dates, names, and events mentioned throughout the text.\n\n"
                question_prefix = "Question: "
                suffix_text = "\n\nPlease provide a comprehensive answer based on all relevant information found in the loaded document context.\nAnswer: "
                full_input_text = instruction_prefix + question_prefix + message + suffix_text
                logging.info("Using refined QA prompt for True KV Cache.")

            # --- Tokenize user input with structure ---
            input_tokens = llm.tokenize(full_input_text.encode('utf-8'))
            logging.info(f"Tokenized user input with structure ({len(input_tokens)} tokens)")

            # --- Evaluate input tokens (Unconditionally) ---
            # Always evaluate the input tokens (containing the appropriate prompt)
            # before starting generation, even for recitation after state load.
            logging.info(f"Evaluating {len(input_tokens)} input tokens...")
            llm.eval(input_tokens)
            logging.info("Input tokens evaluated.")

            # --- Generate response using low-level sampling for ALL cases (QA and Recitation) ---
            
            # Adjust settings based on request type
            if is_recitation_request:
                # Use recitation-specific settings
                sampling_temp = 0.0 # Force deterministic output for recitation
                sampling_repeat_threshold = repeat_threshold # Use enhanced value calculated earlier
                sampling_max_no_output = max_no_output # Use enhanced value calculated earlier
                logging.info(f"Generating Recitation using low-level token sampling (max_tokens={max_tokens}, temp={sampling_temp})")
            else:
                 # Use QA settings
                 sampling_temp = temperature # Use the temperature passed from the UI
                 sampling_repeat_threshold = repeat_threshold # Use standard value calculated earlier
                 sampling_max_no_output = max_no_output # Use standard value calculated earlier
                 logging.info(f"Generating QA response using low-level token sampling (max_tokens={max_tokens}, temp={sampling_temp})")

            # --- Sampling Loop (Common for QA and Recitation) ---
            eos_token = llm.token_eos()
            logging.debug(f"EOS token ID: {eos_token} (type: {type(eos_token)})")
            tokens_generated = []
            # response_text = "" # Initialized earlier

            # Track potential stopping conditions (initialize based on config)
            generated_eos = False
            consecutive_repeats = 0
            # repeat_threshold set from model_config
            no_output_tokens = 0  # Count tokens that don't produce visible output
            # max_no_output set from model_config
            last_response_length = 0
            
            # For better repetition detection
            recent_output_chunks = []  # Store recent output chunks for pattern detection
            max_stored_chunks = 10     # Maximum number of chunks to store
            chunk_min_size = 20        # Minimum size of text to consider for chunk analysis

            # Combine standard EOS with additional stop tokens from config
            stop_token_ids = {int(eos_token)} | {int(t) for t in additional_stop_tokens} # Ensure all are ints
            # --- REMOVED paragraph stop token logic ---

            logging.debug(f"Effective stop token IDs: {stop_token_ids}")

            # Use the max_tokens value passed into the function
            # Get repeat penalty from config, default to 1.1
            repeat_penalty_value = float(self.config_manager.get('LLAMACPP_REPEAT_PENALTY', 1.1))
            logging.info(f"Using repeat_penalty: {repeat_penalty_value}")

            for i in range(max_tokens):
                # Apply the determined temperature and repeat_penalty for sampling
                token_id = llm.sample(temp=sampling_temp, repeat_penalty=repeat_penalty_value) # Apply specific temp and penalty here
                
                # Log sampled token ID periodically or near the end for debugging
                if i >= max_tokens - 20: # Log last 20 tokens sampled
                    logging.debug(f"Sampled token ID at step {i}: {token_id} (type: {type(token_id)})")

                # Enhanced EOS detection using model_config
                is_eos = False
                try:
                    token_id_int = int(token_id) # Convert once for comparisons

                    # Primary check: Is the token in our combined stop set?
                    if token_id_int in stop_token_ids:
                        logging.debug(f"Stop token {token_id_int} encountered based on stop_token_ids set.")
                        is_eos = True

                    # Apply model-specific methods if needed (e.g., 'gemma' might have unique logic)
                    elif eos_detection_method == 'gemma':
                        # Add any Gemma-specific checks here if necessary
                        # For now, relying on additional_stop_tokens is likely sufficient
                        pass
                    elif eos_detection_method == 'strict':
                        # Only check against the official EOS token
                        is_eos = (token_id_int == int(eos_token))
                    elif eos_detection_method == 'flexible':
                        # Add more lenient checks if needed (e.g., string comparison)
                        if str(token_id).strip() == str(eos_token).strip():
                             is_eos = True
                    # Default behavior already covered by stop_token_ids check

                except ValueError:
                    logging.warning(f"Could not convert sampled token ID '{token_id}' to int for EOS check.")
                except Exception as e:
                    logging.error(f"Unexpected error during EOS detection: {e}")

                if is_eos:
                    logging.info(f"Stop token encountered at step {i} (ID: {token_id}). Method: {eos_detection_method}.")
                    generated_eos = True
                    break  # Exit the loop immediately

                # Add token to generated list
                tokens_generated.append(token_id)

                # --- Debug Logging ---
                if self.debug_token_generation and i < 200:  # Limit to first 200 tokens to avoid log bloat
                    try:
                        # Use a temporary list to avoid modifying the main list for detokenization
                        token_text = llm.detokenize([token_id]).decode('utf-8', errors='replace')
                        # Sanitize token text for logging (replace control characters)
                        sanitized_text = ''.join(c if c.isprintable() else f'\\x{ord(c):02x}' for c in token_text)
                        logging.debug(f"Token {i}: ID={token_id}, Text='{sanitized_text}', Hex={hex(token_id)}")
                    except Exception as debug_e:
                        logging.debug(f"Token {i}: ID={token_id}, (Error detokenizing: {debug_e})")
                # --- End Debug Logging ---

                llm.eval([token_id])

                # --- Custom Repetition Logic Temporarily Disabled (2025-03-31) ---
                # Relying solely on repeat_penalty from llama.cpp for now.
                # Periodically check the output text to detect repetition or lack of progress
                # # Check more frequently, e.g., every 8 tokens, or near the end
                # check_interval = 8
                # if (i + 1) % check_interval == 0 or i >= max_tokens - 20:
                #     current_text = llm.detokenize(tokens_generated).decode('utf-8', errors='replace')
                #     new_text = current_text[len(response_text):]

                #     # Check for no new content (only whitespace or empty)
                #     # Use the sampling_max_no_output determined earlier
                #     if not new_text.strip():
                #         no_output_tokens += check_interval # Approximate since we check periodically
                #         if no_output_tokens >= sampling_max_no_output:
                #             logging.info(f"No meaningful output for ~{no_output_tokens} tokens. Stopping early at step {i}.")
                #             break
                #     else:
                #         no_output_tokens = 0  # Reset counter when we get new content

                #     # Enhanced repetition detection - store recent chunks for better analysis
                #     if current_text and len(current_text) > chunk_min_size:
                #         # Get the new text since last check
                #         if new_text and len(new_text) > 0:
                #             # Add new chunk to our buffer
                #             recent_output_chunks.append(new_text)
                #             # Keep only most recent chunks
                #             if len(recent_output_chunks) > max_stored_chunks:
                #                 recent_output_chunks.pop(0)
                                
                #             # Check for repeating patterns in the recent chunks
                #             # Use the sampling_repeat_threshold determined earlier
                #             if stop_on_repetition and len(recent_output_chunks) >= 2:
                #                 # Basic repetition check (same as before but using our chunks)
                #                 if len(recent_output_chunks) >= 2 and recent_output_chunks[-1].strip() == recent_output_chunks[-2].strip() and recent_output_chunks[-1].strip():
                #                     consecutive_repeats += 1
                #                     logging.debug(f"Repetition detected ({consecutive_repeats}/{sampling_repeat_threshold}) at step {i}.")
                #                     if consecutive_repeats >= sampling_repeat_threshold:
                #                         logging.info(f"Repetitive output detected ({consecutive_repeats} times >= threshold {sampling_repeat_threshold}) at step {i}. Stopping early.")
                #                         break
                #                 else:
                #                     # More advanced check: look for repeating sequences
                #                     # This would detect "A B A B A B" type patterns
                #                     repeating_pattern_found = False
                #                     if len(recent_output_chunks) >= 4:  # Need at least 4 chunks to detect a pattern of length 2
                #                         for pattern_len in range(1, min(3, len(recent_output_chunks) // 2)):  # Check patterns of length 1, 2
                #                             last_chunks = recent_output_chunks[-pattern_len*2:]  # Get the last pattern_len*2 chunks
                #                             first_part = ''.join(last_chunks[:pattern_len]).strip()
                #                             second_part = ''.join(last_chunks[pattern_len:]).strip()
                #                             if first_part and first_part == second_part:  # Non-empty pattern repeating
                #                                 repeating_pattern_found = True
                #                                 logging.debug(f"Pattern repetition of length {pattern_len} detected at step {i}.")
                #                                 consecutive_repeats += 1
                #                                 break
                                                
                #                     if not repeating_pattern_found:
                #                         consecutive_repeats = 0  # Reset if no repetition found
                #             else:
                #                 consecutive_repeats = 0  # Reset if repetition check disabled
                #         else:
                #             consecutive_repeats = 0  # Reset if no new text

                #     # Check for repeated content (identical chunks) - Use model_config setting
                #     # Use the sampling_repeat_threshold determined earlier
                #     if stop_on_repetition and current_text and len(current_text) > 20:
                #         # Check for repetitions by comparing the last two equal-length chunks
                #         # Make chunk size dynamic but reasonable
                #         check_len = min(max(20, len(current_text) // 4), 100) # Check 25% up to 100 chars
                #         if len(current_text) >= 2 * check_len: # Ensure enough text for comparison
                #             last_chunk = current_text[-check_len:]
                #             previous_chunk = current_text[-2*check_len:-check_len]
                #             if last_chunk == previous_chunk and last_chunk.strip(): # Avoid stopping on repeated whitespace
                #                 consecutive_repeats += 1
                #                 logging.debug(f"Repetition detected ({consecutive_repeats}/{sampling_repeat_threshold}) at step {i}. Chunks: '{previous_chunk}' == '{last_chunk}'")
                #                 if consecutive_repeats >= sampling_repeat_threshold:
                #                     logging.info(f"Repetitive output detected ({consecutive_repeats} times >= threshold {sampling_repeat_threshold}) at step {i}. Stopping early.")
                #                     break
                #             else:
                #                 # Only reset if not already tracked by our enhanced detection
                #                 if not recent_output_chunks or len(recent_output_chunks) < 2:
                #                     consecutive_repeats = 0 # Reset if chunks differ
                #     elif not stop_on_repetition:
                #          consecutive_repeats = 0 # Ensure counter is reset if check is disabled

                    # Emit chunk and update response text - Moved outside the disabled block
                    # if new_text:
                    #     self.response_chunk.emit(new_text)
                    #     response_text = current_text

                    # QCoreApplication.processEvents() # Moved outside the disabled block

                # Additional early stopping criteria: Token repetition loop (use config)
                # Check if stop_on_repetition is enabled, as this covers a similar case
                # --- Restored Token Loop Check (2025-03-31) ---
                # if stop_on_repetition and i > 100 and len(tokens_generated) >= 5:
                #     last_5_tokens = tokens_generated[-5:]
                #     # Check if all last 5 tokens are identical AND not the EOS/stop tokens
                #     # (Avoid stopping if the model correctly outputs multiple EOS tokens)
                #     try: # Add try-except for int conversion
                #         first_token_int = int(last_5_tokens[0]) # Convert first token once
                #         if first_token_int not in stop_token_ids and all(int(t) == first_token_int for t in last_5_tokens):
                #             logging.info(f"Detected token repetition loop (token ID: {first_token_int}) at step {i}. Stopping early.")
                #             break
                #     except ValueError:
                #          logging.warning(f"Could not convert token ID {last_5_tokens[0]} to int for loop check.")
                # --- End Restored Check ---
                # --- End Custom Repetition Logic Temporarily Disabled ---

                # Emit chunk and update response text (Needs to happen regardless of checks)
                # Check periodically to avoid doing it every single token
                check_interval = 8 # Or some other reasonable interval
                if (i + 1) % check_interval == 0 or i >= max_tokens - 20:
                    current_text = llm.detokenize(tokens_generated).decode('utf-8', errors='replace')
                    new_text = current_text[len(response_text):]
                    if new_text:
                        self.response_chunk.emit(new_text)
                        response_text = current_text
                    QCoreApplication.processEvents()


            # Ensure final text is emitted (if loop finished or broke early)
            final_text = llm.detokenize(tokens_generated).decode('utf-8', errors='replace')
            if len(final_text) > len(response_text):
                 self.response_chunk.emit(final_text[len(response_text):])
            response_text = final_text

            # Add special endings for different conditions
            if not generated_eos and len(tokens_generated) >= max_tokens:
                truncation_msg = f" [... response truncated at {max_tokens} tokens]"
                if is_recitation_request:
                    truncation_msg = f" [... document text continues beyond the {max_tokens} token limit ...]"
                logging.warning(f"Response truncated at {max_tokens} tokens.")
                response_text += truncation_msg
                self.response_chunk.emit(truncation_msg) # Emit the truncation message as a final chunk
            elif not generated_eos and stop_on_repetition and consecutive_repeats >= sampling_repeat_threshold:
                 # If we stopped early due to repetition
                 logging.info(f"Response generation stopped early due to repetition detection.")
                 truncation_msg = " [... response stopped due to repetitive content ...]"
                 response_text += truncation_msg
                 self.response_chunk.emit(truncation_msg)
            elif not generated_eos and no_output_tokens >= sampling_max_no_output:
                 # If we stopped early due to no output tokens
                 logging.info(f"Response generation stopped early due to no meaningful output.")
                 truncation_msg = " [... response stopped, no further content generated ...]"
                 response_text += truncation_msg
                 self.response_chunk.emit(truncation_msg)

            log_msg_type = "Recitation" if is_recitation_request else "QA response"
            logging.info(f"Generated {log_msg_type} with {len(tokens_generated)} tokens. EOS generated: {generated_eos}")

            # --- REMOVED Post-processing for single paragraph recitation ---

            # --- REMOVED Post-Recitation Flag Setting ---

            # --- Finalize (Common for both paths now) ---
            if response_text.strip():
                self.history.append({"role": "assistant", "content": response_text})
                # self.response_complete.emit(response_text, True) # Moved to finally block
            else:
                # Handle case where loop finished but response is empty (e.g., only EOS generated)
                if generated_eos and not response_text.strip():
                     logging.warning("Model generated EOS immediately or only whitespace.")
                     # Decide if this is an error or just an empty valid response
                     # Let's treat it as success with empty content for now
                     # self.error_occurred.emit("Model generated an empty response.")
                else:
                     # Only emit error if not generated_eos or if response_text is still empty
                     if not generated_eos:
                         logging.warning("Model generated an empty response (unknown reason).")
                         self.error_occurred.emit("Model generated an empty response.")
                # self.response_complete.emit("", False) # Moved to finally block

        except Exception as e:
            error_message = f"Error during true KV cache inference: {str(e)}"
            logging.exception(error_message)
            self.error_occurred.emit(error_message)
            # self.response_complete.emit("", False) # Moved to finally block
            self.cache_status_changed.emit("Error")
        finally:
            # --- Handle Cache Reload Logic (After Query) ---
            if acquired_lock and self.cache_behavior_mode == "fresh_after":
                logging.info(f"Cache Mode '{self.cache_behavior_mode}': Reloading KV cache state after query completion/error.")
                self.cache_status_changed.emit("Cache Mode: Reloading After...")
                try:
                    # Re-verify conditions inside lock, just in case
                    if self.persistent_llm and self.warmed_cache_path and Path(self.warmed_cache_path).exists():
                        with open(self.warmed_cache_path, 'rb') as f_pickle:
                            state_data = pickle.load(f_pickle)
                        self.persistent_llm.load_state(state_data)
                        logging.info(f"Cache Mode '{self.cache_behavior_mode}': State reloaded successfully after query.")
                        self.cache_status_changed.emit("Cache Mode: Reload OK (Post)") # Indicate post-query reload
                    else:
                        logging.warning(f"Cache Mode '{self.cache_behavior_mode}': Conditions for post-query reload not met.")
                        self.cache_status_changed.emit("Cache Mode: Reload Failed (Post)")
                except Exception as e_reload_after:
                    # Log error but don't raise, as main operation might have succeeded
                    logging.error(f"Cache Mode '{self.cache_behavior_mode}': Failed to reload state after query: {e_reload_after}")
                    self.error_occurred.emit(f"Cache Reload Failed After Query ({self.cache_behavior_mode}): {e_reload_after}")
                    self.cache_status_changed.emit("Cache Mode: Reload Failed (Post)")

            # --- Release lock ONLY if it was acquired ---
            if acquired_lock: # Ensure this aligns with the 'if acquired_lock and ...' block above it
                self._lock.release()
                logging.debug("Lock released for persistent LLM in true KV thread.")
            # Clean up temporary llm instance if one was created
            if temp_llm:
                logging.info("Releasing temporary Llama instance.")
                temp_llm = None # Allow GC

            # Emit completion signal *after* releasing lock and cleaning up
            emit_start_time = time.perf_counter()
            # Explicitly cast success to bool to fix TypeError
            success = bool(not error_message and response_text.strip())
            self.response_complete.emit(response_text if success else "", success)
            emit_duration = time.perf_counter() - emit_start_time
            logging.debug(f"Emitting response_complete took {emit_duration:.4f}s")

            # Reset status - More informative main status
            main_status = "Ready (Cache Warmed)" if is_using_persistent_llm and not error_message else "Idle"
            self.status_updated.emit(main_status)
            # Reset chat tab status more reliably
            final_chat_status = "Error" if error_message else ("Warmed Up" if is_using_persistent_llm else "Idle")
            self.cache_status_changed.emit(final_chat_status)
            logging.debug("True KV cache inference thread finished.")


    # --- Fallback inference method ---
    # Modified to accept optional pre-loaded llm instance and recitation flags
    def _inference_thread_fallback(self, message: str, model_path: str, model_id: str, context_window: int,
                        kv_cache_path: Optional[str], max_tokens: int, temperature: float,
                        llm: Optional[Llama] = None, is_recitation_request: bool = False,
                        modified_message: str = None): # Removed was_post_recitation_refresh flag
        """
        Fallback inference method using manual context prepending or no context.
        Can optionally receive a pre-loaded Llama instance (less common now).
        Acquires lock if using the persistent instance.
        Uses the passed max_tokens value.
        """
        is_using_persistent_llm = llm is not None # Check if we received a persistent instance
        temp_llm = None # To hold temporarily loaded instance if needed
        error_message = "" # Initialize error_message
        acquired_lock = False # Track if we acquired the lock
        complete_response = "" # Initialize response

        try:
            # --- Acquire lock ONLY if using the persistent instance ---
            # Less likely here, but good practice if llm could be persistent
            if is_using_persistent_llm:
                logging.debug("Attempting to acquire lock for persistent LLM in fallback thread.")
                self._lock.acquire()
                acquired_lock = True
                logging.debug("Lock acquired for persistent LLM in fallback thread.")
                # Verify persistent_llm still exists
                if not self.persistent_llm:
                     raise RuntimeError("Persistent LLM instance disappeared before fallback inference.")
                llm = self.persistent_llm

            # self.response_started.emit() # Moved to before thread start
            self.status_updated.emit("Processing...") # General status update
            self.cache_status_changed.emit("Fallback (Generating)") # Update chat tab status

            # --- Load Model Temporarily (if not passed in) ---
            if not is_using_persistent_llm:
                self.status_updated.emit("Fallback: Loading model...")
                logging.info("Fallback: Loading model temporarily...")
                abs_model_path = str(Path(model_path).resolve())
                if not Path(abs_model_path).exists():
                    raise FileNotFoundError(f"Model file not found: {abs_model_path}")
                
                # Explicitly get settings from config_manager for temporary load
                threads = int(self.config_manager.get('LLAMACPP_THREADS', os.cpu_count() or 4)) 
                batch_size = int(self.config_manager.get('LLAMACPP_BATCH_SIZE', 512)) 
                gpu_layers = int(self.config_manager.get('LLAMACPP_GPU_LAYERS', 0)) 
                logging.info(f"Fallback Load Params: threads={threads}, batch_size={batch_size}, gpu_layers={gpu_layers}")

                temp_llm = Llama(
                   model_path=abs_model_path, n_ctx=context_window, n_threads=threads,
                   n_batch=batch_size, n_gpu_layers=gpu_layers, verbose=False
               )
                llm = temp_llm # Use the temporary instance
                logging.info("Fallback: Temporary model loaded.")
            else:
                 logging.info("Fallback: Using pre-loaded Llama instance.")

            # --- Prepare Chat History with Manual Context Prepending (if cache path provided) ---
            chat_messages = []
            
            # Use different system prompts for recitation vs QA
            if is_recitation_request:
                # Recitation prompt remains the same
                system_prompt_content = "You are a precise document recitation system. Your task is to recite the exact content of the provided text, starting from the beginning. Don't add anything, don't modify anything, just output the exact text."
            else:
                # Default QA prompt if NO context snippet is available
                system_prompt_content = "\n\nYou are an assistant tasked with answering questions based on general knowledge, as no specific document context is available.\n\n"

            if kv_cache_path: # Use kv_cache_path to find original doc for prepending
                logging.info("Fallback: Attempting to prepend original document context.")
                doc_context_text = ""
                try:
                    cache_info = self.cache_manager.get_cache_info(kv_cache_path)
                    if cache_info and 'original_document' in cache_info:
                        original_doc_path_str = cache_info['original_document']
                        if original_doc_path_str != "Unknown":
                            original_doc_path = Path(original_doc_path_str)
                            if original_doc_path.exists():
                                # For recitation, read more of the document (16000 chars instead of 8000)
                                doc_chars = 16000 if is_recitation_request else 8000
                                with open(original_doc_path, 'r', encoding='utf-8', errors='replace') as f_doc:
                                    doc_context_text = f_doc.read(doc_chars) # Read snippet
                                logging.info(f"Fallback: Read {len(doc_context_text)} chars for prepending.")
                            else: logging.warning(f"Fallback: Original doc path not found: {original_doc_path}")
                        else: logging.warning(f"Fallback: Original doc path is 'Unknown' for cache: {kv_cache_path}")
                    else: logging.warning(f"Fallback: No cache info or original doc path for cache: {kv_cache_path}")

                    if doc_context_text:
                        if is_recitation_request:
                            system_prompt_content = (
                                f"You are a precise document recitation system. Your task is to recite the exact content of the provided text, starting from the beginning. "
                                f"Don't add anything, don't modify anything, just output the exact text.\n\n"
                                f"--- TEXT START ---\n{doc_context_text}...\n--- TEXT END ---\n\n"
                                f"Beginning recitation from the start of the document:"
                            )
                        else:
                            system_prompt_content = (
                                f"Use the following text snippet to answer the user's question:\n"
                                f"--- TEXT SNIPPET START ---\n{doc_context_text}...\n--- TEXT SNIPPET END ---\n\n"
                                # Refined QA prompt when using a fallback snippet
                                f"\n\nYou are an assistant answering questions based *only* on the following text snippet. Read the snippet carefully and answer comprehensively based *only* on the information within it.\n\n"
                                f"--- TEXT SNIPPET START ---\n{doc_context_text}...\n--- TEXT SNIPPET END ---\n\n"
                                # The user's question will be added later in the chat history
                            )
                        logging.info("Fallback: Using refined QA prompt with prepended context snippet.")
                    else:
                        logging.warning("Fallback: Failed to read context snippet, using default no-context QA prompt.")
                        # system_prompt_content remains the default no-context prompt defined earlier
                except Exception as e_ctx:
                    logging.error(f"Fallback: Error retrieving context snippet: {e_ctx}")
                    logging.warning("Fallback: Using default no-context QA prompt.")
                    # system_prompt_content remains the default no-context prompt defined earlier
            else:
                 logging.info("Fallback: No cache path provided, using default no-context QA prompt.")
                 # system_prompt_content remains the default no-context prompt defined earlier


            # Add system prompt (which now varies based on context availability)
            chat_messages.append({"role": "system", "content": system_prompt_content})
            
            # For recitation, use a simpler approach with fewer history messages
            if is_recitation_request:
                # Use the modified message that was determined by _is_recitation_command
                if modified_message:
                    chat_messages.append({"role": "user", "content": modified_message})
                else:
                    chat_messages.append({"role": "user", "content": "Please recite the document from the beginning."})
                
                # For recitation, increase temperature slightly to avoid loops
                if temperature < 0.2:
                    temperature = 0.2
                    logging.info(f"Adjusted temperature for recitation to {temperature}")
            else:
                # Add recent history for regular QA
                history_limit = 4
                start_index = max(0, len(self.history) - 1 - history_limit)
                recent_history = self.history[start_index:-1]
                chat_messages.extend(recent_history)
                # Add latest user message
                chat_messages.append(self.history[-1])
                
            logging.info(f"Fallback: Prepared chat history with {len(chat_messages)} messages.")

            # --- Generate Response (Streaming using create_chat_completion) ---
            self.status_updated.emit("Fallback: Generating response...")
            # Get repeat penalty from config, default to 1.1
            repeat_penalty_value = float(self.config_manager.get('LLAMACPP_REPEAT_PENALTY', 1.1))
            logging.info(f"Fallback: Generating response using create_chat_completion (max_tokens={max_tokens}, temp={temperature}, repeat_penalty={repeat_penalty_value})...") # Log max_tokens and penalty
            stream = llm.create_chat_completion(
                messages=chat_messages,
                max_tokens=max_tokens, # Use passed max_tokens
                temperature=temperature, # Use temperature passed from UI
                repeat_penalty=repeat_penalty_value, # Apply repeat penalty
                stream=True
             )

            # complete_response = "" # Moved initialization up
            for chunk in stream:
                try:
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content")
                    if text:
                        self.response_chunk.emit(text)
                        complete_response += text
                except (KeyError, IndexError, TypeError) as e:
                    logging.warning(f"Fallback: Could not extract text from stream chunk: {chunk}, Error: {e}")


            logging.info("Fallback: Response generation complete.")

            # --- Finalize ---
            if complete_response.strip():
                self.history.append({"role": "assistant", "content": complete_response})
                # self.response_complete.emit(complete_response, True) # Moved to finally
            else:
                logging.warning("Fallback: Model stream completed but produced no text.")
                self.error_occurred.emit("Model generated an empty response.")
                # self.response_complete.emit("", False) # Moved to finally
                
            # Set post-recitation flag if needed (original logic)
            if is_recitation_request and is_using_persistent_llm:
                # self._post_recitation = True # Removed - handled by cache mode
                logging.info("Marking model for state refresh on next query (fallback mode)") # Kept log for now

        except Exception as e:
            error_message = f"Error during fallback inference: {str(e)}"
            logging.exception(error_message)
            self.error_occurred.emit(error_message)
            # self.response_complete.emit("", False) # Moved to finally
            self.cache_status_changed.emit("Error")
        finally:
            # --- Release lock ONLY if it was acquired ---
            if acquired_lock:
                self._lock.release()
                logging.debug("Lock released for persistent LLM in fallback thread.")
            # Clean up temporary llm instance if one was created
            if temp_llm:
                logging.info("Releasing temporary Llama instance from fallback.")
                temp_llm = None # Allow GC

            # Emit completion signal *after* releasing lock and cleaning up
            emit_start_time = time.perf_counter()
            # Explicitly cast success to bool to fix TypeError
            success = bool(not error_message and complete_response.strip())
            self.response_complete.emit(complete_response if success else "", success)
            emit_duration = time.perf_counter() - emit_start_time
            logging.debug(f"Emitting response_complete (fallback) took {emit_duration:.4f}s")

            # Reset status - More informative main status
            main_status = "Ready (Cache Warmed)" if is_using_persistent_llm and not error_message else "Idle"
            self.status_updated.emit(main_status)
            # Reset chat tab status more reliably
            final_chat_status = "Error" if error_message else "Idle" # Fallback always ends in Idle or Error
            self.cache_status_changed.emit(final_chat_status)
            logging.debug("Fallback inference thread finished.")
