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
# import shutil # No longer needed?
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

    def __init__(self, config, llama_manager, model_manager, cache_manager):
        """Initialize chat engine"""
        super().__init__()
        self.config = config
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
        self.use_true_kv_cache_logic = self.config.get('USE_TRUE_KV_CACHE', True)
        # Fresh Context Mode setting
        self.fresh_context_mode = self.config.get('USE_FRESH_CONTEXT', False) # Default to False
        logging.info(f"ChatEngine initialized. True KV Cache Logic: {self.use_true_kv_cache_logic}, Fresh Context Mode: {self.fresh_context_mode}")


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

            self.current_kv_cache_path = str(cache_path)
            logging.info(f"Set current KV cache path to {self.current_kv_cache_path}")
            # TODO: Verify cache compatibility with current model?
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
        # self.status_updated.emit(f"KV Cache Usage: {'Enabled' if enabled else 'Disabled'}")
        # Emit specific status for chat tab display
        self.cache_status_changed.emit("Idle" if not self.warmed_cache_path else "Warmed Up")

    def enable_fresh_context_mode(self, enabled: bool):
        """Enable or disable Fresh Context Mode."""
        with self._lock: # Ensure thread safety when changing mode
            self.config['USE_FRESH_CONTEXT'] = enabled # Update runtime config
            self.fresh_context_mode = enabled
            logging.info(f"Fresh Context Mode {'enabled' if enabled else 'disabled'}")
            # Emit status change to update UI elements potentially
            status_msg = f"Fresh Context Mode: {'Enabled' if enabled else 'Disabled'}"
            self.cache_status_changed.emit(status_msg)
            self.status_updated.emit(status_msg) # Also update main status bar


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

            # Get required model info from cache metadata
            cache_info = self.cache_manager.get_cache_info(cache_path)
            if not cache_info:
                logging.error(f"Failed to get cache info for warming up: {cache_path}")
                self.error_occurred.emit(f"Failed to get cache info for: {Path(cache_path).name}")
                self.cache_status_changed.emit("Error")
                return

            required_model_id = cache_info.get('model_id')
            if not required_model_id:
                logging.error(f"Cache info for {cache_path} is missing 'model_id'. Cannot warm up.")
                self.error_occurred.emit(f"Cache '{Path(cache_path).name}' is missing model information.")
                self.cache_status_changed.emit("Error")
                return

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
                    threads = int(self.config.get('LLAMACPP_THREADS', os.cpu_count() or 4))
                    batch_size = int(self.config.get('LLAMACPP_BATCH_SIZE', 512))
                    gpu_layers = int(self.config.get('LLAMACPP_GPU_LAYERS', 0))

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

                # Reset other relevant state if necessary (e.g., history?)
                # self.history = [] # Optional: Decide if reset should clear history too

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
        model_id = self.config.get('CURRENT_MODEL_ID')
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
        if self.fresh_context_mode:
            # In Fresh Context Mode, only document + new message + response matters
            buffer = 256 # Small buffer for prompts/structure
            estimated_total = document_tokens + message_tokens + max_tokens_for_response + buffer
            logging.debug(f"Fresh Context safety check: Estimated total = {estimated_total}")

            if estimated_total > context_size * 0.98: # Use a slightly higher threshold for hard limit
                error_msg = (f"Potential context overflow in Fresh Context Mode. "
                             f"Estimated tokens ({estimated_total}) exceed 98% of limit ({context_size}). "
                             f"Try reducing 'Max Tokens' or using a smaller document.")
                logging.error(error_msg)
                return False, error_msg
            elif estimated_total > context_size * 0.90: # Warning threshold
                warning_msg = (f"High context usage warning (Fresh Context Mode). "
                               f"Estimated tokens ({estimated_total}) > 90% of limit ({context_size}).")
                logging.warning(warning_msg)
                return True, warning_msg # Safe, but warn
            else:
                return True, None # Safe

        else:
            # In Warmed-up Mode (NOT Fresh Context)
            # The primary risk is the initial document size. Precise tracking is hard.
            # Simplified check: Warn if document itself is large.
            if document_tokens > context_size * 0.80:
                warning_msg = (f"High context usage warning (Warmed-up Mode). "
                               f"Document uses {document_tokens} tokens (>80% of limit {context_size}). "
                               f"Conversation may lead to errors. Enable 'Fresh Context Mode' for stability.")
                logging.warning(warning_msg)
                # Emit warning to UI as well
                self.status_updated.emit(warning_msg) # Use main status bar for this persistent warning
                return True, warning_msg # Technically safe to *start*, but warn strongly
            else:
                 # If document is small, assume okay for now in warmed-up mode
                 # (Acknowledging this doesn't prevent eventual overflow from conversation)
                 return True, None


    # --- Send Message Implementation ---
    def send_message(self, message: str, max_tokens: int = 1024, temperature: float = 0.7):
        """Send a message to the model and get a response with true KV caching support"""

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

        # Check conditions for using persistent instance (outside lock initially)
        should_try_persistent = (self.use_kv_cache and
                                 self.persistent_llm and
                                 self.warmed_cache_path and
                                 self.warmed_cache_path == self.current_kv_cache_path)

        if should_try_persistent:
             # If Fresh Context, we need to reset state *before* the thread starts using it.
             # This reset MUST be protected by the lock.
             if self.fresh_context_mode:
                 logging.info("Fresh Context Mode enabled. Attempting state reset before dispatching thread.")
                 self.cache_status_changed.emit("Fresh Context: Resetting...")
                 reset_success = False
                 with self._lock: # Acquire lock *only* for the reset operation
                     try:
                         if self.warmed_cache_path and Path(self.warmed_cache_path).exists():
                             with open(self.warmed_cache_path, 'rb') as f_pickle:
                                 state_data = pickle.load(f_pickle)
                             # Ensure persistent_llm still exists (could have been reset by another thread)
                             if self.persistent_llm:
                                 self.persistent_llm.load_state(state_data)
                                 logging.info("Fresh Context Mode: State reset successful.")
                                 self.cache_status_changed.emit("Fresh Context: Reset OK")
                                 reset_success = True
                             else:
                                 logging.warning("Persistent LLM disappeared during Fresh Context reset attempt.")
                                 raise RuntimeError("Persistent LLM instance no longer available.")
                         else:
                             raise FileNotFoundError(f"Warmed cache path invalid for reset: {self.warmed_cache_path}")
                     except Exception as e_reset:
                         logging.error(f"Failed to reset state for Fresh Context Mode: {e_reset}")
                         self.error_occurred.emit(f"Fresh Context Reset Failed: {e_reset}. Using temporary instance.")
                         self.cache_status_changed.emit("Fresh Context: Reset Failed")
                         # Do NOT proceed with persistent instance if reset failed
                         should_try_persistent = False # Force fallback to temporary

             # If we still intend to use persistent (reset ok or not needed)
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
                     model_id = self.config.get('CURRENT_MODEL_ID')
                     model_info = self.model_manager.get_model_info(model_id) if model_id else None
                     context_window = model_info.get('context_window', 4096) if model_info else 4096
                 logging.info(f"Will use persistent {'(fresh context reset)' if self.fresh_context_mode else 'warmed-up'} instance. Model: {model_path}, Cache: {self.warmed_cache_path}")

        # If not using persistent, get info for temporary load
        if not use_persistent_instance:
            logging.info("Will use temporary instance or fallback.")
            model_id = self.config.get('CURRENT_MODEL_ID')
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
                 master_cache_path_str = self.config.get('MASTER_KV_CACHE_PATH')
                 if master_cache_path_str and Path(master_cache_path_str).exists():
                     actual_kv_cache_path_for_inference = str(master_cache_path_str)
                     logging.info(f"Using master KV cache for inference: {actual_kv_cache_path_for_inference}")
                 else:
                     logging.warning("KV cache enabled, but selected cache invalid and master cache invalid/missing.")
                     # Proceed without cache (will use fallback without context prepending)

        # Add user message to history (do this *before* starting thread)
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

        # Pass the max_tokens value from the UI
        inference_thread = threading.Thread(
            target=target_thread_func,
            args=(message, model_path, context_window, actual_kv_cache_path_for_inference, max_tokens, temperature, llm_arg),
            daemon=True,
        )
        # Emit signal *before* starting thread so UI can disable input
        self.response_started.emit()
        inference_thread.start()

        return True


    # --- Inference thread with true KV cache logic ---
    # Modified to accept optional pre-loaded llm instance
    def _inference_thread_with_true_kv_cache(self, message: str, model_path: str, context_window: int,
                         kv_cache_path: Optional[str], max_tokens: int, temperature: float, llm: Optional[Llama] = None):
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
            # --- Acquire lock ONLY if using the persistent instance ---
            if is_using_persistent_llm:
                logging.debug("Attempting to acquire lock for persistent LLM in true KV thread.")
                self._lock.acquire()
                acquired_lock = True
                logging.debug("Lock acquired for persistent LLM in true KV thread.")
                # Verify persistent_llm still exists after acquiring lock
                if not self.persistent_llm:
                     raise RuntimeError("Persistent LLM instance disappeared before inference.")
                # Use the instance variable now we have the lock
                llm = self.persistent_llm

            # self.response_started.emit() # Moved to before thread start
            self.status_updated.emit("Processing...") # General status update

            if is_using_persistent_llm:
                logging.info(f"True KV cache thread using PERSISTENT instance. Cache: {kv_cache_path}")
                # llm is already loaded and cache state is assumed to be loaded/reset
                self.cache_status_changed.emit("Warmed Up (Generating)") # Update chat tab status
            else:
                # --- Load Model Temporarily ---
                logging.info(f"True KV cache thread loading TEMPORARILY. Model: {model_path}, Cache: {kv_cache_path}")
                self.status_updated.emit("Loading model...")
                abs_model_path = str(Path(model_path).resolve())
                if not Path(abs_model_path).exists():
                    raise FileNotFoundError(f"Model file not found: {abs_model_path}")

                threads = int(self.config.get('LLAMACPP_THREADS', os.cpu_count() or 4))
                batch_size = int(self.config.get('LLAMACPP_BATCH_SIZE', 512))
                gpu_layers = int(self.config.get('LLAMACPP_GPU_LAYERS', 0))

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
                    # --- Check Cache Compatibility Before Loading Temporarily ---
                    cache_info = self.cache_manager.get_cache_info(kv_cache_path)
                    cache_model_id = cache_info.get('model_id') if cache_info else None
                    current_model_id = self.config.get('CURRENT_MODEL_ID') # Model being loaded temporarily

                    if cache_model_id and current_model_id and cache_model_id != current_model_id:
                        logging.warning(f"Cache '{Path(kv_cache_path).name}' was created with model '{cache_model_id}', but current model is '{current_model_id}'. Skipping temporary load_state.")
                        self.error_occurred.emit(f"Cache incompatible with current model ({current_model_id}).") # Notify user
                        # Proceed without loading state
                    else:
                        # Proceed with loading state if compatible or compatibility unknown
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

            # --- Tokenize user input with structure ---
            instruction_prefix = "\n\nBased *only* on the loaded document context, answer the following question:\n"
            question_prefix = "Question: "
            suffix_text = "\n\nAnswer: " # Helps prompt the answer
            full_input_text = instruction_prefix + question_prefix + message + suffix_text

            input_tokens = llm.tokenize(full_input_text.encode('utf-8'))
            logging.info(f"Tokenized user input with structure ({len(input_tokens)} tokens)")

            # --- Evaluate input tokens to update the loaded KV cache state ---
            logging.info("Evaluating input tokens...")
            llm.eval(input_tokens)
            logging.info("Input tokens evaluated.")

            # --- Generate response using low-level token sampling ---
            logging.info(f"Generating response using low-level token sampling (max_tokens={max_tokens})") # Log max_tokens
            eos_token = llm.token_eos()
            logging.debug(f"EOS token ID: {eos_token} (type: {type(eos_token)})") # Log EOS token ID and type
            tokens_generated = []
            # response_text = "" # Moved initialization up

            # Use the max_tokens value passed into the function
            for i in range(max_tokens):
                token_id = llm.sample()
                # Log sampled token ID periodically or near the end for debugging
                if i >= max_tokens - 20: # Log last 20 tokens sampled
                    logging.debug(f"Sampled token ID at step {i}: {token_id} (type: {type(token_id)})")

                # Ensure comparison is between integers
                if int(token_id) == int(eos_token):
                    logging.info(f"EOS token encountered at step {i}.")
                    generated_eos = True
                    break # Exit the loop immediately

                tokens_generated.append(token_id)
                llm.eval([token_id]) # Evaluate the generated token

                # Emit chunks periodically
                if (i + 1) % 8 == 0:
                     current_text = llm.detokenize(tokens_generated).decode('utf-8', errors='replace')
                     new_text = current_text[len(response_text):]
                     if new_text:
                         self.response_chunk.emit(new_text)
                         response_text = current_text
                     QCoreApplication.processEvents()

            # Ensure final text is emitted
            final_text = llm.detokenize(tokens_generated).decode('utf-8', errors='replace')
            if len(final_text) > len(response_text):
                 self.response_chunk.emit(final_text[len(response_text):])
            response_text = final_text

            # Check if truncated
            # Use the max_tokens value passed into the function
            if not generated_eos and len(tokens_generated) >= max_tokens:
                truncation_msg = f" [... response truncated at {max_tokens} tokens]"
                logging.warning(f"Response truncated at {max_tokens} tokens.")
                response_text += truncation_msg
                self.response_chunk.emit(truncation_msg) # Emit the truncation message as a final chunk

            logging.info(f"Generated response with {len(tokens_generated)} tokens. EOS generated: {generated_eos}")

            # --- Finalize ---
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
            # --- Release lock ONLY if it was acquired ---
            if acquired_lock:
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
    # Modified to accept optional pre-loaded llm instance (though less likely to be used now)
    def _inference_thread_fallback(self, message: str, model_path: str, context_window: int,
                        kv_cache_path: Optional[str], max_tokens: int, temperature: float, llm: Optional[Llama] = None):
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
                threads = int(self.config.get('LLAMACPP_THREADS', os.cpu_count() or 4))
                batch_size = int(self.config.get('LLAMACPP_BATCH_SIZE', 512))
                gpu_layers = int(self.config.get('LLAMACPP_GPU_LAYERS', 0))
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
            system_prompt_content = "You are a helpful assistant." # Default system prompt

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
                                with open(original_doc_path, 'r', encoding='utf-8', errors='replace') as f_doc:
                                    doc_context_text = f_doc.read(8000) # Read snippet
                                logging.info(f"Fallback: Read {len(doc_context_text)} chars for prepending.")
                            else: logging.warning(f"Fallback: Original doc path not found: {original_doc_path}")
                        else: logging.warning(f"Fallback: Original doc path is 'Unknown' for cache: {kv_cache_path}")
                    else: logging.warning(f"Fallback: No cache info or original doc path for cache: {kv_cache_path}")

                    if doc_context_text:
                         system_prompt_content = (
                             f"Use the following text snippet to answer the user's question:\n"
                             f"--- TEXT SNIPPET START ---\n{doc_context_text}...\n--- TEXT SNIPPET END ---\n\n"
                             f"Answer based *only* on the text snippet provided."
                         )
                         logging.info("Fallback: Using system prompt with prepended context.")
                    else: logging.warning("Fallback: Failed to read context, using default system prompt.")
                except Exception as e_ctx:
                    logging.error(f"Fallback: Error retrieving context: {e_ctx}")
                    logging.warning("Fallback: Using default system prompt.")
            else:
                 logging.info("Fallback: No cache path provided, using default system prompt without prepending.")


            # Add system prompt
            chat_messages.append({"role": "system", "content": system_prompt_content})
            # Add recent history
            history_limit = 4
            start_index = max(0, len(self.history) - 1 - history_limit)
            recent_history = self.history[start_index:-1]
            chat_messages.extend(recent_history)
            # Add latest user message
            chat_messages.append(self.history[-1])
            logging.info(f"Fallback: Prepared chat history with {len(chat_messages)} messages.")

            # --- Generate Response (Streaming using create_chat_completion) ---
            self.status_updated.emit("Fallback: Generating response...")
            logging.info(f"Fallback: Generating response using create_chat_completion (max_tokens={max_tokens})...") # Log max_tokens
            stream = llm.create_chat_completion(
                messages=chat_messages,
                max_tokens=max_tokens, # Use passed max_tokens
                temperature=temperature,
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


    def clear_history(self):
        self.history = []
        logging.info("Chat history cleared")
        # Also unload cache if one was warmed up? Optional, maybe keep it warm.
        # self.unload_cache()

    def get_history(self) -> List[Dict]:
        return self.history

    def save_history(self, file_path: Union[str, Path]) -> bool:
        try:
            with open(file_path, 'w') as f:
                json.dump({
                    "history": self.history,
                    "model_id": self.config.get('CURRENT_MODEL_ID'),
                    "kv_cache_path": self.current_kv_cache_path,
                    "timestamp": time.time(),
                    "use_kv_cache_setting": self.use_kv_cache,
                    "fresh_context_mode_setting": self.fresh_context_mode # Save fresh context mode
                }, f, indent=2)
            logging.info(f"Chat history saved to {file_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save chat history: {str(e)}")
            return False

    def load_history(self, file_path: Union[str, Path]) -> bool:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            self.history = data.get("history", [])
            kv_cache_path_str = data.get("kv_cache_path")
            if kv_cache_path_str and Path(kv_cache_path_str).exists() and Path(kv_cache_path_str).suffix == '.llama_cache':
                self.current_kv_cache_path = kv_cache_path_str
                logging.info(f"Loaded KV cache path from history: {self.current_kv_cache_path}")
            else:
                 self.current_kv_cache_path = None
            self.use_kv_cache = data.get("use_kv_cache_setting", True)
            logging.info(f"Loaded use_kv_cache setting from history: {self.use_kv_cache}")
            # Also load fresh context mode setting if available
            self.fresh_context_mode = data.get("fresh_context_mode_setting", self.fresh_context_mode) # Keep current if not in file
            logging.info(f"Loaded fresh_context_mode setting from history: {self.fresh_context_mode}")

            logging.info(f"Chat history loaded from {file_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load chat history: {str(e)}")
            return False

    def update_config(self, config):
        self.config = config
        # Update settings from config
        self.use_true_kv_cache_logic = self.config.get('USE_TRUE_KV_CACHE', True)
        self.fresh_context_mode = self.config.get('USE_FRESH_CONTEXT', self.fresh_context_mode) # Update if in config, else keep current
        logging.info(f"ChatEngine configuration updated. True KV Cache Logic: {self.use_true_kv_cache_logic}, Fresh Context Mode: {self.fresh_context_mode}")
