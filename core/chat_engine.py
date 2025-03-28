#!/usr/bin/env python3
"""
Chat functionality for LlamaCag UI

Handles interaction with the model using KV caches.
Includes implementation for true KV cache loading and fallback.
Supports optional persistent model/cache loading.
"""

import os
import sys
import tempfile
import logging
import json
import time
import threading
import re
import pickle
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
    status_updated = pyqtSignal(str)
    # Signals for pre-loading status
    preload_status_update = pyqtSignal(str) # Status message during preload
    preload_finished = pyqtSignal(bool, str) # success, message/error

    def __init__(self, config, llama_manager, model_manager, cache_manager):
        """Initialize chat engine"""
        super().__init__()
        self.config = config
        self.llama_manager = llama_manager
        self.model_manager = model_manager
        self.cache_manager = cache_manager

        # Chat history
        self.history = []

        # Current KV cache for standard chat
        self.current_kv_cache_path = None # Store the path
        self.use_kv_cache = True
        self.use_true_kv_cache_logic = self.config.get('USE_TRUE_KV_CACHE', True)
        logging.info(f"ChatEngine initialized. True KV Cache Logic: {self.use_true_kv_cache_logic}")

        # Persistent model state for pre-loading
        self.persistent_llm: Optional[Llama] = None
        self.preloaded_model_id: Optional[str] = None
        self.preloaded_cache_path: Optional[str] = None
        self.preloading_lock = threading.Lock() # Lock for accessing/modifying persistent_llm

    def is_preloaded(self) -> bool:
        """Check if a model is currently pre-loaded."""
        with self.preloading_lock:
            return self.persistent_llm is not None

    def set_kv_cache(self, kv_cache_path: Optional[Union[str, Path]]):
        """Set the current KV cache path to use for standard chat"""
        if kv_cache_path:
            cache_path = Path(kv_cache_path)
            if not cache_path.exists() or cache_path.suffix != '.llama_cache':
                error_msg = f"KV cache not found or invalid: {cache_path}"
                logging.error(error_msg)
                self.error_occurred.emit(error_msg)
                return False

            self.current_kv_cache_path = str(cache_path)
            logging.info(f"Set current standard chat KV cache path to {self.current_kv_cache_path}")
            return True
        else:
            self.current_kv_cache_path = None
            logging.info("Cleared current standard chat KV cache path")
            return True

    def toggle_kv_cache(self, enabled: bool):
        """Toggle KV cache usage for standard chat"""
        self.use_kv_cache = enabled
        logging.info(f"Standard chat KV cache usage toggled: {enabled}")
        self.status_updated.emit(f"KV Cache Usage: {'Enabled' if enabled else 'Disabled'}")

    def send_message(self, message: str, max_tokens: int = 1024, temperature: float = 0.7):
        """Send a message to the model, prioritizing the pre-loaded model/cache if active."""

        target_model_id = None
        target_model_path = None
        target_context_window = None
        target_kv_cache_path = None

        # --- Check if Pre-loaded Model Should Be Used ---
        use_preloaded = False
        if self.is_preloaded():
            with self.preloading_lock:  # Use a lock for consistent preloaded state check
                use_preloaded = True
                target_model_id = self.preloaded_model_id
                target_kv_cache_path = self.preloaded_cache_path
                # Get model path and context window from preloaded model info
                model_info = self.model_manager.get_model_info(target_model_id)
                if model_info:
                     target_model_path = model_info.get('path')
                     target_context_window = model_info.get('context_window', 4096)
                else:
                     # This *should* not happen if preloading was successful, but handle it
                     logging.error(f"Pre-loaded model info not found for ID: {target_model_id}. Falling back.")
                     use_preloaded = False # Fallback

        if use_preloaded:
            logging.info(f"Using pre-loaded model: {target_model_id}, cache: {target_kv_cache_path}")
            # Note that we don't check if target_model_path is valid at this point. We assume
            # that the model and cache paths were valid at pre-load time.
            if not target_model_path or not Path(target_model_path).exists():
                self.error_occurred.emit(f"Pre-loaded Model file not found: {target_model_path}")
                return False # Pre-load was bad. Should not happen.
        else:
            # --- Standard Logic (Not Pre-Loaded) ---
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

            target_model_id = model_id
            target_model_path = model_path
            target_context_window = context_window

            # --- Determine KV Cache Path for this specific message ---
            target_kv_cache_path = None  # Use this consistently
            if self.use_kv_cache and self.current_kv_cache_path:
                if Path(self.current_kv_cache_path).exists():
                    target_kv_cache_path = self.current_kv_cache_path
                    logging.info(f"Standard chat using cache: {target_kv_cache_path}")
                else:
                    logging.warning(f"Selected standard chat KV cache file not found: {self.current_kv_cache_path}")
                    self.error_occurred.emit(f"Selected KV cache file not found: {Path(self.current_kv_cache_path).name}")
            elif self.use_kv_cache:
                master_cache_path_str = self.config.get('MASTER_KV_CACHE_PATH')
                if master_cache_path_str and Path(master_cache_path_str).exists():
                    target_kv_cache_path = str(master_cache_path_str)
                    logging.info(f"Standard chat using master KV cache: {target_kv_cache_path}")
                else:
                    logging.warning("KV cache enabled for standard chat, but no cache selected and master cache is invalid or missing.")
                    self.error_occurred.emit("KV cache enabled, but no cache selected/master invalid.")

        # Add user message to history (do this *before* starting thread)
        self.history.append({"role": "user", "content": message})

        # --- Start Inference Thread ---
        target_thread_func = self._inference_thread_fallback  # Default to fallback
        if target_kv_cache_path and self.use_true_kv_cache_logic: # Check target_kv_cache_path, not actual_kv_cache_path
            target_thread_func = self._inference_thread_with_true_kv_cache
            logging.info("Dispatching to TRUE KV Cache inference thread.")
        else:
            logging.info("Dispatching to FALLBACK (manual context) inference thread.")

        inference_thread = threading.Thread(
            target=target_thread_func,
            args=(message, target_model_id, target_model_path, target_context_window, target_kv_cache_path, max_tokens, temperature),  # Use target_* variables
            daemon=True,
        )
        inference_thread.start()
        return True

    # --- Modified inference thread with true KV cache logic ---
    def _inference_thread_with_true_kv_cache(self, message: str, model_id: str, model_path: str, context_window: int,
                         kv_cache_path: Optional[str], max_tokens: int, temperature: float):
        """Thread function for model inference using true KV cache loading, potentially using pre-loaded model."""
        llm_instance_to_use: Optional[Llama] = None
        is_using_persistent = False
        acquired_lock = False

        try:
            self.response_started.emit()
            logging.info(f"True KV cache inference requested. Model ID: {model_id}, Cache: {kv_cache_path}")

            # --- Check if pre-loaded model matches and acquire lock ---
            logging.debug("Acquiring preloading lock...")
            acquired_lock = self.preloading_lock.acquire(timeout=5.0) # Wait up to 5s for lock
            if not acquired_lock:
                 logging.error("Timeout acquiring preloading lock. Preload operation might be stuck.")
                 raise TimeoutError("Could not acquire lock for persistent model.")

            logging.debug("Preloading lock acquired.")
            if (self.persistent_llm and
                self.preloaded_model_id == model_id and
                self.preloaded_cache_path == kv_cache_path):
                logging.info("Using pre-loaded persistent model instance.")
                llm_instance_to_use = self.persistent_llm
                is_using_persistent = True
                # Keep lock acquired while using the persistent instance
            else:
                # Pre-loaded doesn't match or doesn't exist, proceed with load-on-demand
                logging.info("Pre-loaded model mismatch or unavailable. Loading model on demand.")
                self.preloading_lock.release() # Release lock if not using persistent
                acquired_lock = False

            # --- Load Model On Demand (if not using persistent) ---
            if not is_using_persistent:
                self.status_updated.emit("Loading model...")
                threads = int(self.config.get('LLAMACPP_THREADS', os.cpu_count() or 4))
                batch_size = int(self.config.get('LLAMACPP_BATCH_SIZE', 512))
                gpu_layers = int(self.config.get('LLAMACPP_GPU_LAYERS', 0))

                logging.info(f"Loading model: {model_path}")
                abs_model_path = str(Path(model_path).resolve())
                if not Path(abs_model_path).exists():
                    raise FileNotFoundError(f"Model file not found: {abs_model_path}")

                llm_instance_to_use = Llama(
                    model_path=abs_model_path,
                    n_ctx=context_window,
                    n_threads=threads,
                    n_batch=batch_size,
                    n_gpu_layers=gpu_layers,
                    verbose=False
                )
                logging.info("Model loaded on demand.")
                self.status_updated.emit("Loading KV cache state...")

                # --- Load KV Cache On Demand ---
                if kv_cache_path and Path(kv_cache_path).exists():
                    logging.info(f"Loading KV cache state from: {kv_cache_path}")
                    try:
                        with open(kv_cache_path, 'rb') as f_pickle:
                            state_data = pickle.load(f_pickle)
                        llm_instance_to_use.load_state(state_data)
                        logging.info("KV cache state loaded successfully on demand.")
                    except Exception as e:
                        logging.error(f"Error loading KV cache on demand: {e}. Falling back.")
                        # Fallback to manual context prepending
                        self._inference_thread_fallback(message, model_path, context_window, kv_cache_path, max_tokens, temperature, llm_instance_to_use)
                        return # Exit this thread as fallback handles it
                else:
                    logging.error("KV cache path invalid or missing for on-demand load. Falling back.")
                    self._inference_thread_fallback(message, model_path, context_window, kv_cache_path, max_tokens, temperature, llm_instance_to_use)
                    return # Exit this thread

            # --- Inference using the selected LLM instance (persistent or on-demand) ---
            self.status_updated.emit("Generating response...")
            if not llm_instance_to_use: # Should not happen, but safety check
                 raise ValueError("LLM instance is unexpectedly None.")

            # --- Tokenize user input with structure ---
            instruction_prefix = "\n\nBased *only* on the loaded document context, answer the following question:\n"
            question_prefix = "Question: "
            suffix_text = "\n\nAnswer: "
            full_input_text = instruction_prefix + question_prefix + message + suffix_text

            input_tokens = llm_instance_to_use.tokenize(full_input_text.encode('utf-8'))
            logging.info(f"Tokenized user input with structure ({len(input_tokens)} tokens)")

            # --- Evaluate input tokens to update the loaded KV cache state ---
            logging.info("Evaluating input tokens...")
            llm_instance_to_use.eval(input_tokens)
            logging.info("Input tokens evaluated.")

            # --- Generate response using low-level token sampling ---
            logging.info("Generating response using low-level token sampling")
            eos_token = llm_instance_to_use.token_eos()
            tokens_generated = []
            response_text = ""
            text = "" # define text here to make sure text exists for the emit call

            for i in range(max_tokens):
                token_id = llm_instance_to_use.sample()

                if token_id == eos_token:
                    logging.info("EOS token encountered.")
                    break

                tokens_generated.append(token_id)
                llm_instance_to_use.eval([token_id])

                if (i + 1) % 8 == 0:
                     current_text = llm_instance_to_use.detokenize(tokens_generated).decode('utf-8', errors='replace')
                     new_text = current_text[len(response_text):]
                     if new_text:
                         text = new_text
                         self.response_chunk.emit(text)
                         response_text = current_text
                     QCoreApplication.processEvents()

            final_text = llm_instance_to_use.detokenize(tokens_generated).decode('utf-8', errors='replace')
            if len(final_text) > len(response_text):
                 text = final_text[len(response_text):]
                 self.response_chunk.emit(text)
            response_text = final_text

            logging.info(f"Generated response with {len(tokens_generated)} tokens using {'persistent' if is_using_persistent else 'on-demand'} KV cache.")

            # --- Finalize ---
            self.status_updated.emit("Idle")
            if response_text.strip():
                self.history.append({"role": "assistant", "content": response_text})
                self.response_complete.emit(response_text, True)
            else:
                logging.warning("Model generated an empty response using true KV cache.")
                self.error_occurred.emit("Model generated an empty response.")
                self.response_complete.emit("", False)

        except Exception as e:
            error_message = f"Error during inference: {str(e)}"
            self.status_updated.emit("Error")
            logging.exception(error_message)
            self.error_occurred.emit(error_message)
            self.response_complete.emit("", False)
        finally:
            # Release lock if acquired
            if acquired_lock:
                logging.debug("Releasing preloading lock.")
                self.preloading_lock.release()
                acquired_lock = False
            # Release on-demand llm instance if created
            if not is_using_persistent and llm_instance_to_use:
                logging.debug("Releasing on-demand LLM instance.")
                del llm_instance_to_use
                llm_instance_to_use = None

            self.status_updated.emit("Idle")
            logging.debug("Inference thread finished.")


    # --- Fallback inference method ---
    def _inference_thread_fallback(self, message: str, model_path: str, context_window: int,
                        kv_cache: Optional[str], max_tokens: int, temperature: float, llm_instance: Optional[Llama] = None):
        """
        Fallback inference method using manual context prepending.
        Can optionally receive a Llama instance (e.g., if true cache load failed).
        Manages its own Llama instance lifecycle if one is not provided.
        """
        local_llm = None # Use a local variable to manage the instance if created here
        try:
            self.status_updated.emit("Using fallback method...")
            # If llm_instance wasn't passed, load it
            if llm_instance is None:
                self.status_updated.emit("Fallback: Loading model...")
                logging.info("Fallback: Loading model...")
                abs_model_path = str(Path(model_path).resolve())
                if not Path(abs_model_path).exists():
                    raise FileNotFoundError(f"Model file not found: {abs_model_path}")
                threads = int(self.config.get('LLAMACPP_THREADS', os.cpu_count() or 4))
                batch_size = int(self.config.get('LLAMACPP_BATCH_SIZE', 512))
                gpu_layers = int(self.config.get('LLAMACPP_GPU_LAYERS', 0))
                local_llm = Llama( # Assign to local_llm
                    model_path=abs_model_path, n_ctx=context_window, n_threads=threads,
                    n_batch=batch_size, n_gpu_layers=gpu_layers, verbose=False
                )
                llm_to_use = local_llm # Use the locally created instance
                logging.info("Fallback: Model loaded.")
                self.status_updated.emit("Fallback: Preparing context...")
            else:
                 llm_to_use = llm_instance # Use the passed instance

            # --- Prepare Chat History with Manual Context Prepending ---
            chat_messages = []
            system_prompt_content = "You are a helpful assistant."

            if kv_cache:
                logging.info("Fallback: Attempting to prepend original document context.")
                doc_context_text = ""
                try:
                    cache_info = self.cache_manager.get_cache_info(kv_cache)
                    if cache_info and 'original_document' in cache_info:
                        original_doc_path_str = cache_info['original_document']
                        if original_doc_path_str != "Unknown":
                            original_doc_path = Path(original_doc_path_str)
                            if original_doc_path.exists():
                                with open(original_doc_path, 'r', encoding='utf-8', errors='replace') as f_doc:
                                    doc_context_text = f_doc.read(8000)
                            else: logging.warning(f"Fallback: Original doc not found: {original_doc_path}")
                        else: logging.warning(f"Fallback: Original doc path is 'Unknown' for cache: {kv_cache}")
                    else: logging.warning(f"Fallback: No cache info or original doc path for cache: {kv_cache}")

                    if doc_context_text:
                         system_prompt_content = (
                             f"You are an assistant tasked with answering questions based *strictly* and *exclusively* on the following provided text snippet. "
                             f"Do not use any prior knowledge or information outside of this text. If the answer cannot be found within the text, state that clearly.\n\n"
                             f"--- TEXT SNIPPET START ---\n{doc_context_text}...\n--- TEXT SNIPPET END ---\n\n"
                             f"Answer the user's question using *only* the information contained within the text snippet above."
                         )
                         logging.info("Fallback: Using STRICT system prompt with prepended context.")
                    else: logging.warning("Fallback: Failed to read original document context.")
                except Exception as e_ctx: logging.error(f"Fallback: Error retrieving context: {e_ctx}")

            chat_messages.append({"role": "system", "content": system_prompt_content})
            history_limit = 4
            start_index = max(0, len(self.history) - 1 - history_limit)
            recent_history = self.history[start_index:-1]
            chat_messages.extend(recent_history)
            chat_messages.append(self.history[-1])
            logging.info(f"Fallback: Prepared chat history with {len(chat_messages)} messages.")

            # --- Generate Response ---
            self.status_updated.emit("Fallback: Generating response...")
            logging.info(f"Fallback: Generating response using create_chat_completion...")
            stream = llm_to_use.create_chat_completion(
                messages=chat_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            complete_response = ""
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
                self.response_complete.emit(complete_response, True)
            else:
                logging.warning("Model stream completed but produced no text.")
                self.error_occurred.emit("Model generated an empty response.")
                self.response_complete.emit("", False)

        except Exception as e:
            error_message = f"Error during fallback inference: {str(e)}"
            self.status_updated.emit("Error")
            logging.exception(error_message)
            self.error_occurred.emit(error_message)
            self.response_complete.emit("", False)
        finally:
            # Release the locally created llm instance if it exists
            if local_llm:
                logging.debug("Releasing fallback-local LLM instance.")
                del local_llm
                llm_to_use = None
            # Don't set status to Idle here, let the caller (_inference_thread_with_true_kv_cache) handle it

    # --- Pre-loading Methods ---
    def preload_model_and_cache(self, model_id: str, cache_path: str) -> Tuple[bool, str]:
        """Loads the specified model and cache into persistent memory."""
        with self.preloading_lock:
            # Unload existing if different
            if self.persistent_llm and (self.preloaded_model_id != model_id or self.preloaded_cache_path != cache_path):
                logging.info("Unloading persistent model and cache.")
                self.preload_status_update.emit("Unloading previous model...")
                del self.persistent_llm
                self.persistent_llm = None
                self.preloaded_model_id = None
                self.preloaded_cache_path = None

            # If already loaded, do nothing
            if self.persistent_llm and self.preloaded_model_id == model_id and self.preloaded_cache_path == cache_path:
                 logging.info(f"Model {model_id} and cache {cache_path} are already pre-loaded.")
                 self.preload_status_update.emit("Already loaded")
                 return True, "Model and cache already loaded."

            logging.info(f"Starting pre-load: Model ID={model_id}, Cache={cache_path}")
            try:
                # --- Get Model Info ---
                model_info = self.model_manager.get_model_info(model_id)
                if not model_info:
                    raise ValueError(f"Model '{model_id}' not found.")
                model_file_path = model_info.get('path')
                if not model_file_path or not Path(model_file_path).exists():
                    raise FileNotFoundError(f"Model file not found for '{model_id}': {model_file_path}")
                context_window = model_info.get('context_window', 4096)

                # --- Validate Cache Path ---
                cache_file = Path(cache_path)
                if not cache_file.exists() or cache_file.suffix != '.llama_cache':
                     raise FileNotFoundError(f"KV cache file not found or invalid: {cache_path}")

                # --- Load Model ---
                self.preload_status_update.emit("Loading model...")
                threads = int(self.config.get('LLAMACPP_THREADS', os.cpu_count() or 4))
                batch_size = int(self.config.get('LLAMACPP_BATCH_SIZE', 512))
                gpu_layers = int(self.config.get('LLAMACPP_GPU_LAYERS', 0))

                logging.info(f"Pre-loading model: {model_file_path}")
                abs_model_path = str(Path(model_file_path).resolve())
                llm = Llama(
                    model_path=abs_model_path,
                    n_ctx=context_window,
                    n_threads=threads,
                    n_batch=batch_size,
                    n_gpu_layers=gpu_layers,
                    verbose=False
                )
                logging.info("Model pre-loaded successfully.")

                # --- Load Cache State ---
                self.preload_status_update.emit("Loading KV cache state...")
                logging.info(f"Pre-loading KV cache state from: {cache_path}")
                with open(cache_path, 'rb') as f_pickle:
                    state_data = pickle.load(f_pickle)
                llm.load_state(state_data)
                logging.info("KV cache state pre-loaded successfully.")

                # --- Store Persistent Instance ---
                self.persistent_llm = llm
                self.preloaded_model_id = model_id
                self.preloaded_cache_path = cache_path
                self.preload_status_update.emit("Load complete")
                logging.info("Pre-loading complete. Model and cache are persistent.")
                return True, "Model and cache loaded successfully."

            except Exception as e:
                logging.exception(f"Failed to pre-load model/cache: {e}")
                # Clean up partially loaded model if error occurred
                if 'llm' in locals() and llm is not None:
                    del llm
                self.persistent_llm = None
                self.preloaded_model_id = None
                self.preloaded_cache_path = None
                return False, str(e)

    def unload_persistent_model(self):
        """Unloads the persistently loaded model and cache."""
        with self.preloading_lock:
            if self.persistent_llm:
                logging.info("Unloading persistent model and cache.")
                del self.persistent_llm
                self.persistent_llm = None
                self.preloaded_model_id = None
                self.preloaded_cache_path = None
                logging.info("Persistent model unloaded.")
            else:
                logging.info("No persistent model was loaded.")

    # --- History and Config Methods ---
    def clear_history(self):
        self.history = []
        logging.info("Chat history cleared")

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
                    "use_kv_cache_setting": self.use_kv_cache
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
            logging.info(f"Chat history loaded from {file_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load chat history: {str(e)}")
            return False

    def update_config(self, config):
        self.config = config
        self.use_true_kv_cache_logic = self.config.get('USE_TRUE_KV_CACHE', True)
        logging.info(f"ChatEngine configuration updated. True KV Cache Logic: {self.use_true_kv_cache_logic}")
        # Potentially trigger unload if config changes affect pre-load?
        # For now, rely on user applying settings in UI.

    def trigger_preload_on_startup(self):
        """Checks config and triggers pre-loading if enabled."""
        if self.config.get('PRELOAD_ENABLED', False):
            model_id = self.config.get('PRELOAD_MODEL_ID')
            cache_path = self.config.get('PRELOAD_CACHE_PATH')
            if model_id and cache_path:
                logging.info(f"Triggering pre-load on startup: Model={model_id}, Cache={cache_path}")
                # Run in a separate thread to avoid blocking main thread
                preload_thread = threading.Thread(
                    target=self._run_preload_worker_sync, # Use a sync wrapper for thread target
                    args=(model_id, cache_path),
                    daemon=True
                )
                preload_thread.start()
            else:
                logging.warning("Pre-load enabled on startup, but model or cache path is missing in config.")

    def _run_preload_worker_sync(self, model_id, cache_path):
        """Synchronous wrapper for preload worker, emits signals."""
        try:
            self.preload_status_update.emit("Loading model (startup)...")
            success, message = self.preload_model_and_cache(model_id, cache_path)
            self.preload_finished.emit(success, message)
        except Exception as e:
            logging.exception("Error during startup pre-loading")
            self.preload_finished.emit(False, f"Startup preload error: {str(e)}")
