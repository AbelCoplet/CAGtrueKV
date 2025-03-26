#!/usr/bin/env python3
"""
Chat functionality for LlamaCag UI

Handles interaction with the model using KV caches.
(Note: True KV cache loading is disabled due to instability.
 Context is manually prepended to the prompt instead.)
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PyQt5.QtCore import QObject, pyqtSignal
from llama_cpp import Llama, LlamaCache


class ChatEngine(QObject):
    """Chat functionality using large context window models with KV caches"""

    # Signals
    response_started = pyqtSignal()
    response_chunk = pyqtSignal(str)  # Text chunk
    response_complete = pyqtSignal(str, bool)  # Full response, success
    error_occurred = pyqtSignal(str)  # Error message

    def __init__(self, config, llama_manager, model_manager, cache_manager):
        """Initialize chat engine"""
        super().__init__()
        self.config = config
        self.llama_manager = llama_manager
        self.model_manager = model_manager
        self.cache_manager = cache_manager

        # Chat history
        self.history = []

        # Current KV cache
        self.current_kv_cache_path = None # Store the path
        self.use_kv_cache = True

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
            self.current_kv_cache_path = None
            logging.info("Cleared current KV cache path")
            return True

    def toggle_kv_cache(self, enabled: bool):
        """Toggle KV cache usage"""
        self.use_kv_cache = enabled
        logging.info(f"KV cache usage toggled: {enabled}")

    def send_message(self, message: str, max_tokens: int = 1024, temperature: float = 0.7):
        """Send a message to the model and get a response"""
        # --- Get Model Info ---
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
        context_window = model_info.get('context_window', 4096) # Default context if not specified

        # --- Determine KV Cache Path ---
        # This path is used to FIND the original document, not for loading state
        actual_kv_cache_path = None
        # Check if cache usage is enabled AND a cache path is set
        if self.use_kv_cache and self.current_kv_cache_path:
            # Check if the specific cache file exists
            if Path(self.current_kv_cache_path).exists():
                 actual_kv_cache_path = self.current_kv_cache_path
                 logging.info(f"Cache selected (for context lookup): {actual_kv_cache_path}")
            else:
                 logging.warning(f"Selected KV cache file not found: {self.current_kv_cache_path}. Will proceed without context.")
                 self.error_occurred.emit(f"Selected KV cache file not found: {Path(self.current_kv_cache_path).name}")
        elif self.use_kv_cache:
             # Cache usage is enabled, but no specific cache is selected. Check for master.
             master_cache_cfg_key = 'MASTER_KV_CACHE_PATH'
             master_cache_path_str = self.config.get(master_cache_cfg_key)
             if master_cache_path_str and Path(master_cache_path_str).exists() and Path(master_cache_path_str).suffix == '.llama_cache':
                 actual_kv_cache_path = str(master_cache_path_str)
                 logging.info(f"Using master KV cache (for context lookup): {actual_kv_cache_path}")
             else:
                 logging.warning("KV cache usage enabled, but no cache selected and master cache is invalid or missing.")
                 self.error_occurred.emit("KV cache enabled, but no cache selected/master invalid.")

        # Add user message to history *before* starting thread
        # We reference self.history directly in _inference_thread now
        # self.history.append({"role": "user", "content": message}) # Moved inside thread prep

        # --- Start Inference Thread ---
        inference_thread = threading.Thread(
            target=self._inference_thread,
            args=(message, model_path, context_window, actual_kv_cache_path, max_tokens, temperature),
            daemon=True,
        )
        inference_thread.start()

        return True

    def _inference_thread(self, message: str, model_path: str, context_window: int,
                        kv_cache_path: Optional[str], max_tokens: int, temperature: float):
        """Thread function for model inference using llama-cpp-python"""
        llm = None # Ensure llm is defined for finally block
        try:
            self.response_started.emit()
            logging.info(f"Inference thread started. Model: {model_path}, Cache selected (for context): {kv_cache_path is not None}")

            # --- Configuration ---
            threads = int(self.config.get('LLAMACPP_THREADS', os.cpu_count() or 4))
            batch_size = int(self.config.get('LLAMACPP_BATCH_SIZE', 512))
            gpu_layers = int(self.config.get('LLAMACPP_GPU_LAYERS', 0))

            # --- Load Model ---
            logging.info(f"Loading model: {model_path}")
            abs_model_path = str(Path(model_path).resolve())
            if not Path(abs_model_path).exists():
                 raise FileNotFoundError(f"Model file not found: {abs_model_path}")

            llm = Llama(
                model_path=abs_model_path,
                n_ctx=context_window,
                n_threads=threads,
                n_batch=batch_size,
                n_gpu_layers=gpu_layers,
                verbose=False
            )
            logging.info("Model loaded.")

            # --- KV Cache Loading (DISABLED) ---
            # Reverted to manual context prepending due to issues with load_state reliability
            logging.info("True KV cache loading is DISABLED. Context will be prepended manually if cache is selected.")
            # if kv_cache_path:
            #     logging.info(f"Attempting to load KV cache state from: {kv_cache_path}")
            #     try:
            #         with open(kv_cache_path, 'rb') as f_pickle:
            #             state_data = pickle.load(f_pickle)
            #         llm.load_state(state_data)
            #         logging.info("KV cache loaded successfully from pickled object.")
            #     except Exception as e:
            #         logging.error(f"Failed to load KV cache state: {e}")
            #         # Fallback or raise error? For now, just log and continue without cache state
            #         kv_cache_path = None # Ensure we don't try to eval based on failed load
            # else:
            #     logging.info("No KV cache specified or cache usage disabled.")

            # --- Prepare Chat History for create_chat_completion ---
            chat_messages = []
            system_prompt_content = "You are a helpful assistant." # Default system prompt

            # If a cache path was determined earlier, try to read context and put in system prompt
            if kv_cache_path:
                logging.info("KV cache is selected. Attempting to prepend original document context to system prompt.")
                doc_context_text = ""
                try:
                    cache_info = self.cache_manager.get_cache_info(kv_cache_path)
                    if cache_info and 'original_document' in cache_info:
                        original_doc_path_str = cache_info['original_document']
                        if original_doc_path_str != "Unknown": # Check if path is known
                            original_doc_path = Path(original_doc_path_str)
                            if original_doc_path.exists():
                                logging.info(f"Reading start of original document: {original_doc_path}")
                                with open(original_doc_path, 'r', encoding='utf-8', errors='replace') as f_doc:
                                    # Read a significant chunk, but consider model's actual context limit
                                    # minus space for history and answer. 8k chars is roughly 2k tokens.
                                    doc_context_text = f_doc.read(8000)
                                logging.info(f"Read {len(doc_context_text)} chars from original document.")
                            else:
                                logging.warning(f"Original document path not found: {original_doc_path}")
                        else:
                             logging.warning(f"Original document path is 'Unknown' for cache: {kv_cache_path}")
                    else:
                        logging.warning(f"Could not find cache info or original document path for cache: {kv_cache_path}")

                    if doc_context_text:
                         # Construct system prompt with explicit context
                         system_prompt_content = (
                             f"Use the following text to answer the user's question:\n"
                             f"--- TEXT START ---\n"
                             f"{doc_context_text}...\n"
                             f"--- TEXT END ---\n\n"
                             f"Answer based *only* on the text provided above."
                         )
                         logging.info("Using system prompt with prepended context (8k chars).")
                    else:
                         logging.warning("Failed to read original document context, using default system prompt.")

                except Exception as e_ctx:
                    logging.error(f"Error retrieving or reading original document context: {e_ctx}")
                    logging.warning("Error during context retrieval, using default system prompt.")

            # Add system prompt
            chat_messages.append({"role": "system", "content": system_prompt_content})

            # Add recent history (e.g., last 4 turns = 2 user, 2 assistant)
            history_limit = 4
            # Calculate start index, ensuring it's not negative
            start_index = max(0, len(self.history) - 1 - history_limit)
            # Get recent turns from self.history (excluding the latest user message which isn't added yet)
            recent_history = self.history[start_index:-1]
            chat_messages.extend(recent_history)

            # Add the latest user message (which is passed as 'message' argument)
            # Note: self.history was updated in send_message before starting thread
            chat_messages.append({"role": "user", "content": message})
            logging.info(f"Prepared chat history with system prompt, {len(recent_history)} recent turns, and current message.")


            # --- Generate Response (Streaming using create_chat_completion) ---
            logging.info(f"Generating response using create_chat_completion (max_tokens={max_tokens}, temp={temperature})...")
            stream = llm.create_chat_completion(
                messages=chat_messages, # Pass the full history
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            complete_response = ""
            for chunk in stream:
                try:
                    # create_chat_completion streams delta content
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content")
                    if text:
                        self.response_chunk.emit(text)
                        complete_response += text
                except (KeyError, IndexError, TypeError) as e:
                    logging.warning(f"Could not extract text from stream chunk: {chunk}, Error: {e}")

            logging.info("Response generation complete.")

            # --- Finalize ---
            if complete_response.strip():
                # Add assistant response to internal history AFTER generation is complete
                self.history.append({"role": "assistant", "content": complete_response})
                # No need to update usage stats if we aren't using the cache loading mechanism
                self.response_complete.emit(complete_response, True)
            else:
                logging.warning("Model stream completed but produced no text.")
                self.error_occurred.emit("Model generated an empty response.")
                self.response_complete.emit("", False)

        except Exception as e:
            error_message = f"Error during inference: {str(e)}"
            logging.exception(error_message)
            self.error_occurred.emit(error_message)
            self.response_complete.emit("", False)
        finally:
            if llm is not None:
                pass
            logging.debug("Inference thread finished.")

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
        logging.info("ChatEngine configuration updated.")
