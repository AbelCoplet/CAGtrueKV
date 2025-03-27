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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PyQt5.QtCore import QObject, pyqtSignal, QCoreApplication # Added QCoreApplication
from llama_cpp import Llama, LlamaCache


class ChatEngine(QObject):
    """Chat functionality using large context window models with KV caches"""

    # Signals
    response_started = pyqtSignal()
    response_chunk = pyqtSignal(str)  # Text chunk
    response_complete = pyqtSignal(str, bool)  # Full response, success
    error_occurred = pyqtSignal(str)  # Error message
    status_updated = pyqtSignal(str) # Added for UI feedback

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
        # Add config option for true KV cache (default to False until proven stable)
        # Let's default to True for testing the new logic as requested
        self.use_true_kv_cache_logic = self.config.get('USE_TRUE_KV_CACHE', True)
        logging.info(f"ChatEngine initialized. True KV Cache Logic: {self.use_true_kv_cache_logic}")


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
        self.status_updated.emit(f"KV Cache Usage: {'Enabled' if enabled else 'Disabled'}")

    # --- New send_message implementation from FIXES ---
    def send_message(self, message: str, max_tokens: int = 1024, temperature: float = 0.7):
        """Send a message to the model and get a response with true KV caching support"""
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
        context_window = model_info.get('context_window', 4096)

        # --- Determine KV Cache Path ---
        actual_kv_cache_path = None
        if self.use_kv_cache and self.current_kv_cache_path:
            if Path(self.current_kv_cache_path).exists():
                actual_kv_cache_path = self.current_kv_cache_path
                logging.info(f"Cache selected: {actual_kv_cache_path}")
            else:
                logging.warning(f"Selected KV cache file not found: {self.current_kv_cache_path}")
                self.error_occurred.emit(f"Selected KV cache file not found: {Path(self.current_kv_cache_path).name}")
        elif self.use_kv_cache:
            master_cache_path_str = self.config.get('MASTER_KV_CACHE_PATH')
            if master_cache_path_str and Path(master_cache_path_str).exists():
                actual_kv_cache_path = str(master_cache_path_str)
                logging.info(f"Using master KV cache: {actual_kv_cache_path}")
            else:
                logging.warning("KV cache enabled, but no cache selected and master cache is invalid or missing.")
                self.error_occurred.emit("KV cache enabled, but no cache selected/master invalid.")

        # Add user message to history (do this *before* starting thread)
        self.history.append({"role": "user", "content": message})

        # --- Start Inference Thread ---
        # Decide which inference method to use based on config/cache availability
        # Use the new method if a cache path exists AND true logic is enabled
        target_thread_func = self._inference_thread_fallback # Default to fallback
        if actual_kv_cache_path and self.use_true_kv_cache_logic:
             target_thread_func = self._inference_thread_with_true_kv_cache
             logging.info("Dispatching to TRUE KV Cache inference thread.")
        else:
             logging.info("Dispatching to FALLBACK (manual context) inference thread.")


        inference_thread = threading.Thread(
            target=target_thread_func, # Use the selected function
            args=(message, model_path, context_window, actual_kv_cache_path, max_tokens, temperature),
            daemon=True,
        )
        inference_thread.start()
        # Status update will happen inside the thread now

        return True

    # --- New inference thread with true KV cache logic from FIXES ---
    def _inference_thread_with_true_kv_cache(self, message: str, model_path: str, context_window: int,
                         kv_cache_path: Optional[str], max_tokens: int, temperature: float):
        """Thread function for model inference using true KV cache loading"""
        llm = None
        try:
            self.response_started.emit()
            self.status_updated.emit("Loading model...") # Status update
            logging.info(f"True KV cache inference thread started. Model: {model_path}, Cache: {kv_cache_path}")

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
                verbose=False # Keep verbose off for cleaner logs
            )
            logging.info("Model loaded.")
            self.status_updated.emit("Loading KV cache state...") # Status update

            # --- Load KV Cache ---
            # This is the core difference: we load the state here.
            if kv_cache_path and Path(kv_cache_path).exists():
                logging.info(f"Loading KV cache state from: {kv_cache_path}")
                try:
                    with open(kv_cache_path, 'rb') as f_pickle:
                        state_data = pickle.load(f_pickle)
                    llm.load_state(state_data)
                    logging.info("KV cache state loaded successfully.")
                    self.status_updated.emit("Generating response...") # Status update

                    # --- Tokenize user input with structure ---
                    # Add explicit instruction to use only loaded context
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
                    logging.info("Generating response using low-level token sampling")
                    eos_token = llm.token_eos()
                    tokens_generated = []
                    response_text = ""

                    for i in range(max_tokens):
                        # Use sample method without temperature (assuming it's set elsewhere or defaults)
                        token_id = llm.sample()

                        if token_id == eos_token:
                            logging.info("EOS token encountered.")
                            break

                        tokens_generated.append(token_id)
                        # Evaluate the generated token to update state for the *next* token
                        llm.eval([token_id])

                        # Emit chunks periodically for responsiveness
                        if (i + 1) % 8 == 0: # Emit every 8 tokens
                             current_text = llm.detokenize(tokens_generated).decode('utf-8', errors='replace')
                             # Send only the *new* part of the text
                             new_text = current_text[len(response_text):]
                             if new_text:
                                 self.response_chunk.emit(new_text)
                                 response_text = current_text # Update the baseline
                             QCoreApplication.processEvents() # Keep UI responsive

                    # Ensure final text is emitted if loop finished without hitting emit condition
                    final_text = llm.detokenize(tokens_generated).decode('utf-8', errors='replace')
                    if len(final_text) > len(response_text):
                         self.response_chunk.emit(final_text[len(response_text):])
                    response_text = final_text # Final full response

                    logging.info(f"Generated response with {len(tokens_generated)} tokens using true KV cache.")

                    # --- Finalize ---
                    self.status_updated.emit("Idle") # Status update
                    if response_text.strip():
                        self.history.append({"role": "assistant", "content": response_text})
                        self.response_complete.emit(response_text, True)
                    else:
                        logging.warning("Model generated an empty response using true KV cache.")
                        self.error_occurred.emit("Model generated an empty response.")
                        self.response_complete.emit("", False)

                    # Successfully used true KV cache, return from function
                    return

                except Exception as e:
                    logging.error(f"Error using true KV cache logic: {e}. Falling back to manual context.")
                    self.status_updated.emit("KV cache error, using fallback...") # Status update
                    # Fall through to the fallback method if any error occurs in the true cache logic

            # --- Fallback if no cache path or true cache logic failed ---
            logging.warning("Falling back to manual context prepending method.")
            # Ensure llm instance is passed if loaded, otherwise fallback loads it
            self._inference_thread_fallback(message, model_path, context_window, kv_cache_path, max_tokens, temperature, llm)

        except Exception as e:
            error_message = f"Error during inference setup or fallback: {str(e)}"
            self.status_updated.emit("Error") # Status update on error
            logging.exception(error_message)
            self.error_occurred.emit(error_message)
            self.response_complete.emit("", False)
        finally:
            # llm object is managed within the scope, should be released.
            # Set status to Idle only if no error occurred previously? Or always? Let's always set to Idle.
            self.status_updated.emit("Idle") # Ensure status returns to Idle
            logging.debug("Inference thread finished.")

    # --- Renamed original _inference_thread to be the fallback ---
    def _inference_thread_fallback(self, message: str, model_path: str, context_window: int,
                        kv_cache_path: Optional[str], max_tokens: int, temperature: float, llm: Optional[Llama] = None):
        """
        Fallback inference method using manual context prepending.
        Can optionally receive a pre-loaded Llama instance.
        """
        try:
            self.status_updated.emit("Using fallback method...") # Status update
            # If llm wasn't passed, load it (this happens if true cache path was None or initial load failed)
            if llm is None:
                self.status_updated.emit("Fallback: Loading model...") # Status update
                logging.info("Fallback: Loading model...")
                abs_model_path = str(Path(model_path).resolve())
                if not Path(abs_model_path).exists():
                    raise FileNotFoundError(f"Model file not found: {abs_model_path}")
                threads = int(self.config.get('LLAMACPP_THREADS', os.cpu_count() or 4))
                batch_size = int(self.config.get('LLAMACPP_BATCH_SIZE', 512))
                gpu_layers = int(self.config.get('LLAMACPP_GPU_LAYERS', 0))
                llm = Llama(
                    model_path=abs_model_path, n_ctx=context_window, n_threads=threads,
                    n_batch=batch_size, n_gpu_layers=gpu_layers, verbose=False
                )
                logging.info("Fallback: Model loaded.")
                self.status_updated.emit("Fallback: Preparing context...") # Status update

            # --- Prepare Chat History with Manual Context Prepending ---
            chat_messages = []
            system_prompt_content = "You are a helpful assistant." # Default system prompt

            if kv_cache_path: # Still use kv_cache_path to find original doc
                logging.info("Fallback: Attempting to prepend original document context to system prompt.")
                doc_context_text = ""
                try:
                    cache_info = self.cache_manager.get_cache_info(kv_cache_path)
                    if cache_info and 'original_document' in cache_info:
                        original_doc_path_str = cache_info['original_document']
                        if original_doc_path_str != "Unknown":
                            original_doc_path = Path(original_doc_path_str)
                            if original_doc_path.exists():
                                logging.info(f"Fallback: Reading start of original document: {original_doc_path}")
                                with open(original_doc_path, 'r', encoding='utf-8', errors='replace') as f_doc:
                                    doc_context_text = f_doc.read(8000)
                                logging.info(f"Fallback: Read {len(doc_context_text)} chars.")
                            else:
                                logging.warning(f"Fallback: Original document path not found: {original_doc_path}")
                        else:
                             logging.warning(f"Fallback: Original document path is 'Unknown' for cache: {kv_cache_path}")
                    else:
                        logging.warning(f"Fallback: Could not find cache info or original document path for cache: {kv_cache_path}")

                    if doc_context_text:
                         system_prompt_content = (
                             f"You are an assistant tasked with answering questions based *strictly* and *exclusively* on the following provided text snippet. "
                             f"Do not use any prior knowledge or information outside of this text. If the answer cannot be found within the text, state that clearly.\n\n"
                             f"--- TEXT SNIPPET START ---\n"
                             f"{doc_context_text}...\n" # Indicate snippet might be truncated
                             f"--- TEXT SNIPPET END ---\n\n"
                             f"Answer the user's question using *only* the information contained within the text snippet above."
                         )
                         logging.info("Fallback: Using STRICT system prompt with prepended context.")
                    else:
                         logging.warning("Fallback: Failed to read original document context, using default system prompt.")
                except Exception as e_ctx:
                    logging.error(f"Fallback: Error retrieving context: {e_ctx}")
                    logging.warning("Fallback: Using default system prompt.")

            # Add system prompt
            chat_messages.append({"role": "system", "content": system_prompt_content})
            # Add recent history
            history_limit = 4
            start_index = max(0, len(self.history) - 1 - history_limit) # Corrected history slicing
            recent_history = self.history[start_index:-1] # Get history BEFORE the last user message
            chat_messages.extend(recent_history)
            # Add latest user message (which is the last one in self.history now)
            chat_messages.append(self.history[-1])
            logging.info(f"Fallback: Prepared chat history with {len(chat_messages)} messages.")

            # --- Generate Response (Streaming using create_chat_completion) ---
            self.status_updated.emit("Fallback: Generating response...") # Status update
            logging.info(f"Fallback: Generating response using create_chat_completion...")
            stream = llm.create_chat_completion(
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
            self.status_updated.emit("Idle") # Status update
            if complete_response.strip():
                # Assistant response is added to history by the caller (send_message) if needed
                self.history.append({"role": "assistant", "content": complete_response}) # Add assistant response here
                self.response_complete.emit(complete_response, True)
            else:
                logging.warning("Fallback: Model stream completed but produced no text.")
                self.error_occurred.emit("Model generated an empty response.")
                self.response_complete.emit("", False)

        except Exception as e:
            # Errors specific to the fallback method
            error_message = f"Error during fallback inference: {str(e)}"
            self.status_updated.emit("Error") # Status update
            logging.exception(error_message)
            self.error_occurred.emit(error_message)
            self.response_complete.emit("", False)
        # No finally block here, llm instance is managed by the caller if passed in

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
        # Update true KV cache setting if present
        self.use_true_kv_cache_logic = self.config.get('USE_TRUE_KV_CACHE', True) # Keep default True for testing
        logging.info(f"ChatEngine configuration updated. True KV Cache Logic: {self.use_true_kv_cache_logic}")
