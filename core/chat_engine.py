#!/usr/bin/env python3
"""
Chat functionality for LlamaCag UI

Handles interaction with the model using KV caches.
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
        actual_kv_cache_path = None
        if self.use_kv_cache:
            if self.current_kv_cache_path:
                actual_kv_cache_path = self.current_kv_cache_path
            else:
                # Try to use master cache path from config
                master_cache_cfg_key = 'MASTER_KV_CACHE_PATH' # Use the key set in document_processor
                master_cache_path_str = self.config.get(master_cache_cfg_key)
                if master_cache_path_str:
                     master_cache_path = Path(master_cache_path_str)
                     if master_cache_path.exists() and master_cache_path.suffix == '.llama_cache':
                         actual_kv_cache_path = str(master_cache_path)
                         logging.info(f"Using master KV cache: {actual_kv_cache_path}")
                     else:
                         logging.warning(f"Master KV cache path invalid or file missing: {master_cache_path_str}")
                         self.error_occurred.emit("Master KV cache is configured but invalid/missing.")
                         return False
                else:
                    self.error_occurred.emit("KV cache usage is enabled, but no cache is selected and no master cache is configured.")
                    return False
            # Final check if cache file exists before starting thread
            if actual_kv_cache_path and not Path(actual_kv_cache_path).exists():
                 self.error_occurred.emit(f"Selected KV cache file not found: {actual_kv_cache_path}")
                 return False

        # Add user message to history *before* starting thread
        self.history.append({"role": "user", "content": message})

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
            logging.info(f"Inference thread started. Model: {model_path}, Cache: {kv_cache_path}")

            # --- Configuration ---
            threads = int(self.config.get('LLAMACPP_THREADS', os.cpu_count() or 4))
            batch_size = int(self.config.get('LLAMACPP_BATCH_SIZE', 512))
            gpu_layers = int(self.config.get('LLAMACPP_GPU_LAYERS', 0))

            # --- Load Model ---
            logging.info(f"Loading model: {model_path}")
            # Ensure model path is absolute
            abs_model_path = str(Path(model_path).resolve())
            if not Path(abs_model_path).exists():
                 raise FileNotFoundError(f"Model file not found: {abs_model_path}")

            llm = Llama(
                model_path=abs_model_path,
                n_ctx=context_window, # Use context window from model info
                n_threads=threads,
                n_batch=batch_size,
                n_gpu_layers=gpu_layers,
                verbose=False # Keep logs clean
            )
            logging.info("Model loaded.")

            # --- Load KV Cache if applicable ---
            if kv_cache_path:
                logging.info(f"Loading KV cache state from: {kv_cache_path}")
                try:
                    # Load the pickled state object
                    with open(kv_cache_path, 'rb') as f_pickle:
                        state_data = pickle.load(f_pickle)
                    # Load the state object into the model
                    llm.load_state(state_data)
                    logging.info("KV cache loaded successfully from pickled object.")
                except pickle.UnpicklingError as e_pickle:
                    logging.error(f"Failed to unpickle KV cache state: {e_pickle}")
                    raise RuntimeError(f"Failed to unpickle KV cache: {e_pickle}") from e_pickle
                except Exception as e:
                    # Log other errors during loading (e.g., incompatible state)
                    logging.error(f"Failed to load KV cache state: {e}")
                    raise RuntimeError(f"Failed to load KV cache: {e}") from e
            else:
                logging.info("No KV cache specified or cache usage disabled.")
                # If no cache, the prompt should ideally contain the context,
                # but current design implies just asking the question directly.

            # --- Prepare Prompt ---
            prompt_for_generation = message # Default to just the message
            if kv_cache_path:
                logging.info("KV cache is active. Attempting to prepend original document context.")
                doc_context_text = ""
                try:
                    cache_info = self.cache_manager.get_cache_info(kv_cache_path)
                    if cache_info and 'original_document' in cache_info:
                        original_doc_path_str = cache_info['original_document']
                        original_doc_path = Path(original_doc_path_str)
                        if original_doc_path.exists():
                            logging.info(f"Reading start of original document: {original_doc_path}")
                            with open(original_doc_path, 'r', encoding='utf-8', errors='replace') as f_doc:
                                doc_context_text = f_doc.read(2000) # Read first 2000 chars as context
                            logging.info(f"Read {len(doc_context_text)} chars from original document.")
                        else:
                            logging.warning(f"Original document path not found: {original_doc_path}")
                    else:
                        logging.warning(f"Could not find cache info or original document path for cache: {kv_cache_path}")

                    if doc_context_text:
                         # Construct prompt with explicit context + question
                         prompt_for_generation = (
                             f"Using the following context:\n\n"
                             f"{doc_context_text}...\n\n" # Add ellipsis to indicate truncation
                             f"Answer the question based *only* on the context provided:\n"
                             f"User: {message}\n"
                             f"Assistant:"
                         )
                         logging.info("Using explicit context prepended prompt structure.")
                    else:
                         # Fallback if context couldn't be read - use simpler prompt but still mention context
                         prompt_for_generation = f"Using the provided context (loaded separately), answer the following question:\n\nUser: {message}\nAssistant:"
                         logging.warning("Failed to read original document context, using fallback prompt.")

                except Exception as e_ctx:
                    logging.error(f"Error retrieving or reading original document context: {e_ctx}")
                    # Fallback prompt if error occurs
                    prompt_for_generation = f"Using the provided context (loaded separately), answer the following question:\n\nUser: {message}\nAssistant:"
                    logging.warning("Error during context retrieval, using fallback prompt.")

            else:
                # If no cache, use the message directly (or potentially a chat template later)
                logging.info("Using simple prompt structure (no KV cache).")


            # --- Generate Response (Streaming using create_completion) ---
            # Reverted back to create_completion as create_chat_completion didn't help with cache
            logging.info(f"Generating response using create_completion (max_tokens={max_tokens}, temp={temperature})...")
            stream = llm.create_completion(
                prompt=prompt_for_generation, # Use the potentially context-prepended prompt
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            complete_response = ""
            for chunk in stream:
                try:
                    # create_completion streams text directly
                    text = chunk["choices"][0]["text"]
                    if text:
                        self.response_chunk.emit(text)
                        complete_response += text
                except (KeyError, IndexError) as e:
                    logging.warning(f"Could not extract text from stream chunk: {chunk}, Error: {e}")

            logging.info("Response generation complete.")

            # --- Finalize ---
            if complete_response.strip():
                # Add assistant response to history
                self.history.append({"role": "assistant", "content": complete_response})

                # Update cache usage stats if cache was used
                if kv_cache_path:
                    # Need the document_id associated with the cache path
                    # This requires looking up the cache path in the cache_manager or document_registry
                    # For now, let's assume cache_manager has a way to map path back to id or handle usage update by path
                    try:
                        self.cache_manager.update_usage_by_path(kv_cache_path)
                    except AttributeError:
                         logging.warning("cache_manager does not have update_usage_by_path method.")
                    except Exception as e:
                         logging.warning(f"Failed to update usage stats for {kv_cache_path}: {e}")


                # Signal completion
                self.response_complete.emit(complete_response, True)
            else:
                # Handle case where stream produced no text
                logging.warning("Model stream completed but produced no text.")
                self.error_occurred.emit("Model generated an empty response.")
                self.response_complete.emit("", False) # Signal completion with empty response

        except Exception as e:
            error_message = f"Error during inference: {str(e)}"
            logging.exception(error_message) # Log full traceback
            self.error_occurred.emit(error_message)
            self.response_complete.emit("", False) # Signal failure
        finally:
            # Ensure model is released (llama-cpp-python handles this when llm goes out of scope,
            # but explicit del might help if managing instances long-term)
            if llm is not None:
                # del llm # Optional: Explicitly delete to free resources sooner
                pass
            logging.debug("Inference thread finished.")


    # Remove the old _extract_answer method as it's no longer needed
    # def _extract_answer(self, full_output: str) -> str: ...
    
    def clear_history(self):
        """Clear chat history"""
        self.history = []
        logging.info("Chat history cleared")

    def get_history(self) -> List[Dict]:
        """Get chat history"""
        return self.history
    
    def save_history(self, file_path: Union[str, Path]) -> bool:
        """Save chat history to a file"""
        try:
            with open(file_path, 'w') as f:
                json.dump({
                    "history": self.history,
                    "model_id": self.config.get('CURRENT_MODEL_ID'),
                    "kv_cache_path": self.current_kv_cache_path, # Save the path
                    "timestamp": time.time(),
                    "use_kv_cache_setting": self.use_kv_cache # Save the toggle state
                }, f, indent=2)
            logging.info(f"Chat history saved to {file_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save chat history: {str(e)}")
            return False

    def load_history(self, file_path: Union[str, Path]) -> bool:
        """Load chat history from a file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            self.history = data.get("history", [])

            # Load KV cache path and toggle state
            kv_cache_path_str = data.get("kv_cache_path")
            if kv_cache_path_str and Path(kv_cache_path_str).exists() and Path(kv_cache_path_str).suffix == '.llama_cache':
                self.current_kv_cache_path = kv_cache_path_str
                logging.info(f"Loaded KV cache path from history: {self.current_kv_cache_path}")
            else:
                 self.current_kv_cache_path = None # Reset if invalid or missing

            self.use_kv_cache = data.get("use_kv_cache_setting", True) # Load toggle state, default to True
            logging.info(f"Loaded use_kv_cache setting from history: {self.use_kv_cache}")

            # TODO: Maybe verify model_id matches current config?

            logging.info(f"Chat history loaded from {file_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load chat history: {str(e)}")
            return False

    def update_config(self, config):
        """Update configuration"""
        self.config = config
        # No longer need to update script path
        logging.info("ChatEngine configuration updated.")
