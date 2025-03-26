#!/usr/bin/env python3
"""
Document processing functionality for LlamaCag UI

Handles document validation, token estimation, and KV cache creation.
"""

import os
import sys
import tempfile
import logging
import shutil
import threading
import json
import re
import time
import pickle # Import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PyQt5.QtCore import QObject, pyqtSignal
from llama_cpp import Llama, LlamaCache

# Assuming utils.token_counter uses tiktoken or similar for a rough estimate
# We'll use llama-cpp's tokenizer for the actual processing count
from utils.token_counter import estimate_tokens


class DocumentProcessor(QObject):
    """Processes documents into KV caches for large context window models"""

    # Signals
    processing_progress = pyqtSignal(str, int)  # document_id, progress percentage
    processing_complete = pyqtSignal(str, bool, str)  # document_id, success, message
    token_estimation_complete = pyqtSignal(str, int, bool)  # document_id, tokens, fits_context

    def __init__(self, config, llama_manager, model_manager, cache_manager):
        """Initialize document processor"""
        super().__init__()
        self.config = config
        self.llama_manager = llama_manager
        self.model_manager = model_manager
        self.cache_manager = cache_manager

        # Set up directories
        self.temp_dir = Path(os.path.expanduser(config.get('LLAMACPP_TEMP_DIR', '~/cag_project/temp_chunks')))
        self.kv_cache_dir = Path(os.path.expanduser(config.get('LLAMACPP_KV_CACHE_DIR', '~/cag_project/kv_caches')))

        # Ensure directories exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.kv_cache_dir.mkdir(parents=True, exist_ok=True)

        # Document registry
        self._document_registry = {}
        self._load_document_registry()

    def _load_document_registry(self):
        """Load document registry from disk"""
        registry_file = self.kv_cache_dir / 'document_registry.json'
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    self._document_registry = json.load(f)
                logging.info(f"Loaded document registry with {len(self._document_registry)} entries")
            except Exception as e:
                logging.error(f"Failed to load document registry: {str(e)}")

    def _save_document_registry(self):
        """Save document registry to disk"""
        registry_file = self.kv_cache_dir / 'document_registry.json'
        try:
            with open(registry_file, 'w') as f:
                json.dump(self._document_registry, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save document registry: {str(e)}")

    def get_document_registry(self) -> Dict:
        """Get the document registry"""
        return self._document_registry

    def estimate_tokens(self, document_path: Union[str, Path]) -> int:
        """Estimate the number of tokens in a document"""
        document_path = Path(document_path)
        if not document_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")

        document_id = self._get_document_id(document_path)

        try:
            # Get current model's context size for preliminary check
            model_id = self.config.get('CURRENT_MODEL_ID', 'gemma-3-4b-128k') # TODO: Use default from model_manager?
            model_info = self.model_manager.get_model_info(model_id) # TODO: Handle model not found?
            context_size = model_info.get('context_window', 128000) if model_info else 128000 # TODO: Use a constant default?

            # Use llama-cpp-python for a more accurate estimate if model is available
            # For now, stick to the rough estimate for the UI feedback
            # TODO: Consider loading the model briefly just to tokenize for estimation? Might be slow.
            try:
                with open(document_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                estimated_tokens = estimate_tokens(content) # Keep using rough estimate for speed
            except Exception as e:
                 logging.warning(f"Could not read document for token estimation: {e}")
                 estimated_tokens = 0

            fits_context = estimated_tokens <= context_size

            # Emit signal with results
            self.token_estimation_complete.emit(document_id, estimated_tokens, fits_context)

            return estimated_tokens

        except Exception as e:
            logging.error(f"Failed to estimate tokens for {document_path}: {str(e)}")
            self.token_estimation_complete.emit(document_id, 0, False)
            return 0


    def process_document(self, document_path: Union[str, Path], set_as_master: bool = False) -> bool:
        """Process a document into a KV cache"""
        document_path = Path(document_path)
        if not document_path.exists():
            self.processing_complete.emit("unknown", False, f"Document not found: {document_path}")
            return False


        document_id = self._get_document_id(document_path)

        try:
            # Create KV cache path (using .llama_cache extension for clarity)
            # TODO: Consider adding model name/hash to cache path for compatibility?
            kv_cache_path = self.kv_cache_dir / f"{document_id}.llama_cache"

            # Get current model path
            model_id = self.config.get('CURRENT_MODEL_ID', 'gemma-3-4b-128k') # TODO: Default handling
            model_info = self.model_manager.get_model_info(model_id)

            if not model_info or not model_info.get('path'):
                error_msg = f"Model not found or path missing: {model_id}"
                logging.error(error_msg)
                self.processing_complete.emit(document_id, False, error_msg)
                return False

            model_path = model_info['path']
            context_window = model_info.get('context_window', 128000) # Use model's context window

            # Start processing in a separate thread
            threading.Thread(
                target=self._process_document_thread,
                args=(document_id, str(document_path), model_path, kv_cache_path, context_window, set_as_master),
                daemon=True,
            ).start()

            return True

        except Exception as e:
            logging.error(f"Failed to start document processing for {document_path}: {str(e)}")
            self.processing_complete.emit(document_id, False, f"Processing failed: {str(e)}")
            return False

    def _process_document_thread(self, document_id: str, document_path: str, model_path: str,
                               kv_cache_path: Path, context_window: int, set_as_master: bool):
        """Thread function for document processing using llama-cpp-python"""
        llm = None # Initialize llm to None for finally block
        try:
            logging.info(f"Starting KV cache creation for {document_id} using model {model_path}")
            self.processing_progress.emit(document_id, 0) # Start progress

            # --- Configuration ---
            threads = int(self.config.get('LLAMACPP_THREADS', os.cpu_count() or 4))
            batch_size = int(self.config.get('LLAMACPP_BATCH_SIZE', 512)) # Default 512 is common
            gpu_layers = int(self.config.get('LLAMACPP_GPU_LAYERS', 0)) # Default to 0 (CPU only)

            # --- Load Model ---
            logging.info(f"Loading model: {model_path} with context size {context_window}")
            self.processing_progress.emit(document_id, 5) # Progress update
            # Ensure model path is absolute and exists
            abs_model_path = str(Path(model_path).resolve())
            if not Path(abs_model_path).exists():
                 raise FileNotFoundError(f"Model file not found: {abs_model_path}")

            llm = Llama(
                model_path=abs_model_path,
                n_ctx=context_window,
                n_threads=threads,
                n_batch=batch_size,
                n_gpu_layers=gpu_layers,
                verbose=False # Keep logs clean, rely on Python logging
            )
            self.processing_progress.emit(document_id, 10) # Progress update

            # --- Read and Tokenize Document ---
            logging.info(f"Reading document: {document_path}")
            with open(document_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            logging.info(f"Tokenizing document...")
            tokens = llm.tokenize(content.encode('utf-8'))
            token_count = len(tokens)
            logging.info(f"Document token count: {token_count}")

            # --- Truncate if Necessary ---
            if token_count > context_window:
                logging.warning(f"Document ({token_count} tokens) exceeds context window ({context_window}). Truncating.")
                tokens = tokens[:context_window]
                token_count = len(tokens) # Update token count after truncation

            # --- Evaluate Tokens (Create KV Cache State) ---
            logging.info(f"Evaluating {token_count} tokens to create KV cache state...")
            eval_start_time = time.time()

            # Simple progress simulation during eval
            total_tokens_to_eval = len(tokens)
            processed_tokens = 0
            for i in range(0, total_tokens_to_eval, batch_size):
                batch = tokens[i:min(i + batch_size, total_tokens_to_eval)]
                if not batch:
                    break
                llm.eval(batch)
                processed_tokens += len(batch)
                progress = 10 + int(80 * (processed_tokens / total_tokens_to_eval)) # Eval is 10% to 90%
                self.processing_progress.emit(document_id, progress)

            eval_time = time.time() - eval_start_time
            logging.info(f"Evaluation complete in {eval_time:.2f} seconds.")
            self.processing_progress.emit(document_id, 90) # Progress update

            # --- Save KV Cache State ---
            logging.info(f"Saving KV cache state to {kv_cache_path}...")
            save_successful = False
            try:
                print("Attempting llm.save_state(kv_cache_path)...")
                llm.save_state(kv_cache_path) # Try with Path object first
                print("KV cache state saved successfully with path argument.") # Corrected indentation
                save_successful = True # Corrected indentation
            except TypeError:
                print("TypeError with save_state(path), attempting save_state() without arguments and pickling...")
                try:
                    state_data = llm.save_state() # Try without arguments, capture the state object
                    # Save the state object using pickle
                    with open(kv_cache_path, 'wb') as f_pickle:
                        pickle.dump(state_data, f_pickle)
                    print("KV cache state object pickled successfully.")
                    save_successful = True
                except AttributeError:
                    # Handle case where save_state() doesn't return a valid object or pickle fails
                    print("Failed to capture or pickle state object from save_state().")
                    # Fall through to error handling
                except pickle.PicklingError as e_pickle:
                    print(f"Error pickling KV cache state object: {e_pickle}")
                    # Fall through to error handling
                except Exception as e_no_args:
                    # This except block seems duplicated and incorrectly indented, removing the inner one
                    print(f"Error calling save_state() without arguments or pickling: {e_no_args}")
                    # Fall through to the error below
            except Exception as e_path_arg:
                 print(f"Error calling save_state(path): {e_path_arg}") # Corrected indentation
                 # Fall through to the error below

            if not save_successful:
                 # If neither method worked or resulted in a file
                 error_msg = f"Failed to save KV cache state to {kv_cache_path} using known methods."
                 logging.error(error_msg)
                 # Create placeholder to prevent subsequent errors trying to load non-existent cache
                 with open(kv_cache_path, 'w') as f:
                     f.write("KV CACHE SAVE FAILED PLACEHOLDER")
                 print("Created placeholder file due to save failure.")
                 raise RuntimeError(error_msg)

            # --- Continue if save was successful ---
            logging.info("KV cache state save process completed.") # Renamed log message
            self.processing_progress.emit(document_id, 95) # Progress update

            # --- Update Document Registry ---
            doc_info = {
                'document_id': document_id,
                'original_file_path': document_path, # Store original path
                'kv_cache_path': str(kv_cache_path),
                'token_count': token_count, # Actual tokens processed
                'context_size': context_window, # Model's context window used
                'model_id': self.config.get('CURRENT_MODEL_ID'),
                'created_at': time.time(),
                'last_used': None,
                'usage_count': 0,
                'is_master': False # Default to false
            }
            self._document_registry[document_id] = doc_info
            self._save_document_registry()

            # --- Set as Master if Requested ---
            if set_as_master:
                if self.set_as_master(document_id):
                     doc_info['is_master'] = True # Update local dict if successful
                     self._document_registry[document_id] = doc_info # Resave registry
                     self._save_document_registry()
                else:
                     logging.warning(f"Failed to set {document_id} as master cache.")
                     # Proceed anyway, but log the warning

            # --- Register with Cache Manager ---
            self.cache_manager.register_cache(document_id, str(kv_cache_path), context_window)

            # --- Notify Completion ---
            self.processing_progress.emit(document_id, 100) # Final progress
            self.processing_complete.emit(
                document_id, True, f"KV cache created successfully at {kv_cache_path}"
            )

        except FileNotFoundError as e: # Specific exception for file not found
             error_message = f"File not found error processing document {document_id}: {str(e)}"
             logging.error(error_message)
             self.processing_complete.emit(document_id, False, error_message)
        except RuntimeError as e: # Specific exception for runtime errors (like model load/eval)
             error_message = f"Runtime error processing document {document_id}: {str(e)}"
             logging.error(error_message)
             self.processing_complete.emit(document_id, False, error_message)
        except Exception as e: # Catch-all for other unexpected exceptions
            error_message = f"Unexpected error processing document {document_id}: {str(e)}"
            logging.exception(error_message) # Log full traceback for unexpected errors
            self.processing_complete.emit(document_id, False, error_message)
        finally:
            # Ensure model is released if loaded
            if llm is not None:
                 # Assuming llama-cpp-python doesn't have an explicit close/del method needed
                 # If it does, call it here. Otherwise, Python's garbage collection handles it.
                 logging.debug(f"Model object for {document_id} going out of scope or cleanup if needed.")
                 pass # No explicit cleanup known for Llama object itself
            logging.debug(f"Finished processing thread for {document_id}")


    def set_as_master(self, document_id: str) -> bool:
        """Set a document as the master KV cache"""
        if document_id not in self._document_registry:
            logging.error(f"Cannot set master: Document {document_id} not found in registry.")
            return False

        doc_info = self._document_registry[document_id]
        kv_cache_path_str = doc_info.get('kv_cache_path')

        if not kv_cache_path_str:
             logging.error(f"Cannot set master: KV cache path missing for {document_id}.")
             return False

        kv_cache_path = Path(kv_cache_path_str)
        # Check if the cache file exists. Removed the read_text check as it's now a binary pickle file.
        # Loading errors will be handled by the chat_engine when it tries to unpickle.
        if not kv_cache_path.exists():
            logging.error(f"Cannot set master: KV cache file not found: {kv_cache_path}")
            return False

        # Define master cache path (using .llama_cache extension)
        master_cache_path = self.kv_cache_dir / 'master_cache.llama_cache'
        try:
            # Copy the actual KV cache file
            shutil.copy2(kv_cache_path, master_cache_path)
            logging.info(f"Set {document_id} ({kv_cache_path}) as master KV cache at {master_cache_path}")

            # Update config (store the path to the master cache file)
            self.config['MASTER_KV_CACHE_PATH'] = str(master_cache_path) # Use a more specific key
            # TODO: Ensure config saving mechanism is triggered if needed

            # Update document info in registry (mark others as not master)
            for doc_id_reg, info in self._document_registry.items():
                 info['is_master'] = (doc_id_reg == document_id)
            self._save_document_registry() # Save updated registry

            # Register the newly created master cache file with the cache manager
            # Pass along details from the original document's info
            try:
                 logging.info(f"Registering master cache '{str(master_cache_path)}' with cache manager.")
                 self.cache_manager.register_cache(
                     document_id="master_cache", # Specific ID for master
                     cache_path=str(master_cache_path),
                     context_size=doc_info.get('context_size', 0),
                     token_count=doc_info.get('token_count', 0),
                     original_file_path=doc_info.get('original_file_path', ''), # Crucial: Pass original path
                     model_id=doc_info.get('model_id', ''),
                     is_master=True
                 )
            except Exception as reg_e:
                 logging.error(f"Failed to register master cache with cache manager: {reg_e}")
                 # Continue anyway, but log the error

            return True

        except Exception as e:
            logging.error(f"Failed to set {document_id} as master KV cache: {str(e)}")
            return False

    def _get_document_id(self, document_path: Path) -> str:
        """Generate a consistent document ID from path"""
        # Use filename without extension
        doc_id = document_path.stem.lower()

        # Clean up non-alphanumeric characters
        doc_id = re.sub(r'[^a-z0-9_]', '_', doc_id)

        return doc_id
