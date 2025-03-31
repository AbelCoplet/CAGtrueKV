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
    cache_ready_for_use = pyqtSignal(str) # New signal: cache_path

    def __init__(self, config_manager, llama_manager, model_manager, cache_manager):
        """Initialize document processor"""
        super().__init__()
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.llama_manager = llama_manager
        self.model_manager = model_manager
        self.cache_manager = cache_manager

        # Set up directories
        self.temp_dir = Path(os.path.expanduser(self.config.get('LLAMACPP_TEMP_DIR', '~/cag_project/temp_chunks')))
        self.kv_cache_dir = Path(os.path.expanduser(self.config.get('LLAMACPP_KV_CACHE_DIR', '~/cag_project/kv_caches')))

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

    # Added use_now parameter, removed set_as_master
    def process_document(self, document_path: Union[str, Path], use_now: bool = False) -> bool:
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
                args=(document_id, str(document_path), model_path, kv_cache_path, context_window, use_now), # Pass use_now
                daemon=True,
            ).start()

            return True

        except Exception as e:
            logging.error(f"Failed to start document processing for {document_path}: {str(e)}")
            self.processing_complete.emit(document_id, False, f"Processing failed: {str(e)}")
            return False

    def _save_kv_cache_state(self, llm, kv_cache_path: Path) -> bool:
        """
        Improved function to save KV cache state using recommended approach.
        Returns True if successful, False otherwise.
        """
        logging.info(f"Saving KV cache state to {kv_cache_path}...")

        # Method 1: Try getting state without arguments first, then pickle
        try:
            logging.info("Using save_state() without arguments and pickling...")
            state_data = llm.save_state()  # Get state data object

            # Verify we got something valid
            if state_data is None:
                logging.error("save_state() returned None")
                return False
                
            # Log state data type and size estimate
            logging.debug(f"State data type: {type(state_data)}")
            try:
                import sys
                state_size_estimate = sys.getsizeof(state_data)
                logging.debug(f"State data approximate size: {state_size_estimate} bytes")
            except Exception as e_size:
                logging.warning(f"Could not estimate state data size: {e_size}")

            # Save with pickle
            try:
                with open(kv_cache_path, 'wb') as f_pickle:
                    pickle.dump(state_data, f_pickle)
                
                # Verify the file was created and has content
                if not kv_cache_path.exists():
                    logging.error("Pickle file was not created!")
                    return False
                    
                if kv_cache_path.stat().st_size == 0:
                    logging.error("Pickle file was created but is empty!")
                    return False
                    
                logging.info(f"KV cache state saved successfully via pickle. File size: {kv_cache_path.stat().st_size} bytes")
                return True
            except (pickle.PicklingError, OSError) as e:
                logging.error(f"Failed to pickle state data: {e}")
                return False
                
        except (AttributeError, Exception) as e:
            logging.error(f"Error in primary KV cache save method: {e}")

        # Method 2: Try direct path argument as fallback
        try:
            logging.info("Trying save_state with direct path argument...")
            llm.save_state(str(kv_cache_path))

            # Check if file was created
            if kv_cache_path.exists() and kv_cache_path.stat().st_size > 0:
                logging.info(f"KV cache saved successfully with path argument. File size: {kv_cache_path.stat().st_size} bytes")
                return True
            else:
                logging.error("save_state(path) did not create a valid file")
        except Exception as e:
            logging.error(f"Error in fallback KV cache save method: {e}")

        # If we get here, both methods failed
        logging.error("All KV cache save methods failed")
        return False

    def _verify_kv_cache_integrity(self, kv_cache_path: Path) -> bool:
        """
        Verify that a saved KV cache file can be loaded correctly.
        This helps ensure the file isn't corrupted or malformed.
        """
        logging.info(f"Verifying KV cache integrity for {kv_cache_path}...")
        try:
            # Try to load the cache file
            with open(kv_cache_path, 'rb') as f:
                state_data = pickle.load(f)
                
            # Check basic properties of state_data
            if state_data is None:
                logging.error("Verification failed: KV cache file loaded as None")
                return False
                
            # Log basic info
            logging.info(f"KV cache verification successful. Cache appears valid.")
            return True
            
        except Exception as e:
            logging.error(f"KV cache verification failed: {str(e)}")
            return False

    # Added use_now parameter, removed set_as_master
    def _process_document_thread(self, document_id: str, document_path: str, model_path: str,
                               kv_cache_path: Path, context_window: int, use_now: bool):
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

            # Log detailed parameters
            logging.info(f"Model load parameters: threads={threads}, batch_size={batch_size}, gpu_layers={gpu_layers}, context_window={context_window}")
            
            # Load the model with the specified parameters
            llm = Llama(
                model_path=abs_model_path,
                n_ctx=context_window,
                n_threads=threads,
                n_batch=batch_size,
                n_gpu_layers=gpu_layers,
                verbose=False # Keep logs clean, rely on Python logging
            )
            self.processing_progress.emit(document_id, 10) # Progress update
            logging.info(f"Model loaded successfully: n_ctx={llm.n_ctx()}")

            # --- Read and Tokenize Document ---
            logging.info(f"Reading document: {document_path}")
            try:
                with open(document_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    
                # Basic document stats
                doc_size_bytes = len(content.encode('utf-8'))
                doc_lines = content.count('\n') + 1
                logging.info(f"Document stats: size={doc_size_bytes} bytes, lines={doc_lines}")
                
                # Check for potential issues like binary data
                if '\x00' in content[:1000]:
                    logging.warning("Document appears to contain binary data (null bytes)")
                    
                # Check for common encoding issues
                try:
                    content.encode('utf-8')
                except UnicodeEncodeError:
                    logging.warning("Document contains characters that cannot be encoded in UTF-8")
                    
            except UnicodeDecodeError as ude:
                logging.error(f"UnicodeDecodeError while reading document: {ude}")
                raise RuntimeError(f"Document appears to have encoding issues: {ude}. Try converting to UTF-8.")
                
            logging.info(f"Tokenizing document...")
            tokens = llm.tokenize(content.encode('utf-8'))
            token_count = len(tokens)
            logging.info(f"Document token count: {token_count}")

            # --- Truncate if Necessary ---
            if token_count > context_window:
                logging.warning(f"Document ({token_count} tokens) exceeds context window ({context_window}). Truncating.")
                tokens = tokens[:context_window]
                token_count = len(tokens) # Update token count after truncation
                truncation_ratio = token_count / token_count * 100
                logging.info(f"Keeping {token_count} tokens ({truncation_ratio:.1f}% of document)")

            # --- Evaluate Tokens (Create KV Cache State) ---
            logging.info(f"Evaluating {token_count} tokens to create KV cache state...")
            eval_start_time = time.time()

            # Simple progress simulation during eval
            total_tokens_to_eval = len(tokens)
            processed_tokens = 0

            # Adjust batch size based on document size for improved performance (as per plan)
            base_batch_size = llm.n_batch # Access attribute directly, not call method
            if token_count > 10000:
                # For large documents, start with smaller batches
                adaptive_batch_size = min(256, base_batch_size) # Use 256 or model's batch size, whichever is smaller
                logging.info(f"Large document detected ({token_count} tokens > 10000), using adaptive batch size: {adaptive_batch_size} (Base: {base_batch_size})")
            else:
                adaptive_batch_size = base_batch_size # Use the model's default batch size
                logging.info(f"Using standard batch size: {adaptive_batch_size}")

            # Process tokens in batches using the adaptive size
            try:
                for i in range(0, total_tokens_to_eval, adaptive_batch_size): # Use adaptive_batch_size
                    batch = tokens[i:min(i + adaptive_batch_size, total_tokens_to_eval)] # Use adaptive_batch_size
                    if not batch:
                        break
                        
                    # Evaluate this batch
                    batch_start_time = time.time()
                    llm.eval(batch)
                    batch_end_time = time.time()
                    
                    # Update progress and log
                    processed_tokens += len(batch)
                    progress = 10 + int(80 * (processed_tokens / total_tokens_to_eval)) # Eval is 10% to 90%
                    self.processing_progress.emit(document_id, progress)
                    
                    # Detailed logging for batches
                    if i % (5 * adaptive_batch_size) == 0 or i == 0:  # Log every 5 batches or first batch, using adaptive size
                        batch_time = batch_end_time - batch_start_time
                        tokens_per_sec = len(batch) / batch_time if batch_time > 0 else 0
                        logging.info(f"Processed batch {i//adaptive_batch_size + 1}/{(total_tokens_to_eval + adaptive_batch_size - 1)//adaptive_batch_size}: " # Use adaptive size in calculation
                                    f"{len(batch)} tokens in {batch_time:.2f}s ({tokens_per_sec:.1f} tokens/sec), "
                                    f"Progress: {processed_tokens}/{total_tokens_to_eval} tokens ({progress}%)")
            except Exception as eval_error:
                logging.error(f"Error during token evaluation: {eval_error}")
                # Check if we processed any tokens at all
                if processed_tokens == 0:
                    raise RuntimeError(f"Failed to process any tokens: {eval_error}")
                else:
                    # Some tokens were processed, log warning and continue (partial cache better than none)
                    logging.warning(f"Created partial KV cache with {processed_tokens}/{total_tokens_to_eval} tokens")

            eval_time = time.time() - eval_start_time
            tokens_per_second = token_count / eval_time if eval_time > 0 else 0
            logging.info(f"Evaluation complete in {eval_time:.2f} seconds ({tokens_per_second:.2f} tokens/sec).")
            self.processing_progress.emit(document_id, 90) # Progress update

            # --- Atomic Cache Save ---
            # 1. Define temporary path
            temp_cache_path = kv_cache_path.with_suffix('.tmp_cache')
            logging.info(f"Saving temporary cache state to: {temp_cache_path}")

            # 2. Save to temporary path
            save_successful = self._save_kv_cache_state(llm, temp_cache_path)

            if not save_successful:
                # Clean up temporary file if save failed
                if temp_cache_path.exists():
                    try:
                        temp_cache_path.unlink()
                    except OSError as e_unlink:
                        logging.warning(f"Could not delete failed temporary cache file {temp_cache_path}: {e_unlink}")
                error_msg = f"Failed to save temporary KV cache state to {temp_cache_path}."
                logging.error(error_msg)
                raise RuntimeError(error_msg)

            # 3. Verify temporary cache integrity
            self.processing_progress.emit(document_id, 95) # Progress update
            verify_successful = self._verify_kv_cache_integrity(temp_cache_path)

            if not verify_successful:
                # Clean up failed temporary file
                if temp_cache_path.exists():
                    try:
                        temp_cache_path.unlink()
                    except OSError as e_unlink:
                        logging.warning(f"Could not delete failed temporary cache file {temp_cache_path}: {e_unlink}")
                logging.error(f"Temporary KV cache verification failed.")
                raise RuntimeError(f"Created temporary KV cache file failed verification check.")

            # 4. Rename temporary file to final path (Atomic operation on most systems)
            try:
                # Ensure final destination doesn't exist (shouldn't, but safety check)
                if kv_cache_path.exists():
                    kv_cache_path.unlink()
                temp_cache_path.rename(kv_cache_path)
                logging.info(f"Successfully renamed temporary cache to final path: {kv_cache_path}")
            except OSError as e_rename:
                # Clean up temporary file if rename failed
                if temp_cache_path.exists():
                    try:
                        temp_cache_path.unlink()
                    except OSError as e_unlink:
                        logging.warning(f"Could not delete temporary cache file after rename error {temp_cache_path}: {e_unlink}")
                error_msg = f"Failed to rename temporary cache to final path: {e_rename}"
                logging.error(error_msg)
                raise RuntimeError(error_msg)

            # --- Cache is now saved and verified ---
            logging.info("KV cache state saved and verified successfully.")
            self.processing_progress.emit(document_id, 97) # Progress update

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

            # --- REMOVED Set as Master Logic ---

            # --- Register with Cache Manager ---
            # Pass context_size to register_cache
            self.cache_manager.register_cache(
                document_id=document_id,
                cache_path=str(kv_cache_path),
                context_size=context_window,
                token_count=token_count,
                original_file_path=document_path,
                model_id=self.config.get('CURRENT_MODEL_ID'),
                is_master=doc_info['is_master'] # Pass final master status
            )

            # --- Notify Completion ---
            self.processing_progress.emit(document_id, 100) # Final progress
            self.processing_complete.emit(
                document_id, True, f"KV cache created successfully at {kv_cache_path}"
            )

            # --- Emit signal to use cache now if requested ---
            if use_now:
                logging.info(f"Emitting signal to use cache now: {kv_cache_path}")
                self.cache_ready_for_use.emit(str(kv_cache_path))

        except FileNotFoundError as e: # Specific exception for file not found
             error_message = f"File not found error processing document {document_id}: {str(e)}"
             logging.error(error_message)
             self.processing_complete.emit(document_id, False, error_message)
        except RuntimeError as e: # Specific exception for runtime errors (like model load/eval/save)
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
                 logging.debug(f"Model object for {document_id} going out of scope or cleanup if needed.")
                 pass # No explicit cleanup known for Llama object itself
            logging.debug(f"Finished processing thread for {document_id}")

    def update_config(self, config_manager):
        """Update configuration from the config manager"""
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        # Update directories if they've changed
        new_temp_dir = Path(os.path.expanduser(self.config.get('LLAMACPP_TEMP_DIR', '~/cag_project/temp_chunks')))
        new_kv_cache_dir = Path(os.path.expanduser(self.config.get('LLAMACPP_KV_CACHE_DIR', '~/cag_project/kv_caches')))
        
        # Check if directories changed
        if new_temp_dir != self.temp_dir:
            self.temp_dir = new_temp_dir
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Updated temp directory to {self.temp_dir}")
            
        if new_kv_cache_dir != self.kv_cache_dir:
            self.kv_cache_dir = new_kv_cache_dir
            self.kv_cache_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Updated KV cache directory to {self.kv_cache_dir}")
            
            # Reload document registry if KV cache directory changed
            self._load_document_registry()

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
        # Check if the cache file exists. Loading errors will be handled by the chat_engine.
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
