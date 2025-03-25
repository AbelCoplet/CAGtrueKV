# LlamaCag UI

## Context-Augmented Generation for Large Language Models

LlamaCag UI is a desktop application that enables context-augmented generation (CAG) with large language models. It allows you to feed documents into a language model's context window and ask questions that leverage that context, creating an experience similar to chatting with your documents.

## Core Concept: Context-Augmented Generation (CAG) with KV Caching

The fundamental idea behind LlamaCag UI is **Context-Augmented Generation (CAG)**, leveraging the power of `llama.cpp`'s KV (Key/Value) caching mechanism. Unlike standard RAG (Retrieval-Augmented Generation) systems that retrieve snippets of text, CAG:

1.  **Processes the entire document** through the language model once to generate its internal state (the KV cache).
2.  **Saves this KV cache** to disk.
3.  **Loads the saved KV cache** for subsequent interactions, allowing the model to "remember" the document context without re-processing the full text.
4.  **Enables deep contextual understanding** by having the model's state primed with the document content.
5.  **Allows fast follow-up questions** as only the new query needs to be processed by the model.

This approach allows models like Gemma 3 and Llama 3 to efficiently utilize their large context windows (e.g., 128K tokens) for in-depth document analysis and question answering, significantly speeding up conversations after the initial document processing.

## Features

- **Model Management**: Download, manage, and select from various large context window models
- **Model Management**: Download, manage, and select from various large context window models (GGUF format).
- **Document Processing**: Load documents and process them into true `llama.cpp` KV caches for efficient context augmentation.
- **Interactive Chat**: Chat with your documents, leveraging the pre-processed KV cache for fast responses.
- **KV Cache Monitor**: Track and manage your document KV caches.
- **Settings**: Configure paths, model parameters (threads, batch size, GPU layers), and application behavior.

### Screenshots

![Screenshot 2025-03-25 at 22.42.34](images/Screenshot%202025-03-25%20at%2022.42.34.png)
![Screenshot 2025-03-25 at 22.42.41](images/Screenshot%202025-03-25%20at%2022.42.41.png)
![Screenshot 2025-03-25 at 22.42.51](images/Screenshot%202025-03-25%20at%2022.42.51.png)
![Screenshot 2025-03-25 at 22.42.56](images/Screenshot%202025-03-25%20at%2022.42.56.png)
![Screenshot 2025-03-25 at 22.43.12](images/Screenshot%202025-03-25%20at%2022.43.12.png)

## Installation

# LlamaCag UI Complete File Structure

```
LlamaCagUI/
├── main.py                  # Application entry point, initializes all components
├── run.sh                   # Script to run the application with correct environment
├── setup_requirements.sh    # Installs all dependencies, llama.cpp, and creates directories
├── cleanup_and_fix.sh       # Utility to clean up installation issues
├── diagnose.sh              # Diagnostic tool to check for proper installation
├── reset.sh                 # Resets application settings to defaults
├── debug_subprocess.py      # Utility for debugging subprocess calls
├── test_app.py              # Simple PyQt test application
├── .env                     # Environment variables and configuration
├── .env.example             # Example configuration file
├── model_urls.txt           # List of model download URLs
├── .gitattributes           # Git attributes configuration
├── .gitignore               # Files to ignore in Git repository
│
├── core/                    # Core functionality components
│   ├── __init__.py          # Package initialization
│   ├── cache_manager.py     # Manages KV caches, including listing, purging and registry
│   ├── chat_engine.py       # Handles chat interaction with models using KV caches
│   ├── document_processor.py # Processes documents into KV caches, estimates tokens
│   ├── llama_manager.py     # Manages llama.cpp installation and updates
│   ├── model_manager.py     # Handles model downloading, importing, and selection
│   └── n8n_interface.py     # Interface for optional n8n workflow integration
│
├── ui/                      # User interface components
│   ├── __init__.py          # Package initialization
│   ├── main_window.py       # Main application window with tabbed interface
│   ├── model_tab.py         # UI for model management and downloading
│   ├── document_tab.py      # UI for document processing and cache creation
│   ├── chat_tab.py          # UI for chatting with documents
│   ├── chat_tab.py.backup   # Backup of chat tab implementation
│   ├── cache_tab.py         # UI for KV cache monitoring and management
│   ├── settings_tab.py      # UI for application configuration
│   │
│   └── components/          # Reusable UI components
│       ├── __init__.py      # Package initialization
│       └── toast.py         # Toast notification component for temporary messages
│
├── utils/                   # Utility functions
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management for app settings
│   ├── logging_utils.py     # Logging setup and utilities
│   └── token_counter.py     # Utilities for estimating tokens in documents
```

*(Note: `scripts/bash` and `utils/script_runner.py` have been removed)*

## Runtime-Created Directories (Not in Repository)

```
~/.llamacag/                 # User configuration directory
├── logs/                    # Application log files with timestamps
├── config.json              # User-specific configuration
└── custom_models.json       # User-defined model configurations

~/Documents/llama.cpp/       # llama.cpp installation directory
├── build/                   # Compiled binaries
│   └── bin/                 # Contains main or llama-cli executables
├── models/                  # Downloaded model files (.gguf format)
└── ... (other llama.cpp files)

~/cag_project/               # Working directory for documents and caches (configurable)
├── kv_caches/               # Stores document KV caches
│   ├── *.llama_cache        # Binary KV cache state files generated by llama.cpp
│   ├── master_cache.llama_cache # Default cache used when none selected
│   ├── cache_registry.json  # Metadata about created caches
│   └── usage_registry.json  # Usage statistics for caches
└── temp_chunks/             # Temporary files used during processing (configurable)
```

### Prerequisites

- macOS (tested on macOS Ventura and later)
- Python 3.8 or higher
- 16GB+ RAM recommended for optimal performance
- Internet connection (for downloading models)

### Memory Requirements

- 8GB RAM: Limited to smaller documents (~25K tokens)
- 16GB RAM: Handles documents up to ~75K tokens
- 32GB+ RAM: Required for utilizing full 128K context window

### Installing Dependencies

```bash
# Required Python dependencies
pip3 install PyQt5 requests python-dotenv llama-cpp-python

# Optional dependencies for enhanced functionality (document parsing, token counting)
# pip3 install tiktoken PyPDF2 python-docx
```
*(Note: `llama-cpp-python` is now a core dependency)*

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/LlamaCagUI.git
   cd LlamaCagUI
   ```

2. **Set up the environment**:
   ```bash
   # Make run script executable
   chmod +x run.sh

   # Install requirements (includes llama-cpp-python and builds llama.cpp)
   # Note: setup_requirements.sh will attempt to install Homebrew if not found
   # You may be prompted for your password during installation
   ./setup_requirements.sh
   ```

3. **Run the application**:
   ```bash
   ./run.sh
   ```

### Verifying Installation

After installation, verify that everything is working correctly:

1. **Run the diagnostics script**:
   ```bash
   ./diagnose.sh
   ```
   This will check for all dependencies and required directories.

2. **Check llama.cpp installation**:
   ```bash
   # Verify llama.cpp is built correctly
   ls ~/Documents/llama.cpp/build/bin
   ```
   You should see `main` or `llama-cli` executable.

3. **Test application launch**:
   Run `./run.sh` and confirm the application opens without errors.

## Directory Structure

LlamaCag UI creates and uses the following directories:

- **~/.llamacag/**: Configuration directory
  - **logs/**: Log files for troubleshooting
  - **config.json**: User configuration
  - **custom_models.json**: Custom model definitions

- **~/Documents/llama.cpp/**: llama.cpp installation
  - **models/**: Downloaded model files (.gguf)

- **~/cag_project/**: Working directories
  - **kv_caches/**: Document caches
  - **temp_chunks/**: Temporary files used during processing

## Usage Guide

### First-Time Setup

1. **Install llama.cpp**: If not already installed, the app will prompt you to install it
2. **Download a model**: Go to the Models tab and download one of the provided models (Gemma 3 4B Instruct recommended)

### Processing Documents (Creating a KV Cache)

1.  Go to the **Documents** tab.
2.  Click **Select File** to choose a document (`.txt`, `.md`, etc.).
3.  The app estimates the token count and indicates if it fits the current model's context window.
4.  Click **Create KV Cache**. This uses `llama.cpp` via `llama-cpp-python` to process the document's tokens and saves the resulting model state (the KV cache) to a `.llama_cache` file. This step can take time depending on document size and hardware.
5.  Optionally check "Set as Master KV Cache" to make this the default cache for new chats.

### Chatting with Documents (Using a KV Cache)

1.  Go to the **Chat** tab.
2.  Ensure **Use KV Cache** is checked. The currently selected or master KV cache will be loaded.
3.  Type your question about the document in the input field.
4.  Click **Send**. The application loads the KV cache and processes *only your query*, resulting in a fast response that leverages the document's context.
5.  Continue the conversation with follow-up questions, which remain fast as the document context is already loaded via the cache.
6.  Adjust temperature and max tokens settings as needed.

### Managing KV Caches

1. Go to the **KV Cache Monitor** tab to view and manage your document caches
2. Select a cache and click **Use Selected** to switch to a different document for your current chat
3. Click **Purge Selected** to delete a cache you no longer need
4. Click **Purge All** to remove all caches and start fresh
5. Use **Refresh** to update the cache list after external changes

## File Management

### Document KV Caches

KV caches are stored in `~/cag_project/kv_caches/` (configurable via Settings). Each cache consists of:

-   A `.llama_cache` file: This binary file contains the internal state (Key/Value cache) of the `llama.cpp` model after processing the document.
-   Registry entries in `cache_registry.json` and `usage_registry.json`.

### Cache Registry Files

Two registry files track cache information within the `kv_caches` directory:

-   `cache_registry.json`: Stores metadata about each cache file (original document, model used, context size, token count, creation time).
-   `usage_registry.json`: Tracks usage statistics (last used time, usage count).

### Cache Management Operations

-   **Viewing Caches**: The KV Cache Monitor tab lists all detected `.llama_cache` files and their metadata from the registry. Use **Refresh** to rescan the directory.
-   **Selecting a Cache**: Use the **Use Selected** button in the monitor tab to load a specific cache for the current chat session.
-   **Deleting Caches**: **Purge Selected** removes a single cache file and its registry entries. **Purge All** removes all `.llama_cache` files and clears the registries.
-   **Setting a Master Cache**: In the Documents tab, check "Set as Master KV Cache" when processing a document to make it the default cache loaded when "Use KV Cache" is enabled in the Chat tab. The master cache file is named `master_cache.llama_cache`.

### Temporary Files

Temporary files created during processing are stored in `~/cag_project/temp_chunks/` and can be safely deleted if you need to free up space.

## Model Management

### Downloading Models

1. Go to the **Models** tab
2. Click **Download Model** to see available models
3. Select a model and click **Download Selected Model**
4. Wait for the download to complete

### Model Recommendations

- **Gemma 3 4B Instruct (Q4_K_M)**: Best balance of performance and memory usage (recommended)
- **Llama 3 8B Instruct**: Higher quality responses for more complex documents
- **Mistral 7B Instruct**: Good alternative with strong reasoning capabilities

### Manual Download

If the automatic download fails:
1. Go to the Models tab and click **Manual Download Info**
2. Follow the instructions to download and place the model files manually
3. Click **Refresh** to detect the manually downloaded models

## Configuration

### Advanced Configuration

LlamaCag UI stores its configuration in:

1.  **User Config**: `~/.llamacag/config.json` - Contains user-specific settings modified via the UI (paths, model parameters).
2.  **Environment Variables**: `.env` file in the application's root directory - Can be used to override defaults or provide settings not in the UI (e.g., `LLAMACPP_GPU_LAYERS`).
3.  **Cache Registries**: `~/cag_project/kv_caches/cache_registry.json` and `usage_registry.json` - Managed automatically by the application.

To modify settings not available in the UI (like GPU layers):

1. Close the application
2. Edit `~/.llamacag/config.json`
3. Restart the application

### Paths

All paths can be configured in the Settings tab:
- **llama.cpp Path**: Location of the llama.cpp installation
- **Models Path**: Directory where models are stored
- **KV Cache Path**: Directory where document caches are stored
- **Temp Path**: Directory for temporary files during processing

### Model Parameters

Configurable in the Settings tab or via `.env`:

-   **Threads**: Number of CPU threads for inference (default: system CPU count).
-   **Batch Size**: Batch size for prompt processing (default: 512).
-   **GPU Layers**: Number of model layers to offload to GPU (set via `LLAMACPP_GPU_LAYERS` in `.env`, requires `llama-cpp-python` built with GPU support). Default is 0 (CPU only).

### n8n Integration

LlamaCag UI includes optional integration with n8n for workflow automation:
- Configure n8n host and port in the Settings tab
- Use the start/stop controls to manage the n8n service
- This feature is optional and not required for core functionality

## Troubleshooting

### Common Issues

#### "No output received from model"

**Cause**: The model might be entering interactive mode or failing to generate output.

**Solution**: 
1. Check the debug logs in ~/.llamacag/logs/
2. Try a shorter document or smaller model
3. Ensure you have sufficient memory (16GB+ recommended)

#### "Model not found"

**Cause**: The model file path is incorrect or the model hasn't been downloaded.

**Solution**:
1. Go to the Models tab and download the model
2. Or manually download and place the model in the ~/Documents/llama.cpp/models/ directory

#### "KV cache not found" or "Invalid KV Cache"

**Cause**: The document hasn't been processed, the `.llama_cache` file is missing, corrupted, or incompatible with the current model/settings.

**Solution**:
1.  Process the document again using the **Documents** tab with the desired model selected.
2.  Check if the `.llama_cache` file exists in `~/cag_project/kv_caches/` (or your configured path).
3.  Ensure the model selected in the **Models** tab is the same one used to create the cache.
4.  Try purging the cache via the **KV Cache Monitor** and re-processing the document.

### Reset and Diagnostics

If you encounter persistent issues:

```bash
# Run diagnostics
./diagnose.sh

# Reset settings
./reset.sh

# For a complete reset, you can also run:
./cleanup_and_fix.sh
```

### Debug Logs

Log files are stored in `~/.llamacag/logs/` with timestamps. When troubleshooting, check the most recent log file for detailed error information.

If reporting issues, please include the relevant log files.

## Technical Details

### How Context-Augmented Generation Works in LlamaCag (New Architecture)

LlamaCag UI now leverages `llama-cpp-python` for true KV caching:

1.  **Cache Creation**:
    *   When you process a document, the application loads the selected language model.
    *   The document text is tokenized.
    *   The model processes these tokens (`llm.eval(tokens)`), populating its internal Key/Value state.
    *   This internal state is saved to disk as a `.llama_cache` file (`llm.save_state(...)`).
2.  **Chatting with Cache**:
    *   When you start a chat with "Use KV Cache" enabled, the application loads the model.
    *   It then loads the pre-computed state from the selected `.llama_cache` file (`llm.load_state(...)`).
    *   Your query is tokenized and processed (`llm.eval(query_tokens)` or `llm.create_completion(...)`). Since the document context is already in the model's state via the cache, only the query needs processing, making responses much faster.

### Document Processing (New Architecture)

When processing a document via the **Documents** tab:

1.  The document is read.
2.  The selected model is loaded via `llama-cpp-python`.
3.  The document content is tokenized using the model's tokenizer.
4.  If the token count exceeds the model's context window, the token list is truncated.
5.  The model evaluates the document tokens to build its internal state.
6.  The model's state is saved to a `.llama_cache` file.
7.  Metadata is recorded in `cache_registry.json`.

### Under the Hood (New Architecture)

-   The application uses the `llama-cpp-python` library for direct interaction with `llama.cpp`.
-   Core logic for cache creation and chat inference resides in `core/document_processor.py` and `core/chat_engine.py`.
-   External bash scripts are no longer used.
-   PyQt5 provides the graphical interface.

## Known Limitations

- **Document Size**: Documents larger than the model's context window will be truncated
- **File Types**: Best support for plain text (.txt) and markdown (.md) files
- **Memory Usage**: Large models and documents require significant RAM
- **Performance**: Initial KV cache creation can be slow for large documents. Chat responses using the cache are significantly faster. Performance depends on CPU/GPU capabilities.
- **GPU Support**: GPU acceleration can be enabled by setting `LLAMACPP_GPU_LAYERS` in the `.env` file (e.g., `LLAMACPP_GPU_LAYERS=30`). This requires `llama-cpp-python` to be installed with the correct GPU support (e.g., Metal for macOS, CUDA for Nvidia). The `setup_requirements.sh` script performs a standard build, which may not include GPU support by default. Manual installation of `llama-cpp-python` with appropriate flags might be needed for optimal GPU usage.
- **Cache Compatibility**: KV caches are generally specific to the model file they were created with. Using a cache created with a different model may lead to errors or unexpected behavior. Context size mismatches might also cause issues.
- **Multiple Documents**: Currently limited to one document context per conversation (via a single KV cache).

## Future Improvements

- [ ] Advanced document processing with chunking for very large documents
- [ ] Multiple document support for combining context from several sources
- [ ] PDF and Word document parsing improvements
- [ ] Custom prompt templates for different use cases
- [ ] Web UI version for remote access
- [ ] Vector database integration for hybrid RAG+CAG approaches
- [ ] Cache organization with folders and tagging
- [ ] Batch processing of document directories
- [ ] GPU layer configuration through the UI Settings tab.
- [ ] Verify KV cache compatibility with the selected model before loading.
- [ ] Export and import of conversations with context.
- [ ] Improved document content visualization.

## License and Credits


The application uses several open-source components:
-   `llama-cpp-python` library and the underlying `llama.cpp` by ggerganov and contributors.
-   PyQt5 for the UI framework.
-   Various language models (Gemma, Llama, Mistral) from their respective creators.

## Feedback and Contributions

Feedback and contributions are welcome! Please submit issues and pull requests on GitHub.

---

*LlamaCag UI: Your documents, augmented by AI.*
