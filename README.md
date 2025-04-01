### Screenshots
![Welcome Screen](https://github.com/AbelCoplet/CAGtrueKV/blob/main/images/Welcome.png)  
![Document Processor](https://github.com/AbelCoplet/CAGtrueKV/blob/main/images/Document.png)  
![Main Chat](https://github.com/AbelCoplet/CAGtrueKV/blob/main/images/Chat.png)  
![Model Select](https://github.com/AbelCoplet/CAGtrueKV/blob/main/images/KV%20Cache.png)
# LlamaCag UI

## Context-Augmented Generation for Large Language Models

LlamaCag UI is a desktop application that enables context-augmented generation (CAG) with large language models. It allows you to feed entire documents into a language model's context window and ask questions that leverage that full context, creating an experience similar to chatting with your documents with unprecedented accuracy.

```markdown
LlamaCagUI/
├── main.py                  # Application entry point, initializes all components
├── run.sh                   # Script to run the application with correct environment
├── setup_requirements.sh    # Installs dependencies, llama.cpp, and creates directories
├── README.md                # Project documentation (to be updated)
├── model_urls.txt           # List of model download URLs
├── .gitattributes           # Git attributes configuration
├── .gitignore               # Files to ignore in Git repository
│
├── core/                    # Core functionality components
│   ├── __init__.py          # Package initialization
│   ├── cache_manager.py     # Manages KV caches, listing, purging and registry
│   ├── chat_engine.py       # Handles chat interaction with models using KV caches
│   ├── document_processor.py # Processes documents into KV caches, estimates tokens
│   ├── llama_manager.py     # Manages llama.cpp installation and updates
│   ├── model_manager.py     # Handles model downloading, importing, and selection
│   └── n8n_interface.py     # Interface for n8n workflow integration
│
├── ui/                      # User interface components
│   ├── __init__.py          # Package initialization
│   ├── main_window.py       # Main application window with tabbed interface
│   ├── model_tab.py         # UI for model management and downloading
│   ├── document_tab.py      # UI for document processing and cache creation
│   ├── chat_tab.py          # UI for chatting with documents
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
│
├── metal/                   # Metal shader files for GPU acceleration (macOS)
│   ├── ggml-metal.metal     # Metal Shading Language source code
│   └── metal_kernels.metallib # Compiled Metal library
│
└── ADDITIONALREADME/        # Additional documentation files
    ├── data_preparation_guide.md # CRITICAL guide for preparing input data
    ├── llamacag-kv-cache-guide.md # Detailed explanation of KV cache modes
    ├── n8n-integration-clarification.md # Notes on N8N integration
    ├── READMECHANGES.md     # Log of README changes
    └── structure.md         # Detailed project structure analysis (this file)

```
**Note:** The `fixes/` directory contains temporary debugging artifacts and is not part of the core application.

For a more detailed breakdown of the project structure and component interactions, please refer to the [Structure Analysis](ADDITIONALREADME/structure.md).

## 📋 Table of Contents

- [Core Concept](#core-concept)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [License and Credits](#license-and-credits)

##  Core Concept

The fundamental idea behind LlamaCag UI is **Context-Augmented Generation (CAG)**, leveraging the power of `llama.cpp`'s KV (Key/Value) caching mechanism. Unlike standard RAG (Retrieval-Augmented Generation) systems that retrieve snippets of text, CAG:

1. **Processes the entire document** through the language model once to generate its internal state (the KV cache)
2. **Saves this KV cache** to disk
3. **Loads the saved KV cache** for subsequent interactions, allowing the model to "remember" the document context without re-processing the full text
4. **Enables deep contextual understanding** by having the model's state primed with the document content
5. **Allows fast follow-up questions** as only the new query needs to be processed by the model

This approach allows models like Gemma 3 and Llama 3 to efficiently utilize their large context windows (e.g., 128K tokens) for in-depth document analysis and question answering, significantly speeding up conversations after the initial document processing.

## ✨ Key Features

- **Model Management**: Download, manage, and select from various large context window models (GGUF format)
- **Document Processing**: Load documents and process them into true `llama.cpp` KV caches for efficient context augmentation
- **Interactive Chat**: Chat with your documents, leveraging the pre-processed KV cache for fast responses
- **KV Cache Monitor**: Track and manage your document KV caches
- **Cache Warm-up**: Pre-load model and cache state into memory for near-instantaneous responses
- **GPU Acceleration**: Configure GPU offloading for significantly improved performance (especially on Apple Silicon)
- **Settings**: Configure paths, model parameters (threads, batch size, GPU layers), and application behavior
- **Data Preparation Focused**: Designed for optimal performance with pre-processed, structured data (See [Data Preparation](#crucial-data-preparation) below).

## ❗ Crucial: Data Preparation

LlamaCag's accuracy and reliability heavily depend on the quality and structure of the input document used to create the KV cache. While it can process generic text, its true potential for precise data retrieval is unlocked with **optimally pre-processed data**.

**Why is this critical?**
- **Context Understanding:** LLMs work best with clear structure (headings, lists, consistent formatting). Poorly formatted or unstructured text can confuse the model, leading to inaccurate answers or incorrect recitation starting points (e.g., mistaking a prologue for the main content).
- **Token Efficiency:** Removing redundant whitespace and using efficient formatting (like Markdown) maximizes the useful information within the model's context window.
- **Avoiding Artifacts:** Ambiguous formatting, hidden characters, or inconsistent structure in the source document can manifest as errors or unexpected behavior during Q&A or recitation.

**How to Prepare Data:**
Please refer to the **detailed [Data Preparation Guide](ADDITIONALREADME/data_preparation_guide.md)** for best practices on converting various data types (text, numbers, tables, technical specs) into an LLM-friendly format using Markdown. Following this guide is essential for achieving high-fidelity results, especially for business data retrieval.

## 📷 Screenshots

![Model Management](https://via.placeholder.com/800x450?text=Model+Management)
![Document Processing](https://via.placeholder.com/800x450?text=Document+Processing) 
![Chat Interface](https://via.placeholder.com/800x450?text=Chat+Interface)
![KV Cache Monitor](https://via.placeholder.com/800x450?text=KV+Cache+Monitor)
![Settings](https://via.placeholder.com/800x450?text=Settings)

##  Installation

### Prerequisites

- **Operating System**: macOS, Linux, or Windows with Python 3.8+
- **Memory Requirements**:
  - **Minimum**: 8GB RAM (limited to ~25K token documents)
  - **Recommended**: 16GB RAM (handles documents up to ~75K tokens)
  - **Optimal**: 32GB+ RAM (required for full 128K context window utilization)
- **Disk Space**: At least 10GB for the application, llama.cpp, and models
- **Internet Connection**: Required for downloading models

> **Note for Apple Silicon (M1/M2/M3/M4) Users:** LlamaCag UI supports GPU acceleration via Metal, which significantly improves performance. See the Performance Optimization section after installation.

### Quick Installation

The easiest way to install LlamaCag UI is using the provided setup script:

```bash
# Clone the repository
git clone https://github.com/AbelCoplet/LlamaCagUI.git
cd LlamaCagUI

# Run the setup script
chmod +x setup_requirements.sh
./setup_requirements.sh

# Run the application
./run.sh
```

The setup script will:
1. Install Homebrew (if not already installed)
2. Install required system dependencies (git, cmake, make, python3, pyqt@5)
3. Install Python packages (PyQt5, requests, python-dotenv, llama-cpp-python)
4. Clone and build llama.cpp
5. Create necessary directories for the application

### Manual Installation

If you prefer to install manually or need more control over the installation process:

#### 1. Install System Dependencies

**macOS**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install git cmake make python3 pyqt@5
```

**Linux (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install git cmake make python3 python3-pip python3-pyqt5
```

**Windows**
- Install [Git for Windows](https://gitforwindows.org/)
- Install [CMake](https://cmake.org/download/)
- Install [Python 3.8+](https://www.python.org/downloads/windows/)
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

#### 2. Install Python Dependencies

```bash
pip install PyQt5 requests python-dotenv llama-cpp-python
```

**Optional Dependencies**
```bash
# For better token estimation and document handling
pip install tiktoken PyPDF2 python-docx
```

#### 3. Set Up llama.cpp

```bash
# Create directory and clone repository
mkdir -p ~/Documents/llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git ~/Documents/llama.cpp

# Build llama.cpp
cd ~/Documents/llama.cpp
mkdir -p build
cd build
cmake ..
cmake --build . -j $(nproc)  # Linux/macOS
# For Windows: cmake --build . --config Release

# Create models directory
mkdir -p ~/Documents/llama.cpp/models
```

#### 4. Create Necessary Directories

```bash
mkdir -p ~/.llamacag/logs
mkdir -p ~/cag_project/kv_caches
mkdir -p ~/cag_project/temp_chunks
```

#### 5. Clone LlamaCag UI Repository

```bash
git clone https://github.com/AbelCoplet/LlamaCagUI.git
cd LlamaCagUI
```

#### 6. Run the Application

```bash
./run.sh  # Linux/macOS
# For Windows: python main.py
```

### Recommended Model: `google/gemma-3-4b-it-Q4_1`

While LlamaCag supports various GGUF models, extensive testing and development for the core CAG features (especially KV caching and state management) have been performed using **`google/gemma-3-4b-it-Q4_1.gguf`**.

**Why this model is recommended:**
- **Tested:** It's the primary model used during development and debugging of the KV cache and chat engine logic.
- **Mac Optimization:** Gemma models often perform well with Metal GPU acceleration on Apple Silicon. The Q4_1 quantization offers a good balance of performance and quality for this size.
- **Context Size:** While labeled 4B, this specific Gemma variant often handles larger contexts effectively within reasonable memory constraints compared to some other models of similar parameter counts, making it suitable for the CAG approach.
- **Instruction Following:** As an instruct-tuned model, it generally follows the specific prompts used by LlamaCag for Q&A and recitation reasonably well (though prompt adherence can still vary).

Using other models might work, but could lead to different performance characteristics or unexpected behavior with KV cache loading, state management, or prompt adherence due to variations in model architecture and training. Sticking to the recommended model ensures the highest likelihood of compatibility with the current implementation. You can download it via the "Download Model" button in the Models tab.

### Performance Optimization

For optimal performance, especially with large documents:

#### Apple Silicon (M1/M2/M3/M4) GPU Acceleration

1. Go to the **Settings** tab after installation
2. Set **GPU Layers** based on your system:
   - For 4B models: Start with 15-20 layers
   - For 8B models: Start with 10-15 layers
   - Adjust based on performance and available memory
3. Set **Threads** to match your CPU's performance core count
4. Click **Save Settings**

## 📝 Usage Guide

### First-Time Setup

1. **Start the application**:
   ```bash
   ./run.sh
   ```

2. **Download a model**: 
   - Go to the Models tab
   - Click "Download Model"
   - Select a model (Gemma 3 4B Instruct recommended for beginners)
   - Wait for the download to complete

3. **Verify installation**:
   - Check that the model appears in the Models tab
   - Select the model by clicking on it

### Processing Documents (Creating a KV Cache)

1. Go to the **Documents** tab
2. Click **Select File** to choose a document (`.txt`, `.md`, etc.)
3. The app estimates the token count and indicates if it fits the current model's context window
4. Click **Create KV Cache**. This processes the document's tokens and saves the resulting model state (the KV cache) to a `.llama_cache` file
5. Optionally check "Set as Master KV Cache" to make this the default cache for new chats

### Chatting with Documents (Using a KV Cache)

1. Go to the **Chat** tab
2. Ensure **Use KV Cache** is checked. The currently selected or master KV cache will be loaded
3. Type your question about the document in the input field
4. Click **Send**. The application loads the KV cache and processes *only your query*, resulting in a fast response that leverages the document's context
5. Continue the conversation with follow-up questions.
6. **Warm Up Cache**: For the fastest responses during a session, click **Warm Up Cache** after selecting a cache. This pre-loads the model and cache into memory.
7. **Cache Behavior (When Warmed Up)**: Select the desired mode using the radio buttons:
    *   **Standard (State Persists)**: Fastest for conversation. Cache state evolves with chat.
    *   **Fresh Context (Reload Before Query)**: Guarantees statelessness. Reloads clean cache *before* each query. Ideal for testing/automation.
    *   **Fresh Context (Reload After Query)**: Experimental. Reloads clean cache *after* each query.

### Managing KV Caches

1. Go to the **KV Cache Monitor** tab to view and manage your document caches
2. Select a cache and click **Use Selected** to switch to a different document for your current chat
3. Click **Purge Selected** to delete a cache you no longer need
4. Click **Purge All** to remove all caches and start fresh
5. Use **Refresh** to update the cache list after external changes

##  Technical Details

### How Context-Augmented Generation Works

LlamaCag UI uses `llama-cpp-python` for true KV caching:

1. **Cache Creation**:
   - When you process a document, the application loads the selected language model
   - The document text is tokenized
   - The model processes these tokens (`llm.eval(tokens)`), populating its internal Key/Value state
   - This internal state is saved to disk as a `.llama_cache` file (`llm.save_state(...)`)

2. **Chatting with Cache**:
   - When you start a chat with "Use KV Cache" enabled, the application loads the model
   - It then loads the pre-computed state from the selected `.llama_cache` file (`llm.load_state(...)`)
   - Your query is tokenized and processed (`llm.eval(query_tokens)` or `llm.create_completion(...)`). Since the document context is already in the model's state via the cache, only the query needs processing, making responses much faster

3. **Warm-Up Mode & Cache Behavior**:
   - When you click "Warm Up Cache", the model and cache are loaded into memory and kept there.
   - The selected **Cache Behavior** mode determines how the state is handled for subsequent queries:
     - **Standard**: State evolves with the conversation.
     - **Fresh Context (Reload Before Query)**: Original state is reloaded from disk *before* processing the query, ensuring a clean slate for the response generation.
     - **Fresh Context (Reload After Query)**: Query is processed using the current state, and the original state is reloaded *after* the response is generated, preparing for the next query.

### Directory Structure

LlamaCag UI creates and uses the following directories:

- **~/.llamacag/**: Configuration directory
  - **logs/**: Log files for troubleshooting
  - **config.json**: User configuration

- **~/Documents/llama.cpp/**: llama.cpp installation
  - **models/**: Downloaded model files (.gguf)

- **~/cag_project/**: Working directories
  - **kv_caches/**: Document caches
  - **temp_chunks/**: Temporary files used during processing

## 🛠️ Troubleshooting

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

**Cause**: The document hasn't been processed, the `.llama_cache` file is missing, or it's incompatible with the current model/settings.

**Solution**:
1. Process the document again using the **Documents** tab with the desired model selected
2. Check if the `.llama_cache` file exists in `~/cag_project/kv_caches/` (or your configured path)
3. Ensure the model selected in the **Models** tab is the same one used to create the cache
4. Try purging the cache via the **KV Cache Monitor** and re-processing the document

#### Slow document processing

**Cause**: Processing large documents requires significant computational resources, especially without GPU acceleration.

**Solution**:
1. Enable GPU acceleration in Settings for Apple Silicon (set GPU Layers to 15-20 for 4B models)
2. Process smaller documents first, especially when testing
3. Be patient - initial processing of a 128K token document can take time, but subsequent chat interactions will be fast

### Reset and Diagnostics

If you encounter persistent issues:

```bash
# Create a fresh configuration
rm -rf ~/.llamacag
rm -rf ~/cag_project/kv_caches
rm -rf ~/cag_project/temp_chunks
mkdir -p ~/.llamacag/logs
mkdir -p ~/cag_project/kv_caches
mkdir -p ~/cag_project/temp_chunks
```

### Debug Logs

Log files are stored in `~/.llamacag/logs/` with timestamps. When troubleshooting, check the most recent log file for detailed error information.

## ❓ FAQ

### Q: What types of documents work best with LlamaCag?
**A:** Text-based documents like `.txt`, `.md`, technical documentation, manuals, research papers, and books work best. The application excels with structured, information-rich content where context is important.

### Q: How large a document can I process?
**A:** This depends on your model's context window and available RAM. With 16GB RAM and Gemma 3 4B, you can typically process documents up to ~75K tokens. With 32GB+ RAM, you can utilize the full 128K context window. Documents that exceed the context window will be truncated.

### Q: Can I use different models with the same KV cache?
**A:** No, KV caches are specific to the model they were created with. If you switch models, you'll need to reprocess your documents to create new caches.

### Q: Does this work on Windows/Linux?
**A:** Yes, though the application was primarily tested on macOS. The core functionality should work on Windows and Linux as long as Python and the required dependencies are installed.

### Q: Does it support GPU acceleration?
**A:** Yes, GPU acceleration can be enabled by setting the GPU Layers parameter in the Settings tab. This requires `llama-cpp-python` to be installed with the correct GPU support (e.g., Metal for macOS, CUDA for Nvidia).

##  Known Limitations

- **Document Size**: Documents larger than the model's context window will be truncated
- **File Types**: Best support for plain text (.txt) and markdown (.md) files
- **Memory Usage**: Large models and documents require significant RAM
- **Performance**: Initial KV cache creation can be slow for large documents. Chat responses using the cache are significantly faster. Performance depends on CPU/GPU capabilities.
- **Cache Compatibility**: KV caches are specific to the exact model file (`.gguf`) they were created with. Using a cache created with a different model *will* lead to errors or unpredictable behavior. Always use the same model for creating and querying a cache.
- **Recitation Limitations**: Precise recitation of specific sections (e.g., "the third paragraph") can be unreliable, as the model may struggle to perfectly identify semantic boundaries within the raw cached text. Full document recitation (starting from the absolute beginning) is more consistent.
- **Multiple Documents**: Currently limited to one document context per conversation (via a single KV cache).

## 🔮 Future Improvements

- Advanced document processing with chunking for very large documents
- Multiple document support for combining context from several sources
- PDF and Word document parsing improvements
- Custom prompt templates for different use cases
- Web UI version for remote access
- Vector database integration for hybrid RAG+CAG approaches
- Cache organization with folders and tagging
- Batch processing of document directories
- GPU layer configuration through the UI Settings tab

# Comparison: True KV Cache vs. Manual Context Prepending

Current Implementation

## 1. True KV Cache Method

```python
# In chat_engine.py
def _inference_thread_with_true_kv_cache(self, message: str, model_path: str, context_window: int,
                     kv_cache_path: Optional[str], max_tokens: int, temperature: float):
    # Load model state
    with open(kv_cache_path, 'rb') as f_pickle:
        state_data = pickle.load(f_pickle)
    llm.load_state(state_data)
    
    # Tokenize and evaluate user input
    llm.eval(input_tokens)
    
    # Generate response using loaded state
    # [generation code...]
```

This method:
- Saves the model's internal state after processing a document
- Loads this internal state directly when querying
- Preserves the full token-level representation of the document
- Optimizes for multiple queries using the same context

## 2. Manual Context Prepending (Fallback)

```python
# In chat_engine.py
def _inference_thread_fallback(self, message: str, model_path: str, context_window: int,
                    kv_cache_path: Optional[str], max_tokens: int, temperature: float, llm: Optional[Llama] = None):
    # Find original document associated with cache
    if kv_cache_path:
        cache_info = self.cache_manager.get_cache_info(kv_cache_path)
        if cache_info and 'original_document' in cache_info:
            original_doc_path_str = cache_info['original_document']
            with open(original_doc_path, 'r', encoding='utf-8', errors='replace') as f_doc:
                doc_context_text = f_doc.read(8000)
                
    # Insert document text into system prompt
    system_prompt_content = (
         f"Use the following text to answer the user's question:\n"
         f"--- TEXT START ---\n"
         f"{doc_context_text}...\n"
         f"--- TEXT END ---\n\n"
         f"Answer based *only* on the text provided above."
    )
    
    # Create chat completion with this augmented prompt
    # [generation code...]
```

This method:
- Reads the beginning of the original document (up to 8000 characters)
- Inserts this text directly into the system prompt
- Reprocesses the context with every query
- Simpler implementation with fewer dependencies on model internals

## Comparative Analysis

### Performance Comparison in CURRENT config

| Aspect | True KV Cache | Manual Context Prepending |
|--------|--------------|--------------------------|
| Setup cost | High (full document processing) | Low (file reading only) |
| Query latency | Lower (reuses processed state) | Higher (reprocesses context every time) |
| Multi-turn efficiency | Excellent (state persists) | Poor (repeats context processing) |
| Memory usage | Higher (stores full KV state) | Lower (only stores text) |
| Context capacity | Full context window | Limited to ~8000 chars (~2000 tokens) |

### Use Case Suitability

For one-shot queries, the manual context prepending offers a reasonable trade-off. It's:

1. **Simpler to implement**: No need for complex state management
2. **More robust**: Less dependent on specific model versions or implementation details
3. **Sufficient for basic needs**: For single questions about a document, prepending works well

For multi-turn conversations or very large documents, the true KV cache method would be significantly more efficient.

## Implementation as an Optional Feature

To make both methods available as options:

```python
# In settings_tab.py
def setup_ui(self):
    # Existing UI elements...
    
    # Add context method selection
    self.context_method_group = QGroupBox("Context Method")
    context_layout = QVBoxLayout(self.context_method_group)
    
    self.true_kv_radio = QRadioButton("True KV Cache (Faster for multiple queries)")
    self.manual_context_radio = QRadioButton("Manual Context Prepending (Simpler, good for one-shot queries)")
    
    if self.config.get('USE_TRUE_KV_CACHE', True):
        self.true_kv_radio.setChecked(True)
    else:
        self.manual_context_radio.setChecked(True)
    
    context_layout.addWidget(self.true_kv_radio)
    context_layout.addWidget(self.manual_context_radio)
    
    model_layout.addRow(self.context_method_group)
```

```python
# Add to save_settings method
def save_settings(self):
    # Existing code...
    
    # Save context method
    self.config['USE_TRUE_KV_CACHE'] = self.true_kv_radio.isChecked()
```

This would allow users to explicitly choose between methods based on their use case.

## Conclusion

Both approaches have their place, and having them both available gives users flexibility:

- **True KV Cache**: For power users, multiple queries, large documents
- **Manual Context Prepending**: For simpler use cases, one-shot queries, or situations where true KV caching has compatibility issues

The current implementation cleverly falls back to manual context prepending when true KV caching isn't available, but making it an explicit choice would give users more control over the performance/simplicity trade-off.


##  License and Credits

### Components and Libraries
The application uses several open-source components:
- `llama-cpp-python` library and the underlying `llama.cpp` by ggerganov and contributors
- PyQt5 for the UI framework
- Various language models (Gemma, Llama, Mistral) from their respective creators

### License
[Your license information here]

---

*LlamaCag UI: Your documents, augmented by AI.*
