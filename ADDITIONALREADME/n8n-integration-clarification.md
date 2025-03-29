# LlamaCag UI - N8N Integration: Clarified Architecture

## Understanding API Operation Flow

Based on your concerns, I'll clarify exactly how the API would interact with your application and propose solutions to avoid conflicts between the API and UI.

### Relationship to Existing Implementation

The API server would use **the exact same core components** that your UI already uses:

1. **Document Processor**: The same code that processes documents through the UI
2. **Cache Manager**: The same code that manages KV caches
3. **Chat Engine**: The same engine that handles inference with warm-up capability

This ensures complete consistency between API and UI operations.

### Document Processing API Flow

When N8N calls the document processing endpoint:

1. **Document Upload**: The API receives the document and saves it temporarily
2. **Document Preprocessing**: The document is preprocessed according to your guidelines
3. **KV Cache Creation**: The API calls `document_processor.process_document()` - the SAME method your UI uses
4. **Master Cache Setting**: If requested, the new cache is set as master via the existing method
5. **Cache Warm-Up**: The API calls `chat_engine.warm_up_cache()` - the SAME method your UI uses

All of these actions trigger the SAME signals that update your UI elements. For example, when the document process completes, both your UI and the API receive the `processing_complete` signal.

### Inference API Flow

When N8N calls the inference endpoint:

1. **Cache Selection**: The API selects the requested KV cache (or uses master) via `chat_engine.set_kv_cache()`
2. **Cache Enabling**: Ensures KV cache usage is enabled via `chat_engine.toggle_kv_cache(true)`
3. **Message Sending**: Sends the query via `chat_engine.send_message()` - the SAME method your UI uses
4. **Response Monitoring**: Listens for the same signals your UI uses (`response_chunk`, `response_complete`)

Again, these actions trigger the SAME signals that update your UI elements.

## Avoiding UI-API Conflicts

There are two potential approaches to handle the relationship between UI and API:

### Approach 1: Integrated Mode (Visible in UI)

In this approach, all API actions are fully visible in the UI:

- Document processing shows progress in the UI Document tab
- Chat inference shows in the Chat tab
- All actions are logged in the UI

**Advantages**:
- Full visibility of automated actions
- No code duplication
- Simpler implementation

**Disadvantages**:
- User may be confused by "automatic" actions appearing in UI
- Potential conflicts if user is actively using the UI while API calls occur

### Approach 2: API Mode (Your Preferred Solution)

Based on your concerns, I recommend implementing an "API Mode" that runs the application with a minimal UI:

```bash
# Run in API mode
./run.sh --api-mode
```

In API mode:
1. The main UI is not shown; instead, a minimal status dashboard appears
2. The same core components (ChatEngine, DocumentProcessor, CacheManager) are used
3. The REST API server is automatically started
4. All operations are logged to the status dashboard

**Implementation in main.py**:

```python
def main():
    """Application entry point"""
    # Setup logging
    setup_logging()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="LlamaCag UI")
    parser.add_argument("--api-mode", action="store_true", help="Run in API mode (headless with REST server)")
    args = parser.parse_args()
    
    # Initialize application
    app = QApplication(sys.argv)
    app.setApplicationName("LlamaCagUI")
    app.setApplicationVersion(VERSION)

    # Load and apply stylesheet
    try:
        style_path = Path(__file__).parent / "ui" / "style.qss"
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())
        else:
            logging.warning(f"Stylesheet not found at {style_path}")
    except Exception as e:
        logging.error(f"Failed to load stylesheet: {str(e)}")
        
    # Load configuration
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        show_error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)
        
    # Initialize core components
    llama_manager = LlamaManager(config)
    model_manager = ModelManager(config)
    cache_manager = CacheManager(config)
    
    # Initialize dependent components
    document_processor = DocumentProcessor(config, llama_manager, model_manager, cache_manager)
    chat_engine = ChatEngine(config, llama_manager, model_manager, cache_manager)
    
    if args.api_mode:
        # Run in API mode
        from api_mode import APIModeDashboard, LlamaCagRestServer
        
        # Initialize REST server
        rest_server = LlamaCagRestServer(config, chat_engine, document_processor, cache_manager)
        
        # Start REST server
        host = config.get('REST_SERVER_HOST', '0.0.0.0')
        port = int(config.get('REST_SERVER_PORT', 8000))
        rest_server.start(host=host, port=port)
        
        # Show minimal dashboard
        dashboard = APIModeDashboard(config, rest_server, chat_engine, document_processor, cache_manager)
        dashboard.show()
        
        # Start the event loop
        sys.exit(app.exec_())
    else:
        # Run in normal UI mode
        # Check if llama.cpp is installed
        if not llama_manager.is_installed():
            response = QMessageBox.question(
                None,
                "LlamaCag UI - Setup",
                "llama.cpp is not installed. Would you like to install it now?",
                QMessageBox.Yes | QMessageBox.No
            )
            if response == QMessageBox.Yes:
                # Show installation dialog
                try:
                    llama_manager.install()
                except Exception as e:
                    logging.error(f"Installation failed: {str(e)}")
                    show_error(f"Installation failed: {str(e)}")
                    sys.exit(1)
            else:
                show_error("llama.cpp is required for this application to function.")
                sys.exit(1)
                
        # Initialize n8n interface
        n8n_interface = N8nInterface(config)
        
        # Create and show main window
        main_window = MainWindow(
            config_manager,
            llama_manager,
            model_manager,
            cache_manager,
            document_processor,
            chat_engine,
            n8n_interface
        )
        main_window.show()
        
        # Start the event loop
        sys.exit(app.exec_())
```

This allows you to run the application in two distinct modes:
1. Normal UI mode (default)
2. API mode (when --api-mode flag is used)

## The API Mode Dashboard

The API Mode dashboard would be a minimal UI that shows:

1. **Server Status**: Whether the REST server is running
2. **Activity Log**: Current and recent API activities
3. **Cache Status**: Current cache in use and warm-up state
4. **API Statistics**: Number of requests processed
5. **Controls**: Start/stop server, view logs, exit

```python
class APIModeDashboard(QMainWindow):
    """Minimal dashboard for API mode"""
    def __init__(self, config, rest_server, chat_engine, document_processor, cache_manager):
        super().__init__()
        self.config = config
        self.rest_server = rest_server
        self.chat_engine = chat_engine
        self.document_processor = document_processor
        self.cache_manager = cache_manager
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Set up the minimal dashboard UI"""
        self.setWindowTitle("LlamaCag UI - API Mode Dashboard")
        self.setMinimumSize(800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Server status
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.StyledPanel)
        status_layout = QHBoxLayout(status_frame)
        
        self.status_label = QLabel("REST Server: Running")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        self.host_port_label = QLabel(f"Host: {self.rest_server.host}, Port: {self.rest_server.port}")
        status_layout.addWidget(self.host_port_label)
        
        status_layout.addStretch()
        
        self.stop_button = QPushButton("Stop Server")
        self.stop_button.clicked.connect(self.stop_server)
        status_layout.addWidget(self.stop_button)
        
        layout.addWidget(status_frame)
        
        # Activity log
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group, 1)
        
        # Cache status
        cache_group = QGroupBox("Cache Status")
        cache_layout = QFormLayout(cache_group)
        
        self.current_cache_label = QLabel("None")
        cache_layout.addRow("Current Cache:", self.current_cache_label)
        
        self.warm_up_status_label = QLabel("Not Warmed Up")
        cache_layout.addRow("Warm-Up Status:", self.warm_up_status_label)
        
        layout.addWidget(cache_group)
        
        # API statistics
        stats_group = QGroupBox("API Statistics")
        stats_layout = QFormLayout(stats_group)
        
        self.doc_requests_label = QLabel("0")
        stats_layout.addRow("Document Requests:", self.doc_requests_label)
        
        self.inference_requests_label = QLabel("0")
        stats_layout.addRow("Inference Requests:", self.inference_requests_label)
        
        layout.addWidget(stats_group)
        
        # Status bar
        self.statusBar().showMessage("LlamaCag UI running in API mode")
        
    def connect_signals(self):
        """Connect signals to update dashboard"""
        # Document processor signals
        self.document_processor.processing_progress.connect(self.on_document_progress)
        self.document_processor.processing_complete.connect(self.on_document_complete)
        
        # Chat engine signals
        self.chat_engine.response_started.connect(self.on_response_started)
        self.chat_engine.response_complete.connect(self.on_response_complete)
        self.chat_engine.cache_status_changed.connect(self.on_cache_status_changed)
        
        # REST server signals (to be implemented)
        self.rest_server.request_received.connect(self.on_request_received)
        
    def log_activity(self, message):
        """Add message to activity log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def on_document_progress(self, document_id, progress):
        """Handle document processing progress"""
        self.log_activity(f"Document {document_id} processing: {progress}%")
        
    def on_document_complete(self, document_id, success, message):
        """Handle document processing completion"""
        if success:
            self.log_activity(f"Document {document_id} processed successfully")
            # Update document request count
            count = int(self.doc_requests_label.text())
            self.doc_requests_label.setText(str(count + 1))
        else:
            self.log_activity(f"Document {document_id} processing failed: {message}")
            
    def on_response_started(self):
        """Handle response generation start"""
        self.log_activity("Response generation started")
        
    def on_response_complete(self, response, success):
        """Handle response generation completion"""
        if success:
            self.log_activity(f"Response generation complete ({len(response)} chars)")
            # Update inference request count
            count = int(self.inference_requests_label.text())
            self.inference_requests_label.setText(str(count + 1))
        else:
            self.log_activity("Response generation failed")
            
    def on_cache_status_changed(self, status):
        """Handle cache status changes"""
        self.warm_up_status_label.setText(status)
        self.log_activity(f"Cache status changed: {status}")
        
    def on_request_received(self, endpoint, method):
        """Handle API request"""
        self.log_activity(f"API request: {method} {endpoint}")
        
    def stop_server(self):
        """Stop the REST server"""
        self.rest_server.stop()
        self.status_label.setText("REST Server: Stopped")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.stop_button.setEnabled(False)
        self.log_activity("REST server stopped")
```

## Changes to the N8N Interface

The N8N interface would be repurposed to also manage the REST server:

```python
class N8nInterface(QObject):
    """Interface for communicating with n8n and managing REST server"""
    # Signals
    status_changed = pyqtSignal(bool)  # is_running
    server_status_changed = pyqtSignal(bool)  # is_server_running
    
    def __init__(self, config, chat_engine=None, document_processor=None, cache_manager=None):
        """Initialize n8n interface"""
        super().__init__()
        self.config = config
        self.chat_engine = chat_engine
        self.document_processor = document_processor
        self.cache_manager = cache_manager
        
        self.n8n_url = f"{config.get('N8N_PROTOCOL', 'http')}://{config.get('N8N_HOST', 'localhost')}:{config.get('N8N_PORT', '5678')}"
        self.rest_server = None
        
        # Start status checking thread
        self._running = True
        self._status_thread = threading.Thread(target=self._check_status_thread, daemon=True)
        self._status_thread.start()
```

## Example N8N Workflow: Document Processing

Here's how a typical N8N workflow for document preprocessing would look:

1. **Trigger**: HTTP Request, File Upload, or Scheduled
2. **File Preparation**: Prepare document file (e.g., from Google Drive, local file)
3. **Document Analysis**: Optionally analyze document size, content, format
4. **HTTP Request**: Send to LlamaCag API
   - Endpoint: `POST /api/documents/process`
   - Parameters:
     - `file`: The document file
     - `set_as_master`: true
     - `preprocess`: true
     - `auto_warmup`: true
5. **Process Response**: Check for success, get cache_path
6. **Next Actions**: Trigger inference, notify user, etc.

## Example N8N Workflow: Document Inference

After document processing:

1. **Trigger**: HTTP Request, User Input, or After Document Processing
2. **Prepare Query**: Get user question or generate query
3. **HTTP Request**: Send to LlamaCag API
   - Endpoint: `POST /api/chat/inference`
   - Body:
     ```json
     {
       "query": "What does the document say about X?",
       "max_tokens": 1024,
       "temperature": 0.7,
       "cache_path": "/path/to/cache.llama_cache", // From previous step
       "use_kv_cache": true
     }
     ```
4. **Process Response**: Get response text
5. **Next Actions**: Format response, send to user, etc.

## Summary: Key Benefits of This Architecture

1. **Consistent Behavior**: Uses the same core components for both UI and API operations
2. **API Mode**: Optional dedicated mode prevents conflicts with interactive use
3. **Document Preprocessing**: Full support for the document trimming techniques
4. **Status Visibility**: Dashboard in API mode shows all current operations
5. **Extensibility**: Easy to add more endpoints or features in the future
