#!/usr/bin/env python3
"""
Welcome dialog for LlamaCag UI shown on first launch.
"""
import sys
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QCheckBox, QPushButton, QDialogButtonBox,
    QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QSettings

class WelcomeDialog(QDialog):
    """
    A dialog window shown on the first launch of the application
    to guide the user through the initial setup and core concepts.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("LlamaCag", "LlamaCagUI")
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface for the dialog."""
        self.setWindowTitle("Welcome to LlamaCag UI!")
        self.setMinimumSize(600, 500) # Adjusted size for content

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # --- Title ---
        title_label = QLabel("👋 Welcome to LlamaCag UI!")
        title_font = self.font()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # --- Introduction ---
        intro_text = """
        <p>Hi there! I'm LlamaCag UI, an application designed to let you 'chat' with your documents using Large Language Models (LLMs) with high accuracy.</p>
        <p><b>Key Concept: Strict Contextual Answering</b></p>
        <p>Unlike many AI tools, LlamaCag is specifically designed to force the LLM to answer questions based <i>only</i> on the content of the document you provide (using its KV Cache). It's artificially limited to prevent the model from using its general knowledge or making things up (hallucinating). This ensures the answers strictly reflect the document's information.</p>
        """
        intro_label = QLabel(intro_text)
        intro_label.setWordWrap(True)
        intro_label.setTextFormat(Qt.RichText)
        intro_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(intro_label)

        # --- Step-by-Step Guide ---
        guide_title = QLabel("🚀 Getting Started: First-Time Setup")
        guide_font = self.font()
        guide_font.setPointSize(13)
        guide_font.setBold(True)
        guide_title.setFont(guide_font)
        layout.addWidget(guide_title)

        guide_text = """
        <ol>
            <li><b>Download/Select a Model:</b> Go to the '<b>Models</b>' tab. You can download recommended models (like Gemma or Llama 3 in GGUF format) or import one you've downloaded manually into the designated folder (check 'Settings'). Select the model you want to use.</li>
            <li><b>Process Your Document:</b> Go to the '<b>Documents</b>' tab. Select a text-based file (<code>.txt</code> or <code>.md</code>). The app will estimate its size. Click '<b>Create KV Cache</b>'. This reads the document and saves the model's 'memory' of it.</li>
            <li><b>Load the Document Context:</b> When a document is processed, its KV Cache is created. To chat with a specific document, go to the '<b>KV Cache Monitor</b>' tab, select the cache corresponding to your document, and click '<b>Use Selected</b>'. Otherwise, the 'Master KV Cache' (if set during document processing) will be used by default.</li>
            <li><b>Start Chatting:</b> Go to the '<b>Chat</b>' tab. Ensure '<b>Use KV Cache</b>' is checked. Now you can ask questions! The model will answer based *only* on the document loaded in the selected KV Cache. <b>Important:</b> If no KV Cache is loaded, the chat won't work, as it requires document context.</li>
        </ol>
        """
        guide_label = QLabel(guide_text)
        guide_label.setWordWrap(True)
        guide_label.setTextFormat(Qt.RichText)
        guide_label.setAlignment(Qt.AlignLeft)
        # Allow links to be opened if any were added (none currently)
        guide_label.setOpenExternalLinks(True)
        layout.addWidget(guide_label)

        # --- Spacer ---
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # --- Don't Show Again Checkbox ---
        self.dont_show_checkbox = QCheckBox("Don't show this message again")
        layout.addWidget(self.dont_show_checkbox)

        # --- Buttons ---
        button_box = QDialogButtonBox()
        close_button = button_box.addButton("Close", QDialogButtonBox.AcceptRole)
        layout.addWidget(button_box)

        # Connect signals
        close_button.clicked.connect(self.accept) # Use accept to handle closing

    def accept(self):
        """Handle dialog acceptance (Close button clicked)."""
        if self.dont_show_checkbox.isChecked():
            self.settings.setValue("showWelcomeDialog", False)
        else:
            # Ensure the setting is True if the box is unchecked when closing
            self.settings.setValue("showWelcomeDialog", True)
        super().accept()

    @staticmethod
    def should_show(default=True):
        """Check QSettings to see if the dialog should be shown."""
        settings = QSettings("LlamaCag", "LlamaCagUI")
        return settings.value("showWelcomeDialog", defaultValue=default, type=bool)

# Example usage for testing (optional)
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    # To test persistence, uncomment the next line to reset the setting
    # QSettings("LlamaCag", "LlamaCagUI").setValue("showWelcomeDialog", True)
    if WelcomeDialog.should_show():
        dialog = WelcomeDialog()
        dialog.exec_() # Use exec_ for modal testing
    else:
        print("Welcome dialog is set to not show.")
    sys.exit()
