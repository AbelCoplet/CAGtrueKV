# LlamaCag Data Preparation Guide

## Objective

The primary strength of LlamaCag lies in its ability to provide high-fidelity, precise answers based on the **entire content** of a document stored in its KV cache. To maximize accuracy and retrieval capabilities, especially for technical specifications, financial data, procedures, or other structured business information, it is crucial to **pre-process and structure your source documents** optimally *before* creating the KV cache.

This guide outlines best practices for preparing your data for LlamaCag. While the system can handle generic text, following these guidelines will significantly improve performance and reliability for data-intensive tasks.

## Core Principles

1.  **Structure is Key:** Use clear, consistent structure that the LLM can easily parse. Markdown is highly recommended.
2.  **Explicitness:** Don't rely on the LLM to infer implicit relationships or data types. Label things clearly.
3.  **Consistency:** Use the same formatting and terminology throughout a document.
4.  **Token Efficiency:** Remove redundant characters and formatting, but prioritize clarity over extreme compression.
5.  **Full Precision:** Retain all necessary digits, identifiers, and details. Do not summarize or round data during preparation.

## Recommended Formatting: Markdown

Markdown provides a good balance of human readability and machine-parsable structure.

### 1. Headings for Hierarchy

Use Markdown headings (`#`, `##`, `###`, etc.) to define the document's structure logically.

```markdown
# Project Specification: XYZ-123

## 1. Overview
   ... text ...

## 2. Technical Requirements

### 2.1 Hardware
   ... text ...

### 2.2 Software
   ... text ...

## 3. Financial Data

### 3.1 Budget Allocation
   ... text ...

# Appendix A: Error Codes
...
```

### 2. Lists for Enumeration

Use ordered (`1.`, `2.`) or unordered (`*`, `-`) lists for procedures, requirements, or itemized data. Ensure consistent indentation.

```markdown
**Safety Procedure:**

1.  Ensure power is disconnected.
2.  Verify grounding strap is attached.
3.  Wear appropriate PPE:
    *   Safety glasses
    *   Gloves (Nitrile)
4.  Proceed with component removal.

**Component List:**

*   Resistor R101: 10k Ohm, 1%
*   Capacitor C203: 0.1uF, 50V X7R
*   IC U5: Part# ABC-789 Rev B
```

### 3. Tables for Tabular Data

Use Markdown tables for structured data. Keep them simple and ensure clear headers.

```markdown
**Budget Allocation:**

| Department  | Q1 Budget | Q2 Budget | Notes                  |
|-------------|-----------|-----------|------------------------|
| Engineering | $50,000   | $55,000   | Includes new test rig  |
| Marketing   | $30,000   | $30,000   | Trade show expenses    |
| Operations  | $25,000   | $27,500   | Increased shipping     |
```

**Alternative for Complex Tables:** If Markdown tables become too complex or wide, consider representing rows as key-value lists under a relevant heading:

```markdown
### Table: Component Specifications

**Item ID: R101**
*   Type: Resistor
*   Value: 10k Ohm
*   Tolerance: 1%
*   Package: 0805

**Item ID: C203**
*   Type: Capacitor
*   Value: 0.1uF
*   Voltage: 50V
*   Dielectric: X7R
*   Package: 0603
```

### 4. Code Blocks for Code/Logs

Use fenced code blocks (```) with optional language identifiers for code snippets, log excerpts, or pre-formatted text.

````markdown
```python
def calculate_checksum(data):
    # Implementation details...
    checksum = sum(bytearray(data)) % 256
    return checksum
```

```log
[2025-03-30 22:00:01] INFO: System startup sequence initiated.
[2025-03-30 22:00:02] WARN: Sensor B7 reading out of range (Value: 105.3).
[2025-03-30 22:00:03] INFO: Calibration routine started.
```
````

### 5. Emphasis and Clarity

*   Use **bold** (`**bold**` or `__bold__`) for emphasis on key terms or labels.
*   Use *italics* (`*italics*` or `_italics_`) sparingly for definitions or titles.
*   Ensure clear, unambiguous language. Define acronyms on first use.

## Handling Specific Data Types

### Numbers and Financial Data
*   Retain full precision (e.g., `$1,234.56`, `3.14159`).
*   Clearly label units (e.g., `Voltage: 5V`, `Weight: 2.5kg`, `Amount: $USD 500.00`).
*   Use consistent formatting for dates (e.g., `YYYY-MM-DD`).

### Technical Specifications
*   Use lists or key-value pairs for parameters.
*   Clearly label specifications and their values.
*   Include part numbers, revision numbers, and standards where applicable.

### Procedures and Instructions
*   Use ordered lists (`1.`, `2.`, `3.`).
*   Keep steps clear, concise, and sequential.
*   Use bold text for critical actions or warnings.

## Whitespace and Token Efficiency

*   **Remove Trailing Whitespace:** Eliminate spaces at the end of lines.
*   **Consistent Newlines:**
    *   Use a single newline between items in a list or closely related lines.
    *   Use **two newlines** to signify a clear paragraph break or separation between distinct blocks of information (like between a heading and text, or between different list items if they are complex). This helps the LLM (and potentially future parsing logic) understand separation.
*   **Avoid Excessive Blank Lines:** Don't use more than two consecutive newlines.
*   **Indentation:** Use spaces (typically 2 or 4) for indentation in lists or code blocks. Be consistent.

## Optional: Using Text Markers

For highly structured data or complex documents, consider adding simple text markers to explicitly demarcate sections or data types. This is optional but can aid both human readability and potential future automated extraction. Use a consistent format.

```markdown
[SECTION_START: Financials Q1]
... Q1 financial data ...
[SECTION_END: Financials Q1]

[TABLE_START: Error Codes]
| Code | Description | Severity |
|------|-------------|----------|
| E001 | Timeout     | High     |
| E002 | Invalid CRC | Medium   |
[TABLE_END: Error Codes]

[ITEM_START: Part XYZ-123]
Type: Widget
Material: Steel
Weight: 1.5kg
[ITEM_END: Part XYZ-123]
```

Choose markers that are unlikely to appear in the actual data.

## Workflow Summary

1.  **Convert:** Convert source documents (Word, PDF, Excel) to plain text or Markdown.
2.  **Structure:** Apply consistent Markdown formatting (headings, lists, tables, code blocks).
3.  **Refine:** Clean up whitespace, ensure clarity, add labels/units.
4.  **Optimize (Optional):** Add markers if needed for complex data.
5.  **Verify:** Review the prepared text file for accuracy and completeness.
6.  **Process:** Use the LlamaCag UI to create the KV cache from this optimized text file.

By investing time in preparing your data according to these guidelines, you will significantly enhance the ability of LlamaCag to act as a precise, reliable knowledge base for your specific information needs.
