# Document Parser

A tool for processing documents and extracting structured information.

## Requirements

- Python >= 3.8
- CUDA >= 11.1 (for GPU version)
- Tesseract OCR

## Installation

### 1. Installing Tesseract OCR

For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-rus  # for Russian language
sudo apt-get install tesseract-ocr-nld  # for Dutch language
```

For other operating systems, please refer to the [official documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).

### 2. Installing Python Package

#### Basic Installation (CPU version):
```bash
pip install .
```

#### Installation with GPU support:
```bash
pip install ".[gpu]"
```

## Usage

```python
from document_parser import TwoStageParser

# Initialize parser
parser = TwoStageParser(cuda=True)  # use cuda=False for CPU version

# Process image
result, predictions, details = parser.process_image(
    "path/to/your/image.jpg",
    mode="two_stage",
    target_classes=["Invoice_detail"]
)
```

## Project Structure

```
DocumentParser/
├── LayoutDetectron/
│   ├── Invoices/
│   │   ├── config.yaml
│   │   └── model_final.pth
│   └── InvoiceDetails/
│       ├── config.yaml
│       └── model_final.pth
├── YoloDetectron/
│   ├── Invoices/
│   │   └── best.pt
│   └── InvoiceDetails/
│       └── best.pt
├── config.json
├── setup.py
└── README.md
```

## Notes

- Make sure all required models and configuration files are in their respective directories
- When using the GPU version, ensure CUDA versions are compatible
- Properly installed Tesseract is required for OCR functionality 