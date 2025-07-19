# TextParser

Module for extracting and organizing text from images while preserving column structure.

## Requirements

- Python >= 3.8
- Tesseract OCR

## Installation

### 1. Installing Tesseract OCR

For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-rus  # for Russian language
```

For other operating systems, please refer to the [official documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).

### 2. Installing Python Package

```bash
# Using pip with requirements.txt
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Usage

```python
from text_parser import process_image, main_algo

# Process a single image
columns, text_lines = process_image("path/to/your/image.jpg")

# Output results
print("Results by columns:")
for i, column in enumerate(columns):
    print(f"\nColumn {i+1}:")
    for word in column:
        print(f"Text: {word['text']}")
        print(f"Coordinates: {word['bbox']}")

print("\nText lines:")
for line in text_lines:
    print(line)
```

## Project Structure

```
TextParser/
├── __init__.py
├── parser.py
├── text_processing.py
├── setup.py
├── requirements.txt
└── README.md
```

## Main Functions

### process_image(image_path)
- Loads and rotates the image if necessary
- Extracts text while preserving coordinates
- Sorts words into columns
- Returns a tuple (columns with text and coordinates, text lines)

### sort_words_into_columns(words, max_x_distance=30.0, max_word_gap=50.0, verbose=False)
- Groups words into columns based on their position
- Merges closely positioned words
- Supports customization of grouping parameters

### extract_text_from_image(image, x1=0, y1=0, x2=None, y2=None)
- Extracts text from an image or its region
- Preserves coordinates for each word
- Supports both PIL.Image and numpy.ndarray inputs

## Parameters

- `max_x_distance`: maximum X-axis distance between words in the same column (default 30.0)
- `max_word_gap`: maximum distance between words for merging them (default 50.0)
- `verbose`: output detailed information about the processing

## Notes

- Properly installed Tesseract is required for OCR functionality
- All coordinates in the output correspond to the original image
- Word merging takes into account their relative positions and distances between them 