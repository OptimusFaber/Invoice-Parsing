# Invoice Parsing Project

This project is designed for comprehensive invoice parsing and information extraction, consisting of several specialized modules:

## Project Structure

### Binarization
Module for document binarization using the Bradley-Roth method. This preprocessing step helps improve the quality of subsequent text recognition and parsing.

### DocumentParser
Module for document parsing and extraction of main zones. Implements two methods:
- YOLO-based detection: Uses standard YOLO training approach for object detection
- LayoutDetectron-based detection: Based on the LayoutParser framework. For detailed information about LayoutParser implementation and training, please refer to [Layout-Parser/layout-parser](https://github.com/Layout-Parser/layout-parser)

The module identifies key areas in invoices such as company details, customer information, invoice details, tables, and totals.

### TextParser
Module for extracting text from the identified document zones. Currently uses pytesseract as the OCR engine to convert image regions into machine-readable text.

### SpacyParser
A trained model for processing information from specific document zones. This module performs named entity recognition (NER) and information extraction from the extracted text.