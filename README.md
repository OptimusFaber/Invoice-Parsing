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
Module for extracting text from the identified document zones. Currently uses pytesseract as the OCR engine to convert image regions into machine-readable text. Future plans include testing alternative OCR engines like PaddleOCR and DocTR for improved accuracy and performance.

#### OCR Engine Comparison

| Feature | PaddleOCR | DocTR | EasyOCR | Pytesseract |
|---------|-----------|--------|----------|-------------|
| Architecture | PP-OCR series (DB + CRNN) | Transformers-based | CRAFT + CRNN | Tesseract LSTM |
| CPU Support | ✅ Good performance | ✅ Light models available | ✅ Native support | ✅ Native support |
| GPU Support | ✅ Via PaddlePaddle | ✅ Via PyTorch | ✅ Via PyTorch | ❌ Limited |
| PyTorch Support | ❌ Uses PaddlePaddle | ✅ Native support | ✅ Native support | N/A |
| Layout Analysis | ✅ Built-in | ✅ Layout-aware OCR | ❌ Basic | ❌ Basic |
| Language Support | 80+ languages | Multi-language | 80+ languages | 100+ languages |
| Model Size | Ultra-lightweight (3.5M) | Varies (light options available) | ~100MB | ~30MB |
| GitHub Stats | 51.9k stars (2020) | 5k stars (2021) | 27.3k stars (2020) | Part of Tesseract |
| Repository | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | [DocTR](https://github.com/mindee/doctr) | [EasyOCR](https://github.com/JaidedAI/EasyOCR) | [Tesseract](https://github.com/tesseract-ocr/tesseract) |

##### PaddleOCR
✅ Advantages:
- Highly accurate, especially with ch_PP-OCRv3 and en_PP-OCRv3 models
- Supports horizontal and vertical text
- Built-in layout analysis (tables, blocks, text direction)
- Good CPU performance (faster than Pytesseract with better accuracy)

❌ Disadvantages:
- Requires understanding of installation and PaddlePaddle dependencies

##### DocTR (Document Text Recognition)
✅ Advantages:
- Modern architecture using Transformers
- Light models available for CPU usage
- Easy integration and layout-aware OCR support
- Native PyTorch support

❌ Disadvantages:
- Newer project with less community support
- Fewer pre-trained models compared to PaddleOCR

##### EasyOCR
✅ Advantages:
- Simple Python API and very easy to use
- Excellent multi-language support out of the box
- Native PyTorch integration
- Active community and good documentation
- Ready-to-use models for 80+ languages

❌ Disadvantages:
- Larger model size compared to PaddleOCR
- Basic layout analysis capabilities
- Can be slower than PaddleOCR on CPU

### SpacyParser
A trained model for processing information from specific document zones. This module performs named entity recognition (NER) and information extraction from the extracted text.