from setuptools import setup, find_packages

setup(
    name="document_parser",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "layoutparser>=0.3.4",
        "torch>=1.9.0+cu118",  # Устанавливаем PyTorch с CUDA 11.8
        "torchvision>=0.10.0+cu118",  # Устанавливаем torchvision с CUDA 11.8
        "ultralytics>=8.0.0",  # для YOLO
        "pytesseract>=0.3.8",
        "imutils>=0.5.4",
        "tqdm>=4.65.0",
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/cu118",  # Добавляем PyTorch репозиторий
    ],
    extras_require={
        "gpu": [
            "torch>=1.9.0+cu118",  # CUDA 11.8
            "torchvision>=0.10.0+cu118",
        ],
        "detectron": [
            "detectron2 @ git+https://github.com/facebookresearch/detectron2.git",  # для layout parser
        ],
    },
    package_data={
        "document_parser": [
            "config.json",
            "LayoutDetectron/Invoices/*.yaml",
            "LayoutDetectron/Invoices/*.pth",
            "LayoutDetectron/InvoiceDetails/*.yaml",
            "LayoutDetectron/InvoiceDetails/*.pth",
            "YoloDetectron/Invoices/*.pt",
            "YoloDetectron/InvoiceDetails/*.pt",
        ],
    },
    description="Document Parser for processing invoices and extracting structured information",
    author="Rodrick Kalkopf",
    author_email="r.shkokov@gmail.com",
    url="https://github.com/OptimusFaber/document-parser",
) 