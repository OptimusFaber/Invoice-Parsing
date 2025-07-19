from setuptools import setup, find_packages

setup(
    name="text_parser",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "pytesseract>=0.3.8",
        "imutils>=0.5.4",
        "tqdm>=4.65.0",
    ],
    description="Text Parser for extracting and organizing text from images",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/text-parser",
) 