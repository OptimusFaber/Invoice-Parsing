# Binarization

Module for document image binarization with multiple thresholding algorithms, optimized for OCR preprocessing.

## Features

- Multiple binarization algorithms:
  - Bradley-Roth (optimized with Numba)
  - Sauvola
  - Niblack
  - Adaptive (Gaussian and Mean)
  - Otsu
- Parallel processing support using Numba
- Integral image optimization
- Additional preprocessing options:
  - Contrast enhancement (CLAHE)
  - Noise reduction
  - Gaussian blur

## Requirements

- Python >= 3.8
- NumPy
- OpenCV (cv2)
- Numba
- Matplotlib

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from Binarization.image_preprocessing import (
    bradley_threshold,
    sauvola_threshold,
    niblack_threshold,
    adaptive_threshold,
    otsu_threshold
)

# Bradley-Roth method (recommended for most cases)
binary = bradley_threshold(
    img,
    s=15,      # window size (odd number)
    t=0.15     # threshold value (0 to 1)
)

# Sauvola method (good for varying backgrounds)
binary = sauvola_threshold(
    img,
    window_size=15,  # local window size
    k=0.2           # sensitivity parameter
)

# Niblack method
binary = niblack_threshold(
    img,
    window_size=15,  # local window size
    k=-0.2          # weight parameter
)

# Adaptive thresholding
binary = adaptive_threshold(
    img,
    window_size=15,     # local window size
    c=10,              # constant subtracted from mean
    method='gaussian'   # 'gaussian' or 'mean'
)

# Otsu's method
binary = otsu_threshold(
    img,
    blur_kernel=5  # Gaussian blur kernel size
)
```

### Advanced OCR Preprocessing

```python
from Binarization.image_preprocessing import preprocess_image_for_ocr

processed = preprocess_image_for_ocr(
    img,
    blur_kernel=3,
    denoise=True,
    contrast_enhance=True
)
```

## Algorithm Details

### Bradley-Roth Implementation

The Bradley-Roth algorithm is implemented with several optimizations:
- Numba JIT compilation for faster processing
- Parallel processing support
- Integral image calculation for efficient window operations
- Optimized memory usage

Key parameters:
- `s`: Window size (should be odd, typically 1/8 of image width)
- `t`: Threshold value (typically 0.15)

### Other Methods

1. **Sauvola**: Adapts threshold based on local mean and standard deviation
   - Good for documents with varying backgrounds
   - More computationally intensive than Bradley-Roth

2. **Niblack**: Similar to Sauvola but more sensitive to noise
   - Uses local mean and standard deviation
   - Requires post-processing for noise reduction

3. **Adaptive**: OpenCV's implementation
   - Supports both Gaussian and mean methods
   - Fast but less accurate than Bradley-Roth for documents

4. **Otsu**: Global thresholding method
   - Fast but doesn't handle varying backgrounds well
   - Includes optional Gaussian blur preprocessing

## Performance Tips

1. For best OCR results:
   - Start with Bradley-Roth method
   - If results are unsatisfactory, try Sauvola
   - Use contrast enhancement for low-contrast images

2. Parameter tuning:
   - Window size should be odd and approximately 1/8 of image width
   - Increase threshold for darker images
   - Decrease threshold for lighter images

3. Optimization:
   - Bradley-Roth is optimized for parallel processing
   - Use appropriate window sizes (larger windows = slower processing)
   - Enable contrast enhancement only when needed

## Notes

- All methods automatically convert color images to grayscale
- Window sizes are automatically adjusted to be odd numbers
- Bradley-Roth implementation is optimized with Numba for performance
- OCR preprocessing includes optional denoising and contrast enhancement 