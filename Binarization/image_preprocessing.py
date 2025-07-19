import numpy as np
import cv2
from numba import njit, prange

@njit(parallel=True)
def calculate_integral_image(img):
    """
    Fast integral image calculation using numba.
    """
    height, width = img.shape
    intImg = np.zeros((height + 1, width + 1), dtype=np.int32)
    
    for i in prange(height):
        for j in prange(width):
            intImg[i + 1, j + 1] = intImg[i, j + 1] + intImg[i + 1, j] - intImg[i, j] + img[i, j]
            
    return intImg

@njit(parallel=True)
def bradley_threshold_optimized(img, integral_img, s, t):
    """
    Optimized Bradley thresholding implementation with parallel processing.
    """
    height, width = img.shape
    result = np.zeros_like(img, dtype=np.uint8)
    s2 = s // 2
    
    for i in prange(height):
        for j in prange(width):
            # Define window boundaries
            x1 = max(0, i - s2)
            x2 = min(height - 1, i + s2)
            y1 = max(0, j - s2)
            y2 = min(width - 1, j + s2)
            
            # Calculate window area
            count = (x2 - x1 + 1) * (y2 - y1 + 1)
            
            # Calculate sum of pixels in window using integral image
            sum_region = (integral_img[x2 + 1, y2 + 1] - 
                         integral_img[x2 + 1, y1] - 
                         integral_img[x1, y2 + 1] + 
                         integral_img[x1, y1])
            
            # Apply threshold
            if img[i, j] * count < sum_region * (1.0 - t):
                result[i, j] = 0
            else:
                result[i, j] = 255
                
    return result

def bradley_threshold(img, s=15, t=0.15):
    """
    Improved Bradley thresholding algorithm.
    
    Args:
        img: grayscale input image
        s: window size (should be odd)
        t: threshold value (0 to 1)
    
    Returns:
        Binarized image
    """
    # Input validation
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure s is odd
    if s % 2 == 0:
        s += 1
    
    # Calculate integral image
    integral_img = calculate_integral_image(img)
    
    # Apply optimized thresholding
    result = bradley_threshold_optimized(img, integral_img, s, t)
    
    return result

def sauvola_threshold(img, window_size=15, k=0.2):
    """
    Sauvola thresholding algorithm - better for document images with varying backgrounds.
    
    Args:
        img: grayscale input image
        window_size: size of the local window
        k: positive parameter controlling threshold sensitivity
        
    Returns:
        Binarized image
    """
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
        
    # Calculate mean and standard deviation
    mean = cv2.boxFilter(img.astype(np.float32), -1, (window_size, window_size), 
                         borderType=cv2.BORDER_REFLECT)
    
    mean_sq = cv2.boxFilter(img.astype(np.float32) * img.astype(np.float32), -1, 
                           (window_size, window_size), borderType=cv2.BORDER_REFLECT)
    
    # Calculate standard deviation
    std = np.sqrt(mean_sq - mean * mean)
    
    # Calculate threshold
    threshold = mean * (1.0 + k * ((std / 128.0) - 1.0))
    
    # Apply threshold
    result = np.zeros_like(img, dtype=np.uint8)
    result[img > threshold] = 255
    
    return result

def niblack_threshold(img, window_size=15, k=-0.2):
    """
    Niblack thresholding algorithm.
    
    Args:
        img: grayscale input image
        window_size: size of the local window
        k: weight parameter (typically negative)
        
    Returns:
        Binarized image
    """
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
        
    # Calculate mean and standard deviation
    mean = cv2.boxFilter(img.astype(np.float32), -1, (window_size, window_size), 
                         borderType=cv2.BORDER_REFLECT)
    
    mean_sq = cv2.boxFilter(img.astype(np.float32) * img.astype(np.float32), -1, 
                           (window_size, window_size), borderType=cv2.BORDER_REFLECT)
    
    # Calculate standard deviation
    std = np.sqrt(mean_sq - mean * mean)
    
    # Calculate threshold
    threshold = mean + k * std
    
    # Apply threshold
    result = np.zeros_like(img, dtype=np.uint8)
    result[img > threshold] = 255
    
    return result

def adaptive_threshold(img, window_size=15, c=10, method='gaussian'):
    """
    Enhanced adaptive thresholding using OpenCV.
    
    Args:
        img: grayscale input image
        window_size: size of the local window (must be odd)
        c: constant subtracted from mean or weighted mean
        method: 'mean' or 'gaussian' for different weighting methods
        
    Returns:
        Binarized image
    """
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Choose adaptive method
    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == 'gaussian' else cv2.ADAPTIVE_THRESH_MEAN_C
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(img, 255, adaptive_method, cv2.THRESH_BINARY, window_size, c)
    
    return binary

def otsu_threshold(img, blur_kernel=5):
    """
    Otsu's thresholding with Gaussian blur preprocessing.
    
    Args:
        img: input image
        blur_kernel: size of the Gaussian blur kernel
    
    Returns:
        Binarized image
    """
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply Gaussian blur
    if blur_kernel > 0:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def preprocess_image_for_ocr(img, method='sauvola', window_size=15, k=0.2, c=10, 
                            blur_kernel=3, denoise=True, contrast_enhance=True,
                            adaptive_method='gaussian'):
    """
    Enhanced image preprocessing for improved OCR text recognition.
    
    Args:
        img: input image
        method: ignored (kept for backward compatibility)
        window_size: ignored (kept for backward compatibility)
        k: ignored (kept for backward compatibility)
        c: ignored (kept for backward compatibility)
        blur_kernel: kernel size for Gaussian blur
        denoise: apply additional denoising
        contrast_enhance: enhance contrast before binarization
        adaptive_method: ignored (kept for backward compatibility)
        
    Returns:
        Processed grayscale image (without binarization)
    """
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply contrast enhancement if requested
    if contrast_enhance:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Apply denoising if requested
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Apply Gaussian blur to reduce noise
    if blur_kernel > 0:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    
    # Return the preprocessed grayscale image without binarization
    return gray 