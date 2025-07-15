"""
Image Loader Module
==================

Handles image loading, format conversion, and EXIF data extraction
following international standards.
"""

import cv2
import numpy as np
import os
from PIL import Image, ExifTags
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ImageLoader:
    """Professional image loader with EXIF support"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp']
    
    def load_image(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[Image.Image]]:
        """
        Load image using OpenCV and PIL
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (opencv_image, pil_image) or (None, None) if failed
        """
        try:
            # Validate file existence
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Validate file extension
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.supported_formats:
                raise ValueError(f"Unsupported format: {ext}")
            
            # Load with OpenCV
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                raise ValueError(f"Cannot load image with OpenCV: {image_path}")
            
            # Convert BGR to RGB
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Load with PIL for EXIF
            img_pil = Image.open(image_path)
            
            return img_cv, img_pil
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None
    
    def get_image_info(self, img_cv: np.ndarray, img_pil: Image.Image) -> Dict[str, Any]:
        """
        Extract basic image information
        
        Args:
            img_cv: OpenCV image array
            img_pil: PIL image object
            
        Returns:
            Dictionary containing image information
        """
        height, width = img_cv.shape[:2]
        channels = img_cv.shape[2] if len(img_cv.shape) > 2 else 1
        
        # Calculate megapixels
        megapixels = (width * height) / 1000000
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Get file size
        file_size = 0
        if hasattr(img_pil, 'filename') and img_pil.filename:
            file_size = os.path.getsize(img_pil.filename)
        
        # Get DPI information
        dpi_info = {}
        if hasattr(img_pil, 'info') and 'dpi' in img_pil.info:
            dpi_info['dpi'] = img_pil.info['dpi']
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'megapixels': round(megapixels, 2),
            'aspect_ratio': round(aspect_ratio, 3),
            'file_size': file_size,
            'dpi_info': dpi_info,
            'color_mode': img_pil.mode if img_pil else 'RGB'
        }
    
    def extract_exif_data(self, img_pil: Image.Image) -> Dict[str, Any]:
        """
        Extract comprehensive EXIF data
        
        Args:
            img_pil: PIL image object
            
        Returns:
            Dictionary containing EXIF data and camera information
        """
        exif_data = {}
        camera_info = {}
        
        if not hasattr(img_pil, '_getexif') or not img_pil._getexif():
            return {'exif_data': exif_data, 'camera_info': camera_info}
        
        exif = img_pil._getexif()
        
        for tag, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            exif_data[tag_name] = value
            
            # Extract important camera information
            if tag_name == 'Make':
                camera_info['camera_make'] = str(value).strip()
            elif tag_name == 'Model':
                camera_info['camera_model'] = str(value).strip()
            elif tag_name == 'FNumber':
                camera_info['aperture'] = f"f/{float(value):.1f}"
            elif tag_name == 'ExposureTime':
                if isinstance(value, tuple) and len(value) == 2:
                    camera_info['shutter_speed'] = f"{value[0]}/{value[1]}s"
                else:
                    camera_info['shutter_speed'] = f"{value}s"
            elif tag_name == 'ISOSpeedRatings':
                camera_info['iso'] = int(value)
            elif tag_name == 'FocalLength':
                if isinstance(value, tuple) and len(value) == 2:
                    camera_info['focal_length'] = f"{float(value[0]/value[1]):.1f}mm"
                else:
                    camera_info['focal_length'] = f"{float(value):.1f}mm"
            elif tag_name == 'WhiteBalance':
                camera_info['white_balance'] = 'Auto' if value == 0 else 'Manual'
            elif tag_name == 'Flash':
                camera_info['flash'] = 'On' if value & 1 else 'Off'
            elif tag_name == 'ExposureMode':
                modes = ['Auto', 'Manual', 'Auto bracket']
                camera_info['exposure_mode'] = modes[value] if value < len(modes) else 'Unknown'
            elif tag_name == 'MeteringMode':
                modes = ['Unknown', 'Average', 'Center-weighted', 'Spot', 
                        'Multi-spot', 'Pattern', 'Partial']
                camera_info['metering_mode'] = modes[value] if value < len(modes) else 'Unknown'
            elif tag_name == 'DateTime':
                camera_info['date_time'] = str(value)
            elif tag_name == 'ColorSpace':
                camera_info['color_space'] = 'sRGB' if value == 1 else 'Adobe RGB'
            elif tag_name == 'LensModel':
                camera_info['lens_model'] = str(value).strip()
            elif tag_name == 'FocalLengthIn35mmFilm':
                camera_info['focal_length_35mm'] = f"{int(value)}mm"
            elif tag_name == 'ExposureBiasValue':
                if isinstance(value, tuple) and len(value) == 2:
                    bias = float(value[0]) / float(value[1])
                    camera_info['exposure_bias'] = f"{bias:+.1f} EV"
            elif tag_name == 'MaxApertureValue':
                if isinstance(value, tuple) and len(value) == 2:
                    max_aperture = float(value[0]) / float(value[1])
                    camera_info['max_aperture'] = f"f/{max_aperture:.1f}"
        
        return {
            'exif_data': exif_data,
            'camera_info': camera_info
        }
    
    def validate_image(self, img_cv: np.ndarray) -> bool:
        """
        Validate image for processing
        
        Args:
            img_cv: OpenCV image array
            
        Returns:
            True if image is valid for processing
        """
        if img_cv is None:
            return False
        
        if len(img_cv.shape) < 2:
            return False
        
        height, width = img_cv.shape[:2]
        if height < 100 or width < 100:
            return False
        
        return True
