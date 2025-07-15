#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Image Quality Analyzer - Advanced Edition
=====================================================

A comprehensive image quality assessment tool with advanced pixel-level analysis.
Features state-of-the-art evaluation techniques for professional photography.

Usage:
    python main.py <image1_path> <image2_path>
    python main.py --analyze-single <image_path>

Advanced Features:
- Pixel-level noise analysis with SNR mapping
- Depth of field and bokeh quality assessment
- Object detection and background analysis
- Advanced color space evaluation (RGB, HSV, LAB)
- Texture and detail analysis with local binary patterns
- Professional composition analysis
- Multi-region exposure analysis
- Advanced sharpness metrics (Laplacian, Sobel, Tenengrad, Brenner)

Author: AI Assistant
Date: July 15, 2025
Version: 2.0.0 (Advanced Edition)
"""

import sys
import os
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Only essential imports
try:
    import cv2
    import numpy as np
    from PIL import Image, ExifTags
    print("✅ Essential libraries loaded successfully")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Please install: pip install opencv-python numpy pillow")
    sys.exit(1)


class SimpleImageAnalyzer:
    """Simplified image quality analyzer using only OpenCV and NumPy"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp']
    
    def load_image(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[Image.Image]]:
        """Load image with OpenCV and PIL"""
        try:
            # Validate file
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return None, None
            
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.supported_formats:
                print(f"❌ Unsupported format: {ext}")
                return None, None
            
            # Load with OpenCV
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                print(f"❌ Cannot load image: {image_path}")
                return None, None
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Load with PIL for EXIF
            img_pil = Image.open(image_path)
            
            return img_rgb, img_pil
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None, None
    
    def extract_exif_data(self, img_pil: Image.Image) -> Dict[str, Any]:
        """Extract EXIF data from image"""
        exif_data = {}
        camera_info = {}
        
        try:
            if hasattr(img_pil, '_getexif') and img_pil._getexif():
                exif = img_pil._getexif()
                
                for tag, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    exif_data[tag_name] = value
                    
                    # Extract camera information
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
                    elif tag_name == 'DateTime':
                        camera_info['date_time'] = str(value)
        
        except Exception as e:
            print(f"⚠️ EXIF extraction error: {e}")
        
        return {'exif_data': exif_data, 'camera_info': camera_info}
    
    def analyze_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Comprehensive advanced image analysis"""
        print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
        
        # Load image
        img_rgb, img_pil = self.load_image(image_path)
        if img_rgb is None:
            return None
        
        # Convert to grayscale and other color spaces
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        height, width = gray.shape
        
        # Basic information
        megapixels = (width * height) / 1000000
        aspect_ratio = width / height
        file_size = os.path.getsize(image_path)
        
        # Extract EXIF data
        exif_info = self.extract_exif_data(img_pil)
        
        # ADVANCED ANALYSIS MODULES
        print("   🔬 Running advanced pixel-level analysis...")
        
        # 1. ADVANCED SHARPNESS ANALYSIS
        sharpness_metrics = self.analyze_sharpness_advanced(gray)
        
        # 2. PIXEL-LEVEL NOISE ANALYSIS
        noise_metrics = self.analyze_noise_per_pixel(gray, img_rgb)
        
        # 3. DEPTH OF FIELD & BOKEH ANALYSIS
        depth_metrics = self.analyze_depth_of_field(gray, img_rgb)
        
        # 4. OBJECT DETECTION & BACKGROUND ANALYSIS
        object_metrics = self.analyze_objects_and_background(gray, img_rgb)
        
        # 5. ADVANCED COLOR SPACE ANALYSIS
        color_metrics = self.analyze_color_advanced(img_rgb, img_hsv, img_lab)
        
        # 6. TEXTURE & DETAIL ANALYSIS
        texture_metrics = self.analyze_texture_detail(gray)
        
        # 7. EXPOSURE & DYNAMIC RANGE ANALYSIS
        exposure_metrics = self.analyze_exposure_advanced(gray, img_rgb)
        
        # 8. COMPOSITION & VISUAL BALANCE
        composition_metrics = self.analyze_composition_advanced(gray, img_rgb)
        
        # CALCULATE ADVANCED SCORES
        print("   📊 Calculating advanced quality scores...")
        
        # Normalize and score each metric (0-100 scale)
        scores = {}
        
        # Sharpness scoring
        sharpness_score = min(100, max(0, sharpness_metrics['laplacian_variance'] / 100))
        scores['sharpness'] = sharpness_score
        
        # Noise scoring (lower noise = higher score)
        noise_score = max(0, 100 - noise_metrics['high_frequency_noise'] * 2)
        scores['noise'] = noise_score
        
        # Depth of field scoring
        depth_score = min(100, max(0, depth_metrics['depth_separation'] / 50))
        scores['depth_of_field'] = depth_score
        
        # Object clarity scoring
        object_score = min(100, max(0, object_metrics['contour_complexity'] / 20))
        scores['object_clarity'] = object_score
        
        # Color quality scoring
        color_score = max(0, 100 - color_metrics['white_balance_error'] * 2)
        scores['color_quality'] = color_score
        
        # Texture scoring
        texture_score = min(100, max(0, texture_metrics['texture_energy'] / 50))
        scores['texture'] = texture_score
        
        # Exposure scoring
        exposure_score = exposure_metrics['midtones'] * 100
        scores['exposure'] = exposure_score
        
        # Composition scoring
        composition_score = composition_metrics['thirds_balance'] * 100
        scores['composition'] = composition_score
        
        # Overall score with advanced weights
        weights = {
            'sharpness': 0.20,
            'noise': 0.15,
            'depth_of_field': 0.15,
            'object_clarity': 0.10,
            'color_quality': 0.15,
            'texture': 0.10,
            'exposure': 0.10,
            'composition': 0.05
        }
        
        overall_score = sum(scores[metric] * weights[metric] for metric in weights.keys())
        scores['overall'] = overall_score
        
        # Grade assignment
        grade = self.get_grade(overall_score)
        
        # Compile comprehensive results
        analysis = {
            'image_info': {
                'filename': os.path.basename(image_path),
                'full_path': image_path,
                'width': width,
                'height': height,
                'megapixels': round(megapixels, 2),
                'aspect_ratio': round(aspect_ratio, 3),
                'file_size_kb': round(file_size / 1024, 1)
            },
            'camera_info': exif_info['camera_info'],
            'advanced_metrics': {
                'sharpness': sharpness_metrics,
                'noise': noise_metrics,
                'depth_of_field': depth_metrics,
                'objects_background': object_metrics,
                'color_analysis': color_metrics,
                'texture': texture_metrics,
                'exposure': exposure_metrics,
                'composition': composition_metrics
            },
            'quality_scores': scores,
            'grade': grade,
            'analysis_date': datetime.now().isoformat()
        }
        
        return analysis
    
    def get_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 95:
            return "A++ (Outstanding)"
        elif score >= 90:
            return "A+ (Excellent)"
        elif score >= 85:
            return "A (Very Good)"
        elif score >= 80:
            return "B+ (Good)"
        elif score >= 75:
            return "B (Above Average)"
        elif score >= 70:
            return "B- (Average)"
        elif score >= 65:
            return "C+ (Below Average)"
        elif score >= 60:
            return "C (Fair)"
        elif score >= 50:
            return "D (Poor)"
        else:
            return "F (Unacceptable)"
    
    def compare_images(self, image1_path: str, image2_path: str) -> Dict[str, Any]:
        """Compare two images"""
        print("🥊 COMPARING IMAGES")
        print("=" * 50)
        
        # Analyze both images
        analysis1 = self.analyze_image(image1_path)
        analysis2 = self.analyze_image(image2_path)
        
        if not analysis1 or not analysis2:
            return {}
        
        # Extract scores
        score1 = analysis1['quality_scores']['overall']
        score2 = analysis2['quality_scores']['overall']
        
        # Determine winner
        if score1 > score2:
            winner = 1
            margin = score1 - score2
        elif score2 > score1:
            winner = 2
            margin = score2 - score1
        else:
            winner = 0
            margin = 0
        
        # Significance
        if margin > 10:
            significance = "significant"
        elif margin > 5:
            significance = "moderate"
        elif margin > 0:
            significance = "slight"
        else:
            significance = "tie"
        
        # Category comparison
        category_comparison = {}
        for category in ['sharpness', 'noise', 'depth_of_field', 'object_clarity', 'color_quality', 'texture', 'exposure', 'composition']:
            score1_cat = analysis1['quality_scores'][category]
            score2_cat = analysis2['quality_scores'][category]
            diff = score1_cat - score2_cat
            
            category_comparison[category] = {
                'score1': score1_cat,
                'score2': score2_cat,
                'difference': round(diff, 1),
                'winner': 1 if diff > 0 else 2 if diff < 0 else 0
            }
        
        return {
            'image1': analysis1,
            'image2': analysis2,
            'comparison': {
                'winner': winner,
                'margin': round(margin, 2),
                'significance': significance,
                'category_comparison': category_comparison
            }
        }
    
    def analyze_sharpness_advanced(self, gray: np.ndarray) -> Dict[str, Any]:
        """Advanced sharpness analysis with multiple metrics"""
        print("     📐 Analyzing sharpness...")
        
        # 1. Laplacian variance (global sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        # 2. Sobel gradient magnitude
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_mean = np.mean(sobel_magnitude)
        gradient_std = np.std(sobel_magnitude)
        
        # 3. Tenengrad sharpness
        tenengrad = np.mean(sobel_x**2 + sobel_y**2)
        
        # 4. Brenner sharpness
        brenner = np.mean((gray[2:, :] - gray[:-2, :])**2)
        
        # 5. Regional sharpness analysis
        h, w = gray.shape
        regions = []
        for i in range(0, h, h//4):
            for j in range(0, w, w//4):
                region = gray[i:i+h//4, j:j+w//4]
                if region.size > 0:
                    region_lap = cv2.Laplacian(region, cv2.CV_64F)
                    regions.append(np.var(region_lap))
        
        regional_sharpness = {
            'mean': np.mean(regions),
            'std': np.std(regions),
            'max': np.max(regions),
            'min': np.min(regions)
        }
        
        # 6. Focus measure using high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray, -1, kernel)
        focus_measure = np.mean(np.abs(filtered))
        
        return {
            'laplacian_variance': float(laplacian_var),
            'gradient_mean': float(gradient_mean),
            'gradient_std': float(gradient_std),
            'tenengrad': float(tenengrad),
            'brenner': float(brenner),
            'focus_measure': float(focus_measure),
            'regional_sharpness': regional_sharpness
        }
    
    def analyze_noise_per_pixel(self, gray: np.ndarray, img_rgb: np.ndarray) -> Dict[str, Any]:
        """Pixel-level noise analysis"""
        print("     🔊 Analyzing pixel-level noise...")
        
        # 1. High-frequency noise estimation
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(gray, -1, kernel)
        noise_level = np.std(high_freq)
        
        # 2. Median filter noise estimation
        median_filtered = cv2.medianBlur(gray, 3)
        noise_map = np.abs(gray.astype(float) - median_filtered.astype(float))
        pixel_noise_mean = np.mean(noise_map)
        pixel_noise_std = np.std(noise_map)
        
        # 3. Gaussian noise estimation
        gaussian_filtered = cv2.GaussianBlur(gray, (3, 3), 0)
        gaussian_noise = np.abs(gray.astype(float) - gaussian_filtered.astype(float))
        gaussian_noise_level = np.std(gaussian_noise)
        
        # 4. SNR calculation per region
        h, w = gray.shape
        snr_regions = []
        for i in range(0, h, h//8):
            for j in range(0, w, w//8):
                region = gray[i:i+h//8, j:j+w//8]
                if region.size > 0:
                    signal_power = np.mean(region**2)
                    noise_power = np.var(region)
                    if noise_power > 0:
                        snr = 10 * np.log10(signal_power / noise_power)
                        snr_regions.append(snr)
        
        # 5. Color noise analysis
        r_noise = np.std(cv2.filter2D(img_rgb[:, :, 0], -1, kernel))
        g_noise = np.std(cv2.filter2D(img_rgb[:, :, 1], -1, kernel))
        b_noise = np.std(cv2.filter2D(img_rgb[:, :, 2], -1, kernel))
        
        # 6. Temporal noise estimation (simulate)
        temporal_noise = np.std([r_noise, g_noise, b_noise])
        
        return {
            'high_frequency_noise': float(noise_level),
            'pixel_noise_mean': float(pixel_noise_mean),
            'pixel_noise_std': float(pixel_noise_std),
            'gaussian_noise': float(gaussian_noise_level),
            'snr_mean': float(np.mean(snr_regions)) if snr_regions else 0,
            'snr_std': float(np.std(snr_regions)) if snr_regions else 0,
            'color_noise': {
                'red': float(r_noise),
                'green': float(g_noise),
                'blue': float(b_noise),
                'temporal': float(temporal_noise)
            }
        }
    
    def analyze_depth_of_field(self, gray: np.ndarray, img_rgb: np.ndarray) -> Dict[str, Any]:
        """Depth of field and bokeh analysis"""
        print("     🎯 Analyzing depth of field...")
        
        # 1. Depth map estimation using gradient
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 2. Focus regions detection
        blur_kernel = cv2.getGaussianKernel(5, 1)
        blurred = cv2.filter2D(gray, -1, blur_kernel)
        sharpness_map = np.abs(gray.astype(float) - blurred.astype(float))
        
        # 3. Bokeh quality analysis
        # Find circular shapes (potential bokeh)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, 
                                  param1=50, param2=30, minRadius=5, maxRadius=50)
        
        bokeh_count = 0
        bokeh_quality = 0
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            bokeh_count = len(circles)
            
            # Analyze bokeh quality
            for (x, y, r) in circles:
                if 0 <= x-r and x+r < gray.shape[1] and 0 <= y-r and y+r < gray.shape[0]:
                    bokeh_region = gray[y-r:y+r, x-r:x+r]
                    if bokeh_region.size > 0:
                        bokeh_quality += np.std(bokeh_region)
            
            bokeh_quality /= bokeh_count if bokeh_count > 0 else 1
        
        # 4. Background blur analysis
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        border_region = np.concatenate([
            gray[:h//4, :].flatten(),
            gray[3*h//4:, :].flatten(),
            gray[:, :w//4].flatten(),
            gray[:, 3*w//4:].flatten()
        ])
        
        center_sharpness = np.var(cv2.Laplacian(center_region, cv2.CV_64F))
        border_sharpness = np.var(cv2.Laplacian(border_region.reshape(-1, 1), cv2.CV_64F))
        
        depth_separation = center_sharpness - border_sharpness
        
        # 5. Focus transition analysis
        focus_transitions = []
        for i in range(0, h, h//10):
            row = gray[i, :]
            transitions = np.diff(np.diff(row))
            focus_transitions.append(np.std(transitions))
        
        return {
            'gradient_magnitude_mean': float(np.mean(gradient_magnitude)),
            'gradient_magnitude_std': float(np.std(gradient_magnitude)),
            'sharpness_map_mean': float(np.mean(sharpness_map)),
            'sharpness_map_std': float(np.std(sharpness_map)),
            'bokeh_count': int(bokeh_count),
            'bokeh_quality': float(bokeh_quality),
            'depth_separation': float(depth_separation),
            'center_sharpness': float(center_sharpness),
            'border_sharpness': float(border_sharpness),
            'focus_transitions': float(np.mean(focus_transitions))
        }
    
    def analyze_objects_and_background(self, gray: np.ndarray, img_rgb: np.ndarray) -> Dict[str, Any]:
        """Object detection and background analysis"""
        print("     🎭 Analyzing objects and background...")
        
        # 1. Edge detection for object boundaries
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        
        # 2. Contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours
        contour_areas = []
        contour_perimeters = []
        contour_complexities = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area > 100:  # Filter small contours
                contour_areas.append(area)
                contour_perimeters.append(perimeter)
                if perimeter > 0:
                    complexity = (perimeter ** 2) / area
                    contour_complexities.append(complexity)
        
        # 3. Object-background separation
        # Use Otsu's thresholding for segmentation
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate object vs background ratio
        object_pixels = np.sum(thresh == 255)
        background_pixels = np.sum(thresh == 0)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # 4. Background uniformity
        background_mask = thresh == 0
        if np.any(background_mask):
            background_values = gray[background_mask]
            background_uniformity = 1 - (np.std(background_values) / 255)
        else:
            background_uniformity = 0
        
        # 5. Object complexity analysis
        object_mask = thresh == 255
        if np.any(object_mask):
            object_values = gray[object_mask]
            object_complexity = np.std(object_values) / 255
        else:
            object_complexity = 0
        
        return {
            'edge_density': float(edge_density),
            'contour_count': len(contours),
            'contour_areas': {
                'mean': float(np.mean(contour_areas)) if contour_areas else 0,
                'std': float(np.std(contour_areas)) if contour_areas else 0,
                'max': float(np.max(contour_areas)) if contour_areas else 0
            },
            'contour_complexity': float(np.mean(contour_complexities)) if contour_complexities else 0,
            'object_background_ratio': float(object_pixels / total_pixels),
            'background_uniformity': float(background_uniformity),
            'object_complexity': float(object_complexity)
        }
    
    def analyze_color_advanced(self, img_rgb: np.ndarray, img_hsv: np.ndarray, img_lab: np.ndarray) -> Dict[str, Any]:
        """Advanced color space analysis"""
        print("     🎨 Analyzing color spaces...")
        
        # 1. RGB analysis
        r_channel = img_rgb[:, :, 0]
        g_channel = img_rgb[:, :, 1]
        b_channel = img_rgb[:, :, 2]
        
        rgb_stats = {
            'r_mean': float(np.mean(r_channel)),
            'g_mean': float(np.mean(g_channel)),
            'b_mean': float(np.mean(b_channel)),
            'r_std': float(np.std(r_channel)),
            'g_std': float(np.std(g_channel)),
            'b_std': float(np.std(b_channel))
        }
        
        # 2. HSV analysis
        h_channel = img_hsv[:, :, 0]
        s_channel = img_hsv[:, :, 1]
        v_channel = img_hsv[:, :, 2]
        
        hsv_stats = {
            'hue_mean': float(np.mean(h_channel)),
            'saturation_mean': float(np.mean(s_channel)),
            'value_mean': float(np.mean(v_channel)),
            'hue_std': float(np.std(h_channel)),
            'saturation_std': float(np.std(s_channel)),
            'value_std': float(np.std(v_channel))
        }
        
        # 3. LAB analysis
        l_channel = img_lab[:, :, 0]
        a_channel = img_lab[:, :, 1]
        b_channel_lab = img_lab[:, :, 2]
        
        lab_stats = {
            'lightness_mean': float(np.mean(l_channel)),
            'a_mean': float(np.mean(a_channel)),
            'b_mean': float(np.mean(b_channel_lab)),
            'lightness_std': float(np.std(l_channel)),
            'a_std': float(np.std(a_channel)),
            'b_std': float(np.std(b_channel_lab))
        }
        
        # 4. Color harmony analysis
        # Calculate color distribution
        hue_hist = cv2.calcHist([h_channel], [0], None, [180], [0, 180])
        dominant_hues = np.argsort(hue_hist.flatten())[-5:]  # Top 5 hues
        
        # 5. Color temperature estimation
        r_mean = rgb_stats['r_mean']
        g_mean = rgb_stats['g_mean']
        b_mean = rgb_stats['b_mean']
        
        if b_mean > 0:
            color_temp_ratio = r_mean / b_mean
            color_temperature = 5500 - (color_temp_ratio - 1) * 1000
            color_temperature = max(2000, min(10000, color_temperature))
        else:
            color_temperature = 5500
        
        # 6. Color balance analysis
        white_balance_error = np.std([r_mean, g_mean, b_mean])
        
        # 7. Color saturation analysis
        saturation_regions = []
        h, w = img_hsv.shape[:2]
        for i in range(0, h, h//8):
            for j in range(0, w, w//8):
                region = s_channel[i:i+h//8, j:j+w//8]
                if region.size > 0:
                    saturation_regions.append(np.mean(region))
        
        return {
            'rgb_stats': rgb_stats,
            'hsv_stats': hsv_stats,
            'lab_stats': lab_stats,
            'color_temperature': float(color_temperature),
            'white_balance_error': float(white_balance_error),
            'dominant_hues': dominant_hues.tolist(),
            'saturation_uniformity': float(np.std(saturation_regions)) if saturation_regions else 0
        }
    
    def analyze_texture_detail(self, gray: np.ndarray) -> Dict[str, Any]:
        """Texture and detail analysis"""
        print("     🧩 Analyzing texture and detail...")
        
        # 1. Local Binary Pattern (simplified)
        def local_binary_pattern(img, radius=1):
            rows, cols = img.shape
            lbp = np.zeros_like(img)
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = img[i, j]
                    code = 0
                    for k in range(8):
                        x = i + int(radius * np.cos(2 * np.pi * k / 8))
                        y = j + int(radius * np.sin(2 * np.pi * k / 8))
                        if img[x, y] >= center:
                            code |= (1 << k)
                    lbp[i, j] = code
            return lbp
        
        # Sample small region for LBP to avoid performance issues
        h, w = gray.shape
        sample_region = gray[h//4:3*h//4:4, w//4:3*w//4:4]  # Downsample
        lbp = local_binary_pattern(sample_region)
        lbp_histogram = np.histogram(lbp.flatten(), bins=256)[0]
        lbp_uniformity = np.sum(lbp_histogram**2) / (np.sum(lbp_histogram)**2)
        
        # 2. Gabor filter responses
        gabor_responses = []
        for theta in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            gabor_responses.append(np.std(filtered))
        
        # 3. Texture energy
        texture_energy = np.var(gray)
        
        # 4. Edge density per region
        edges = cv2.Canny(gray, 50, 150)
        edge_regions = []
        for i in range(0, h, h//8):
            for j in range(0, w, w//8):
                region = edges[i:i+h//8, j:j+w//8]
                if region.size > 0:
                    edge_regions.append(np.sum(region) / region.size)
        
        return {
            'lbp_uniformity': float(lbp_uniformity),
            'gabor_responses': [float(x) for x in gabor_responses],
            'texture_energy': float(texture_energy),
            'edge_density_mean': float(np.mean(edge_regions)) if edge_regions else 0,
            'edge_density_std': float(np.std(edge_regions)) if edge_regions else 0
        }
    
    def analyze_exposure_advanced(self, gray: np.ndarray, img_rgb: np.ndarray) -> Dict[str, Any]:
        """Advanced exposure analysis"""
        print("     💡 Analyzing exposure...")
        
        # 1. Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / hist.sum()
        
        # 2. Exposure zones
        shadows = np.sum(hist_norm[:64])  # 0-25%
        midtones = np.sum(hist_norm[64:192])  # 25-75%
        highlights = np.sum(hist_norm[192:])  # 75-100%
        
        # 3. Clipping analysis
        underexposed = np.sum(hist_norm[:5])  # Pure black
        overexposed = np.sum(hist_norm[250:])  # Pure white
        
        # 4. Dynamic range
        dynamic_range = np.max(gray) - np.min(gray)
        
        # 5. Contrast analysis
        rms_contrast = np.sqrt(np.mean((gray - np.mean(gray))**2))
        michelson_contrast = (np.max(gray) - np.min(gray)) / (np.max(gray) + np.min(gray))
        
        # 6. Regional exposure analysis
        h, w = gray.shape
        exposure_regions = []
        for i in range(0, h, h//8):
            for j in range(0, w, w//8):
                region = gray[i:i+h//8, j:j+w//8]
                if region.size > 0:
                    exposure_regions.append(np.mean(region))
        
        exposure_uniformity = 1 - (np.std(exposure_regions) / 255) if exposure_regions else 0
        
        return {
            'shadows': float(shadows),
            'midtones': float(midtones),
            'highlights': float(highlights),
            'underexposed': float(underexposed),
            'overexposed': float(overexposed),
            'dynamic_range': float(dynamic_range),
            'rms_contrast': float(rms_contrast),
            'michelson_contrast': float(michelson_contrast),
            'exposure_uniformity': float(exposure_uniformity)
        }
    
    def analyze_composition_advanced(self, gray: np.ndarray, img_rgb: np.ndarray) -> Dict[str, Any]:
        """Advanced composition analysis"""
        print("     🎨 Analyzing composition...")
        
        h, w = gray.shape
        
        # 1. Rule of thirds analysis
        third_h, third_w = h // 3, w // 3
        regions = []
        for i in range(3):
            for j in range(3):
                region = gray[i*third_h:(i+1)*third_h, j*third_w:(j+1)*third_w]
                regions.append(np.mean(region))
        
        thirds_balance = 1 - (np.std(regions) / np.mean(regions))
        
        # 2. Center weight analysis
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_brightness = np.mean(center_region)
        edge_brightness = np.mean([
            np.mean(gray[:h//4, :]),
            np.mean(gray[3*h//4:, :]),
            np.mean(gray[:, :w//4]),
            np.mean(gray[:, 3*w//4:])
        ])
        
        center_emphasis = center_brightness - edge_brightness
        
        # 3. Symmetry analysis
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])
        
        if left_half.shape == right_half.shape:
            horizontal_symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        else:
            horizontal_symmetry = 0
        
        # 4. Leading lines detection
        lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi/180, 100, 
                               minLineLength=50, maxLineGap=10)
        
        line_count = len(lines) if lines is not None else 0
        
        return {
            'thirds_balance': float(thirds_balance),
            'center_emphasis': float(center_emphasis),
            'horizontal_symmetry': float(horizontal_symmetry) if not np.isnan(horizontal_symmetry) else 0,
            'line_count': int(line_count)
        }
    

def print_single_analysis(analysis: Dict[str, Any]):
    """Print single image analysis results"""
    info = analysis['image_info']
    camera = analysis['camera_info']
    metrics = analysis['advanced_metrics']
    scores = analysis['quality_scores']
    
    print(f"\n📊 ADVANCED ANALYSIS RESULTS")
    print("=" * 70)
    
    # Basic info
    print(f"📐 Image: {info['filename']}")
    print(f"📏 Size: {info['width']}x{info['height']} ({info['megapixels']} MP)")
    print(f"💾 File Size: {info['file_size_kb']} KB")
    print(f"📐 Aspect Ratio: {info['aspect_ratio']}")
    
    # Camera info
    if camera:
        print(f"\n📷 Camera Information:")
        for key, value in camera.items():
            if value and str(value) != "0" and str(value) != "0.0":
                print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    # Advanced Quality Scores
    print(f"\n🏆 Advanced Quality Scores:")
    for category, score in scores.items():
        if category != 'overall':
            status = get_status_emoji(score)
            print(f"   • {category.replace('_', ' ').title()}: {score:.1f}/100 {status}")
    
    print(f"\n⭐ Overall Score: {scores['overall']:.1f}/100")
    print(f"🎯 Grade: {analysis['grade']}")
    
    # Detailed Metrics
    print(f"\n🔍 Detailed Analysis:")
    
    # Sharpness details
    sharpness = metrics['sharpness']
    print(f"   📐 Sharpness:")
    print(f"      • Laplacian Variance: {sharpness['laplacian_variance']:.1f}")
    print(f"      • Gradient Magnitude: {sharpness['gradient_mean']:.1f}")
    print(f"      • Focus Measure: {sharpness['focus_measure']:.1f}")
    
    # Noise details
    noise = metrics['noise']
    print(f"   🔊 Noise Analysis:")
    print(f"      • Pixel Noise Level: {noise['pixel_noise_mean']:.2f}")
    print(f"      • SNR Mean: {noise['snr_mean']:.1f} dB")
    print(f"      • Color Noise (R/G/B): {noise['color_noise']['red']:.1f}/{noise['color_noise']['green']:.1f}/{noise['color_noise']['blue']:.1f}")
    
    # Depth of field
    depth = metrics['depth_of_field']
    print(f"   🎯 Depth of Field:")
    print(f"      • Depth Separation: {depth['depth_separation']:.1f}")
    print(f"      • Bokeh Count: {depth['bokeh_count']}")
    print(f"      • Bokeh Quality: {depth['bokeh_quality']:.1f}")
    
    # Objects and background
    objects = metrics['objects_background']
    print(f"   🎭 Objects & Background:")
    print(f"      • Object Count: {objects['contour_count']}")
    print(f"      • Background Uniformity: {objects['background_uniformity']:.2f}")
    print(f"      • Object Complexity: {objects['object_complexity']:.2f}")
    
    # Color analysis
    color = metrics['color_analysis']
    print(f"   🎨 Color Analysis:")
    print(f"      • Color Temperature: {color['color_temperature']:.0f}K")
    print(f"      • White Balance Error: {color['white_balance_error']:.1f}")
    print(f"      • Saturation Uniformity: {color['saturation_uniformity']:.2f}")
    
    # Texture
    texture = metrics['texture']
    print(f"   🧩 Texture Analysis:")
    print(f"      • Texture Energy: {texture['texture_energy']:.1f}")
    print(f"      • Edge Density: {texture['edge_density_mean']:.4f}")
    
    # Exposure
    exposure = metrics['exposure']
    print(f"   💡 Exposure Analysis:")
    print(f"      • Shadows/Midtones/Highlights: {exposure['shadows']:.2f}/{exposure['midtones']:.2f}/{exposure['highlights']:.2f}")
    print(f"      • Dynamic Range: {exposure['dynamic_range']:.1f}")
    print(f"      • Clipping (Under/Over): {exposure['underexposed']:.3f}/{exposure['overexposed']:.3f}")
    
    # Composition
    composition = metrics['composition']
    print(f"   🎨 Composition:")
    print(f"      • Rule of Thirds Balance: {composition['thirds_balance']:.2f}")
    print(f"      • Center Emphasis: {composition['center_emphasis']:.1f}")
    print(f"      • Leading Lines: {composition['line_count']}")


def print_comparison_results(comparison: Dict[str, Any]):
    """Print comparison results"""
    analysis1 = comparison['image1']
    analysis2 = comparison['image2']
    comp_result = comparison['comparison']
    
    img1_name = analysis1['image_info']['filename']
    img2_name = analysis2['image_info']['filename']
    
    print(f"\n🥊 COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"📸 Image 1: {img1_name}")
    print(f"📸 Image 2: {img2_name}")
    
    # Overall scores
    score1 = analysis1['quality_scores']['overall']
    score2 = analysis2['quality_scores']['overall']
    
    print(f"\n🏆 Overall Scores:")
    print(f"   {img1_name}: {score1:.1f}/100")
    print(f"   {img2_name}: {score2:.1f}/100")
    
    # Winner
    winner = comp_result['winner']
    margin = comp_result['margin']
    significance = comp_result['significance']
    
    if winner == 1:
        winner_name = img1_name
    elif winner == 2:
        winner_name = img2_name
    else:
        winner_name = "Tie"
    
    print(f"\n🥇 Winner: {winner_name}")
    if margin > 0:
        print(f"📊 Margin: {margin:.1f} points ({significance})")
    
    # Category breakdown
    print(f"\n📊 Category Breakdown:")
    for category, data in comp_result['category_comparison'].items():
        winner_cat = data['winner']
        diff = data['difference']
        
        if winner_cat == 1:
            winner_text = f"{img1_name} wins"
        elif winner_cat == 2:
            winner_text = f"{img2_name} wins"
        else:
            winner_text = "Tie"
        
        print(f"   • {category.title()}: {winner_text} ({diff:+.1f})")


def get_status_emoji(score: float) -> str:
    """Get emoji status based on score"""
    if score >= 90:
        return "🟢"
    elif score >= 80:
        return "🔵"
    elif score >= 70:
        return "🟡"
    elif score >= 60:
        return "🟠"
    else:
        return "🔴"


def save_analysis_json(analysis: Dict[str, Any], output_path: str):
    """Save analysis to JSON file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"💾 Analysis saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description='Professional Image Quality Analyzer v2.0.0 (Simplified)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py image1.jpg image2.jpg
  python main.py --analyze-single image.jpg
  python main.py --save-json image1.jpg image2.jpg
        """
    )
    
    parser.add_argument('images', nargs='*', help='Image paths (1 or 2 images)')
    parser.add_argument('--analyze-single', '-s', metavar='IMAGE', 
                       help='Analyze a single image')
    parser.add_argument('--save-json', '-j', action='store_true',
                       help='Save analysis results to JSON files')
    parser.add_argument('--version', '-v', action='version', version='2.0.0')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("🖼️  PROFESSIONAL IMAGE QUALITY ANALYZER v2.0.0 (ADVANCED)")
    print("=" * 80)
    print("Advanced pixel-level analysis with professional metrics")
    print("Features: Sharpness, Noise, Depth, Objects, Color, Texture, Exposure")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = SimpleImageAnalyzer()
    
    # Handle single image analysis
    if args.analyze_single:
        analysis = analyzer.analyze_image(args.analyze_single)
        if analysis:
            print_single_analysis(analysis)
            
            if args.save_json:
                json_path = f"{os.path.splitext(args.analyze_single)[0]}_analysis.json"
                save_analysis_json(analysis, json_path)
        return 0
    
    # Handle image arguments
    if len(args.images) == 1:
        # Single image analysis
        analysis = analyzer.analyze_image(args.images[0])
        if analysis:
            print_single_analysis(analysis)
            
            if args.save_json:
                json_path = f"{os.path.splitext(args.images[0])[0]}_analysis.json"
                save_analysis_json(analysis, json_path)
        
    elif len(args.images) == 2:
        # Two image comparison
        comparison = analyzer.compare_images(args.images[0], args.images[1])
        if comparison:
            print_comparison_results(comparison)
            
            if args.save_json:
                json_path = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_analysis_json(comparison, json_path)
        
    else:
        # No valid arguments
        print("❌ Please provide 1 or 2 image paths")
        print("\nUsage:")
        print("  python main.py image1.jpg image2.jpg")
        print("  python main.py --analyze-single image.jpg")
        print("  python main.py --help")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
