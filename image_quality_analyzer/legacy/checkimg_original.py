#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chương trình so sánh chất lượng ảnh chuyên nghiệp
Đánh giá ảnh chụp từ điện thoại với tất cả các tiêu chí có thể
Author: AI Assistant
Date: July 12, 2025
"""

import cv2
import numpy as np
import os
import sys
from PIL import Image, ImageStat, ExifTags
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, filters, feature, metrics
from skimage.restoration import estimate_sigma
import math
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class ImageQualityAnalyzer:
    """Lớp phân tích chất lượng ảnh chuyên nghiệp"""
    
    def __init__(self):
        self.results = {}
        
    def load_image(self, image_path: str) -> Tuple[np.ndarray, Image.Image]:
        """Tải ảnh và chuyển đổi định dạng"""
        try:
            # Tải bằng OpenCV
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                raise ValueError(f"Không thể tải ảnh: {image_path}")
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Tải bằng PIL để lấy EXIF
            img_pil = Image.open(image_path)
            
            return img_cv, img_pil
        except Exception as e:
            print(f"Lỗi khi tải ảnh {image_path}: {e}")
            return None, None
    
    def get_basic_info(self, img_cv: np.ndarray, img_pil: Image.Image) -> Dict[str, Any]:
        """Lấy thông tin cơ bản và EXIF của ảnh"""
        height, width = img_cv.shape[:2]
        channels = img_cv.shape[2] if len(img_cv.shape) > 2 else 1
        
        # Tính megapixels
        megapixels = (width * height) / 1000000
        
        # Tính tỷ lệ khung hình
        aspect_ratio = width / height
        
        # Lấy thông tin EXIF chi tiết
        exif_data = {}
        camera_info = {}
        
        if hasattr(img_pil, '_getexif') and img_pil._getexif():
            exif = img_pil._getexif()
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                exif_data[tag_name] = value
                
                # Trích xuất thông tin camera quan trọng
                if tag_name == 'Make':
                    camera_info['camera_make'] = str(value)
                elif tag_name == 'Model':
                    camera_info['camera_model'] = str(value)
                elif tag_name == 'FNumber':
                    camera_info['aperture'] = f"f/{float(value):.1f}"
                elif tag_name == 'ExposureTime':
                    if isinstance(value, tuple):
                        camera_info['shutter_speed'] = f"{value[0]}/{value[1]}s"
                    else:
                        camera_info['shutter_speed'] = f"{value}s"
                elif tag_name == 'ISOSpeedRatings':
                    camera_info['iso'] = value
                elif tag_name == 'FocalLength':
                    if isinstance(value, tuple):
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
                    modes = ['Unknown', 'Average', 'Center-weighted', 'Spot', 'Multi-spot', 'Pattern', 'Partial']
                    camera_info['metering_mode'] = modes[value] if value < len(modes) else 'Unknown'
                elif tag_name == 'DateTime':
                    camera_info['date_time'] = str(value)
                elif tag_name == 'ColorSpace':
                    camera_info['color_space'] = 'sRGB' if value == 1 else 'Adobe RGB'
        
        # Tính DPI nếu có
        dpi_info = {}
        if hasattr(img_pil, 'info') and 'dpi' in img_pil.info:
            dpi_info['dpi'] = img_pil.info['dpi']
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'megapixels': round(megapixels, 2),
            'aspect_ratio': round(aspect_ratio, 2),
            'file_size': os.path.getsize(img_pil.filename) if hasattr(img_pil, 'filename') else 0,
            'camera_info': camera_info,
            'dpi_info': dpi_info,
            'exif_data': exif_data
        }
    
    def calculate_sharpness(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Tính độ sắc nét của ảnh"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Laplacian variance (phương pháp chuẩn)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Sobel variance
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = np.sqrt(sobel_x**2 + sobel_y**2).var()
        
        # Brenner gradient
        brenner = np.sum((gray[2:, :] - gray[:-2, :])**2)
        
        # Tenengrad
        tenengrad = np.sum(sobel_x**2 + sobel_y**2)
        
        return {
            'laplacian_variance': round(laplacian_var, 2),
            'sobel_variance': round(sobel_var, 2),
            'brenner_gradient': round(brenner, 2),
            'tenengrad': round(tenengrad, 2)
        }
    
    def calculate_noise(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Tính mức độ nhiễu của ảnh"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Ước tính sigma noise bằng phương pháp Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sigma = np.sqrt(np.mean(laplacian**2)) / 6.0  # Hệ số 6 cho phân phối Laplacian
        
        # SNR (Signal-to-Noise Ratio)
        signal_power = np.mean(gray**2)
        noise_power = sigma**2
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # PSNR peak signal-to-noise ratio
        mse = np.mean((gray - np.mean(gray))**2)
        psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Thêm phương pháp ước tính nhiễu khác
        # Robust median estimator
        median_filter = cv2.medianBlur(gray, 5)
        noise_estimate = np.median(np.abs(gray - median_filter)) * 1.4826
        
        return {
            'noise_sigma': round(sigma, 4),
            'noise_estimate': round(noise_estimate, 4),
            'snr_db': round(snr, 2),
            'psnr_db': round(psnr, 2)
        }
    
    def calculate_brightness_contrast(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Tính độ sáng và độ tương phản"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Độ sáng trung bình
        brightness = np.mean(gray)
        
        # Độ tương phản RMS
        contrast_rms = np.sqrt(np.mean((gray - brightness)**2))
        
        # Độ tương phản Michelson
        max_val = np.max(gray)
        min_val = np.min(gray)
        michelson_contrast = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) > 0 else 0
        
        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        
        # Entropy
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
        
        return {
            'brightness': round(brightness, 2),
            'contrast_rms': round(contrast_rms, 2),
            'michelson_contrast': round(michelson_contrast, 4),
            'entropy': round(entropy, 2)
        }
    
    def calculate_color_quality(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Tính chất lượng màu sắc"""
        # Chuyển sang HSV
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
        
        # Độ bão hòa trung bình
        saturation_mean = np.mean(hsv[:,:,1])
        saturation_std = np.std(hsv[:,:,1])
        
        # Độ rực rỡ màu sắc
        vibrance = np.mean(np.max(img_cv, axis=2) - np.min(img_cv, axis=2))
        
        # Color cast detection
        r_mean = np.mean(img_cv[:,:,0])
        g_mean = np.mean(img_cv[:,:,1])
        b_mean = np.mean(img_cv[:,:,2])
        
        color_cast = np.std([r_mean, g_mean, b_mean])
        
        # Gamut coverage (ước tính)
        unique_colors = len(np.unique(img_cv.reshape(-1, 3), axis=0))
        gamut_coverage = unique_colors / (img_cv.shape[0] * img_cv.shape[1])
        
        return {
            'saturation_mean': round(saturation_mean, 2),
            'saturation_std': round(saturation_std, 2),
            'vibrance': round(vibrance, 2),
            'color_cast': round(color_cast, 2),
            'gamut_coverage': round(gamut_coverage, 4),
            'unique_colors': unique_colors
        }
    
    def calculate_exposure(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Tính chất lượng phơi sáng"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        
        # Underexposure (pixels < 5% of max)
        underexposed = np.sum(hist_norm[:13])  # 0-12 (5% of 255)
        
        # Overexposure (pixels > 95% of max)
        overexposed = np.sum(hist_norm[242:])  # 242-255 (95% of 255)
        
        # Well-exposed (middle range)
        well_exposed = np.sum(hist_norm[51:204])  # 20%-80% range
        
        # Dynamic range
        dynamic_range = np.max(gray) - np.min(gray)
        
        return {
            'underexposed_ratio': round(underexposed, 4),
            'overexposed_ratio': round(overexposed, 4),
            'well_exposed_ratio': round(well_exposed, 4),
            'dynamic_range': round(dynamic_range, 2)
        }
    
    def calculate_composition(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Tính chất lượng bố cục"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Rule of thirds
        # Chia ảnh thành 9 phần và tính độ cân bằng
        third_h, third_w = h // 3, w // 3
        
        regions = []
        for i in range(3):
            for j in range(3):
                region = gray[i*third_h:(i+1)*third_h, j*third_w:(j+1)*third_w]
                regions.append(np.mean(region))
        
        # Tính độ cân bằng
        balance = 1 - (np.std(regions) / np.mean(regions))
        
        # Edge density
        edges = feature.canny(gray, sigma=1)
        edge_density = np.sum(edges) / (h * w)
        
        # Symmetry analysis
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])
        min_width = min(left_half.shape[1], right_half.shape[1])
        symmetry = np.corrcoef(left_half[:, :min_width].flatten(), 
                              right_half[:, :min_width].flatten())[0, 1]
        
        return {
            'composition_balance': round(balance, 4),
            'edge_density': round(edge_density, 6),
            'symmetry': round(symmetry, 4) if not np.isnan(symmetry) else 0
        }
    
    def calculate_depth_of_field(self, img_cv: np.ndarray, camera_info: Dict) -> Dict[str, Any]:
        """Tính toán độ sâu trường ảnh và bokeh"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Phân tích độ mờ nền (bokeh quality)
        # Sử dụng gradient để tìm vùng nét và mờ
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Tính toán độ sâu trường ảnh
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        edge_region = np.concatenate([
            gray[:h//4, :].flatten(),
            gray[3*h//4:, :].flatten(),
            gray[:, :w//4].flatten(),
            gray[:, 3*w//4:].flatten()
        ])
        
        center_sharpness = np.var(center_region)
        edge_sharpness = np.var(edge_region)
        dof_ratio = center_sharpness / (edge_sharpness + 1e-6)
        
        # Phân tích bokeh
        # Tìm các vùng mờ có độ tương phản thấp
        blur_kernel = cv2.GaussianBlur(gray, (15, 15), 0)
        blur_diff = np.abs(gray.astype(float) - blur_kernel.astype(float))
        bokeh_quality = np.mean(blur_diff)
        
        # Tính toán focus peaking
        focus_mask = gradient_magnitude > np.percentile(gradient_magnitude, 80)
        focus_percentage = np.sum(focus_mask) / (h * w) * 100
        
        # Ước tính khẩu độ từ độ mờ nền
        estimated_aperture = None
        if 'aperture' in camera_info:
            estimated_aperture = camera_info['aperture']
        
        return {
            'dof_ratio': round(dof_ratio, 2),
            'bokeh_quality': round(bokeh_quality, 2),
            'focus_percentage': round(focus_percentage, 2),
            'estimated_aperture': estimated_aperture,
            'center_sharpness': round(center_sharpness, 2),
            'edge_sharpness': round(edge_sharpness, 2)
        }
    
    def calculate_color_science(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Phân tích khoa học màu sắc chuyên nghiệp"""
        # Chuyển sang các không gian màu khác nhau
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
        yuv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2YUV)
        
        # Phân tích Lab color space
        l_channel = lab[:, :, 0]  # Lightness
        a_channel = lab[:, :, 1]  # Green-Red
        b_channel = lab[:, :, 2]  # Blue-Yellow
        
        # Tính toán Color Temperature (ước tính)
        r_mean = np.mean(img_cv[:, :, 0])
        g_mean = np.mean(img_cv[:, :, 1])
        b_mean = np.mean(img_cv[:, :, 2])
        
        # Ước tính color temperature từ RGB ratio
        if b_mean > 0:
            color_temp_ratio = r_mean / b_mean
            estimated_color_temp = 5500 - (color_temp_ratio - 1) * 1000
            estimated_color_temp = max(2000, min(10000, estimated_color_temp))
        else:
            estimated_color_temp = 5500
        
        # Tính toán Tint
        tint = (g_mean - (r_mean + b_mean) / 2) / 255 * 100
        
        # Phân tích Color Grading
        shadows = np.mean(img_cv[img_cv < 85])
        midtones = np.mean(img_cv[(img_cv >= 85) & (img_cv < 170)])
        highlights = np.mean(img_cv[img_cv >= 170])
        
        # Tính toán Color Harmony
        hue_values = hsv[:, :, 0].flatten()
        hue_hist = np.histogram(hue_values, bins=36, range=(0, 180))[0]
        dominant_hues = np.argsort(hue_hist)[-3:]  # Top 3 dominant hues
        
        # Color Contrast trong Lab space
        a_contrast = np.std(a_channel)
        b_contrast = np.std(b_channel)
        
        # Skin tone analysis (nếu có)
        skin_tone_quality = self.analyze_skin_tones(img_cv)
        
        return {
            'estimated_color_temp': round(estimated_color_temp, 0),
            'tint': round(tint, 2),
            'shadows_avg': round(shadows, 2),
            'midtones_avg': round(midtones, 2),
            'highlights_avg': round(highlights, 2),
            'lab_a_contrast': round(a_contrast, 2),
            'lab_b_contrast': round(b_contrast, 2),
            'dominant_hues': dominant_hues.tolist(),
            'skin_tone_quality': skin_tone_quality
        }
    
    def analyze_skin_tones(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Phân tích chất lượng skin tone"""
        # Chuyển sang YCrCb để phát hiện da
        ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_RGB2YCrCb)
        
        # Skin tone detection trong YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        skin_percentage = np.sum(skin_mask > 0) / (img_cv.shape[0] * img_cv.shape[1]) * 100
        
        if skin_percentage > 1:  # Có skin tone
            # Phân tích chất lượng skin tone
            skin_pixels = img_cv[skin_mask > 0]
            if len(skin_pixels) > 0:
                skin_r = np.mean(skin_pixels[:, 0])
                skin_g = np.mean(skin_pixels[:, 1])
                skin_b = np.mean(skin_pixels[:, 2])
                
                # Tính toán skin tone naturalness
                skin_ratio_rg = skin_r / (skin_g + 1e-6)
                skin_ratio_rb = skin_r / (skin_b + 1e-6)
                
                # Ideal skin tone ratios
                ideal_rg = 1.15
                ideal_rb = 1.2
                
                naturalness = 100 - abs(skin_ratio_rg - ideal_rg) * 50 - abs(skin_ratio_rb - ideal_rb) * 50
                naturalness = max(0, min(100, naturalness))
                
                return {
                    'skin_percentage': round(skin_percentage, 2),
                    'skin_naturalness': round(naturalness, 2),
                    'skin_r_avg': round(skin_r, 2),
                    'skin_g_avg': round(skin_g, 2),
                    'skin_b_avg': round(skin_b, 2)
                }
        
        return {
            'skin_percentage': round(skin_percentage, 2),
            'skin_naturalness': 0,
            'skin_r_avg': 0,
            'skin_g_avg': 0,
            'skin_b_avg': 0
        }
    
    def calculate_professional_metrics(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Tính toán các chỉ số chuyên nghiệp"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Tính toán MTF (Modulation Transfer Function)
        # Sử dụng edge response để ước tính MTF
        edges = cv2.Canny(gray, 50, 150)
        edge_response = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        
        # Tính toán Chromatic Aberration
        r_channel = img_cv[:, :, 0]
        g_channel = img_cv[:, :, 1]
        b_channel = img_cv[:, :, 2]
        
        # Tính correlation giữa các channel
        rg_correlation = np.corrcoef(r_channel.flatten(), g_channel.flatten())[0, 1]
        rb_correlation = np.corrcoef(r_channel.flatten(), b_channel.flatten())[0, 1]
        gb_correlation = np.corrcoef(g_channel.flatten(), b_channel.flatten())[0, 1]
        
        chromatic_aberration = 100 - np.mean([rg_correlation, rb_correlation, gb_correlation]) * 100
        
        # Tính toán Vignetting
        h, w = gray.shape
        center = (h // 2, w // 2)
        
        # Tạo mask hình tròn từ center
        Y, X = np.ogrid[:h, :w]
        center_mask = (X - center[1]) ** 2 + (Y - center[0]) ** 2 <= (min(h, w) // 4) ** 2
        edge_mask = (X - center[1]) ** 2 + (Y - center[0]) ** 2 >= (min(h, w) // 3) ** 2
        
        center_brightness = np.mean(gray[center_mask])
        edge_brightness = np.mean(gray[edge_mask])
        
        vignetting = (center_brightness - edge_brightness) / center_brightness * 100
        
        # Tính toán Distortion (barrel/pincushion)
        # Sử dụng grid analysis
        distortion_score = self.calculate_distortion(gray)
        
        # Tính toán Noise pattern analysis
        noise_pattern = self.analyze_noise_pattern(gray)
        
        return {
            'mtf_estimate': round(edge_response * 1000, 2),
            'chromatic_aberration': round(chromatic_aberration, 2),
            'vignetting': round(vignetting, 2),
            'distortion_score': round(distortion_score, 2),
            'noise_pattern': noise_pattern
        }
    
    def calculate_distortion(self, gray: np.ndarray) -> float:
        """Tính toán distortion"""
        h, w = gray.shape
        
        # Tìm edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Tìm lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Phân tích độ cong của các đường thẳng
            line_angles = []
            for line in lines:
                rho, theta = line[0]
                line_angles.append(theta)
            
            # Tính độ lệch chuẩn của các góc
            angle_std = np.std(line_angles)
            distortion = angle_std * 180 / np.pi
            return min(distortion, 10)  # Cap at 10 degrees
        
        return 0
    
    def analyze_noise_pattern(self, gray: np.ndarray) -> Dict[str, float]:
        """Phân tích pattern nhiễu"""
        # High frequency analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Tính toán noise characteristics
        h, w = gray.shape
        center = (h // 2, w // 2)
        
        # High frequency noise
        high_freq_mask = np.zeros((h, w))
        cv2.circle(high_freq_mask, center, min(h, w) // 4, 1, -1)
        high_freq_noise = np.mean(magnitude_spectrum[high_freq_mask == 0])
        
        # Banding analysis
        row_variance = np.var(np.mean(gray, axis=1))
        col_variance = np.var(np.mean(gray, axis=0))
        banding_score = (row_variance + col_variance) / 2
        
        return {
            'high_freq_noise': round(high_freq_noise, 2),
            'banding_score': round(banding_score, 2),
            'row_variance': round(row_variance, 2),
            'col_variance': round(col_variance, 2)
        }
    
    def calculate_overall_score(self, analysis: Dict[str, Any]) -> float:
        """Tính điểm tổng kết chung với nhiều tiêu chí hơn"""
        scores = {}
        
        # Điểm độ sắc nét (0-100)
        sharpness_score = min(100, analysis['sharpness']['laplacian_variance'] / 10)
        scores['sharpness'] = sharpness_score
        
        # Điểm nhiễu (0-100, càng ít nhiễu càng tốt)
        noise_score = min(100, max(0, analysis['noise']['snr_db'] - 10) * 2)
        scores['noise'] = noise_score
        
        # Điểm độ tương phản (0-100)
        contrast_score = min(100, analysis['brightness_contrast']['contrast_rms'] / 2)
        scores['contrast'] = contrast_score
        
        # Điểm màu sắc (0-100)
        color_score = min(100, analysis['color']['saturation_mean'] / 2.55)
        scores['color'] = color_score
        
        # Điểm phơi sáng (0-100)
        exposure_score = analysis['exposure']['well_exposed_ratio'] * 100
        scores['exposure'] = exposure_score
        
        # Điểm bố cục (0-100)
        composition_score = analysis['composition']['composition_balance'] * 100
        scores['composition'] = composition_score
        
        # Điểm độ sâu trường ảnh (0-100)
        dof_score = min(100, analysis['depth_of_field']['bokeh_quality'] / 2)
        scores['depth_of_field'] = dof_score
        
        # Điểm khoa học màu sắc (0-100)
        color_science_score = min(100, analysis['color_science']['skin_tone_quality']['skin_naturalness'])
        scores['color_science'] = color_science_score
        
        # Điểm chuyên nghiệp (0-100)
        professional_score = 100 - min(50, analysis['professional_metrics']['chromatic_aberration'])
        scores['professional'] = professional_score
        
        # Điểm tổng (trọng số cập nhật)
        weights = {
            'sharpness': 0.20,
            'noise': 0.15,
            'contrast': 0.12,
            'color': 0.12,
            'exposure': 0.12,
            'composition': 0.08,
            'depth_of_field': 0.08,
            'color_science': 0.08,
            'professional': 0.05
        }
        
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        return {
            'individual_scores': scores,
            'total_score': round(total_score, 2),
            'grade': self.get_grade(total_score)
        }
    
    def get_grade(self, score: float) -> str:
        """Chuyển điểm thành xếp hạng"""
        if score >= 90:
            return "Xuất sắc (A+)"
        elif score >= 80:
            return "Tốt (A)"
        elif score >= 70:
            return "Khá (B+)"
        elif score >= 60:
            return "Trung bình khá (B)"
        elif score >= 50:
            return "Trung bình (C)"
        else:
            return "Cần cải thiện (D)"
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Phân tích toàn diện một ảnh"""
        print(f"\n🔍 Đang phân tích ảnh: {os.path.basename(image_path)}")
        
        img_cv, img_pil = self.load_image(image_path)
        if img_cv is None:
            return None
        
        analysis = {}
        
        # Thông tin cơ bản
        analysis['basic_info'] = self.get_basic_info(img_cv, img_pil)
        
        # Độ sắc nét
        analysis['sharpness'] = self.calculate_sharpness(img_cv)
        
        # Nhiễu
        analysis['noise'] = self.calculate_noise(img_cv)
        
        # Độ sáng và tương phản
        analysis['brightness_contrast'] = self.calculate_brightness_contrast(img_cv)
        
        # Chất lượng màu sắc
        analysis['color'] = self.calculate_color_quality(img_cv)
        
        # Phơi sáng
        analysis['exposure'] = self.calculate_exposure(img_cv)
        
        # Bố cục
        analysis['composition'] = self.calculate_composition(img_cv)
        
        # Độ sâu trường ảnh và bokeh
        analysis['depth_of_field'] = self.calculate_depth_of_field(img_cv, analysis['basic_info']['camera_info'])
        
        # Khoa học màu sắc
        analysis['color_science'] = self.calculate_color_science(img_cv)
        
        # Chỉ số chuyên nghiệp
        analysis['professional_metrics'] = self.calculate_professional_metrics(img_cv)
        
        # Điểm tổng kết
        analysis['overall_score'] = self.calculate_overall_score(analysis)
        
        return analysis
    
    def compare_images(self, image1_path: str, image2_path: str):
        """So sánh hai ảnh và đưa ra kết luận"""
        print("=" * 80)
        print("🖼️  CHƯƠNG TRÌNH SO SÁNH CHẤT LƯỢNG ẢNH CHUYÊN NGHIỆP")
        print("=" * 80)
        
        # Phân tích ảnh 1
        analysis1 = self.analyze_image(image1_path)
        if analysis1 is None:
            return
        
        # Phân tích ảnh 2
        analysis2 = self.analyze_image(image2_path)
        if analysis2 is None:
            return
        
        # In kết quả chi tiết
        self.print_detailed_results(image1_path, analysis1, image2_path, analysis2)
        
        # In kết quả so sánh
        self.print_comparison_results(image1_path, analysis1, image2_path, analysis2)
    
    def print_detailed_results(self, img1_path: str, analysis1: Dict, img2_path: str, analysis2: Dict):
        """In kết quả chi tiết cho từng ảnh"""
        
        for i, (path, analysis) in enumerate([(img1_path, analysis1), (img2_path, analysis2)], 1):
            print(f"\n📊 KẾT QUẢ PHÂN TÍCH CHI TIẾT - ẢNH {i}: {os.path.basename(path)}")
            print("-" * 70)
            
            # Thông tin cơ bản
            basic = analysis['basic_info']
            camera = basic['camera_info']
            print(f"📐 Thông tin cơ bản:")
            print(f"   • Kích thước: {basic['width']}x{basic['height']} ({basic['megapixels']} MP)")
            print(f"   • Tỷ lệ khung hình: {basic['aspect_ratio']}")
            print(f"   • Kênh màu: {basic['channels']}")
            print(f"   • Kích thước file: {basic['file_size'] / 1024:.1f} KB")
            
            # Thông tin camera
            if camera:
                print(f"\n📷 Thông tin camera:")
                if 'camera_make' in camera:
                    print(f"   • Hãng: {camera['camera_make']}")
                if 'camera_model' in camera:
                    print(f"   • Model: {camera['camera_model']}")
                if 'aperture' in camera:
                    print(f"   • Khẩu độ: {camera['aperture']}")
                if 'shutter_speed' in camera:
                    print(f"   • Tốc độ màn trập: {camera['shutter_speed']}")
                if 'iso' in camera:
                    print(f"   • ISO: {camera['iso']}")
                if 'focal_length' in camera:
                    print(f"   • Tiêu cự: {camera['focal_length']}")
                if 'white_balance' in camera:
                    print(f"   • White Balance: {camera['white_balance']}")
                if 'flash' in camera:
                    print(f"   • Flash: {camera['flash']}")
                if 'exposure_mode' in camera:
                    print(f"   • Chế độ phơi sáng: {camera['exposure_mode']}")
                if 'metering_mode' in camera:
                    print(f"   • Chế độ đo sáng: {camera['metering_mode']}")
                if 'color_space' in camera:
                    print(f"   • Không gian màu: {camera['color_space']}")
            
            # Độ sắc nét
            sharpness = analysis['sharpness']
            print(f"\n🔍 Độ sắc nét:")
            print(f"   • Laplacian Variance: {sharpness['laplacian_variance']}")
            print(f"   • Sobel Variance: {sharpness['sobel_variance']}")
            print(f"   • Brenner Gradient: {sharpness['brenner_gradient']}")
            print(f"   • Tenengrad: {sharpness['tenengrad']}")
            
            # Nhiễu
            noise = analysis['noise']
            print(f"\n📡 Chất lượng tín hiệu:")
            print(f"   • Mức nhiễu (Laplacian): {noise['noise_sigma']}")
            print(f"   • Mức nhiễu (Median): {noise['noise_estimate']}")
            print(f"   • SNR: {noise['snr_db']} dB")
            print(f"   • PSNR: {noise['psnr_db']} dB")
            
            # Độ sáng và tương phản
            brightness = analysis['brightness_contrast']
            print(f"\n💡 Độ sáng & Tương phản:")
            print(f"   • Độ sáng trung bình: {brightness['brightness']}")
            print(f"   • Độ tương phản RMS: {brightness['contrast_rms']}")
            print(f"   • Độ tương phản Michelson: {brightness['michelson_contrast']}")
            print(f"   • Entropy: {brightness['entropy']}")
            
            # Màu sắc
            color = analysis['color']
            print(f"\n🎨 Chất lượng màu sắc:")
            print(f"   • Độ bão hòa TB: {color['saturation_mean']}")
            print(f"   • Độ lệch chuẩn bão hòa: {color['saturation_std']}")
            print(f"   • Độ rực rỡ: {color['vibrance']}")
            print(f"   • Độ lệch màu: {color['color_cast']}")
            print(f"   • Số màu độc nhất: {color['unique_colors']}")
            
            # Phơi sáng
            exposure = analysis['exposure']
            print(f"\n📸 Chất lượng phơi sáng:")
            print(f"   • Tỷ lệ thiếu sáng: {exposure['underexposed_ratio']:.2%}")
            print(f"   • Tỷ lệ quá sáng: {exposure['overexposed_ratio']:.2%}")
            print(f"   • Tỷ lệ phơi sáng tốt: {exposure['well_exposed_ratio']:.2%}")
            print(f"   • Dải động: {exposure['dynamic_range']}")
            
            # Bố cục
            composition = analysis['composition']
            print(f"\n🖼️ Chất lượng bố cục:")
            print(f"   • Cân bằng bố cục: {composition['composition_balance']:.3f}")
            print(f"   • Mật độ cạnh: {composition['edge_density']:.6f}")
            print(f"   • Tính đối xứng: {composition['symmetry']:.3f}")
            
            # Độ sâu trường ảnh
            dof = analysis['depth_of_field']
            print(f"\n🎯 Độ sâu trường ảnh & Bokeh:")
            print(f"   • Tỷ lệ DOF: {dof['dof_ratio']}")
            print(f"   • Chất lượng bokeh: {dof['bokeh_quality']}")
            print(f"   • Phần trăm vùng nét: {dof['focus_percentage']:.1f}%")
            if dof['estimated_aperture']:
                print(f"   • Khẩu độ ước tính: {dof['estimated_aperture']}")
            print(f"   • Độ nét trung tâm: {dof['center_sharpness']}")
            print(f"   • Độ nét viền: {dof['edge_sharpness']}")
            
            # Khoa học màu sắc
            color_sci = analysis['color_science']
            print(f"\n🔬 Khoa học màu sắc:")
            print(f"   • Nhiệt độ màu ước tính: {color_sci['estimated_color_temp']:.0f}K")
            print(f"   • Tint: {color_sci['tint']:.2f}")
            print(f"   • Shadows trung bình: {color_sci['shadows_avg']}")
            print(f"   • Midtones trung bình: {color_sci['midtones_avg']}")
            print(f"   • Highlights trung bình: {color_sci['highlights_avg']}")
            print(f"   • Lab A contrast: {color_sci['lab_a_contrast']}")
            print(f"   • Lab B contrast: {color_sci['lab_b_contrast']}")
            
            # Skin tone analysis
            skin = color_sci['skin_tone_quality']
            if skin['skin_percentage'] > 1:
                print(f"\n👤 Phân tích skin tone:")
                print(f"   • Phần trăm skin: {skin['skin_percentage']:.1f}%")
                print(f"   • Tự nhiên skin: {skin['skin_naturalness']:.1f}/100")
                print(f"   • Skin R/G/B: {skin['skin_r_avg']:.1f}/{skin['skin_g_avg']:.1f}/{skin['skin_b_avg']:.1f}")
            
            # Chỉ số chuyên nghiệp
            pro = analysis['professional_metrics']
            print(f"\n🔧 Chỉ số chuyên nghiệp:")
            print(f"   • MTF ước tính: {pro['mtf_estimate']}")
            print(f"   • Chromatic Aberration: {pro['chromatic_aberration']:.2f}")
            print(f"   • Vignetting: {pro['vignetting']:.2f}%")
            print(f"   • Distortion: {pro['distortion_score']:.2f}°")
            
            # Noise pattern
            noise_pattern = pro['noise_pattern']
            print(f"\n🎭 Pattern nhiễu:")
            print(f"   • High freq noise: {noise_pattern['high_freq_noise']:.2f}")
            print(f"   • Banding score: {noise_pattern['banding_score']:.2f}")
            
            # Điểm tổng kết
            score = analysis['overall_score']
            print(f"\n⭐ ĐIỂM TỔNG KẾT:")
            print(f"   • Độ sắc nét: {score['individual_scores']['sharpness']:.1f}/100")
            print(f"   • Chất lượng tín hiệu: {score['individual_scores']['noise']:.1f}/100")
            print(f"   • Tương phản: {score['individual_scores']['contrast']:.1f}/100")
            print(f"   • Màu sắc: {score['individual_scores']['color']:.1f}/100")
            print(f"   • Phơi sáng: {score['individual_scores']['exposure']:.1f}/100")
            print(f"   • Bố cục: {score['individual_scores']['composition']:.1f}/100")
            print(f"   • Độ sâu trường ảnh: {score['individual_scores']['depth_of_field']:.1f}/100")
            print(f"   • Khoa học màu sắc: {score['individual_scores']['color_science']:.1f}/100")
            print(f"   • Chuyên nghiệp: {score['individual_scores']['professional']:.1f}/100")
            print(f"\n🏆 ĐIỂM TỔNG: {score['total_score']}/100 - {score['grade']}")
    
    def print_comparison_results(self, img1_path: str, analysis1: Dict, img2_path: str, analysis2: Dict):
        """In kết quả so sánh giữa hai ảnh"""
        print("\n" + "=" * 80)
        print("🥊 KẾT QUẢ SO SÁNH GIỮA HAI ẢNH")
        print("=" * 80)
        
        img1_name = os.path.basename(img1_path)
        img2_name = os.path.basename(img2_path)
        
        score1 = analysis1['overall_score']['total_score']
        score2 = analysis2['overall_score']['total_score']
        
        # So sánh tổng thể
        print(f"\n🏆 ĐIỂM TỔNG KẾT:")
        print(f"   📸 {img1_name}: {score1}/100")
        print(f"   📸 {img2_name}: {score2}/100")
        
        if score1 > score2:
            winner = img1_name
            diff = score1 - score2
        elif score2 > score1:
            winner = img2_name  
            diff = score2 - score1
        else:
            winner = "Hòa"
            diff = 0
        
        print(f"\n🥇 NGƯỜI THẮNG: {winner}")
        if diff > 0:
            print(f"   📊 Cách biệt: {diff:.1f} điểm")
        
        # So sánh chi tiết từng tiêu chí
        print(f"\n📊 SO SÁNH CHI TIẾT:")
        
        categories = ['sharpness', 'noise', 'contrast', 'color', 'exposure', 'composition', 'depth_of_field', 'color_science', 'professional']
        category_names = ['Độ sắc nét', 'Chất lượng tín hiệu', 'Tương phản', 'Màu sắc', 'Phơi sáng', 'Bố cục', 'Độ sâu trường ảnh', 'Khoa học màu sắc', 'Chuyên nghiệp']
        
        for cat, name in zip(categories, category_names):
            score1_cat = analysis1['overall_score']['individual_scores'][cat]
            score2_cat = analysis2['overall_score']['individual_scores'][cat]
            
            print(f"\n   🔸 {name}:")
            print(f"      {img1_name}: {score1_cat:.1f}/100")
            print(f"      {img2_name}: {score2_cat:.1f}/100")
            
            if score1_cat > score2_cat:
                print(f"      ✅ {img1_name} thắng (+{score1_cat - score2_cat:.1f})")
            elif score2_cat > score1_cat:
                print(f"      ✅ {img2_name} thắng (+{score2_cat - score1_cat:.1f})")
            else:
                print(f"      🤝 Hòa")
        
        # Khuyến nghị
        print(f"\n💡 KHUYẾN NGHỊ:")
        if diff > 10:
            print(f"   • Ảnh {winner} có chất lượng vượt trội hơn rõ rệt")
        elif diff > 5:
            print(f"   • Ảnh {winner} có chất lượng tốt hơn đáng kể")
        elif diff > 0:
            print(f"   • Ảnh {winner} có chất lượng tốt hơn một chút")
        else:
            print(f"   • Hai ảnh có chất lượng tương đương nhau")
        
        # Phân tích điểm mạnh/yếu
        print(f"\n📈 PHÂN TÍCH ĐIỂM MẠNH/YẾU:")
        
        for i, (name, analysis) in enumerate([(img1_name, analysis1), (img2_name, analysis2)], 1):
            scores = analysis['overall_score']['individual_scores']
            print(f"\n   📸 {name}:")
            
            # Tìm điểm mạnh nhất
            best_category = max(scores, key=scores.get)
            best_score = scores[best_category]
            best_name = dict(zip(categories, category_names))[best_category]
            
            # Tìm điểm yếu nhất
            worst_category = min(scores, key=scores.get)
            worst_score = scores[worst_category]
            worst_name = dict(zip(categories, category_names))[worst_category]
            
            print(f"      ✅ Điểm mạnh: {best_name} ({best_score:.1f}/100)")
            print(f"      ⚠️  Điểm yếu: {worst_name} ({worst_score:.1f}/100)")


def main():
    """Hàm chính của chương trình"""
    if len(sys.argv) != 3:
        print("Cách sử dụng: python checkimg.py <đường_dẫn_ảnh_1> <đường_dẫn_ảnh_2>")
        print("Ví dụ: python checkimg.py image1.jpg image2.jpg")
        return
    
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    
    # Kiểm tra file tồn tại
    if not os.path.exists(image1_path):
        print(f"❌ Không tìm thấy ảnh: {image1_path}")
        return
    
    if not os.path.exists(image2_path):
        print(f"❌ Không tìm thấy ảnh: {image2_path}")
        return
    
    # Khởi tạo analyzer và so sánh
    analyzer = ImageQualityAnalyzer()
    analyzer.compare_images(image1_path, image2_path)


if __name__ == "__main__":
    main()