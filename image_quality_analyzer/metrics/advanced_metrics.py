"""
Advanced Image Quality Metrics
==============================

Advanced metrics for professional image quality assessment including:
- Perceptual metrics (SSIM, MS-SSIM)
- Bokeh quality analysis
- Texture analysis
- Color science metrics
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from scipy import ndimage, signal, stats
from skimage import filters, feature, metrics
from skimage.feature import greycomatrix, greycoprops
import warnings
warnings.filterwarnings('ignore')


class AdvancedMetrics:
    """Advanced image quality metrics"""
    
    def __init__(self):
        self.standard_illuminants = {
            'D65': [0.3127, 0.3290],  # Daylight 6500K
            'A': [0.4476, 0.4074],    # Incandescent 2856K
            'F2': [0.3721, 0.3751]    # Fluorescent
        }
    
    def calculate_perceptual_metrics(self, img_cv: np.ndarray) -> Dict[str, float]:
        """
        Calculate perceptual quality metrics
        
        Args:
            img_cv: Input image array
            
        Returns:
            Dictionary containing perceptual metrics
        """
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Create reference image (slightly blurred version)
        reference = cv2.GaussianBlur(gray, (3, 3), 0.5)
        
        # SSIM (Structural Similarity Index)
        ssim_score = metrics.structural_similarity(gray, reference, data_range=255)
        
        # PSNR (Peak Signal-to-Noise Ratio)
        psnr_score = metrics.peak_signal_noise_ratio(gray, reference, data_range=255)
        
        # Calculate gradient similarity
        grad_sim = self._calculate_gradient_similarity(gray, reference)
        
        # Calculate texture similarity
        texture_sim = self._calculate_texture_similarity(gray, reference)
        
        # Perceptual sharpness
        perceptual_sharpness = self._calculate_perceptual_sharpness(gray)
        
        return {
            'ssim': round(ssim_score, 4),
            'psnr': round(psnr_score, 2),
            'gradient_similarity': round(grad_sim, 4),
            'texture_similarity': round(texture_sim, 4),
            'perceptual_sharpness': round(perceptual_sharpness, 3)
        }
    
    def _calculate_gradient_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate gradient similarity between two images"""
        # Calculate gradients
        grad1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        grad2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitudes
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
        
        # Calculate correlation
        correlation = np.corrcoef(grad1_mag.flatten(), grad2_mag.flatten())[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_texture_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate texture similarity using LBP"""
        # Calculate Local Binary Patterns
        lbp1 = feature.local_binary_pattern(img1, 24, 8, method='uniform')
        lbp2 = feature.local_binary_pattern(img2, 24, 8, method='uniform')
        
        # Calculate histograms
        hist1 = np.histogram(lbp1, bins=26, range=(0, 26))[0]
        hist2 = np.histogram(lbp2, bins=26, range=(0, 26))[0]
        
        # Normalize histograms
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # Calculate correlation
        correlation = np.corrcoef(hist1, hist2)[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_perceptual_sharpness(self, gray: np.ndarray) -> float:
        """Calculate perceptual sharpness"""
        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate perceptual sharpness
        sharpness = np.var(laplacian)
        
        return sharpness
    
    def calculate_bokeh_quality(self, img_cv: np.ndarray) -> Dict[str, float]:
        """
        Analyze bokeh quality in detail
        
        Args:
            img_cv: Input image array
            
        Returns:
            Dictionary containing bokeh quality metrics
        """
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Detect out-of-focus areas
        blur_map = self._create_blur_map(gray)
        
        # Analyze bokeh characteristics
        bokeh_smoothness = self._calculate_bokeh_smoothness(gray, blur_map)
        bokeh_shape = self._analyze_bokeh_shape(gray, blur_map)
        depth_transition = self._calculate_depth_transition(gray)
        
        # Calculate focus peaking
        focus_peaking = self._calculate_focus_peaking(gray)
        
        # Analyze depth of field
        dof_analysis = self._analyze_depth_of_field(gray)
        
        return {
            'bokeh_smoothness': round(bokeh_smoothness, 3),
            'bokeh_shape_quality': round(bokeh_shape, 3),
            'depth_transition': round(depth_transition, 3),
            'focus_peaking': round(focus_peaking, 3),
            'dof_ratio': round(dof_analysis['dof_ratio'], 3),
            'background_blur': round(dof_analysis['background_blur'], 3)
        }
    
    def _create_blur_map(self, gray: np.ndarray) -> np.ndarray:
        """Create blur map of the image"""
        # Calculate local variance
        kernel = np.ones((9, 9), np.float32) / 81
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        # Normalize
        blur_map = 1 / (1 + local_variance)
        
        return blur_map
    
    def _calculate_bokeh_smoothness(self, gray: np.ndarray, blur_map: np.ndarray) -> float:
        """Calculate bokeh smoothness"""
        # Find blurred regions
        blur_threshold = np.percentile(blur_map, 80)
        blurred_regions = blur_map > blur_threshold
        
        if np.sum(blurred_regions) > 0:
            # Calculate smoothness in blurred regions
            blurred_pixels = gray[blurred_regions]
            smoothness = 1 / (1 + np.var(blurred_pixels))
        else:
            smoothness = 0.5
        
        return smoothness
    
    def _analyze_bokeh_shape(self, gray: np.ndarray, blur_map: np.ndarray) -> float:
        """Analyze bokeh shape quality"""
        # Find potential bokeh balls (bright spots in blurred areas)
        blur_threshold = np.percentile(blur_map, 85)
        bright_threshold = np.percentile(gray, 90)
        
        bokeh_candidates = (blur_map > blur_threshold) & (gray > bright_threshold)
        
        if np.sum(bokeh_candidates) > 0:
            # Analyze shape quality (circular vs. polygonal)
            contours, _ = cv2.findContours(bokeh_candidates.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            shape_scores = []
            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Filter small contours
                    # Calculate circularity
                    perimeter = cv2.arcLength(contour, True)
                    area = cv2.contourArea(contour)
                    
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        shape_scores.append(circularity)
            
            if shape_scores:
                return np.mean(shape_scores)
        
        return 0.5
    
    def _calculate_depth_transition(self, gray: np.ndarray) -> float:
        """Calculate depth transition quality"""
        # Calculate gradient of focus
        focus_gradient = self._calculate_focus_gradient(gray)
        
        # Smooth transitions have gradual focus changes
        transition_quality = 1 / (1 + np.std(focus_gradient))
        
        return transition_quality
    
    def _calculate_focus_gradient(self, gray: np.ndarray) -> np.ndarray:
        """Calculate focus gradient map"""
        # Calculate local sharpness
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        sharpness = cv2.filter2D(gray, -1, kernel)
        
        # Calculate gradient of sharpness
        grad_x = cv2.Sobel(sharpness, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(sharpness, cv2.CV_64F, 0, 1, ksize=3)
        
        focus_gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        return focus_gradient
    
    def _calculate_focus_peaking(self, gray: np.ndarray) -> float:
        """Calculate focus peaking percentage"""
        # Calculate high frequency content
        high_freq = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Threshold for focus peaking
        threshold = np.percentile(np.abs(high_freq), 90)
        focus_mask = np.abs(high_freq) > threshold
        
        # Calculate percentage
        focus_percentage = np.sum(focus_mask) / focus_mask.size * 100
        
        return focus_percentage
    
    def _analyze_depth_of_field(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze depth of field characteristics"""
        h, w = gray.shape
        
        # Define regions
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        edge_regions = np.concatenate([
            gray[:h//4, :].flatten(),
            gray[3*h//4:, :].flatten(),
            gray[:, :w//4].flatten(),
            gray[:, 3*w//4:].flatten()
        ])
        
        # Calculate sharpness in each region
        center_sharpness = np.var(cv2.Laplacian(center_region, cv2.CV_64F))
        edge_sharpness = np.var(cv2.Laplacian(edge_regions.reshape(-1, 1), cv2.CV_64F))
        
        # Calculate DOF ratio
        dof_ratio = center_sharpness / (edge_sharpness + 1e-6)
        
        # Calculate background blur
        background_blur = 1 / (1 + edge_sharpness)
        
        return {
            'dof_ratio': dof_ratio,
            'background_blur': background_blur
        }
    
    def calculate_texture_analysis(self, img_cv: np.ndarray) -> Dict[str, float]:
        """
        Perform detailed texture analysis
        
        Args:
            img_cv: Input image array
            
        Returns:
            Dictionary containing texture metrics
        """
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # GLCM (Gray-Level Co-occurrence Matrix) features
        glcm_features = self._calculate_glcm_features(gray)
        
        # Gabor filter responses
        gabor_responses = self._calculate_gabor_responses(gray)
        
        # Local Binary Pattern analysis
        lbp_features = self._calculate_lbp_features(gray)
        
        # Fractal dimension
        fractal_dimension = self._calculate_fractal_dimension(gray)
        
        # Texture energy
        texture_energy = self._calculate_texture_energy(gray)
        
        return {
            'glcm_contrast': round(glcm_features['contrast'], 3),
            'glcm_correlation': round(glcm_features['correlation'], 3),
            'glcm_energy': round(glcm_features['energy'], 3),
            'glcm_homogeneity': round(glcm_features['homogeneity'], 3),
            'gabor_mean_response': round(gabor_responses['mean_response'], 3),
            'gabor_std_response': round(gabor_responses['std_response'], 3),
            'lbp_uniformity': round(lbp_features['uniformity'], 3),
            'lbp_variance': round(lbp_features['variance'], 3),
            'fractal_dimension': round(fractal_dimension, 3),
            'texture_energy': round(texture_energy, 3)
        }
    
    def _calculate_glcm_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Calculate GLCM texture features"""
        # Reduce image size for faster computation
        small_gray = cv2.resize(gray, (256, 256))
        
        # Calculate GLCM
        glcm = greycomatrix(small_gray, [1], [0], 256, symmetric=True, normed=True)
        
        # Calculate features
        contrast = greycoprops(glcm, 'contrast')[0, 0]
        correlation = greycoprops(glcm, 'correlation')[0, 0]
        energy = greycoprops(glcm, 'energy')[0, 0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
        
        return {
            'contrast': contrast,
            'correlation': correlation,
            'energy': energy,
            'homogeneity': homogeneity
        }
    
    def _calculate_gabor_responses(self, gray: np.ndarray) -> Dict[str, float]:
        """Calculate Gabor filter responses"""
        responses = []
        
        # Multiple orientations and frequencies
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            for frequency in [0.1, 0.3, 0.5]:
                gabor_real, _ = filters.gabor(gray, frequency=frequency, theta=theta)
                responses.append(np.abs(gabor_real))
        
        if responses:
            all_responses = np.stack(responses, axis=0)
            mean_response = np.mean(all_responses)
            std_response = np.std(all_responses)
        else:
            mean_response = 0.0
            std_response = 0.0
        
        return {
            'mean_response': mean_response,
            'std_response': std_response
        }
    
    def _calculate_lbp_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Calculate Local Binary Pattern features"""
        # Calculate LBP
        lbp = feature.local_binary_pattern(gray, 24, 8, method='uniform')
        
        # Calculate histogram
        hist = np.histogram(lbp, bins=26, range=(0, 26))[0]
        hist = hist / np.sum(hist)
        
        # Calculate uniformity and variance
        uniformity = np.sum(hist**2)
        variance = np.var(hist)
        
        return {
            'uniformity': uniformity,
            'variance': variance
        }
    
    def _calculate_fractal_dimension(self, gray: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        # Threshold image
        threshold = np.mean(gray)
        binary = (gray > threshold).astype(int)
        
        # Box-counting
        scales = [1, 2, 4, 8, 16, 32]
        counts = []
        
        for scale in scales:
            # Downsample
            h, w = binary.shape
            h_new, w_new = h // scale, w // scale
            
            if h_new > 0 and w_new > 0:
                downsampled = binary[:h_new*scale, :w_new*scale].reshape(h_new, scale, w_new, scale)
                boxes = np.sum(downsampled, axis=(1, 3)) > 0
                count = np.sum(boxes)
                counts.append(count)
        
        if len(counts) > 1:
            # Linear regression to find fractal dimension
            log_scales = np.log(scales[:len(counts)])
            log_counts = np.log(counts)
            
            slope, _, _, _, _ = stats.linregress(log_scales, log_counts)
            fractal_dimension = -slope
        else:
            fractal_dimension = 1.0
        
        return fractal_dimension
    
    def _calculate_texture_energy(self, gray: np.ndarray) -> float:
        """Calculate texture energy"""
        # Calculate energy in different frequency bands
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Calculate energy in high frequency region
        h, w = gray.shape
        center = (h // 2, w // 2)
        
        # Create high frequency mask
        y, x = np.ogrid[:h, :w]
        mask = ((x - center[1])**2 + (y - center[0])**2) > (min(h, w) // 4)**2
        
        # Calculate energy
        high_freq_energy = np.sum(magnitude[mask]**2)
        total_energy = np.sum(magnitude**2)
        
        texture_energy = high_freq_energy / total_energy if total_energy > 0 else 0
        
        return texture_energy
    
    def calculate_color_science_metrics(self, img_cv: np.ndarray) -> Dict[str, float]:
        """
        Calculate advanced color science metrics
        
        Args:
            img_cv: Input image array
            
        Returns:
            Dictionary containing color science metrics
        """
        # Convert to different color spaces
        lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
        
        # Color rendering index approximation
        cri = self._calculate_cri_approximation(img_cv)
        
        # Metamerism index
        metamerism = self._calculate_metamerism_index(img_cv)
        
        # Color constancy
        color_constancy = self._calculate_color_constancy(img_cv)
        
        # Skin tone analysis
        skin_tone_analysis = self._analyze_skin_tones(img_cv)
        
        # Memory color analysis
        memory_color_analysis = self._analyze_memory_colors(img_cv)
        
        return {
            'cri_approximation': round(cri, 1),
            'metamerism_index': round(metamerism, 3),
            'color_constancy': round(color_constancy, 3),
            'skin_tone_naturalness': round(skin_tone_analysis['naturalness'], 3),
            'skin_tone_coverage': round(skin_tone_analysis['coverage'], 3),
            'memory_color_accuracy': round(memory_color_analysis['accuracy'], 3),
            'memory_color_preference': round(memory_color_analysis['preference'], 3)
        }
    
    def _calculate_cri_approximation(self, img_cv: np.ndarray) -> float:
        """Calculate Color Rendering Index approximation"""
        # This is a simplified approximation
        # Real CRI requires specific test colors
        
        # Convert to Lab
        lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
        
        # Calculate color distribution
        a_range = np.max(lab[:, :, 1]) - np.min(lab[:, :, 1])
        b_range = np.max(lab[:, :, 2]) - np.min(lab[:, :, 2])
        
        # Approximate CRI based on color range
        color_range = np.sqrt(a_range**2 + b_range**2)
        cri = 100 - (color_range / 10)  # Simplified mapping
        
        return max(0, min(100, cri))
    
    def _calculate_metamerism_index(self, img_cv: np.ndarray) -> float:
        """Calculate metamerism index"""
        # Analyze color stability under different illuminants
        # This is a simplified implementation
        
        # Calculate color variations
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
        
        # Calculate color stability
        hue_std = np.std(hsv[:, :, 0])
        saturation_std = np.std(hsv[:, :, 1])
        
        # Metamerism index (lower is better)
        metamerism = (hue_std + saturation_std) / 2
        
        return metamerism
    
    def _calculate_color_constancy(self, img_cv: np.ndarray) -> float:
        """Calculate color constancy"""
        # Gray world assumption
        r_mean = np.mean(img_cv[:, :, 0])
        g_mean = np.mean(img_cv[:, :, 1])
        b_mean = np.mean(img_cv[:, :, 2])
        
        # Calculate deviation from gray
        gray_deviation = np.std([r_mean, g_mean, b_mean])
        
        # Color constancy (lower deviation is better)
        constancy = 1 / (1 + gray_deviation / 50)
        
        return constancy
    
    def _analyze_skin_tones(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Analyze skin tone quality"""
        # Convert to YCrCb for skin detection
        ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_RGB2YCrCb)
        
        # Skin tone detection
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        coverage = np.sum(skin_mask > 0) / (img_cv.shape[0] * img_cv.shape[1])
        
        if coverage > 0.01:  # At least 1% skin
            skin_pixels = img_cv[skin_mask > 0]
            
            # Calculate skin tone naturalness
            r_mean = np.mean(skin_pixels[:, 0])
            g_mean = np.mean(skin_pixels[:, 1])
            b_mean = np.mean(skin_pixels[:, 2])
            
            # Ideal skin tone ratios
            rg_ratio = r_mean / (g_mean + 1e-6)
            rb_ratio = r_mean / (b_mean + 1e-6)
            
            # Compare with ideal ratios
            ideal_rg = 1.15
            ideal_rb = 1.2
            
            naturalness = 1 - (abs(rg_ratio - ideal_rg) + abs(rb_ratio - ideal_rb)) / 2
            naturalness = max(0, naturalness)
        else:
            naturalness = 0.5
        
        return {
            'naturalness': naturalness,
            'coverage': coverage
        }
    
    def _analyze_memory_colors(self, img_cv: np.ndarray) -> Dict[str, float]:
        """Analyze memory colors (sky, grass, etc.)"""
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
        
        # Define memory color ranges in HSV
        memory_colors = {
            'sky': {'hue': (90, 130), 'sat': (50, 255), 'val': (100, 255)},
            'grass': {'hue': (35, 85), 'sat': (40, 255), 'val': (40, 200)},
            'skin': {'hue': (5, 25), 'sat': (20, 150), 'val': (80, 255)}
        }
        
        accuracy_scores = []
        preference_scores = []
        
        for color_name, ranges in memory_colors.items():
            # Create mask for this color
            mask = ((hsv[:, :, 0] >= ranges['hue'][0]) & (hsv[:, :, 0] <= ranges['hue'][1]) &
                   (hsv[:, :, 1] >= ranges['sat'][0]) & (hsv[:, :, 1] <= ranges['sat'][1]) &
                   (hsv[:, :, 2] >= ranges['val'][0]) & (hsv[:, :, 2] <= ranges['val'][1]))
            
            if np.sum(mask) > 0:
                # Calculate accuracy (how close to ideal)
                color_pixels = hsv[mask]
                mean_hue = np.mean(color_pixels[:, 0])
                mean_sat = np.mean(color_pixels[:, 1])
                mean_val = np.mean(color_pixels[:, 2])
                
                # Ideal values (simplified)
                ideal_hue = (ranges['hue'][0] + ranges['hue'][1]) / 2
                ideal_sat = (ranges['sat'][0] + ranges['sat'][1]) / 2
                ideal_val = (ranges['val'][0] + ranges['val'][1]) / 2
                
                # Calculate deviations
                hue_dev = abs(mean_hue - ideal_hue) / 180
                sat_dev = abs(mean_sat - ideal_sat) / 255
                val_dev = abs(mean_val - ideal_val) / 255
                
                accuracy = 1 - (hue_dev + sat_dev + val_dev) / 3
                accuracy_scores.append(accuracy)
                
                # Preference score (slightly enhanced saturation and value)
                preference = 1 - (hue_dev + max(0, (ideal_sat - mean_sat) / 255) + 
                                max(0, (ideal_val - mean_val) / 255)) / 3
                preference_scores.append(preference)
        
        avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.5
        avg_preference = np.mean(preference_scores) if preference_scores else 0.5
        
        return {
            'accuracy': avg_accuracy,
            'preference': avg_preference
        }
