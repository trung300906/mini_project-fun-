"""
Main Image Quality Analyzer
===========================

Core analyzer class that coordinates all quality assessments
"""

import os
import sys
from typing import Dict, Any, Optional
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_loader import ImageLoader
from metrics.iso_standards import ISOStandardsMetrics
from metrics.advanced_metrics import AdvancedMetrics


class ImageQualityAnalyzer:
    """
    Professional Image Quality Analyzer
    
    Provides comprehensive image quality assessment using international standards
    and advanced metrics.
    """
    
    def __init__(self):
        self.image_loader = ImageLoader()
        self.iso_metrics = ISOStandardsMetrics()
        self.advanced_metrics = AdvancedMetrics()
        
        # Quality weights for overall scoring
        self.quality_weights = {
            'sharpness': 0.20,
            'noise': 0.15,
            'color_accuracy': 0.12,
            'exposure': 0.12,
            'distortion': 0.10,
            'perceptual': 0.08,
            'bokeh': 0.08,
            'texture': 0.08,
            'color_science': 0.07
        }
    
    def analyze_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Perform comprehensive image quality analysis
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing all analysis results or None if failed
        """
        print(f"\n🔍 Analyzing image: {os.path.basename(image_path)}")
        
        # Load image
        img_cv, img_pil = self.image_loader.load_image(image_path)
        if img_cv is None or img_pil is None:
            return None
        
        # Validate image
        if not self.image_loader.validate_image(img_cv):
            print("❌ Image validation failed")
            return None
        
        analysis_results = {}
        
        # Basic image information
        analysis_results['basic_info'] = self.image_loader.get_image_info(img_cv, img_pil)
        
        # EXIF data
        exif_data = self.image_loader.extract_exif_data(img_pil)
        analysis_results['exif_data'] = exif_data['exif_data']
        analysis_results['camera_info'] = exif_data['camera_info']
        
        # ISO Standards metrics
        print("   📏 Calculating ISO standards metrics...")
        analysis_results['iso_sfr'] = self.iso_metrics.calculate_sfr_iso12233(img_cv)
        analysis_results['iso_noise'] = self.iso_metrics.calculate_noise_iso15739(img_cv)
        analysis_results['iso_color'] = self.iso_metrics.calculate_color_reproduction_iso14524(img_cv)
        analysis_results['iso_distortion'] = self.iso_metrics.calculate_distortion_iso20462(img_cv)
        
        # Advanced metrics
        print("   🧠 Calculating advanced metrics...")
        analysis_results['perceptual'] = self.advanced_metrics.calculate_perceptual_metrics(img_cv)
        analysis_results['bokeh'] = self.advanced_metrics.calculate_bokeh_quality(img_cv)
        analysis_results['texture'] = self.advanced_metrics.calculate_texture_analysis(img_cv)
        analysis_results['color_science'] = self.advanced_metrics.calculate_color_science_metrics(img_cv)
        
        # Calculate overall quality score
        print("   📊 Calculating overall quality score...")
        analysis_results['overall_score'] = self.calculate_overall_quality_score(analysis_results)
        
        print("   ✅ Analysis completed!")
        return analysis_results
    
    def calculate_overall_quality_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall quality score based on all metrics
        
        Args:
            analysis: Dictionary containing all analysis results
            
        Returns:
            Dictionary containing individual scores and overall score
        """
        individual_scores = {}
        
        # Sharpness score (0-100)
        if 'iso_sfr' in analysis:
            sfr = analysis['iso_sfr']
            sharpness_score = min(100, sfr['mtf50'] * 100)
            individual_scores['sharpness'] = sharpness_score
        
        # Noise score (0-100, higher SNR is better)
        if 'iso_noise' in analysis:
            noise = analysis['iso_noise']
            noise_score = min(100, max(0, (noise['snr_db'] - 10) * 2))
            individual_scores['noise'] = noise_score
        
        # Color accuracy score (0-100)
        if 'iso_color' in analysis:
            color = analysis['iso_color']
            color_score = color['color_accuracy'] * 100
            individual_scores['color_accuracy'] = color_score
        
        # Exposure score (0-100)
        exposure_score = self._calculate_exposure_score(analysis)
        individual_scores['exposure'] = exposure_score
        
        # Distortion score (0-100, lower distortion is better)
        if 'iso_distortion' in analysis:
            distortion = analysis['iso_distortion']
            distortion_score = max(0, 100 - distortion['geometric_distortion'] * 10)
            individual_scores['distortion'] = distortion_score
        
        # Perceptual score (0-100)
        if 'perceptual' in analysis:
            perceptual = analysis['perceptual']
            perceptual_score = perceptual['ssim'] * 100
            individual_scores['perceptual'] = perceptual_score
        
        # Bokeh score (0-100)
        if 'bokeh' in analysis:
            bokeh = analysis['bokeh']
            bokeh_score = (bokeh['bokeh_smoothness'] + bokeh['bokeh_shape_quality']) * 50
            individual_scores['bokeh'] = bokeh_score
        
        # Texture score (0-100)
        if 'texture' in analysis:
            texture = analysis['texture']
            texture_score = texture['glcm_energy'] * 100
            individual_scores['texture'] = texture_score
        
        # Color science score (0-100)
        if 'color_science' in analysis:
            color_science = analysis['color_science']
            color_science_score = color_science['skin_tone_naturalness'] * 100
            individual_scores['color_science'] = color_science_score
        
        # Calculate weighted total score
        total_score = 0
        total_weight = 0
        
        for category, weight in self.quality_weights.items():
            if category in individual_scores:
                total_score += individual_scores[category] * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0
        
        return {
            'individual_scores': individual_scores,
            'total_score': round(final_score, 2),
            'grade': self._get_quality_grade(final_score),
            'scoring_weights': self.quality_weights
        }
    
    def _calculate_exposure_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate exposure quality score"""
        # This would analyze histogram, dynamic range, etc.
        # Simplified implementation
        
        if 'basic_info' in analysis:
            # Use basic brightness analysis
            # This is a placeholder - would need actual exposure analysis
            return 75.0  # Default score
        
        return 50.0
    
    def _get_quality_grade(self, score: float) -> str:
        """
        Convert numerical score to quality grade
        
        Args:
            score: Numerical score (0-100)
            
        Returns:
            Quality grade string
        """
        if score >= 95:
            return "Outstanding (A++)"
        elif score >= 90:
            return "Excellent (A+)"
        elif score >= 85:
            return "Very Good (A)"
        elif score >= 80:
            return "Good (A-)"
        elif score >= 75:
            return "Above Average (B+)"
        elif score >= 70:
            return "Average (B)"
        elif score >= 65:
            return "Below Average (B-)"
        elif score >= 60:
            return "Fair (C+)"
        elif score >= 55:
            return "Poor (C)"
        elif score >= 50:
            return "Very Poor (C-)"
        else:
            return "Unacceptable (F)"
    
    def compare_images(self, image1_path: str, image2_path: str) -> Dict[str, Any]:
        """
        Compare two images and provide detailed analysis
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Dictionary containing comparison results
        """
        print("\n" + "=" * 80)
        print("🖼️  PROFESSIONAL IMAGE QUALITY COMPARISON")
        print("=" * 80)
        
        # Analyze both images
        analysis1 = self.analyze_image(image1_path)
        analysis2 = self.analyze_image(image2_path)
        
        if analysis1 is None or analysis2 is None:
            return {}
        
        # Prepare comparison results
        comparison_results = {
            'image1': {
                'path': image1_path,
                'analysis': analysis1
            },
            'image2': {
                'path': image2_path,
                'analysis': analysis2
            },
            'comparison': self._generate_comparison_summary(analysis1, analysis2),
            'winner': self._determine_winner(analysis1, analysis2)
        }
        
        return comparison_results
    
    def _generate_comparison_summary(self, analysis1: Dict[str, Any], 
                                   analysis2: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison summary between two analyses"""
        summary = {}
        
        # Compare overall scores
        score1 = analysis1['overall_score']['total_score']
        score2 = analysis2['overall_score']['total_score']
        
        summary['score_difference'] = abs(score1 - score2)
        summary['better_image'] = 1 if score1 > score2 else 2
        
        # Compare individual categories
        individual_comparison = {}
        scores1 = analysis1['overall_score']['individual_scores']
        scores2 = analysis2['overall_score']['individual_scores']
        
        for category in scores1:
            if category in scores2:
                diff = scores1[category] - scores2[category]
                individual_comparison[category] = {
                    'difference': round(diff, 2),
                    'winner': 1 if diff > 0 else 2 if diff < 0 else 0
                }
        
        summary['individual_comparison'] = individual_comparison
        
        return summary
    
    def _determine_winner(self, analysis1: Dict[str, Any], 
                         analysis2: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the winner based on comprehensive analysis"""
        score1 = analysis1['overall_score']['total_score']
        score2 = analysis2['overall_score']['total_score']
        
        if score1 > score2:
            winner = 1
            margin = score1 - score2
        elif score2 > score1:
            winner = 2
            margin = score2 - score1
        else:
            winner = 0
            margin = 0
        
        # Determine significance of difference
        if margin > 10:
            significance = "significant"
        elif margin > 5:
            significance = "moderate"
        elif margin > 0:
            significance = "slight"
        else:
            significance = "tie"
        
        return {
            'winner': winner,
            'margin': round(margin, 2),
            'significance': significance,
            'score1': score1,
            'score2': score2
        }
