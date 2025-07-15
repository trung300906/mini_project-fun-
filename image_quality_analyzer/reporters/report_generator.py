"""
Report Generator Module
=======================

Generates detailed reports and visualizations for image quality analysis
"""

import os
import json
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class ReportGenerator:
    """
    Professional report generator for image quality analysis
    """
    
    def __init__(self):
        self.report_style = {
            'font_size': 10,
            'title_size': 14,
            'header_size': 12,
            'colors': {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'accent': '#F18F01',
                'success': '#C73E1D',
                'warning': '#F77F00',
                'neutral': '#6C757D'
            }
        }
    
    def generate_detailed_report(self, analysis: Dict[str, Any], 
                               image_path: str, 
                               output_dir: str = None) -> str:
        """
        Generate a detailed analysis report
        
        Args:
            analysis: Analysis results dictionary
            image_path: Path to the analyzed image
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report file
        """
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        
        # Create report filename
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        report_filename = f"{image_name}_quality_report.txt"
        report_path = os.path.join(output_dir, report_filename)
        
        # Generate report content
        report_content = self._generate_report_content(analysis, image_path)
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path
    
    def _generate_report_content(self, analysis: Dict[str, Any], 
                               image_path: str) -> str:
        """Generate the text content for the report"""
        content = []
        
        # Header
        content.append("=" * 100)
        content.append("PROFESSIONAL IMAGE QUALITY ANALYSIS REPORT")
        content.append("=" * 100)
        content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"Image: {os.path.basename(image_path)}")
        content.append(f"Full Path: {image_path}")
        content.append("")
        
        # Executive Summary
        content.append("🎯 EXECUTIVE SUMMARY")
        content.append("-" * 50)
        overall = analysis.get('overall_score', {})
        content.append(f"Overall Quality Score: {overall.get('total_score', 0)}/100")
        content.append(f"Quality Grade: {overall.get('grade', 'N/A')}")
        content.append("")
        
        # Quick Assessment
        content.append("⚡ QUICK ASSESSMENT")
        content.append("-" * 50)
        individual_scores = overall.get('individual_scores', {})
        
        for category, score in individual_scores.items():
            status = self._get_status_indicator(score)
            content.append(f"{category.replace('_', ' ').title()}: {score:.1f}/100 {status}")
        content.append("")
        
        # Basic Information
        content.append("📊 BASIC IMAGE INFORMATION")
        content.append("-" * 50)
        basic_info = analysis.get('basic_info', {})
        content.append(f"Dimensions: {basic_info.get('width', 0)}x{basic_info.get('height', 0)}")
        content.append(f"Megapixels: {basic_info.get('megapixels', 0)} MP")
        content.append(f"Aspect Ratio: {basic_info.get('aspect_ratio', 0)}")
        content.append(f"Color Channels: {basic_info.get('channels', 0)}")
        content.append(f"File Size: {basic_info.get('file_size', 0) / 1024:.1f} KB")
        content.append("")
        
        # Camera Information
        camera_info = analysis.get('camera_info', {})
        if camera_info:
            content.append("📷 CAMERA INFORMATION")
            content.append("-" * 50)
            for key, value in camera_info.items():
                content.append(f"{key.replace('_', ' ').title()}: {value}")
            content.append("")
        
        # ISO Standards Analysis
        content.append("📏 ISO STANDARDS ANALYSIS")
        content.append("-" * 50)
        
        # Sharpness (ISO 12233)
        iso_sfr = analysis.get('iso_sfr', {})
        if iso_sfr:
            content.append("🔍 Sharpness (ISO 12233):")
            content.append(f"   • MTF50: {iso_sfr.get('mtf50', 0):.3f}")
            content.append(f"   • SFR Horizontal: {iso_sfr.get('sfr_horizontal', 0):.3f}")
            content.append(f"   • SFR Vertical: {iso_sfr.get('sfr_vertical', 0):.3f}")
            content.append(f"   • Acutance: {iso_sfr.get('acutance', 0):.3f}")
            content.append("")
        
        # Noise (ISO 15739)
        iso_noise = analysis.get('iso_noise', {})
        if iso_noise:
            content.append("📡 Noise Analysis (ISO 15739):")
            content.append(f"   • Temporal Noise: {iso_noise.get('temporal_noise', 0):.4f}")
            content.append(f"   • Spatial Noise: {iso_noise.get('spatial_noise', 0):.4f}")
            content.append(f"   • SNR: {iso_noise.get('snr_db', 0):.2f} dB")
            content.append(f"   • Visual Noise: {iso_noise.get('visual_noise', 0):.4f}")
            content.append("")
        
        # Color Reproduction (ISO 14524)
        iso_color = analysis.get('iso_color', {})
        if iso_color:
            content.append("🎨 Color Reproduction (ISO 14524):")
            content.append(f"   • Color Accuracy: {iso_color.get('color_accuracy', 0):.3f}")
            content.append(f"   • Gamut Coverage: {iso_color.get('gamut_coverage', 0):.3f}")
            content.append(f"   • Color Consistency: {iso_color.get('color_consistency', 0):.3f}")
            content.append(f"   • White Balance Accuracy: {iso_color.get('white_balance_accuracy', 0):.3f}")
            content.append(f"   • Color Temperature: {iso_color.get('color_temperature', 0):.0f}K")
            content.append("")
        
        # Distortion (ISO 20462)
        iso_distortion = analysis.get('iso_distortion', {})
        if iso_distortion:
            content.append("📐 Distortion Analysis (ISO 20462):")
            content.append(f"   • Geometric Distortion: {iso_distortion.get('geometric_distortion', 0):.3f}°")
            content.append(f"   • Chromatic Aberration: {iso_distortion.get('chromatic_aberration', 0):.3f}")
            content.append(f"   • Vignetting: {iso_distortion.get('vignetting', 0):.3f}%")
            content.append(f"   • Perspective Distortion: {iso_distortion.get('perspective_distortion', 0):.3f}°")
            content.append("")
        
        # Advanced Analysis
        content.append("🧠 ADVANCED ANALYSIS")
        content.append("-" * 50)
        
        # Perceptual Metrics
        perceptual = analysis.get('perceptual', {})
        if perceptual:
            content.append("👁️ Perceptual Quality:")
            content.append(f"   • SSIM: {perceptual.get('ssim', 0):.4f}")
            content.append(f"   • PSNR: {perceptual.get('psnr', 0):.2f} dB")
            content.append(f"   • Gradient Similarity: {perceptual.get('gradient_similarity', 0):.4f}")
            content.append(f"   • Texture Similarity: {perceptual.get('texture_similarity', 0):.4f}")
            content.append(f"   • Perceptual Sharpness: {perceptual.get('perceptual_sharpness', 0):.3f}")
            content.append("")
        
        # Bokeh Analysis
        bokeh = analysis.get('bokeh', {})
        if bokeh:
            content.append("💫 Bokeh Quality:")
            content.append(f"   • Bokeh Smoothness: {bokeh.get('bokeh_smoothness', 0):.3f}")
            content.append(f"   • Bokeh Shape Quality: {bokeh.get('bokeh_shape_quality', 0):.3f}")
            content.append(f"   • Depth Transition: {bokeh.get('depth_transition', 0):.3f}")
            content.append(f"   • Focus Peaking: {bokeh.get('focus_peaking', 0):.3f}%")
            content.append(f"   • DOF Ratio: {bokeh.get('dof_ratio', 0):.3f}")
            content.append("")
        
        # Texture Analysis
        texture = analysis.get('texture', {})
        if texture:
            content.append("🎭 Texture Analysis:")
            content.append(f"   • GLCM Contrast: {texture.get('glcm_contrast', 0):.3f}")
            content.append(f"   • GLCM Correlation: {texture.get('glcm_correlation', 0):.3f}")
            content.append(f"   • GLCM Energy: {texture.get('glcm_energy', 0):.3f}")
            content.append(f"   • GLCM Homogeneity: {texture.get('glcm_homogeneity', 0):.3f}")
            content.append(f"   • Fractal Dimension: {texture.get('fractal_dimension', 0):.3f}")
            content.append("")
        
        # Color Science
        color_science = analysis.get('color_science', {})
        if color_science:
            content.append("🔬 Color Science:")
            content.append(f"   • CRI Approximation: {color_science.get('cri_approximation', 0):.1f}")
            content.append(f"   • Metamerism Index: {color_science.get('metamerism_index', 0):.3f}")
            content.append(f"   • Color Constancy: {color_science.get('color_constancy', 0):.3f}")
            content.append(f"   • Skin Tone Naturalness: {color_science.get('skin_tone_naturalness', 0):.3f}")
            content.append(f"   • Memory Color Accuracy: {color_science.get('memory_color_accuracy', 0):.3f}")
            content.append("")
        
        # Quality Assessment
        content.append("🎯 QUALITY ASSESSMENT")
        content.append("-" * 50)
        content.append(self._generate_quality_assessment(analysis))
        content.append("")
        
        # Recommendations
        content.append("💡 RECOMMENDATIONS")
        content.append("-" * 50)
        content.append(self._generate_recommendations(analysis))
        content.append("")
        
        # Technical Notes
        content.append("📝 TECHNICAL NOTES")
        content.append("-" * 50)
        content.append("• This analysis follows international standards including ISO 12233, ISO 15739, ISO 14524, and ISO 20462")
        content.append("• Scores are calibrated for professional photography standards")
        content.append("• Advanced metrics include perceptual quality, bokeh analysis, and color science")
        content.append("• Results may vary depending on viewing conditions and display characteristics")
        content.append("")
        
        content.append("=" * 100)
        content.append("END OF REPORT")
        content.append("=" * 100)
        
        return "\n".join(content)
    
    def _get_status_indicator(self, score: float) -> str:
        """Get status indicator based on score"""
        if score >= 90:
            return "🟢 Excellent"
        elif score >= 80:
            return "🔵 Very Good"
        elif score >= 70:
            return "🟡 Good"
        elif score >= 60:
            return "🟠 Average"
        else:
            return "🔴 Poor"
    
    def _generate_quality_assessment(self, analysis: Dict[str, Any]) -> str:
        """Generate quality assessment text"""
        overall = analysis.get('overall_score', {})
        total_score = overall.get('total_score', 0)
        
        assessment = []
        
        if total_score >= 90:
            assessment.append("This image demonstrates exceptional quality with professional-grade characteristics.")
        elif total_score >= 80:
            assessment.append("This image shows very good quality suitable for most professional applications.")
        elif total_score >= 70:
            assessment.append("This image has good quality with some areas for improvement.")
        elif total_score >= 60:
            assessment.append("This image has average quality with several areas needing attention.")
        else:
            assessment.append("This image has quality issues that significantly impact its professional usability.")
        
        # Add specific strengths and weaknesses
        individual_scores = overall.get('individual_scores', {})
        
        strengths = [k for k, v in individual_scores.items() if v >= 85]
        weaknesses = [k for k, v in individual_scores.items() if v < 65]
        
        if strengths:
            assessment.append(f"\nStrengths: {', '.join(strengths)}")
        
        if weaknesses:
            assessment.append(f"\nWeaknesses: {', '.join(weaknesses)}")
        
        return "\n".join(assessment)
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> str:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        individual_scores = analysis.get('overall_score', {}).get('individual_scores', {})
        
        # Sharpness recommendations
        if individual_scores.get('sharpness', 100) < 70:
            recommendations.append("• Consider using a tripod or faster shutter speed to improve sharpness")
            recommendations.append("• Check lens calibration and ensure proper focusing")
        
        # Noise recommendations
        if individual_scores.get('noise', 100) < 70:
            recommendations.append("• Use lower ISO settings when possible")
            recommendations.append("• Consider noise reduction in post-processing")
        
        # Color recommendations
        if individual_scores.get('color_accuracy', 100) < 70:
            recommendations.append("• Use proper white balance settings")
            recommendations.append("• Consider color calibration of your display")
        
        # Distortion recommendations
        if individual_scores.get('distortion', 100) < 70:
            recommendations.append("• Use lens correction profiles in post-processing")
            recommendations.append("• Consider lens distortion correction")
        
        # General recommendations
        if not recommendations:
            recommendations.append("• Image quality is good overall")
            recommendations.append("• Consider fine-tuning based on specific use case")
        
        return "\n".join(recommendations)
    
    def generate_comparison_report(self, comparison_results: Dict[str, Any], 
                                 output_dir: str = None) -> str:
        """
        Generate a comparison report between two images
        
        Args:
            comparison_results: Results from image comparison
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated comparison report
        """
        if output_dir is None:
            output_dir = os.getcwd()
        
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"image_comparison_{timestamp}.txt"
        report_path = os.path.join(output_dir, report_filename)
        
        # Generate report content
        report_content = self._generate_comparison_content(comparison_results)
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path
    
    def _generate_comparison_content(self, comparison_results: Dict[str, Any]) -> str:
        """Generate comparison report content"""
        content = []
        
        # Header
        content.append("=" * 100)
        content.append("PROFESSIONAL IMAGE QUALITY COMPARISON REPORT")
        content.append("=" * 100)
        content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        # Images being compared
        image1 = comparison_results.get('image1', {})
        image2 = comparison_results.get('image2', {})
        
        content.append("📸 IMAGES BEING COMPARED")
        content.append("-" * 50)
        content.append(f"Image 1: {os.path.basename(image1.get('path', 'N/A'))}")
        content.append(f"Image 2: {os.path.basename(image2.get('path', 'N/A'))}")
        content.append("")
        
        # Winner announcement
        winner = comparison_results.get('winner', {})
        content.append("🏆 COMPARISON RESULTS")
        content.append("-" * 50)
        
        if winner.get('winner', 0) == 1:
            winner_name = os.path.basename(image1.get('path', 'Image 1'))
            content.append(f"Winner: {winner_name}")
        elif winner.get('winner', 0) == 2:
            winner_name = os.path.basename(image2.get('path', 'Image 2'))
            content.append(f"Winner: {winner_name}")
        else:
            content.append("Result: Tie")
        
        content.append(f"Margin: {winner.get('margin', 0):.2f} points")
        content.append(f"Significance: {winner.get('significance', 'N/A')}")
        content.append("")
        
        # Detailed comparison
        comparison = comparison_results.get('comparison', {})
        individual_comparison = comparison.get('individual_comparison', {})
        
        content.append("📊 DETAILED COMPARISON")
        content.append("-" * 50)
        
        for category, comp_data in individual_comparison.items():
            winner_num = comp_data.get('winner', 0)
            difference = comp_data.get('difference', 0)
            
            if winner_num == 1:
                winner_text = f"{os.path.basename(image1.get('path', 'Image 1'))} wins"
            elif winner_num == 2:
                winner_text = f"{os.path.basename(image2.get('path', 'Image 2'))} wins"
            else:
                winner_text = "Tie"
            
            content.append(f"{category.replace('_', ' ').title()}: {winner_text} ({difference:+.2f})")
        
        content.append("")
        
        # Individual scores
        content.append("🎯 INDIVIDUAL SCORES")
        content.append("-" * 50)
        
        scores1 = image1.get('analysis', {}).get('overall_score', {}).get('individual_scores', {})
        scores2 = image2.get('analysis', {}).get('overall_score', {}).get('individual_scores', {})
        
        content.append(f"{'Category':<20} {'Image 1':<10} {'Image 2':<10} {'Difference':<12}")
        content.append("-" * 55)
        
        for category in scores1:
            if category in scores2:
                score1 = scores1[category]
                score2 = scores2[category]
                diff = score1 - score2
                
                content.append(f"{category.replace('_', ' ').title():<20} {score1:<10.1f} {score2:<10.1f} {diff:+.1f}")
        
        content.append("")
        
        # Overall scores
        total1 = image1.get('analysis', {}).get('overall_score', {}).get('total_score', 0)
        total2 = image2.get('analysis', {}).get('overall_score', {}).get('total_score', 0)
        
        content.append("🏆 OVERALL SCORES")
        content.append("-" * 50)
        content.append(f"Image 1: {total1:.2f}/100")
        content.append(f"Image 2: {total2:.2f}/100")
        content.append(f"Difference: {total1 - total2:+.2f}")
        content.append("")
        
        content.append("=" * 100)
        content.append("END OF COMPARISON REPORT")
        content.append("=" * 100)
        
        return "\n".join(content)
    
    def save_analysis_json(self, analysis: Dict[str, Any], 
                          image_path: str, 
                          output_dir: str = None) -> str:
        """
        Save analysis results as JSON
        
        Args:
            analysis: Analysis results
            image_path: Path to the analyzed image
            output_dir: Directory to save the JSON file
            
        Returns:
            Path to the saved JSON file
        """
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        
        # Create filename
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        json_filename = f"{image_name}_analysis.json"
        json_path = os.path.join(output_dir, json_filename)
        
        # Add metadata
        analysis_with_metadata = {
            'metadata': {
                'image_path': image_path,
                'analysis_date': datetime.now().isoformat(),
                'analyzer_version': '2.0.0'
            },
            'analysis': analysis
        }
        
        # Save JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_with_metadata, f, indent=2, ensure_ascii=False, default=str)
        
        return json_path
