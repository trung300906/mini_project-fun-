#!/usr/bin/env python3
"""
Basic phone number validators
"""

import re
import phonenumbers
from phonenumbers import carrier, geocoder, timezone
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BasicPhoneValidator:
    """Validator cơ bản cho số điện thoại"""
    
    def __init__(self):
        self.default_region = "VN"
        
    def validate_format(self, phone_number: str, region: Optional[str] = None) -> Tuple[bool, Dict]:
        """Kiểm tra định dạng số điện thoại"""
        try:
            # Thử parse với region cung cấp hoặc None
            try:
                parsed = phonenumbers.parse(phone_number, region)
            except phonenumbers.NumberParseException:
                # Nếu không có country code, thử với region mặc định
                parsed = phonenumbers.parse(phone_number, self.default_region)
            
            is_valid = phonenumbers.is_valid_number(parsed)
            is_possible = phonenumbers.is_possible_number(parsed)
            
            country = geocoder.description_for_number(parsed, "en")
            carrier_name = carrier.name_for_number(parsed, "en")
            timezones = timezone.time_zones_for_number(parsed)
            
            # Thông tin số điện thoại
            number_type = phonenumbers.number_type(parsed)
            region_code = phonenumbers.region_code_for_number(parsed)
            
            return is_valid, {
                'is_possible': is_possible,
                'country': country,
                'carrier': carrier_name,
                'timezones': list(timezones),
                'number_type': number_type,
                'region_code': region_code,
                'formatted_national': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL),
                'formatted_international': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                'formatted_e164': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
                'country_code': parsed.country_code,
                'national_number': parsed.national_number,
                'extension': parsed.extension,
                'italian_leading_zero': parsed.italian_leading_zero,
                'number_of_leading_zeros': parsed.number_of_leading_zeros,
                'raw_input': parsed.raw_input
            }
        except Exception as e:
            logger.error(f"Format validation error: {e}")
            return False, {'error': str(e)}
    
    def clean_phone_number(self, phone_number: str) -> str:
        """Làm sạch số điện thoại"""
        # Loại bỏ tất cả ký tự không phải số
        cleaned = re.sub(r'\D', '', phone_number)
        return cleaned
    
    def normalize_phone_number(self, phone_number: str, region: str = "VN") -> str:
        """Chuẩn hóa số điện thoại"""
        try:
            parsed = phonenumbers.parse(phone_number, region)
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        except:
            return self.clean_phone_number(phone_number)
    
    def get_phone_type(self, phone_number: str, region: str = "VN") -> str:
        """Lấy loại số điện thoại"""
        try:
            parsed = phonenumbers.parse(phone_number, region)
            phone_type = phonenumbers.number_type(parsed)
            
            type_mapping = {
                phonenumbers.PhoneNumberType.MOBILE: "Mobile",
                phonenumbers.PhoneNumberType.FIXED_LINE: "Landline",
                phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE: "Fixed Line or Mobile",
                phonenumbers.PhoneNumberType.TOLL_FREE: "Toll Free",
                phonenumbers.PhoneNumberType.PREMIUM_RATE: "Premium Rate",
                phonenumbers.PhoneNumberType.SHARED_COST: "Shared Cost",
                phonenumbers.PhoneNumberType.VOIP: "VoIP",
                phonenumbers.PhoneNumberType.PERSONAL_NUMBER: "Personal Number",
                phonenumbers.PhoneNumberType.PAGER: "Pager",
                phonenumbers.PhoneNumberType.UAN: "UAN",
                phonenumbers.PhoneNumberType.VOICEMAIL: "Voicemail",
                phonenumbers.PhoneNumberType.UNKNOWN: "Unknown"
            }
            
            return type_mapping.get(phone_type, "Unknown")
        except:
            return "Unknown"
    
    def is_valid_vietnam_mobile(self, phone_number: str) -> bool:
        """Kiểm tra số di động Việt Nam hợp lệ"""
        cleaned = self.clean_phone_number(phone_number)
        
        # Các đầu số di động VN hiện tại
        vietnam_mobile_prefixes = [
            '84', '032', '033', '034', '035', '036', '037', '038', '039',
            '070', '071', '072', '073', '074', '075', '076', '077', '078', '079',
            '081', '082', '083', '084', '085', '086', '087', '088', '089',
            '090', '091', '092', '093', '094', '095', '096', '097', '098', '099',
            '056', '058', '059', '086', '096', '097', '098'
        ]
        
        # Kiểm tra pattern
        for prefix in vietnam_mobile_prefixes:
            if cleaned.startswith(prefix) or cleaned.startswith(f"84{prefix}"):
                return True
        
        # Kiểm tra pattern cụ thể
        vn_patterns = [
            r'^84[3-9]\d{8}$',      # +84 + 9 digits
            r'^0[3-9]\d{8}$',       # 0 + 9 digits
            r'^[3-9]\d{8}$',        # 9 digits
        ]
        
        for pattern in vn_patterns:
            if re.match(pattern, cleaned):
                return True
        
        return False
    
    def extract_country_code(self, phone_number: str) -> Optional[str]:
        """Trích xuất mã quốc gia"""
        try:
            parsed = phonenumbers.parse(phone_number, None)
            return str(parsed.country_code)
        except:
            return None
    
    def get_timezone_info(self, phone_number: str) -> list:
        """Lấy thông tin múi giờ"""
        try:
            parsed = phonenumbers.parse(phone_number, None)
            return list(timezone.time_zones_for_number(parsed))
        except:
            return []
    
    def validate_international_format(self, phone_number: str) -> bool:
        """Kiểm tra định dạng quốc tế"""
        # Định dạng E.164
        pattern = r'^\+[1-9]\d{1,14}$'
        return bool(re.match(pattern, phone_number))
    
    def get_carrier_info(self, phone_number: str, region: str = "VN") -> Dict:
        """Lấy thông tin nhà mạng chi tiết"""
        try:
            parsed = phonenumbers.parse(phone_number, region)
            carrier_name = carrier.name_for_number(parsed, "en")
            
            # Thông tin nhà mạng Việt Nam
            vn_carriers = {
                "viettel": {
                    "name": "Viettel",
                    "type": "Mobile Network Operator",
                    "country": "Vietnam",
                    "website": "https://viettel.com.vn"
                },
                "vinaphone": {
                    "name": "Vinaphone",
                    "type": "Mobile Network Operator", 
                    "country": "Vietnam",
                    "website": "https://vinaphone.com.vn"
                },
                "mobifone": {
                    "name": "Mobifone",
                    "type": "Mobile Network Operator",
                    "country": "Vietnam", 
                    "website": "https://mobifone.vn"
                },
                "vietnamobile": {
                    "name": "Vietnamobile",
                    "type": "Mobile Network Operator",
                    "country": "Vietnam",
                    "website": "https://vietnamobile.com.vn"
                },
                "gmobile": {
                    "name": "Gmobile",
                    "type": "Mobile Network Operator",
                    "country": "Vietnam",
                    "website": "https://gmobile.vn"
                }
            }
            
            carrier_info = vn_carriers.get(carrier_name.lower(), {
                "name": carrier_name,
                "type": "Unknown",
                "country": "Unknown",
                "website": "Unknown"
            })
            
            return carrier_info
            
        except Exception as e:
            logger.error(f"Carrier info error: {e}")
            return {"error": str(e)}
