#!/usr/bin/env python3
"""
API validation module for phone numbers
"""

import asyncio
import aiohttp
import requests
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

class APIValidator:
    """API validator cho phone numbers"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PhoneValidator/1.0',
            'Accept': 'application/json'
        })
        self.timeout = 10
        self.max_retries = 3
        
        # API configurations
        self.api_configs = {
            'numverify': {
                'base_url': 'http://apilayer.net/api/validate',
                'api_key': '',  # Cần API key thật
                'rate_limit': 100,  # requests per hour
                'requires_auth': True
            },
            'phonevalidator': {
                'base_url': 'https://phonevalidation.abstractapi.com/v1/',
                'api_key': '',  # Cần API key thật
                'rate_limit': 1000,
                'requires_auth': True
            },
            'twilio': {
                'base_url': 'https://lookups.twilio.com/v1/PhoneNumbers/',
                'account_sid': '',
                'auth_token': '',
                'rate_limit': 10000,
                'requires_auth': True
            },
            'ipqualityscore': {
                'base_url': 'https://ipqualityscore.com/api/json/phone/',
                'api_key': '',
                'rate_limit': 5000,
                'requires_auth': True
            }
        }
        
        # Rate limiting
        self.rate_limits = {}
        self.last_requests = {}
    
    def set_api_key(self, service: str, api_key: str, **kwargs):
        """Thiết lập API key cho service"""
        if service in self.api_configs:
            self.api_configs[service]['api_key'] = api_key
            
            # Thiết lập thêm credentials nếu cần
            if service == 'twilio':
                self.api_configs[service]['account_sid'] = kwargs.get('account_sid', '')
                self.api_configs[service]['auth_token'] = kwargs.get('auth_token', '')
            
            logger.info(f"API key set for {service}")
        else:
            logger.warning(f"Unknown service: {service}")
    
    def _check_rate_limit(self, service: str) -> bool:
        """Kiểm tra rate limit"""
        current_time = time.time()
        
        if service not in self.last_requests:
            self.last_requests[service] = []
        
        # Loại bỏ requests cũ hơn 1 giờ
        hour_ago = current_time - 3600
        self.last_requests[service] = [
            req_time for req_time in self.last_requests[service]
            if req_time > hour_ago
        ]
        
        # Kiểm tra rate limit
        rate_limit = self.api_configs[service]['rate_limit']
        if len(self.last_requests[service]) >= rate_limit:
            return False
        
        # Thêm request hiện tại
        self.last_requests[service].append(current_time)
        return True
    
    def validate_with_numverify(self, phone_number: str) -> Dict:
        """Validate với Numverify API"""
        service = 'numverify'
        
        if not self._check_rate_limit(service):
            return {'error': 'Rate limit exceeded'}
        
        config = self.api_configs[service]
        
        if not config['api_key']:
            return {'error': 'API key not configured'}
        
        try:
            params = {
                'access_key': config['api_key'],
                'number': phone_number,
                'country_code': '',
                'format': 1
            }
            
            response = self.session.get(
                config['base_url'],
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'service': service,
                    'valid': data.get('valid', False),
                    'number': data.get('number'),
                    'local_format': data.get('local_format'),
                    'international_format': data.get('international_format'),
                    'country_prefix': data.get('country_prefix'),
                    'country_code': data.get('country_code'),
                    'country_name': data.get('country_name'),
                    'location': data.get('location'),
                    'carrier': data.get('carrier'),
                    'line_type': data.get('line_type'),
                    'success': True
                }
            else:
                return {
                    'service': service,
                    'error': f'HTTP {response.status_code}',
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"Numverify API error: {e}")
            return {
                'service': service,
                'error': str(e),
                'success': False
            }
    
    def validate_with_phonevalidator(self, phone_number: str) -> Dict:
        """Validate với PhoneValidator API"""
        service = 'phonevalidator'
        
        if not self._check_rate_limit(service):
            return {'error': 'Rate limit exceeded'}
        
        config = self.api_configs[service]
        
        if not config['api_key']:
            return {'error': 'API key not configured'}
        
        try:
            params = {
                'api_key': config['api_key'],
                'phone': phone_number
            }
            
            response = self.session.get(
                config['base_url'],
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'service': service,
                    'valid': data.get('valid', False),
                    'format': data.get('format'),
                    'country': data.get('country'),
                    'location': data.get('location'),
                    'type': data.get('type'),
                    'carrier': data.get('carrier'),
                    'success': True
                }
            else:
                return {
                    'service': service,
                    'error': f'HTTP {response.status_code}',
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"PhoneValidator API error: {e}")
            return {
                'service': service,
                'error': str(e),
                'success': False
            }
    
    def validate_with_twilio(self, phone_number: str) -> Dict:
        """Validate với Twilio Lookup API"""
        service = 'twilio'
        
        if not self._check_rate_limit(service):
            return {'error': 'Rate limit exceeded'}
        
        config = self.api_configs[service]
        
        if not config['account_sid'] or not config['auth_token']:
            return {'error': 'Twilio credentials not configured'}
        
        try:
            from requests.auth import HTTPBasicAuth
            
            url = f"{config['base_url']}{phone_number}"
            params = {
                'Type': 'carrier'
            }
            
            auth = HTTPBasicAuth(config['account_sid'], config['auth_token'])
            
            response = self.session.get(
                url,
                params=params,
                auth=auth,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'service': service,
                    'phone_number': data.get('phone_number'),
                    'national_format': data.get('national_format'),
                    'country_code': data.get('country_code'),
                    'carrier': data.get('carrier', {}).get('name'),
                    'carrier_type': data.get('carrier', {}).get('type'),
                    'mobile_country_code': data.get('carrier', {}).get('mobile_country_code'),
                    'mobile_network_code': data.get('carrier', {}).get('mobile_network_code'),
                    'success': True
                }
            else:
                return {
                    'service': service,
                    'error': f'HTTP {response.status_code}',
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"Twilio API error: {e}")
            return {
                'service': service,
                'error': str(e),
                'success': False
            }
    
    def validate_with_ipqualityscore(self, phone_number: str) -> Dict:
        """Validate với IPQualityScore API"""
        service = 'ipqualityscore'
        
        if not self._check_rate_limit(service):
            return {'error': 'Rate limit exceeded'}
        
        config = self.api_configs[service]
        
        if not config['api_key']:
            return {'error': 'API key not configured'}
        
        try:
            url = f"{config['base_url']}{config['api_key']}/{phone_number}"
            
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'service': service,
                    'valid': data.get('valid', False),
                    'active': data.get('active', False),
                    'fraud_score': data.get('fraud_score', 0),
                    'recent_abuse': data.get('recent_abuse', False),
                    'VOIP': data.get('VOIP', False),
                    'prepaid': data.get('prepaid', False),
                    'risky': data.get('risky', False),
                    'name': data.get('name'),
                    'timezone': data.get('timezone'),
                    'country': data.get('country'),
                    'region': data.get('region'),
                    'city': data.get('city'),
                    'zip_code': data.get('zip_code'),
                    'dialing_code': data.get('dialing_code'),
                    'carrier': data.get('carrier'),
                    'line_type': data.get('line_type'),
                    'success': True
                }
            else:
                return {
                    'service': service,
                    'error': f'HTTP {response.status_code}',
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"IPQualityScore API error: {e}")
            return {
                'service': service,
                'error': str(e),
                'success': False
            }
    
    def validate_with_custom_api(self, phone_number: str, api_url: str, headers: Dict = None) -> Dict:
        """Validate với custom API"""
        try:
            if headers:
                self.session.headers.update(headers)
            
            response = self.session.get(
                api_url,
                params={'phone': phone_number},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return {
                    'service': 'custom',
                    'data': response.json(),
                    'success': True
                }
            else:
                return {
                    'service': 'custom',
                    'error': f'HTTP {response.status_code}',
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"Custom API error: {e}")
            return {
                'service': 'custom',
                'error': str(e),
                'success': False
            }
    
    def validate_single_phone(self, phone_number: str, services: List[str] = None) -> Dict:
        """Validate một số với nhiều API"""
        if services is None:
            services = ['numverify', 'phonevalidator', 'twilio', 'ipqualityscore']
        
        results = {}
        
        # Validate với từng service
        for service in services:
            if service == 'numverify':
                results[service] = self.validate_with_numverify(phone_number)
            elif service == 'phonevalidator':
                results[service] = self.validate_with_phonevalidator(phone_number)
            elif service == 'twilio':
                results[service] = self.validate_with_twilio(phone_number)
            elif service == 'ipqualityscore':
                results[service] = self.validate_with_ipqualityscore(phone_number)
            
            # Thêm delay để tránh rate limit
            time.sleep(0.1)
        
        # Tổng hợp kết quả
        consensus_result = self._analyze_api_consensus(results)
        
        return {
            'phone_number': phone_number,
            'api_results': results,
            'consensus': consensus_result,
            'validation_time': time.time()
        }
    
    def validate_batch_phones(self, phone_numbers: List[str], services: List[str] = None, max_workers: int = 5) -> List[Dict]:
        """Validate batch phones với threading"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.validate_single_phone, phone, services): phone
                for phone in phone_numbers
            }
            
            for future in as_completed(futures):
                phone = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error validating {phone}: {e}")
                    results.append({
                        'phone_number': phone,
                        'error': str(e),
                        'success': False
                    })
        
        return results
    
    async def async_validate_single_phone(self, phone_number: str, services: List[str] = None) -> Dict:
        """Async validate một số điện thoại"""
        if services is None:
            services = ['numverify', 'phonevalidator', 'twilio', 'ipqualityscore']
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for service in services:
                if service == 'numverify':
                    tasks.append(self._async_validate_numverify(session, phone_number))
                elif service == 'phonevalidator':
                    tasks.append(self._async_validate_phonevalidator(session, phone_number))
                elif service == 'twilio':
                    tasks.append(self._async_validate_twilio(session, phone_number))
                elif service == 'ipqualityscore':
                    tasks.append(self._async_validate_ipqualityscore(session, phone_number))
            
            api_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(api_results):
                service = services[i]
                if not isinstance(result, Exception):
                    results[service] = result
                else:
                    results[service] = {
                        'service': service,
                        'error': str(result),
                        'success': False
                    }
        
        # Tổng hợp kết quả
        consensus_result = self._analyze_api_consensus(results)
        
        return {
            'phone_number': phone_number,
            'api_results': results,
            'consensus': consensus_result,
            'validation_time': time.time()
        }
    
    async def async_validate_batch_phones(self, phone_numbers: List[str], services: List[str] = None) -> List[Dict]:
        """Async validate batch phones"""
        tasks = [
            self.async_validate_single_phone(phone, services)
            for phone in phone_numbers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_results = []
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                final_results.append(result)
            else:
                final_results.append({
                    'phone_number': phone_numbers[i],
                    'error': str(result),
                    'success': False
                })
        
        return final_results
    
    async def _async_validate_numverify(self, session: aiohttp.ClientSession, phone_number: str) -> Dict:
        """Async validate với Numverify"""
        service = 'numverify'
        config = self.api_configs[service]
        
        if not config['api_key']:
            return {'service': service, 'error': 'API key not configured', 'success': False}
        
        try:
            params = {
                'access_key': config['api_key'],
                'number': phone_number,
                'country_code': '',
                'format': 1
            }
            
            async with session.get(config['base_url'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'service': service,
                        'valid': data.get('valid', False),
                        'number': data.get('number'),
                        'country_name': data.get('country_name'),
                        'location': data.get('location'),
                        'carrier': data.get('carrier'),
                        'line_type': data.get('line_type'),
                        'success': True
                    }
                else:
                    return {
                        'service': service,
                        'error': f'HTTP {response.status}',
                        'success': False
                    }
                    
        except Exception as e:
            return {
                'service': service,
                'error': str(e),
                'success': False
            }
    
    async def _async_validate_phonevalidator(self, session: aiohttp.ClientSession, phone_number: str) -> Dict:
        """Async validate với PhoneValidator"""
        service = 'phonevalidator'
        config = self.api_configs[service]
        
        if not config['api_key']:
            return {'service': service, 'error': 'API key not configured', 'success': False}
        
        try:
            params = {
                'api_key': config['api_key'],
                'phone': phone_number
            }
            
            async with session.get(config['base_url'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'service': service,
                        'valid': data.get('valid', False),
                        'country': data.get('country'),
                        'location': data.get('location'),
                        'type': data.get('type'),
                        'carrier': data.get('carrier'),
                        'success': True
                    }
                else:
                    return {
                        'service': service,
                        'error': f'HTTP {response.status}',
                        'success': False
                    }
                    
        except Exception as e:
            return {
                'service': service,
                'error': str(e),
                'success': False
            }
    
    async def _async_validate_twilio(self, session: aiohttp.ClientSession, phone_number: str) -> Dict:
        """Async validate với Twilio"""
        service = 'twilio'
        config = self.api_configs[service]
        
        if not config['account_sid'] or not config['auth_token']:
            return {'service': service, 'error': 'Twilio credentials not configured', 'success': False}
        
        try:
            import base64
            
            # Basic auth
            credentials = f"{config['account_sid']}:{config['auth_token']}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                'Authorization': f'Basic {encoded_credentials}'
            }
            
            url = f"{config['base_url']}{phone_number}"
            params = {'Type': 'carrier'}
            
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'service': service,
                        'phone_number': data.get('phone_number'),
                        'country_code': data.get('country_code'),
                        'carrier': data.get('carrier', {}).get('name'),
                        'carrier_type': data.get('carrier', {}).get('type'),
                        'success': True
                    }
                else:
                    return {
                        'service': service,
                        'error': f'HTTP {response.status}',
                        'success': False
                    }
                    
        except Exception as e:
            return {
                'service': service,
                'error': str(e),
                'success': False
            }
    
    async def _async_validate_ipqualityscore(self, session: aiohttp.ClientSession, phone_number: str) -> Dict:
        """Async validate với IPQualityScore"""
        service = 'ipqualityscore'
        config = self.api_configs[service]
        
        if not config['api_key']:
            return {'service': service, 'error': 'API key not configured', 'success': False}
        
        try:
            url = f"{config['base_url']}{config['api_key']}/{phone_number}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'service': service,
                        'valid': data.get('valid', False),
                        'fraud_score': data.get('fraud_score', 0),
                        'risky': data.get('risky', False),
                        'country': data.get('country'),
                        'carrier': data.get('carrier'),
                        'line_type': data.get('line_type'),
                        'success': True
                    }
                else:
                    return {
                        'service': service,
                        'error': f'HTTP {response.status}',
                        'success': False
                    }
                    
        except Exception as e:
            return {
                'service': service,
                'error': str(e),
                'success': False
            }
    
    def _analyze_api_consensus(self, results: Dict) -> Dict:
        """Phân tích consensus từ multiple API results"""
        successful_results = {
            service: result for service, result in results.items()
            if result.get('success', False)
        }
        
        if not successful_results:
            return {
                'consensus_valid': False,
                'confidence': 0.0,
                'agreement_score': 0.0,
                'majority_vote': 'unknown',
                'details': 'No successful API responses'
            }
        
        # Tổng hợp validity votes
        validity_votes = []
        for service, result in successful_results.items():
            if 'valid' in result:
                validity_votes.append(result['valid'])
        
        # Tổng hợp carrier information
        carriers = []
        for service, result in successful_results.items():
            if 'carrier' in result and result['carrier']:
                carriers.append(result['carrier'])
        
        # Tổng hợp country information
        countries = []
        for service, result in successful_results.items():
            if 'country' in result and result['country']:
                countries.append(result['country'])
            elif 'country_name' in result and result['country_name']:
                countries.append(result['country_name'])
        
        # Tính consensus
        if validity_votes:
            valid_count = sum(validity_votes)
            total_votes = len(validity_votes)
            consensus_valid = valid_count > total_votes / 2
            confidence = valid_count / total_votes if consensus_valid else (total_votes - valid_count) / total_votes
            agreement_score = max(valid_count, total_votes - valid_count) / total_votes
        else:
            consensus_valid = False
            confidence = 0.0
            agreement_score = 0.0
        
        # Majority vote
        if validity_votes:
            majority_vote = 'valid' if consensus_valid else 'invalid'
        else:
            majority_vote = 'unknown'
        
        # Most common carrier và country
        most_common_carrier = max(set(carriers), key=carriers.count) if carriers else 'unknown'
        most_common_country = max(set(countries), key=countries.count) if countries else 'unknown'
        
        return {
            'consensus_valid': consensus_valid,
            'confidence': confidence,
            'agreement_score': agreement_score,
            'majority_vote': majority_vote,
            'total_apis': len(results),
            'successful_apis': len(successful_results),
            'validity_votes': validity_votes,
            'most_common_carrier': most_common_carrier,
            'most_common_country': most_common_country,
            'details': f'{len(successful_results)} out of {len(results)} APIs responded successfully'
        }
    
    def get_api_status(self) -> Dict:
        """Lấy trạng thái các API"""
        status = {}
        
        for service, config in self.api_configs.items():
            has_credentials = False
            
            if service == 'twilio':
                has_credentials = bool(config['account_sid'] and config['auth_token'])
            else:
                has_credentials = bool(config['api_key'])
            
            current_time = time.time()
            hour_ago = current_time - 3600
            
            recent_requests = len([
                req_time for req_time in self.last_requests.get(service, [])
                if req_time > hour_ago
            ])
            
            status[service] = {
                'configured': has_credentials,
                'rate_limit': config['rate_limit'],
                'recent_requests': recent_requests,
                'remaining_requests': max(0, config['rate_limit'] - recent_requests),
                'base_url': config['base_url']
            }
        
        return status
