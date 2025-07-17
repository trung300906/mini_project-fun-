#!/usr/bin/env python3
"""
Cache utilities for phone validator
"""

import time
import json
import pickle
import hashlib
import threading
from typing import Any, Dict, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    value: Any
    timestamp: float
    ttl: float
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return time.time() - self.timestamp > self.ttl

class MemoryCache:
    """In-memory cache with TTL support"""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'expired': 0,
            'evicted': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    del self.cache[key]
                    self.stats['expired'] += 1
                    return None
                
                self.stats['hits'] += 1
                return entry.value
            
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache"""
        with self.lock:
            if ttl is None:
                ttl = self.default_ttl
            
            entry = CacheEntry(value, time.time(), ttl)
            self.cache[key] = entry
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats = {
                'hits': 0,
                'misses': 0,
                'expired': 0,
                'evicted': 0
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
                self.stats['expired'] += 1
            
            return len(expired_keys)
    
    def size(self) -> int:
        """Get cache size"""
        with self.lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'expired': self.stats['expired'],
                'evicted': self.stats['evicted']
            }

class FileCache:
    """File-based cache with TTL support"""
    
    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'expired': 0,
            'errors': 0
        }
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key"""
        # Hash key to create filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_path = self._get_cache_path(key)
        
        with self.lock:
            try:
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        entry = pickle.load(f)
                    
                    if entry.is_expired():
                        cache_path.unlink()
                        self.stats['expired'] += 1
                        return None
                    
                    self.stats['hits'] += 1
                    return entry.value
                
                self.stats['misses'] += 1
                return None
                
            except Exception as e:
                logger.error(f"Error reading cache file {cache_path}: {e}")
                self.stats['errors'] += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache"""
        cache_path = self._get_cache_path(key)
        
        with self.lock:
            try:
                if ttl is None:
                    ttl = self.default_ttl
                
                entry = CacheEntry(value, time.time(), ttl)
                
                with open(cache_path, 'wb') as f:
                    pickle.dump(entry, f)
                    
            except Exception as e:
                logger.error(f"Error writing cache file {cache_path}: {e}")
                self.stats['errors'] += 1
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        cache_path = self._get_cache_path(key)
        
        with self.lock:
            try:
                if cache_path.exists():
                    cache_path.unlink()
                    return True
                return False
                
            except Exception as e:
                logger.error(f"Error deleting cache file {cache_path}: {e}")
                self.stats['errors'] += 1
                return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            try:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                
                self.stats = {
                    'hits': 0,
                    'misses': 0,
                    'expired': 0,
                    'errors': 0
                }
                
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                self.stats['errors'] += 1
    
    def cleanup_expired(self) -> int:
        """Remove expired cache files"""
        with self.lock:
            expired_count = 0
            
            try:
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        with open(cache_file, 'rb') as f:
                            entry = pickle.load(f)
                        
                        if entry.is_expired():
                            cache_file.unlink()
                            expired_count += 1
                            self.stats['expired'] += 1
                            
                    except Exception as e:
                        logger.error(f"Error checking cache file {cache_file}: {e}")
                        # Delete corrupted cache files
                        cache_file.unlink()
                        expired_count += 1
                        
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")
                self.stats['errors'] += 1
            
            return expired_count
    
    def size(self) -> int:
        """Get cache size"""
        with self.lock:
            try:
                return len(list(self.cache_dir.glob("*.cache")))
            except Exception as e:
                logger.error(f"Error getting cache size: {e}")
                return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': self.size(),
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'expired': self.stats['expired'],
                'errors': self.stats['errors']
            }

class CacheManager:
    """Cache manager with multiple backends"""
    
    def __init__(self, 
                 memory_cache: bool = True,
                 file_cache: bool = True,
                 memory_ttl: int = 3600,
                 file_ttl: int = 86400,
                 cache_dir: str = "cache"):
        
        self.memory_cache = MemoryCache(memory_ttl) if memory_cache else None
        self.file_cache = FileCache(cache_dir, file_ttl) if file_cache else None
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then file)"""
        # Try memory cache first
        if self.memory_cache:
            value = self.memory_cache.get(key)
            if value is not None:
                return value
        
        # Try file cache
        if self.file_cache:
            value = self.file_cache.get(key)
            if value is not None:
                # Store in memory cache for faster access
                if self.memory_cache:
                    self.memory_cache.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, memory_ttl: Optional[float] = None, file_ttl: Optional[float] = None) -> None:
        """Set value in cache"""
        # Store in memory cache
        if self.memory_cache:
            self.memory_cache.set(key, value, memory_ttl)
        
        # Store in file cache
        if self.file_cache:
            self.file_cache.set(key, value, file_ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from all caches"""
        deleted = False
        
        if self.memory_cache:
            deleted = self.memory_cache.delete(key) or deleted
        
        if self.file_cache:
            deleted = self.file_cache.delete(key) or deleted
        
        return deleted
    
    def clear(self) -> None:
        """Clear all caches"""
        if self.memory_cache:
            self.memory_cache.clear()
        
        if self.file_cache:
            self.file_cache.clear()
    
    def cleanup_expired(self) -> Dict[str, int]:
        """Remove expired entries from all caches"""
        result = {}
        
        if self.memory_cache:
            result['memory'] = self.memory_cache.cleanup_expired()
        
        if self.file_cache:
            result['file'] = self.file_cache.cleanup_expired()
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches"""
        stats = {}
        
        if self.memory_cache:
            stats['memory'] = self.memory_cache.get_stats()
        
        if self.file_cache:
            stats['file'] = self.file_cache.get_stats()
        
        return stats
    
    def _cleanup_worker(self):
        """Background cleanup worker"""
        while True:
            try:
                time.sleep(300)  # Cleanup every 5 minutes
                self.cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")

def cache_result(cache_manager: CacheManager, 
                 key_func: Optional[Callable] = None,
                 memory_ttl: Optional[float] = None,
                 file_ttl: Optional[float] = None):
    """Decorator to cache function results"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_manager.set(cache_key, result, memory_ttl, file_ttl)
            
            return result
        
        return wrapper
    return decorator

# Phone validation specific cache functions
def get_phone_cache_key(phone_number: str, validation_type: str = "full") -> str:
    """Generate cache key for phone validation"""
    normalized_phone = ''.join(c for c in phone_number if c.isdigit())
    return f"phone_validation:{validation_type}:{normalized_phone}"

def get_ml_cache_key(phone_number: str, model_name: str) -> str:
    """Generate cache key for ML predictions"""
    normalized_phone = ''.join(c for c in phone_number if c.isdigit())
    return f"ml_prediction:{model_name}:{normalized_phone}"

def get_api_cache_key(phone_number: str, api_provider: str) -> str:
    """Generate cache key for API responses"""
    normalized_phone = ''.join(c for c in phone_number if c.isdigit())
    return f"api_response:{api_provider}:{normalized_phone}"

class PhoneValidationCache:
    """Specialized cache for phone validation results"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    def get_validation_result(self, phone_number: str, validation_type: str = "full") -> Optional[Dict[str, Any]]:
        """Get cached validation result"""
        cache_key = get_phone_cache_key(phone_number, validation_type)
        return self.cache_manager.get(cache_key)
    
    def set_validation_result(self, phone_number: str, result: Dict[str, Any], 
                             validation_type: str = "full", ttl: Optional[float] = None) -> None:
        """Cache validation result"""
        cache_key = get_phone_cache_key(phone_number, validation_type)
        self.cache_manager.set(cache_key, result, ttl, ttl)
    
    def get_ml_prediction(self, phone_number: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get cached ML prediction"""
        cache_key = get_ml_cache_key(phone_number, model_name)
        return self.cache_manager.get(cache_key)
    
    def set_ml_prediction(self, phone_number: str, model_name: str, 
                         prediction: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Cache ML prediction"""
        cache_key = get_ml_cache_key(phone_number, model_name)
        self.cache_manager.set(cache_key, prediction, ttl, ttl)
    
    def get_api_response(self, phone_number: str, api_provider: str) -> Optional[Dict[str, Any]]:
        """Get cached API response"""
        cache_key = get_api_cache_key(phone_number, api_provider)
        return self.cache_manager.get(cache_key)
    
    def set_api_response(self, phone_number: str, api_provider: str, 
                        response: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Cache API response"""
        cache_key = get_api_cache_key(phone_number, api_provider)
        self.cache_manager.set(cache_key, response, ttl, ttl)
    
    def clear_phone_cache(self, phone_number: str) -> None:
        """Clear all cached data for a phone number"""
        normalized_phone = ''.join(c for c in phone_number if c.isdigit())
        
        # This is a simplified implementation
        # In a real implementation, you might want to track keys by phone number
        # or use pattern matching if the cache backend supports it
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache_manager.get_stats()

# Global cache instance
global_cache_manager = CacheManager()
phone_validation_cache = PhoneValidationCache(global_cache_manager)
