"""
Gemini API Key Manager - Handles multiple API keys with automatic failover
"""

import os
from typing import Optional

try:
    from google import genai
except ImportError:
    print("[ERROR] google-genai not installed")
    exit(1)


class GeminiKeyManager:
    """Manages multiple Gemini API keys with automatic failover on quota errors."""
    
    def __init__(self):
        """Initialize with API keys from environment."""
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.client = None
        
        if not self.api_keys:
            raise ValueError("No Gemini API keys found in environment variables")
        
        print(f"[INFO] Loaded {len(self.api_keys)} Gemini API key(s)")
        self._initialize_client()
    
    def _load_api_keys(self) -> list:
        """Load all API keys from environment variables."""
        keys = []
        
        # Try loading GEMINI_API_KEY_1 through GEMINI_API_KEY_7
        for i in range(1, 8):
            key = os.getenv(f'GEMINI_API_KEY_{i}')
            if key:
                keys.append(key)
        
        # Fallback to single GEMINI_API_KEY if no numbered keys found
        if not keys:
            single_key = os.getenv('GEMINI_API_KEY')
            if single_key:
                keys.append(single_key)
        
        return keys
    
    def _initialize_client(self):
        """Initialize Gemini client with current API key."""
        current_key = self.api_keys[self.current_key_index]
        self.client = genai.Client(api_key=current_key)
        print(f"[INFO] Using Gemini API key #{self.current_key_index + 1}")
    
    def _switch_to_next_key(self) -> bool:
        """Switch to the next available API key. Returns True if successful."""
        if self.current_key_index >= len(self.api_keys) - 1:
            # No more keys available
            return False
        
        self.current_key_index += 1
        self._initialize_client()
        print(f"[INFO] Switched to Gemini API key #{self.current_key_index + 1}")
        return True
    
    def generate_content(self, model: str, contents: str, **kwargs):
        """
        Generate content with automatic key switching on quota errors.
        
        Args:
            model: Model name (e.g., 'gemini-2.5-flash')
            contents: Prompt content
            **kwargs: Additional arguments for generate_content
            
        Returns:
            Response from Gemini API
            
        Raises:
            Exception: If all API keys have been exhausted
        """
        max_retries = len(self.api_keys)
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    **kwargs
                )
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a quota/rate limit error
                is_quota_error = any(keyword in error_str for keyword in [
                    'quota', 'rate limit', 'resource exhausted', '429', 'too many requests'
                ])
                
                if is_quota_error:
                    print(f"[WARNING] API key #{self.current_key_index + 1} quota exceeded: {e}")
                    
                    # Try switching to next key
                    if self._switch_to_next_key():
                        print(f"[INFO] Retrying with API key #{self.current_key_index + 1}...")
                        continue
                    else:
                        print("[ERROR] All API keys have reached their quota limit")
                        raise Exception("All Gemini API keys exhausted") from e
                else:
                    # Not a quota error, raise immediately
                    raise
        
        # Should not reach here
        raise Exception("Failed to generate content after all retries")
    
    def get_client(self):
        """Get the current Gemini client instance."""
        return self.client
