"""
ElevenLabs API Key Manager - Handles multiple API keys with automatic failover
"""

import os

try:
    from elevenlabs import ElevenLabs
except ImportError:
    print("[ERROR] elevenlabs not installed")
    exit(1)


class ElevenLabsKeyManager:
    """Manages multiple ElevenLabs API keys with automatic failover on errors."""
    
    def __init__(self):
        """Initialize with API keys from environment."""
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.client = None
        
        if not self.api_keys:
            raise ValueError("No ElevenLabs API keys found in environment variables")
        
        print(f"[INFO] Loaded {len(self.api_keys)} ElevenLabs API key(s)")
        self._initialize_client()
    
    def _load_api_keys(self) -> list:
        """Load all API keys from environment variables."""
        keys = []
        
        # Try loading ELEVENLABS_API_KEY_1 and ELEVENLABS_API_KEY_2
        for i in range(1, 3):
            key = os.getenv(f'ELEVENLABS_API_KEY_{i}')
            if key:
                keys.append(key)
        
        # Fallback to single ELEVENLABS_API_KEY if no numbered keys found
        if not keys:
            single_key = os.getenv('ELEVENLABS_API_KEY')
            if single_key:
                keys.append(single_key)
        
        return keys
    
    def _initialize_client(self):
        """Initialize ElevenLabs client with current API key."""
        current_key = self.api_keys[self.current_key_index]
        self.client = ElevenLabs(api_key=current_key)
        print(f"[INFO] Using ElevenLabs API key #{self.current_key_index + 1}")
    
    def _switch_to_next_key(self) -> bool:
        """Switch to the next available API key. Returns True if successful."""
        if self.current_key_index >= len(self.api_keys) - 1:
            # No more keys available
            return False
        
        self.current_key_index += 1
        self._initialize_client()
        print(f"[INFO] Switched to ElevenLabs API key #{self.current_key_index + 1}")
        return True
    
    def text_to_speech_convert(self, voice_id: str, text: str, model_id: str = "eleven_turbo_v2_5"):
        """
        Convert text to speech with automatic key switching on errors.
        
        Args:
            voice_id: Voice ID to use
            text: Text to convert
            model_id: Model ID to use (default: eleven_turbo_v2_5)
            
        Returns:
            Audio data from ElevenLabs API
            
        Raises:
            Exception: If all API keys have been exhausted
        """
        max_retries = len(self.api_keys)
        
        for attempt in range(max_retries):
            try:
                audio = self.client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=model_id
                )
                return audio
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a quota/rate limit error
                is_quota_error = any(keyword in error_str for keyword in [
                    'quota', 'rate limit', 'resource exhausted', '429', 
                    'too many requests', 'limit exceeded', 'insufficient'
                ])
                
                if is_quota_error or attempt == 0:  # Try next key on any error for first attempt
                    print(f"[WARNING] ElevenLabs API key #{self.current_key_index + 1} error: {e}")
                    
                    # Try switching to next key
                    if self._switch_to_next_key():
                        print(f"[INFO] Retrying with ElevenLabs API key #{self.current_key_index + 1}...")
                        continue
                    else:
                        print("[ERROR] All ElevenLabs API keys have failed")
                        raise Exception("All ElevenLabs API keys exhausted") from e
                else:
                    # Not a quota error, raise immediately
                    raise
        
        # Should not reach here
        raise Exception("Failed to generate audio after all retries")
    
    def get_client(self):
        """Get the current ElevenLabs client instance."""
        return self.client
