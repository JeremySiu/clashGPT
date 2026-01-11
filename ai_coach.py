"""
Clash Royale AI Coach - Uses Gemini to provide gameplay feedback
Monitors game state and sends updates to Gemini every 10 seconds
"""

import os
import time
import tempfile
import subprocess
from ocr_reader import ClashOCRReader

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
except ImportError:
    print("[WARNING] python-dotenv not installed. Install with: pip install python-dotenv")
    print("[INFO] Will try to use GEMINI_API_KEY from environment variables")

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("[ERROR] google-genai not installed")
    print("Install with: pip install google-genai")
    exit(1)

from gemini_key_manager import GeminiKeyManager

try:
    from elevenlabs import ElevenLabs, save
except ImportError:
    print("[ERROR] elevenlabs not installed")
    print("Install with: pip install elevenlabs")
    exit(1)

from elevenlabs_key_manager import ElevenLabsKeyManager


class ClashAICoach:
    """AI coach that analyzes game state and provides feedback."""
    
    def __init__(self, api_key: str = None):
        """Initialize AI coach."""
        # Initialize Gemini with key manager for automatic failover
        try:
            self.key_manager = GeminiKeyManager()
            self.client = self.key_manager.get_client()
        except Exception as e:
            print(f"[ERROR] Failed to initialize Gemini: {e}")
            print("Make sure to set GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. in your .env file")
            exit(1)
        
        # Initialize ElevenLabs client with key manager
        try:
            self.elevenlabs_manager = ElevenLabsKeyManager()
            self.tts_enabled = True
            print("[INFO] ElevenLabs TTS enabled")
        except Exception as e:
            print(f"[WARNING] ElevenLabs initialization failed: {e}")
            print("[INFO] TTS will be disabled.")
            self.tts_enabled = False
            self.elevenlabs_manager = None
        
        # Initialize OCR reader
        self.ocr = ClashOCRReader()
        
        # Game state tracking
        self.game_history = []
        self.last_feedback = None
        
        print("[INFO] AI Coach initialized with Gemini 2.5 Flash")
        print("[INFO] ElevenLabs TTS initialized")
    
    def format_game_state(self, timer: int, hp_data: dict) -> str:
        """Format game state into a readable string."""
        if timer is not None:
            time_str = f"{timer//60}:{timer%60:02d}"
        else:
            time_str = "Unknown"
        
        opp_left = f"{hp_data['opponent_left_hp']:.1f}%" if hp_data.get('opponent_left_hp') else "N/A"
        opp_right = f"{hp_data['opponent_right_hp']:.1f}%" if hp_data.get('opponent_right_hp') else "N/A"
        player_left = f"{hp_data['player_left_hp']:.1f}%" if hp_data.get('player_left_hp') else "N/A"
        player_right = f"{hp_data['player_right_hp']:.1f}%" if hp_data.get('player_right_hp') else "N/A"
        
        return f"Time: {time_str} | Opponent Towers: Left {opp_left}, Right {opp_right} | Your Towers: Left {player_left}, Right {player_right}"
    
    def speak(self, text: str):
        """Convert text to speech and play it using ElevenLabs."""
        if not self.tts_enabled:
            return
        
        try:
            # Generate audio using key manager
            audio = self.elevenlabs_manager.text_to_speech_convert(
                voice_id="EXAVITQu4vr4xnSDxMaL",  # Rachel voice ID
                text=text,
                model_id="eleven_turbo_v2_5"  # Free tier model
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                save(audio, temp_file.name)
                temp_path = temp_file.name
            
            # Play using Windows Media Player (wmplayer.exe) in silent mode
            subprocess.Popen(["powershell", "-WindowStyle", "Hidden", "-c", 
                            f"Add-Type -AssemblyName presentationCore; $player = New-Object System.Windows.Media.MediaPlayer; $player.Open('{temp_path}'); $player.Play(); Start-Sleep -Seconds 10"], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        except Exception as e:
            print(f"[ERROR] Failed to play audio: {e}")
    
    def get_ai_feedback(self, current_state: str) -> str:
        """Get gameplay feedback from Gemini."""
        # Build context with recent history
        history_text = "\n".join(self.game_history[-3:]) if self.game_history else "No previous data"
        
        prompt = f"""You are a Clash Royale coach. Analyze this game state and provide SHORT tactical advice (max 2 sentences).

Recent history:
{history_text}

Current state:
{current_state}

Give brief, actionable advice about:
- Which tower to focus on
- Whether to play aggressive or defensive
- Any urgent concerns

Keep it SHORT and tactical."""
        
        try:
            response = self.key_manager.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"[Error getting AI feedback: {e}]"
    
    def run(self):
        """Main loop - monitor game and provide feedback."""
        print("[INFO] Starting AI Coach...")
        print("[INFO] Waiting 10 seconds before calibration...")
        
        # Wait and calibrate
        for i in range(10, 0, -1):
            print(f"  Starting in {i}...")
            time.sleep(1)
        
        print("\n[INFO] Calibrating HP bars...")
        calibration_attempts = 0
        while not self.ocr.calibrated:
            calibration_attempts += 1
            print(f"[ATTEMPT {calibration_attempts}] Searching for HP bars...")
            
            if self.ocr.calibrate_hp_bars():
                print("[SUCCESS] HP bars calibrated! Starting monitoring...")
                break
            else:
                print("[WAITING] Bars not detected. Retrying in 1 second...")
                time.sleep(1)
        
        print("\n" + "="*70)
        print("AI COACH ACTIVE - Providing feedback every 10 seconds")
        print("="*70)
        
        try:
            while True:
                loop_start = time.time()
                
                # Read game state
                timer_seconds = self.ocr.read_timer_only()
                hp_data = self.ocr.read_tower_hp_bars()
                
                # Format current state
                current_state = self.format_game_state(timer_seconds, hp_data)
                
                # Display current state
                print(f"\n[GAME STATE]")
                print(f"  {current_state}")
                
                # Get AI feedback
                print(f"\n[AI COACH] Analyzing...")
                feedback = self.get_ai_feedback(current_state)
                
                print(f"[AI FEEDBACK]")
                print(f"  {feedback}")
                
                # Speak the feedback
                self.speak(feedback)
                
                # Store in history
                self.game_history.append(current_state)
                if len(self.game_history) > 10:
                    self.game_history.pop(0)
                
                # Wait 10 seconds
                elapsed = time.time() - loop_start
                sleep_time = max(0, 10.0 - elapsed)
                
                print(f"\n[NEXT UPDATE IN {sleep_time:.1f}s]")
                print("-"*70)
                
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n\n[INFO] AI Coach stopped.")


if __name__ == "__main__":
    # Usage: Set GEMINI_API_KEY environment variable
    # Or pass directly: coach = ClashAICoach(api_key="your-key-here")
    
    coach = ClashAICoach()
    coach.run()
