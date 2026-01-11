"""
Web interface for Clash Royale AI Coach
"""

import os
import time
import base64
import tempfile
from threading import Thread, Event
from flask import Flask, render_template, jsonify, send_file
from flask_socketio import SocketIO, emit
from ocr_reader import ClashOCRReader

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    print("[ERROR] google-genai not installed")
    exit(1)

from gemini_key_manager import GeminiKeyManager

try:
    from elevenlabs import ElevenLabs, save
except ImportError:
    print("[ERROR] elevenlabs not installed")
    exit(1)

from elevenlabs_key_manager import ElevenLabsKeyManager


app = Flask(__name__)
app.config['SECRET_KEY'] = 'clash-royale-ai-coach'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
battle_active = False
battle_thread = None
stop_event = Event()
selected_character = 'optimist'  # Default character

# Coaching Characters with unique voices and personalities
COACHING_CHARACTERS = {
    'optimist': {
        'name': 'Sunny - The Optimist',
        'voice_id': 'EXAVITQu4vr4xnSDxMaL',  # Rachel
        'emoji': '😊',
        'color': '#4CAF50',
        'prompt_style': '''You are Sunny, an incredibly optimistic and encouraging Clash Royale coach. 
You ALWAYS find the positive in every situation. Even when towers are low, you focus on comeback potential.
You're supportive, upbeat, and believe in the player. Keep advice SHORT (max 2 sentences) but motivational.'''
    },
    'critic': {
        'name': 'Maven - The Analyst',
        'voice_id': 'pNInz6obpgDQGcFmaJgB',  # Adam
        'emoji': '🤔',
        'color': '#2196F3',
        'prompt_style': '''You are Maven, a critical and analytical Clash Royale coach.
You point out mistakes, inefficiencies, and missed opportunities with precision.
You're blunt, honest, and focused on improvement. Keep advice SHORT (max 2 sentences) but cutting.'''
    },
    'trash_talker': {
        'name': 'Blaze - The Trash Talker',
        'voice_id': 'VR6AewLTigWG4xSOukaG',  # Antoni
        'emoji': '🔥',
        'color': '#f44336',
        'prompt_style': '''You are Blaze, an aggressive trash-talking Clash Royale coach.
You mock the opponent relentlessly and hype up aggressive plays. You're cocky, bold, and in-your-face.
Make fun of the opponent's low towers. Keep it SHORT (max 2 sentences) but savage.'''
    },
    'strategist': {
        'name': 'Atlas - The Strategist',
        'voice_id': 'ErXwobaYiN019PkySvjV',  # Antoni (different style)
        'emoji': '🧠',
        'color': '#9C27B0',
        'prompt_style': '''You are Atlas, a cerebral and strategic Clash Royale coach.
You focus on tactical positioning, resource management, and calculated plays.
You're calm, methodical, and data-driven. Keep advice SHORT (max 2 sentences) but tactical.'''
    },
    'hype_man': {
        'name': 'Volt - The Hype Man',
        'voice_id': 'TxGEqnHWrfWFTfGW9XjX',  # Josh
        'emoji': '⚡',
        'color': '#FF9800',
        'prompt_style': '''You are Volt, an explosive and energetic Clash Royale coach!
You're HYPED about EVERYTHING! You scream encouragement and get excited about big plays.
You're loud, enthusiastic, and all CAPS energy! Keep it SHORT (max 2 sentences) but ELECTRIC!'''
    }
}


class WebAICoach:
    """AI coach that sends updates via WebSocket"""
    
    def __init__(self, character_id='optimist'):
        # Initialize Gemini with key manager for automatic failover
        try:
            self.key_manager = GeminiKeyManager()
            self.client = self.key_manager.get_client()
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini: {e}")
        
        # Set character
        self.character = COACHING_CHARACTERS.get(character_id, COACHING_CHARACTERS['optimist'])
        
        # Initialize ElevenLabs with key manager
        try:
            self.elevenlabs_manager = ElevenLabsKeyManager()
            self.tts_enabled = True
        except Exception as e:
            print(f"[WARNING] ElevenLabs initialization failed: {e}")
            self.tts_enabled = False
            self.elevenlabs_manager = None
        
        # Initialize OCR
        self.ocr = ClashOCRReader()
        self.game_history = []
    
    def format_game_state(self, timer: int, hp_data: dict) -> str:
        """Format game state into readable string"""
        if timer is not None:
            time_str = f"{timer//60}:{timer%60:02d}"
        else:
            time_str = "Unknown"
        
        opp_left = f"{hp_data['opponent_left_hp']:.1f}%" if hp_data.get('opponent_left_hp') else "N/A"
        opp_right = f"{hp_data['opponent_right_hp']:.1f}%" if hp_data.get('opponent_right_hp') else "N/A"
        player_left = f"{hp_data['player_left_hp']:.1f}%" if hp_data.get('player_left_hp') else "N/A"
        player_right = f"{hp_data['player_right_hp']:.1f}%" if hp_data.get('player_right_hp') else "N/A"
        
        return f"Time: {time_str} | Opponent: Left {opp_left}, Right {opp_right} | You: Left {player_left}, Right {player_right}"
    
    def get_ai_feedback(self, current_state: str) -> str:
        """Get AI feedback from Gemini using character's personality"""
        history_text = "\n".join(self.game_history[-3:]) if self.game_history else "No previous data"
        
        prompt = f"""{self.character['prompt_style']}

Recent history:
{history_text}

Current state:
{current_state}

Provide your coaching feedback now (max 2 sentences):"""
        
        try:
            response = self.key_manager.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"[Error: {e}]"
    
    def generate_audio(self, text: str) -> tuple:
        """Generate audio and return base64 encoded data with duration"""
        if not self.tts_enabled:
            return None, 0
        
        try:
            audio = self.elevenlabs_manager.text_to_speech_convert(
                voice_id=self.character['voice_id'],
                text=text,
                model_id="eleven_turbo_v2_5"
            )
            
            # Save to temp file and read as base64
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_path = temp_file.name
                save(audio, temp_path)
            
            # Read the file
            time.sleep(0.1)  # Small delay to ensure file is written
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
                audio_data = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Estimate duration (rough approximation: ~1000 bytes per 0.1 second for MP3)
            estimated_duration = max(3, len(audio_bytes) / 10000)  # At least 3 seconds
            
            # Clean up - try multiple times if needed
            for _ in range(3):
                try:
                    os.unlink(temp_path)
                    break
                except PermissionError:
                    time.sleep(0.1)
            
            return audio_data, estimated_duration
        except Exception as e:
            print(f"[ERROR] Audio generation failed: {e}")
            return None, 0
    
    def run(self):
        """Main monitoring loop"""
        global battle_active
        
        socketio.emit('status', {'message': 'Waiting 10 seconds...', 'type': 'info'})
        
        # Wait before calibration
        for i in range(10, 0, -1):
            if stop_event.is_set():
                return
            socketio.emit('status', {'message': f'Starting in {i}...', 'type': 'info'})
            time.sleep(1)
        
        # Calibrate
        socketio.emit('status', {'message': 'Calibrating HP bars...', 'type': 'info'})
        calibration_attempts = 0
        while not self.ocr.calibrated and calibration_attempts < 10:
            if stop_event.is_set():
                return
            calibration_attempts += 1
            if self.ocr.calibrate_hp_bars():
                socketio.emit('status', {'message': 'Calibrated! Monitoring...', 'type': 'success'})
                break
            time.sleep(1)
        
        if not self.ocr.calibrated:
            socketio.emit('status', {'message': 'Calibration failed', 'type': 'error'})
            battle_active = False
            return
        
        # Monitor loop
        while battle_active and not stop_event.is_set():
            loop_start = time.time()
            
            # Read game state
            timer_seconds = self.ocr.read_timer_only()
            hp_data = self.ocr.read_tower_hp_bars()
            current_state = self.format_game_state(timer_seconds, hp_data)
            
            # Send game state
            socketio.emit('game_state', {
                'state': current_state,
                'timer': timer_seconds,
                'hp': hp_data
            })
            
            # Get AI feedback
            feedback = self.get_ai_feedback(current_state)
            
            # Generate audio
            audio_data, audio_duration = self.generate_audio(feedback)
            
            # Send feedback with audio
            socketio.emit('ai_feedback', {
                'text': feedback,
                'audio': audio_data,
                'duration': audio_duration,
                'character': {
                    'name': self.character['name'],
                    'emoji': self.character['emoji'],
                    'color': self.character['color']
                }
            })
            
            # Store history
            self.game_history.append(current_state)
            if len(self.game_history) > 10:
                self.game_history.pop(0)
            
            # Calculate wait time (15 seconds total, but ensure audio finishes)
            elapsed = time.time() - loop_start
            min_wait = max(audio_duration + 1, 15.0)  # At least audio duration + 1 sec buffer, or 15 seconds
            sleep_time = max(0, min_wait - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)


def battle_worker():
    """Background worker for battle monitoring"""
    global selected_character
    try:
        coach = WebAICoach(character_id=selected_character)
        coach.run()
    except Exception as e:
        socketio.emit('status', {'message': f'Error: {str(e)}', 'type': 'error'})
    finally:
        global battle_active
        battle_active = False
        socketio.emit('battle_stopped', {})


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    """Client connected"""
    emit('status', {'message': 'Connected to AI Coach', 'type': 'info'})
    emit('characters', {'characters': COACHING_CHARACTERS, 'selected': selected_character})


@socketio.on('select_character')
def handle_select_character(data):
    """Change the coaching character"""
    global selected_character
    
    character_id = data.get('character_id')
    if character_id in COACHING_CHARACTERS:
        selected_character = character_id
        character = COACHING_CHARACTERS[character_id]
        emit('character_selected', {
            'character_id': character_id,
            'name': character['name'],
            'emoji': character['emoji']
        }, broadcast=True)
        emit('status', {'message': f'Selected {character["name"]}', 'type': 'success'})


@socketio.on('start_battle')
def handle_start_battle():
    """Start battle monitoring"""
    global battle_active, battle_thread, stop_event
    
    if battle_active:
        emit('status', {'message': 'Battle already active', 'type': 'warning'})
        return
    
    battle_active = True
    stop_event.clear()
    battle_thread = Thread(target=battle_worker)
    battle_thread.start()
    emit('battle_started', {})


@socketio.on('stop_battle')
def handle_stop_battle():
    """Stop battle monitoring"""
    global battle_active, stop_event
    
    if not battle_active:
        emit('status', {'message': 'No battle active', 'type': 'warning'})
        return
    
    battle_active = False
    stop_event.set()
    emit('battle_stopped', {})


if __name__ == '__main__':
    print("[INFO] Starting Clash Royale AI Coach Web Interface")
    print("[INFO] Open http://localhost:5000 in your browser")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
