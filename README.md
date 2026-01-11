# ClashGPT - AI Coach for Clash Royale

An AI-powered real-time coach for Clash Royale that uses OCR to read your game state and provides tactical feedback using Google's Gemini AI, with text-to-speech powered by ElevenLabs.

## Features

- 🎮 Real-time game state monitoring via OCR
- 🤖 AI-powered tactical coaching using Gemini 2.5 Flash
- 🗣️ Multiple coach personalities with unique voices
- 🔄 Automatic API key failover for uninterrupted service
- 📊 Tower HP tracking and battle timer display
- 🎨 Beautiful web interface with character selection

## Requirements

- Windows OS (uses Windows Media Player for audio)
- Python 3.8+
- Clash Royale running on your screen
- Gemini API key(s) (up to 7 for failover)
- ElevenLabs API key(s) (up to 2 for failover)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ClashGPT.git
cd ClashGPT
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install flask flask-socketio google-genai elevenlabs python-dotenv easyocr opencv-python numpy Pillow mss
```

4. **Create a `.env` file with your API keys:**
```env
# Gemini API Keys (add up to 7 for automatic failover)
GEMINI_API_KEY_1=your_first_gemini_key_here
GEMINI_API_KEY_2=your_second_gemini_key_here
GEMINI_API_KEY_3=
GEMINI_API_KEY_4=
GEMINI_API_KEY_5=
GEMINI_API_KEY_6=
GEMINI_API_KEY_7=

# ElevenLabs API Keys (add up to 2 for automatic failover)
ELEVENLABS_API_KEY_1=your_elevenlabs_key_here
ELEVENLABS_API_KEY_2=
```

## Usage

1. **Start the application:**
```bash
python web_app.py
```

2. **Open your browser:**
Navigate to `http://localhost:5000`

3. **Position Clash Royale:**
Move your Clash Royale window to the **right side** of your screen for optimal OCR detection

4. **Select your coach:**
Choose from 5 different coaching personalities, each with unique voices and styles

5. **Start a battle:**
Click "Start Battle" then immediately begin a match in Clash Royale

## Coaching Characters

- 😊 **Sunny** - The Optimist: Always encouraging and positive
- 😈 **Blaze** - The Savage: Brutally honest and aggressive
- 🧠 **Atlas** - The Analyst: Data-driven and strategic
- 😎 **Maverick** - The Chill: Laid-back and casual
- 🎩 **Sterling** - The Gentleman: Polite and sophisticated

## API Key Failover

The app automatically switches to backup API keys when quota limits are reached:
- **Gemini**: Supports up to 7 API keys
- **ElevenLabs**: Supports up to 2 API keys

Console output will show which key is currently active and when switches occur.

## Troubleshooting

- **OCR not detecting game**: Make sure Clash Royale is on the right side of your screen
- **No audio**: Check that your ElevenLabs API key is valid
- **Quota errors**: Add additional API keys to your `.env` file
- **Calibration failing**: Start the battle immediately after clicking "Start Battle"

## Technologies Used

- **Flask & Flask-SocketIO** - Web framework and real-time communication
- **Google Gemini 2.5 Flash** - AI coaching intelligence
- **ElevenLabs** - Text-to-speech generation
- **EasyOCR** - Optical character recognition
- **OpenCV & MSS** - Screen capture and image processing

## Note

This application is designed to run **locally** on your machine. It requires access to your screen to monitor the game, which cannot work on cloud hosting platforms like Vercel or Heroku.

## License

MIT License - Feel free to use and modify!

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
