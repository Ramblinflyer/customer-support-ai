# app.py
# Main application file for the Support Call Assistant

import os
import json
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

# Import our components
from components.audio_capture import AudioCaptureService
from components.transcription import DeepgramTranscriber
from components.guidance import AgentGuidanceSystem
from components.analyzer import ConversationAnalyzer

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Initialize components
audio_capture = AudioCaptureService()
transcriber = DeepgramTranscriber(api_key=os.getenv('DEEPGRAM_API_KEY'))
guidance_system = AgentGuidanceSystem()
analyzer = ConversationAnalyzer(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Global variables
is_recording = False


@app.route('/')
def index():
    """Render the main interface"""
    return render_template('index.html')


@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    """Start audio recording"""
    global is_recording
    
    try:
        audio_capture.start_capture()
        is_recording = True
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    """Stop audio recording and process the captured audio"""
    global is_recording
    
    try:
        audio_capture.stop_capture()
        is_recording = False
        
        # Get the audio data
        audio_data = audio_capture.get_audio_data()
        
        if audio_data:
            # Transcribe the audio
            transcript = transcriber.transcribe_audio(audio_data)
            
            if transcript:
                # Analyze the transcript to extract information
                extracted_info = analyzer.analyze_transcript(transcript)
                
                # Update guidance system with extracted information
                for field, value in extracted_info.items():
                    if value:
                        guidance_system.update_field(field, value)
                
                # Send transcript and extracted info to client
                socketio.emit('transcript_update', {
                    'text': transcript,
                    'extracted_info': extracted_info
                })
                
                # Send updated completion status
                completion_status = guidance_system.get_completion_status()
                socketio.emit('status_update', completion_status)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/update_field', methods=['POST'])
def update_field():
    """Update a field with collected information"""
    data = request.json
    
    field_name = data.get('field')
    value = data.get('value')
    
    if not field_name:
        return jsonify({'status': 'error', 'message': 'Field name is required'})
    
    try:
        guidance_system.update_field(field_name, value)
        
        # Send updated completion status
        completion_status = guidance_system.get_completion_status()
        socketio.emit('status_update', completion_status)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error updating field: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/advance_stage', methods=['POST'])
def advance_stage():
    """Advance to the next conversation stage"""
    data = request.json
    stage = data.get('stage')
    
    try:
        # If a specific stage was provided, set it directly
        if stage:
            guidance_system.set_stage(stage)
        else:
            # Otherwise advance to the next stage
            guidance_system.advance_stage()
        
        # Send new prompt
        next_prompt = guidance_system.get_next_prompt()
        socketio.emit('prompt_update', {
            'stage': guidance_system.current_stage,
            'prompt': next_prompt
        })
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error advancing stage: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/get_ticket_data', methods=['GET'])
def get_ticket_data():
    """Get the formatted ticket data"""
    try:
        ticket_data = guidance_system.format_ticket_data()
        return jsonify({'status': 'success', 'data': ticket_data})
    except Exception as e:
        logger.error(f"Error getting ticket data: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset the session to start fresh"""
    global guidance_system
    
    try:
        # Create a new guidance system instance
        guidance_system = AgentGuidanceSystem()
        
        # Send new prompt
        next_prompt = guidance_system.get_next_prompt()
        socketio.emit('prompt_update', {
            'stage': guidance_system.current_stage,
            'prompt': next_prompt
        })
        
        # Send updated completion status
        completion_status = guidance_system.get_completion_status()
        socketio.emit('status_update', completion_status)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
