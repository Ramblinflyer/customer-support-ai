# Customer Support AI Assistant - Sample Implementation
# This sample demonstrates core functionality for the audio capture, 
# transcription, and analysis components

import pyaudio
import wave
import numpy as np
import threading
import time
import json
import requests
import base64
from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioCapture:
    """Captures system audio input and output"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024, channels=2, device_index=None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.lock = threading.Lock()
        
    def start_recording(self):
        """Start audio recording"""
        self.is_recording = True
        self.frames = []
        
        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback
        )
        
        logger.info("Recording started")
        
    def _callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream processing"""
        with self.lock:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def stop_recording(self):
        """Stop audio recording"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        self.is_recording = False
        logger.info("Recording stopped")
        
    def get_recent_audio(self, seconds=5):
        """Get the most recent N seconds of audio"""
        with self.lock:
            # Calculate how many frames we need for N seconds
            frames_needed = int((self.sample_rate / self.chunk_size) * seconds)
            recent_frames = self.frames[-frames_needed:] if frames_needed < len(self.frames) else self.frames
            
        return b''.join(recent_frames)
    
    def save_recording(self, filename="recording.wav"):
        """Save recording to a WAV file"""
        with self.lock:
            if not self.frames:
                logger.warning("No audio data to save")
                return False
            
            try:
                wf = wave.open(filename, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
                wf.close()
                logger.info(f"Recording saved to {filename}")
                return True
            except Exception as e:
                logger.error(f"Error saving recording: {e}")
                return False
    
    def __del__(self):
        """Clean up resources"""
        if self.stream:
            self.stream.close()
        self.audio.terminate()


class Transcriber:
    """Transcribes audio to text using Whisper API"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.transcript_buffer = []
        self.last_transcription_time = 0
        
    def transcribe_audio(self, audio_data, sample_rate=44100):
        """Transcribe audio data using Whisper API"""
        # Convert audio data to proper format
        # For OpenAI Whisper API, we need to save it as a file first
        temp_filename = "temp_audio.wav"
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(2)  # Stereo
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        
        try:
            # Call Whisper API
            with open(temp_filename, "rb") as audio_file:
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": audio_file},
                    data={
                        "model": "whisper-large-v3",
                        "language": "en",
                        "response_format": "verbose_json",
                        "timestamp_granularities": ["segment"]
                    }
                )
            
            if response.status_code != 200:
                logger.error(f"API error: {response.text}")
                return None
            
            result = response.json()
            # Update transcript buffer with new text
            current_time = time.time()
            
            # Extract segments with timestamps
            for segment in result.get("segments", []):
                segment_text = segment.get("text", "").strip()
                if segment_text:
                    self.transcript_buffer.append({
                        "text": segment_text,
                        "timestamp": current_time,
                        "start": segment.get("start"),
                        "end": segment.get("end"),
                        # In a real implementation, we'd use a speaker diarization service
                        # For now, we're just using a placeholder
                        "speaker": "unknown"
                    })
            
            self.last_transcription_time = current_time
            
            # Return the latest transcription
            return result.get("text", "")
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
        finally:
            # Clean up temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    
    def get_full_transcript(self):
        """Get the full transcript with timestamps"""
        return self.transcript_buffer
    
    def get_recent_transcript(self, seconds=60):
        """Get transcript from the last N seconds"""
        current_time = time.time()
        cutoff_time = current_time - seconds
        
        recent = [
            segment for segment in self.transcript_buffer 
            if segment["timestamp"] >= cutoff_time
        ]
        
        return recent
    
    def get_text_only_transcript(self, recent_only=True, seconds=60):
        """Get transcript text only without metadata"""
        segments = self.get_recent_transcript(seconds) if recent_only else self.transcript_buffer
        return " ".join([segment["text"] for segment in segments])


class ConversationAnalyzer:
    """Analyzes conversation transcripts using Claude API"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
    
    def _call_claude_api(self, system_prompt, user_message):
        """Make a call to Claude API"""
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "claude-3-sonnet-20240229",
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_message}],
                    "temperature": 0.2,
                    "max_tokens": 1000
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Claude API error: {response.text}")
                return None
            
            result = response.json()
            return result["content"][0]["text"]
            
        except Exception as e:
            logger.error(f"Claude API call error: {e}")
            return None
    
    def extract_information(self, transcript):
        """Extract key information from the transcript"""
        system_prompt = """
        You are an AI assistant for customer support. Extract the following information from the conversation transcript:
        - Customer name and contact information
        - Product or service being discussed
        - Issue description
        - Steps already taken by the customer
        - Relevant account information
        - Action items and next steps
        
        Return the information in JSON format only with these keys:
        customer_name, contact_info, product, issue_description, steps_taken, account_info, action_items
        """
        
        result = self._call_claude_api(system_prompt, transcript)
        
        if result:
            try:
                # Extract JSON from the response
                # Sometimes Claude might wrap the JSON in markdown code blocks
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    json_str = result.split("```")[1].strip()
                else:
                    json_str = result
                
                extracted_info = json.loads(json_str)
                return extracted_info
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}, Response: {result}")
                return None
        
        return None
    
    def categorize_issue(self, transcript):
        """Categorize the customer issue"""
        system_prompt = """
        You are an AI assistant for customer support. Categorize this customer support conversation into one of the following categories:
        - Technical Issue
        - Billing Question
        - Product Information
        - Feature Request
        - Complaint
        - Other
        
        Also identify any specific products or services mentioned.
        
        Return the information in JSON format only with these keys:
        category, subcategory, products, priority (low, medium, high)
        """
        
        result = self._call_claude_api(system_prompt, transcript)
        
        if result:
            try:
                # Extract JSON similarly to extract_information
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    json_str = result.split("```")[1].strip()
                else:
                    json_str = result
                
                category_info = json.loads(json_str)
                return category_info
            except json.JSONDecodeError:
                logger.error(f"JSON decode error in categorization: {result}")
                return None
        
        return None
    
    def analyze_sentiment(self, transcript):
        """Analyze sentiment of the conversation"""
        system_prompt = """
        You are an AI assistant for customer support. Analyze the sentiment of this conversation.
        
        Return the analysis in JSON format only with these keys:
        overall_sentiment (numeric score from -1.0 to 1.0),
        customer_sentiment (numeric score from -1.0 to 1.0),
        agent_sentiment (numeric score from -1.0 to 1.0),
        sentiment_trends (list of key observations)
        """
        
        result = self._call_claude_api(system_prompt, transcript)
        
        if result:
            try:
                # Extract JSON similarly
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    json_str = result.split("```")[1].strip()
                else:
                    json_str = result
                
                sentiment_info = json.loads(json_str)
                return sentiment_info
            except json.JSONDecodeError:
                logger.error(f"JSON decode error in sentiment analysis: {result}")
                return None
        
        return None


class AgentAssistant:
    """Provides real-time assistance to support agents"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.analyzer = ConversationAnalyzer(self.api_key)
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
    
    def generate_suggestions(self, transcript, customer_info=None, issue_category=None):
        """Generate helpful suggestions for the agent"""
        context = ""
        if customer_info:
            context += f"\nCustomer Info: {json.dumps(customer_info)}"
        if issue_category:
            context += f"\nIssue Category: {json.dumps(issue_category)}"
        
        system_prompt = """
        You are an AI assistant for customer support agents.
        Based on the ongoing conversation, provide helpful suggestions to the agent.
        These may include:
        - Information that should be collected from the customer
        - Potential solutions to try
        - Relevant policy information
        - Tone and sentiment guidance
        - Additional questions to ask
        
        Keep suggestions concise and actionable. Format your response as a JSON array with each suggestion as an object with "type" and "text" fields.
        """
        
        user_message = f"Here is the current conversation transcript:\n\n{transcript}{context}"
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "claude-3-haiku-20240307",  # Using a faster model for real-time suggestions
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_message}],
                    "temperature": 0.3,
                    "max_tokens": 500
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Claude API error: {response.text}")
                return []
            
            result = response.json()
            content = result["content"][0]["text"]
            
            try:
                # Parse the JSON response
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].strip()
                else:
                    json_str = content
                
                suggestions = json.loads(json_str)
                return suggestions
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in suggestions: {e}")
                return []
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []


class CRMIntegration:
    """Integrates with CRM systems to log customer interactions"""
    
    def __init__(self, api_url=None, api_key=None, crm_type="e-automate"):
        self.api_url = api_url or os.getenv('CRM_API_URL')
        self.api_key = api_key or os.getenv('CRM_API_KEY')
        self.crm_type = crm_type
        
    def create_ticket(self, customer_info, issue_info, transcript=None):
        """Create a new ticket in the CRM system"""
        # For demonstration, we'll just simulate the API call
        
        # In a real implementation, you would:
        # 1. Format the data according to your CRM's API requirements
        # 2. Make an HTTP request to create the ticket
        # 3. Handle the response and return the ticket ID
        
        # Simulate successful ticket creation
        ticket_id = f"TICKET-{int(time.time())}"
        logger.info(f"Created ticket {ticket_id} in {self.crm_type}")
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "message": f"Ticket created successfully in {self.crm_type}"
        }
    
    def update_ticket(self, ticket_id, update_data):
        """Update an existing ticket"""
        # Simulate ticket update
        logger.info(f"Updated ticket {ticket_id} with {update_data}")
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "message": "Ticket updated successfully"
        }
    
    def log_interaction(self, ticket_id, transcript, sentiment=None):
        """Log the conversation in the ticket"""
        # Simulate logging the interaction
        logger.info(f"Logged interaction for ticket {ticket_id}")
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "message": "Interaction logged successfully"
        }


# Flask web application for the dashboard and agent interface
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables to maintain state
audio_capture = None
transcriber = None
analyzer = None
agent_assistant = None
crm = None
current_ticket_id = None
is_processing = False
processing_thread = None


@app.route('/')
def index():
    """Render the main dashboard"""
    return render_template('index.html')


@app.route('/agent')
def agent_view():
    """Render the agent interface"""
    return render_template('agent.html')


@app.route('/admin')
def admin_view():
    """Render the admin interface"""
    return render_template('admin.html')


@app.route('/api/start', methods=['POST'])
def start_recording():
    """Start the recording and processing"""
    global audio_capture, transcriber, analyzer, agent_assistant, crm, is_processing, processing_thread
    
    if is_processing:
        return jsonify({"success": False, "message": "Already recording"})
    
    # Initialize components if needed
    if not audio_capture:
        audio_capture = AudioCapture()
    
    if not transcriber:
        transcriber = Transcriber()
    
    if not analyzer:
        analyzer = ConversationAnalyzer()
    
    if not agent_assistant:
        agent_assistant = AgentAssistant()
    
    if not crm:
        crm = CRMIntegration()
    
    # Start audio capture
    audio_capture.start_recording()
    
    # Start processing thread
    is_processing = True
    processing_thread = threading.Thread(target=process_audio_stream)
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({"success": True, "message": "Recording started"})


@app.route('/api/stop', methods=['POST'])
def stop_recording():
    """Stop the recording and processing"""
    global audio_capture, is_processing, current_ticket_id
    
    if not is_processing:
        return jsonify({"success": False, "message": "Not recording"})
    
    # Stop processing
    is_processing = False
    
    # Wait for processing thread to complete
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=5.0)
    
    # Stop audio capture
    if audio_capture:
        audio_capture.stop_recording()
        audio_capture.save_recording(f"recording_{current_ticket_id}.wav")
    
    return jsonify({"success": True, "message": "Recording stopped"})


@app.route('/api/transcript', methods=['GET'])
def get_transcript():
    """Get the current transcript"""
    global transcriber
    
    if not transcriber:
        return jsonify({"success": False, "message": "Transcriber not initialized"})
    
    full_transcript = transcriber.get_full_transcript()
    return jsonify({"success": True, "transcript": full_transcript})


@app.route('/api/analysis', methods=['GET'])
def get_analysis():
    """Get the current conversation analysis"""
    global analyzer, transcriber
    
    if not analyzer or not transcriber:
        return jsonify({"success": False, "message": "Analyzer not initialized"})
    
    # Get the recent transcript
    transcript_text = transcriber.get_text_only_transcript(recent_only=True)
    
    if not transcript_text:
        return jsonify({"success": False, "message": "No transcript available"})
    
    # Get information extraction
    extracted_info = analyzer.extract_information(transcript_text)
    
    # Get issue categorization
    category_info = analyzer.categorize_issue(transcript_text)
    
    # Get sentiment analysis
    sentiment_info = analyzer.analyze_sentiment(transcript_text)
    
    return jsonify({
        "success": True,
        "extracted_info": extracted_info,
        "category_info": category_info,
        "sentiment_info": sentiment_info
    })


@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    """Get suggestions for the agent"""
    global agent_assistant, transcriber, analyzer
    
    if not agent_assistant or not transcriber:
        return jsonify({"success": False, "message": "Agent assistant not initialized"})
    
    # Get the transcript
    transcript_text = transcriber.get_text_only_transcript()
    
    if not transcript_text:
        return jsonify({"success": False, "message": "No transcript available"})
    
    # Get customer info and issue category if available
    customer_info = None
    issue_category = None
    
    if analyzer:
        customer_info = analyzer.extract_information(transcript_text)
        issue_category = analyzer.categorize_issue(transcript_text)
    
    # Get suggestions
    suggestions = agent_assistant.generate_suggestions(transcript_text, customer_info, issue_category)
    
    return jsonify({
        "success": True,
        "suggestions": suggestions
    })


@app.route('/api/create_ticket', methods=['POST'])
def create_new_ticket():
    """Create a new ticket in the CRM"""
    global crm, analyzer, transcriber, current_ticket_id
    
    if not crm or not analyzer or not transcriber:
        return jsonify({"success": False, "message": "CRM integration not initialized"})
    
    # Get the transcript
    transcript_text = transcriber.get_text_only_transcript(recent_only=False)
    
    if not transcript_text:
        return jsonify({"success": False, "message": "No transcript available"})
    
    # Get customer info and issue details
    customer_info = analyzer.extract_information(transcript_text)
    issue_info = analyzer.categorize_issue(transcript_text)
    
    if not customer_info or not issue_info:
        return jsonify({"success": False, "message": "Unable to extract required information"})
    
    # Create ticket
    result = crm.create_ticket(customer_info, issue_info, transcript_text)
    
    if result["success"]:
        current_ticket_id = result["ticket_id"]
    
    return jsonify(result)


def process_audio_stream():
    """Process the audio stream continuously"""
    global audio_capture, transcriber, analyzer, agent_assistant, is_processing
    
    logger.info("Starting audio processing thread")
    
    chunk_interval = 5  # Process in 5-second chunks
    last_analysis_time = 0
    analysis_interval = 15  # Full analysis every 15 seconds
    
    while is_processing:
        try:
            # Get recent audio chunk
            audio_chunk = audio_capture.get_recent_audio(seconds=chunk_interval)
            
            if audio_chunk and len(audio_chunk) > 0:
                # Transcribe the audio
                transcription = transcriber.transcribe_audio(audio_chunk)
                
                if transcription:
                    # Emit transcription update to connected clients
                    socketio.emit('transcription_update', {
                        'text': transcription,
                        'timestamp': time.time()
                    })
                    
                    current_time = time.time()
                    
                    # Run full analysis periodically
                    if current_time - last_analysis_time >= analysis_interval:
                        # Get full transcript for analysis
                        full_transcript = transcriber.get_text_only_transcript()
                        
                        # Run information extraction
                        if analyzer:
                            info = analyzer.extract_information(full_transcript)
                            if info:
                                socketio.emit('info_update', info)
                            
                            # Run sentiment analysis
                            sentiment = analyzer.analyze_sentiment(full_transcript)
                            if sentiment:
                                socketio.emit('sentiment_update', sentiment)
                            
                            # Run categorization
                            category = analyzer.categorize_issue(full_transcript)
                            if category:
                                socketio.emit('category_update', category)
                        
                        # Generate suggestions
                        if agent_assistant:
                            suggestions = agent_assistant.generate_suggestions(full_transcript)
                            if suggestions:
                                socketio.emit('suggestions_update', suggestions)
                        
                        last_analysis_time = current_time
            
            # Small sleep to prevent CPU overuse
            time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            time.sleep(1)  # Sleep on error to prevent tight loop
    
    logger.info("Audio processing thread stopped")


@app.route('/api/dashboard_data', methods=['GET'])
def get_dashboard_data():
    """Get data for the dashboard"""
    # In a real implementation, this would query the metrics database
    # For this sample, we'll return simulated data
    
    return jsonify({
        "success": True,
        "metrics": {
            "avg_resolution_time": 15.3,  # minutes
            "sentiment_trend": [
                {"date": "2025-03-22", "score": 0.65},
                {"date": "2025-03-23", "score": 0.72},
                {"date": "2025-03-24", "score": 0.68},
                {"date": "2025-03-25", "score": 0.71},
                {"date": "2025-03-26", "score": 0.75},
                {"date": "2025-03-27", "score": 0.73},
                {"date": "2025-03-28", "score": 0.74},
            ],
            "issue_categories": [
                {"category": "Technical Issue", "count": 45},
                {"category": "Billing Question", "count": 32},
                {"category": "Product Information", "count": 18},
                {"category": "Feature Request", "count": 12},
                {"category": "Complaint", "count": 8},
                {"category": "Other", "count": 5},
            ],
            "agent_performance": [
                {"agent": "John Smith", "score": 92},
                {"agent": "Sarah Johnson", "score": 87},
                {"agent": "Michael Brown", "score": 85},
                {"agent": "Emily Davis", "score": 91},
                {"agent": "Robert Wilson", "score": 83},
            ]
        }
    })


@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get the gamification leaderboard"""
    # In a real implementation, this would query the database
    
    return jsonify({
        "success": True,
        "leaderboard": [
            {
                "agent": "Emily Davis",
                "score": 1250,
                "badges": ["Speed Demon", "Customer Whisperer"],
                "metrics": {
                    "avg_resolution_time": 12.1,
                    "customer_satisfaction": 4.8,
                    "tickets_resolved": 52
                }
            },
            {
                "agent": "John Smith",
                "score": 1180,
                "badges": ["Knowledge Guru", "Efficiency Expert"],
                "metrics": {
                    "avg_resolution_time": 14.3,
                    "customer_satisfaction": 4.7,
                    "tickets_resolved": 48
                }
            },
            {
                "agent": "Sarah Johnson",
                "score": 1050,
                "badges": ["Customer Whisperer"],
                "metrics": {
                    "avg_resolution_time": 15.2,
                    "customer_satisfaction": 4.6,
                    "tickets_resolved": 45
                }
            },
            {
                "agent": "Michael Brown",
                "score": 980,
                "badges": ["Efficiency Expert"],
                "metrics": {
                    "avg_resolution_time": 13.8,
                    "customer_satisfaction": 4.5,
                    "tickets_resolved": 43
                }
            },
            {
                "agent": "Robert Wilson",
                "score": 920,
                "badges": [],
                "metrics": {
                    "avg_resolution_time": 16.4,
                    "customer_satisfaction": 4.4,
                    "tickets_resolved": 40
                }
            }
        ]
    })


if __name__ == '__main__':
    # Start the Flask app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

    