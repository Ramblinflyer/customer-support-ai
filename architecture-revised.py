# Support Call Assistant - Revised Architecture
# Windows-compatible implementation using Deepgram instead of PyAudio

# ===================================
# Component 1: Audio Capture Service (Windows-compatible)
# ===================================
class AudioCaptureService:
    """
    Captures system audio using Windows-compatible methods.
    Based on a combination of sounddevice and webrtcvad for voice detection.
    """
    
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.frames = []
        
    def start_capture(self):
        """Start capturing audio from microphone"""
        import sounddevice as sd
        import numpy as np
        import threading
        
        self.is_recording = True
        self.frames = []
        
        def callback(indata, frames, time, status):
            """This is called for each audio frame"""
            if self.is_recording:
                self.frames.append(indata.copy())
        
        # Start the recording stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=callback
        )
        self.stream.start()
        
    def stop_capture(self):
        """Stop audio capture"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
    
    def get_audio_data(self):
        """Get the recorded audio data as bytes"""
        import numpy as np
        
        if not self.frames:
            return None
            
        # Combine all frames into a single numpy array
        audio_data = np.concatenate(self.frames, axis=0)
        
        # Convert to int16 format expected by most speech APIs
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Convert to bytes
        return audio_int16.tobytes()
    
    def save_recording(self, filename="recording.wav"):
        """Save the recording to a WAV file"""
        import numpy as np
        import wave
        import struct
        
        if not self.frames:
            return False
            
        try:
            audio_data = np.concatenate(self.frames, axis=0)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            return True
        except Exception as e:
            print(f"Error saving recording: {e}")
            return False


# ===================================
# Component 2: Deepgram Transcription Service
# ===================================
class DeepgramTranscriber:
    """
    Handles real-time transcription of audio using Deepgram API.
    """
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.transcript_buffer = []
        
    def transcribe_audio(self, audio_data, sample_rate=16000):
        """
        Transcribe audio data using Deepgram API.
        """
        from deepgram import Deepgram
        import asyncio
        import json
        
        dg_client = Deepgram(self.api_key)
        
        # Define options for transcription
        options = {
            "punctuate": True,
            "model": "nova-2",
            "language": "en-US",
            "diarize": True,  # For speaker identification
            "smart_format": True
        }
        
        # Create a function to handle the async Deepgram request
        async def process_audio():
            source = {"buffer": audio_data, "mimetype": "audio/raw"}
            response = await dg_client.transcription.prerecorded(source, options)
            return response
            
        # Run the async function and get the result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(process_audio())
        loop.close()
        
        # Process the response
        if response and 'results' in response:
            result = response['results']
            transcript = result['channels'][0]['alternatives'][0]['transcript']
            
            # Process speaker diarization if available
            if 'speakers' in result:
                speaker_segments = []
                for segment in result['channels'][0]['alternatives'][0]['words']:
                    if 'speaker' in segment:
                        speaker = segment['speaker']
                        word = segment['word']
                        start = segment['start']
                        end = segment['end']
                        
                        # Append to the right speaker segment or create a new one
                        if not speaker_segments or speaker_segments[-1]['speaker'] != speaker:
                            speaker_segments.append({
                                'speaker': speaker,
                                'text': word,
                                'start': start,
                                'end': end
                            })
                        else:
                            speaker_segments[-1]['text'] += ' ' + word
                            speaker_segments[-1]['end'] = end
                
                # Add speaker segments to transcript buffer
                for segment in speaker_segments:
                    self.transcript_buffer.append({
                        'speaker': f"Speaker {segment['speaker']}",
                        'text': segment['text'],
                        'timestamp': segment['start'],
                        'duration': segment['end'] - segment['start']
                    })
            else:
                # If no diarization, just add the full transcript
                self.transcript_buffer.append({
                    'speaker': 'unknown',
                    'text': transcript,
                    'timestamp': 0,
                    'duration': 0
                })
            
            return transcript
        
        return None
    
    def get_full_transcript(self):
        """Get the full transcript with speaker information"""
        return self.transcript_buffer
    
    def get_text_only_transcript(self):
        """Get the transcript text only without metadata"""
        return " ".join([segment['text'] for segment in self.transcript_buffer])


# ===================================
# Component 3: Agent Guidance System
# ===================================
class AgentGuidanceSystem:
    """
    Provides structured prompts to guide agents through information collection.
    Tracks completion status of required information fields.
    """
    
    def __init__(self):
        # Define the required fields for a complete ticket
        self.required_fields = {
            "caller_first_name": {"completed": False, "value": None},
            "caller_last_name": {"completed": False, "value": None},
            "company_name": {"completed": False, "value": None},
            "software_make": {"completed": False, "value": None},
            "software_model": {"completed": False, "value": None},
            "software_version": {"completed": False, "value": None},
            "issue_type": {"completed": False, "value": None},  # "how-to" or "error"
            "issue_description": {"completed": False, "value": None},
            "error_messages": {"completed": False, "value": None},
            "troubleshooting_steps": {"completed": False, "value": None}
        }
        
        # Define conversation stages
        self.stages = [
            "intro",
            "caller_identity",
            "company_info",
            "software_details",
            "issue_details",
            "error_messages",
            "troubleshooting_history",
            "technician_availability",
            "support_options",
            "billing_confirmation",
            "summary",
            "next_steps"
        ]
        
        self.current_stage = "intro"
    
    def get_next_prompt(self):
        """
        Get the next prompt to guide the agent based on current stage
        and collected information.
        """
        prompts = {
            "intro": "Welcome to technical support. Please collect the caller's information.",
            
            "caller_identity": "Ask for the caller's first and last name.",
            
            "company_info": "Confirm the company name the caller is calling from.",
            
            "software_details": (
                "Please ask for the software make, model, and version.\n"
                "Example: Bluebeam Revu 21, SolidWorks 2019, AutoCAD LT 2024"
            ),
            
            "issue_details": (
                "Ask the caller to describe their issue.\n"
                "Clarify if this is a 'how-to' question or an actual software error."
            ),
            
            "error_messages": "Ask if there are any error messages displayed. If yes, ask for the exact text.",
            
            "troubleshooting_history": "Ask what troubleshooting steps they have already attempted.",
            
            "technician_availability": (
                "Check technician availability for this type of issue.\n"
                "Is a technician currently available to assist?"
            ),
            
            "support_options": (
                "Based on the issue type, consider directing to:\n"
                "- Manufacturer (MFG) support chat\n"
                "- AEC support chat\n"
                "- Internal technician support"
            ),
            
            "billing_confirmation": (
                "Reminder: Check Salesforce for TPM1 hours and guide the caller accordingly.\n"
                "Confirm whether this case will be billable and communicate this to the caller."
            ),
            
            "summary": "Confirm all collected information with the caller.",
            
            "next_steps": "Explain the next steps and expected timeline for resolution."
        }
        
        return prompts.get(self.current_stage, "Continue helping the caller.")
    
    def advance_stage(self):
        """Move to the next conversation stage"""
        current_index = self.stages.index(self.current_stage)
        if current_index < len(self.stages) - 1:
            self.current_stage = self.stages[current_index + 1]
    
    def update_field(self, field_name, value):
        """Update a field with collected information"""
        if field_name in self.required_fields:
            self.required_fields[field_name]["value"] = value
            self.required_fields[field_name]["completed"] = True
    
    def get_completion_status(self):
        """Get the status of required field completion"""
        total_fields = len(self.required_fields)
        completed_fields = sum(1 for field in self.required_fields.values() if field["completed"])
        
        return {
            "total": total_fields,
            "completed": completed_fields,
            "percentage": (completed_fields / total_fields) * 100,
            "fields": self.required_fields
        }
    
    def format_ticket_data(self):
        """Format collected data into a structured ticket format"""
        ticket_template = """
SUPPORT TICKET

CALLER INFORMATION
------------------
First Name: {caller_first_name}
Last Name: {caller_last_name}
Company: {company_name}

SOFTWARE DETAILS
---------------
Software: {software_make} {software_model}
Version: {software_version}

ISSUE DETAILS
------------
Type: {issue_type}
Description: {issue_description}

Error Messages: {error_messages}

Troubleshooting Steps Attempted: {troubleshooting_steps}

SUPPORT NOTES
------------
        """
        
        # Replace placeholders with collected values
        formatted_ticket = ticket_template.format(
            caller_first_name=self.required_fields["caller_first_name"]["value"] or "N/A",
            caller_last_name=self.required_fields["caller_last_name"]["value"] or "N/A",
            company_name=self.required_fields["company_name"]["value"] or "N/A",
            software_make=self.required_fields["software_make"]["value"] or "N/A",
            software_model=self.required_fields["software_model"]["value"] or "N/A",
            software_version=self.required_fields["software_version"]["value"] or "N/A",
            issue_type=self.required_fields["issue_type"]["value"] or "N/A",
            issue_description=self.required_fields["issue_description"]["value"] or "N/A",
            error_messages=self.required_fields["error_messages"]["value"] or "None reported",
            troubleshooting_steps=self.required_fields["troubleshooting_steps"]["value"] or "None attempted"
        )
        
        return formatted_ticket


# ===================================
# Component 4: Conversation Analyzer
# ===================================
class ConversationAnalyzer:
    """
    Analyzes conversation transcripts to automatically extract
    required information fields.
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        
    def analyze_transcript(self, transcript):
        """
        Analyze transcript to extract key information using Claude API.
        """
        import requests
        import json
        
        system_prompt = """
        You are an assistant for customer support agents. Extract the following information from the conversation transcript:
        
        1. Caller's first name
        2. Caller's last name
        3. Company name
        4. Software make (e.g., Bluebeam, SolidWorks, AutoCAD)
        5. Software model (e.g., Revu, Standard, LT)
        6. Software version (e.g., 21, 2019, 2024)
        7. Issue type ("how-to" or "error")
        8. Issue description (brief summary of the problem)
        9. Error messages (any specific error text mentioned)
        10. Troubleshooting steps already attempted
        
        Return the information in JSON format with these exact keys:
        caller_first_name, caller_last_name, company_name, software_make, software_model, software_version, issue_type, issue_description, error_messages, troubleshooting_steps
        
        If any field is not found in the transcript, set its value to null.
        """
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json={
                    "model": "claude-3-haiku-20240307",
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": transcript}],
                    "temperature": 0.2,
                    "max_tokens": 1000
                }
            )
            
            if response.status_code != 200:
                print(f"API error: {response.text}")
                return {}
            
            result = response.json()
            content = result["content"][0]["text"]
            
            # Parse the JSON response
            try:
                # Extract JSON from the response
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].strip()
                else:
                    json_str = content
                
                extracted_info = json.loads(json_str)
                return extracted_info
            except json.JSONDecodeError:
                print(f"JSON decode error: {content}")
                return {}
            
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return {}


# ===================================
# Component 5: Web Interface
# ===================================
class WebInterface:
    """
    Provides a web-based interface for the support agent
    to interact with the AI assistant.
    """
    
    def __init__(self, port=5000):
        from flask import Flask
        from flask_socketio import SocketIO
        
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.port = port
        
        # Set up routes and socket handlers
        self.setup_routes()
    
    def setup_routes(self):
        """Set up the Flask routes and Socket.IO handlers"""
        from flask import render_template, jsonify, request
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/start_recording', methods=['POST'])
        def start_recording():
            # Start audio recording
            # This will be implemented in the main application
            return jsonify({'status': 'success'})
        
        @self.app.route('/api/stop_recording', methods=['POST'])
        def stop_recording():
            # Stop audio recording
            # This will be implemented in the main application
            return jsonify({'status': 'success'})
        
        @self.app.route('/api/update_field', methods=['POST'])
        def update_field():
            data = request.json
            # Update field in guidance system
            # This will be implemented in the main application
            return jsonify({'status': 'success'})
        
        @self.app.route('/api/advance_stage', methods=['POST'])
        def advance_stage():
            # Advance to next conversation stage
            # This will be implemented in the main application
            return jsonify({'status': 'success'})
        
        @self.app.route('/api/get_ticket_data', methods=['GET'])
        def get_ticket_data():
            # Get formatted ticket data
            # This will be implemented in the main application
            return jsonify({'status': 'success', 'data': ''})
    
    def run(self):
        """Run the web interface"""
        self.socketio.run(self.app, host='0.0.0.0', port=self.port)


# ===================================
# Component 6: Main Application
# ===================================
class SupportCallAssistant:
    """
    Main application class that coordinates all components.
    """
    
    def __init__(self, config):
        self.config = config
        self.audio_capture = None
        self.transcriber = None
        self.guidance_system = None
        self.analyzer = None
        self.web_interface = None
        
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all system components"""
        # Initialize audio capture
        self.audio_capture = AudioCaptureService(
            sample_rate=16000,
            channels=1
        )
        
        # Initialize transcriber
        self.transcriber = DeepgramTranscriber(
            api_key=self.config.get('deepgram_api_key')
        )
        
        # Initialize guidance system
        self.guidance_system = AgentGuidanceSystem()
        
        # Initialize conversation analyzer
        self.analyzer = ConversationAnalyzer(
            api_key=self.config.get('claude_api_key')
        )
        
        # Initialize web interface
        self.web_interface = WebInterface(
            port=self.config.get('web_port', 5000)
        )
    
    def start(self):
        """Start the support call assistant"""
        # Set up API routes with callback functions
        
        @self.web_interface.app.route('/api/start_recording', methods=['POST'])
        def start_recording():
            self.audio_capture.start_capture()
            return jsonify({'status': 'success'})
        
        @self.web_interface.app.route('/api/stop_recording', methods=['POST'])
        def stop_recording():
            self.audio_capture.stop_capture()
            audio_data = self.audio_capture.get_audio_data()
            
            if audio_data:
                transcript = self.transcriber.transcribe_audio(audio_data)
                
                if transcript:
                    # Analyze transcript to extract information
                    extracted_info = self.analyzer.analyze_transcript(transcript)
                    
                    # Update guidance system with extracted information
                    for field, value in extracted_info.items():
                        if value:
                            self.guidance_system.update_field(field, value)
                    
                    # Send transcript and extracted info to client
                    self.web_interface.socketio.emit('transcript_update', {
                        'text': transcript,
                        'extracted_info': extracted_info
                    })
            
            return jsonify({'status': 'success'})
        
        @self.web_interface.app.route('/api/update_field', methods=['POST'])
        def update_field():
            from flask import request
            data = request.json
            
            field_name = data.get('field')
            value = data.get('value')
            
            if field_name and value:
                self.guidance_system.update_field(field_name, value)
                
                # Send updated completion status
                completion_status = self.guidance_system.get_completion_status()
                self.web_interface.socketio.emit('status_update', completion_status)
            
            return jsonify({'status': 'success'})
        
        @self.web_interface.app.route('/api/advance_stage', methods=['POST'])
        def advance_stage():
            self.guidance_system.advance_stage()
            
            # Send new prompt
            next_prompt = self.guidance_system.get_next_prompt()
            self.web_interface.socketio.emit('prompt_update', {
                'stage': self.guidance_system.current_stage,
                'prompt': next_prompt
            })
            
            return jsonify({'status': 'success'})
        
        @self.web_interface.app.route('/api/get_ticket_data', methods=['GET'])
        def get_ticket_data():
            ticket_data = self.guidance_system.format_ticket_data()
            return jsonify({'status': 'success', 'data': ticket_data})
        
        # Start the web interface
        self.web_interface.run()
