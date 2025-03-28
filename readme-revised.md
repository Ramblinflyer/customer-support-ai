# Support Call Assistant

A Windows-compatible application that assists customer support agents during calls by capturing and transcribing conversations, providing guided prompts, and generating structured support tickets.

## Features

- **Windows-Compatible Audio Capture**: Uses `sounddevice` for reliable audio capture on Windows systems
- **Deepgram Integration**: Real-time transcription with high accuracy and speaker diarization
- **Structured Information Collection**: Guides agents through collecting essential details:
  - Caller name and company
  - Software details (make, model, version)
  - Issue description and type (how-to vs. error)
  - Error messages
  - Troubleshooting steps already taken
- **Guided Workflow**: Step-by-step prompts for the complete support call flow
- **Ticket Generation**: Creates formatted tickets ready to copy into ticketing systems
- **AI-Powered Information Extraction**: Uses Claude AI to automatically extract key details from conversations

## System Requirements

- Python 3.9+
- Windows operating system (for sounddevice support)
- Deepgram API key
- Anthropic API key (for Claude)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/support-call-assistant.git
   cd support-call-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   DEEPGRAM_API_KEY=your_deepgram_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Project Structure

```
support-call-assistant/
├── app.py                  # Main Flask application
├── components/             # Core components
│   ├── audio_capture.py    # Audio capture functionality
│   ├── transcription.py    # Deepgram transcription
│   ├── guidance.py         # Agent guidance system
│   └── analyzer.py         # Conversation analysis with Claude
├── templates/              # HTML templates
│   └── index.html          # Main interface
├── static/                 # Static assets
│   ├── css/                # Stylesheets
│   └── js/                 # JavaScript files
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000/
   ```

3. Using the interface:
   - Click "Start Recording" to begin capturing audio
   - Follow the prompts to gather necessary information
   - Use the "Next Stage" button to advance through the call flow
   - Fields will be automatically populated when possible
   - Copy the formatted ticket when complete

## Call Flow Stages

1. **Introduction**: Initial greeting and starting the conversation
2. **Caller Identity**: Gathering caller's first and last name
3. **Company Info**: Recording the company name
4. **Software Details**: Collecting software make, model, and version
5. **Issue Details**: Understanding the problem and classifying it
6. **Error Messages**: Recording any specific error messages
7. **Troubleshooting History**: Documenting what the caller has already tried
8. **Technician Availability**: Checking if a technician is available
9. **Support Options**: Directing to appropriate support channels
10. **Billing Confirmation**: Confirming TPM1 hours and billing status
11. **Summary**: Reviewing all collected information
12. **Next Steps**: Explaining what happens next

## Customization

### Modifying Prompts

Edit the prompts in `components/guidance.py` to adjust the guidance provided to agents.

### Adding Fields

To add new information fields:

1. Update the `required_fields` dictionary in `components/guidance.py`
2. Add the corresponding input fields in `templates/index.html`
3. Update the ticket template in `format_ticket_data()` method

### CRM Integration

The system is designed to be expanded with CRM integration in the future. To add this:

1. Create a new component in `components/crm.py`
2. Implement methods for creating and updating tickets in your CRM
3. Update `app.py` to include these integrations

## Troubleshooting

### Audio Capture Issues

If you experience problems with audio capture:

1. Ensure your microphone is properly set as the default recording device
2. Try adjusting the sample rate in `AudioCaptureService` (e.g., to 44100 Hz)
3. Check Windows permissions for microphone access

### Transcription Accuracy

To improve transcription accuracy:

1. Use a good quality microphone
2. Reduce background noise
3. Speak clearly and at a moderate pace
4. Consider adjusting Deepgram model parameters in `transcribe_audio()` method

## License

This project is licensed under the MIT License - see the LICENSE file for details.
