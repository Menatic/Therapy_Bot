from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
import re
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('therapy_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Gemini Flash Model
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("Missing Gemini API key in environment variables")
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')  # Using only the Flash model
    logger.info("Gemini Flash model successfully initialized")

    # Therapeutic context prompt
    THERAPY_CONTEXT = """
    You are Serenity AI, a compassionate mental health assistant. Provide brief, supportive responses that:
    1. Validate feelings ("I hear you're feeling...")
    2. Ask one open-ended question
    3. Offer one simple coping strategy
    4. Keep responses under 3 sentences
    
    For crisis situations:
    1. Acknowledge the pain ("This sounds really difficult")
    2. Provide emergency resources
    3. Encourage immediate action
    4. Keep response under 4 sentences
    """
    
except Exception as e:
    logger.error(f"Initialization error: {str(e)}")
    raise

# Crisis resources
CRISIS_RESOURCES = {
    "suicide": {
        "response": "I'm deeply concerned about you. Please call the 988 Suicide & Crisis Lifeline now at 988 (US) or your local emergency number.",
        "resources": ["988 Suicide & Crisis Lifeline", "Crisis Text Line: Text HOME to 741741"]
    },
    "self-harm": {
        "response": "Your safety matters. Please reach out to someone you trust or text HOME to 741741 to connect with a crisis counselor.",
        "resources": ["Crisis Text Line: Text HOME to 741741"]
    }
}

def is_crisis_message(text: str) -> bool:
    """Detect crisis keywords with context awareness"""
    text = text.lower()
    crisis_indicators = [
        'kill myself', 'end it all', 'want to die',
        'cut myself', 'self harm', 'hurt myself',
        'suicide', 'ending my life'
    ]
    return any(indicator in text for indicator in crisis_indicators)

def generate_response(user_input: str) -> str:
    """Generate therapeutic response using only the Flash model"""
    try:
        # Check for crisis first
        if is_crisis_message(user_input):
            crisis_type = "suicide" if any(word in user_input.lower() for word in ['suicide', 'die', 'end it']) else "self-harm"
            return CRISIS_RESOURCES[crisis_type]["response"]
        
        # Generate regular therapeutic response
        prompt = f"""
        User message: {user_input}
        
        Please respond with:
        1. Brief emotional validation
        2. One open-ended question
        3. One simple coping suggestion
        """
        
        response = model.generate_content(
            THERAPY_CONTEXT + prompt,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 150
            }
        )
        
        if not response.text:
            raise ValueError("Empty response from model")
            
        # Clean up response
        cleaned = re.sub(r'\*+', '', response.text)  # Remove markdown
        return cleaned.strip()
        
    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}")
        return "I'm having trouble responding. Could you rephrase that or try again later?"

@app.route('/chat', methods=['POST'])
def chat():
    """Therapeutic chat endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "JSON input required"}), 400

        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({"response": "I'm here when you're ready to talk"}), 200
        
        # Determine if crisis
        is_crisis = is_crisis_message(user_input)
        crisis_type = None
        
        if is_crisis:
            crisis_type = "suicide" if any(word in user_input.lower() for word in ['suicide', 'die', 'end it']) else "self-harm"
        
        # Generate response
        if is_crisis:
            response = CRISIS_RESOURCES[crisis_type]["response"]
            resources = CRISIS_RESOURCES[crisis_type]["resources"]
        else:
            response = generate_response(user_input)
            resources = []
        
        # Log interaction
        logger.info(f"User: {user_input}")
        logger.info(f"AI: {response}")
        
        return jsonify({
            "response": response,
            "is_crisis": is_crisis,
            "resources": resources,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        return jsonify({
            "response": "I'm unable to respond properly right now. Please try again later.",
            "success": False
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)