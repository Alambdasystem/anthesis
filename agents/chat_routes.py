"""
Chat Routes for Anthesis AI Agent System
"""

import os
import json
import time
from flask import Blueprint, request, jsonify, current_app
from .auth import token_required
from .registry import get_agent_registry
import jwt

chat_bp = Blueprint('chat', __name__, url_prefix='/api/chat')

def get_current_user_id():
    """Extract user ID from JWT token"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return None
            
        secret_key = current_app.config.get('SECRET_KEY', 'your_secret_here')
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        return payload.get('user_data', {}).get('id')
    except Exception:
        return None

def load_messages():
    """Load chat messages from JSON file"""
    chat_file = os.path.join(current_app.root_path, 'data', 'chat_inbox.json')
    if not os.path.exists(chat_file):
        return []
    with open(chat_file, 'r') as f:
        return json.load(f)

def save_messages(msgs):
    """Save chat messages to JSON file"""
    chat_dir = os.path.join(current_app.root_path, 'data')
    os.makedirs(chat_dir, exist_ok=True)
    chat_file = os.path.join(chat_dir, 'chat_inbox.json')
    with open(chat_file, 'w') as f:
        json.dump(msgs, f, indent=2)

@chat_bp.route('/inbox', methods=['GET'])
@token_required
def inbox():
    """Get conversation between current user and peer"""
    user = get_current_user_id()
    peer = request.args.get('peer')
    
    if not user:
        return jsonify({"success": False, "error": "User not authenticated"}), 401
    
    all_msgs = load_messages()
    conv = [m for m in all_msgs
            if (m['from']==user and m['to']==peer) or (m['from']==peer and m['to']==user)]
    
    # Sort by timestamp
    conv.sort(key=lambda m: m.get('ts', 0))
    
    return jsonify(conv)

@chat_bp.route('/send', methods=['POST'])
@token_required
def send():
    """Send a message and get AI response if peer is an agent"""
    user = get_current_user_id()
    
    if not user:
        return jsonify({"success": False, "error": "User not authenticated"}), 401
    
    data = request.get_json()
    peer = data.get('to')
    text = data.get('text')
    
    if not peer or not text:
        return jsonify({"success": False, "error": "Missing recipient or message text"}), 400
    
    msgs = load_messages()

    # 1) Store user message
    user_msg = {
        "from": user,
        "to": peer,
        "text": text,
        "ts": int(time.time())
    }
    msgs.append(user_msg)

    # 2) If peer is a registered agent, delegate to it
    registry = get_agent_registry()
    agent = registry.get_agent(peer)
    if agent:
        try:
            # Execute the agent with the user's message
            result = agent.execute(text, {"message": text, "sender": user})
            
            # Extract response text from result
            response_text = "I'm processing your request..."
            if isinstance(result, dict):
                if result.get('success') and result.get('data'):
                    response_text = result['data'].get('response', response_text)
                elif result.get('data'):
                    response_text = str(result['data'])
                elif result.get('error'):
                    response_text = f"Error: {result['error']}"
            
            # Store AI response
            ai_msg = {
                "from": peer,
                "to": user,
                "text": response_text,
                "ts": int(time.time())
            }
            msgs.append(ai_msg)
            
        except Exception as e:
            # If agent execution fails, send error message
            error_msg = {
                "from": peer,
                "to": user,
                "text": f"I'm having trouble processing your request right now. Error: {str(e)}",
                "ts": int(time.time())
            }
            msgs.append(error_msg)

    save_messages(msgs)
    return jsonify({"success": True})
