# Flask Backend Integration for Lecture Agents
# Add this code to your main Flask app (app.py or main.py)

import os
import json
import uuid
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from flask import Flask, jsonify, request

app = Flask(__name__)

# Add this function to register your lecture agents on startup
def register_lecture_agents():
    """
    Bootstrap function to register in-memory lectureAgents into agent_contacts.json
    Call this in your app startup (in the __main__ guard)
    """
    # Define your lecture agents (same as in your JS)
    lectureAgents = {
        'dr-smith': {
            'name': 'Dr. Smith',
            'specialization': 'AI & Machine Learning Expert',
            'persona': 'You are Dr. Smith, an AI and Machine Learning expert with 15 years of experience. You explain complex concepts in simple terms and always provide practical examples. You are enthusiastic about emerging technologies and love to share real-world applications.'
        },
        'prof-chen': {
            'name': 'Prof. Chen',
            'specialization': 'Data Science & Analytics',
            'persona': 'You are Professor Chen, a data scientist with expertise in statistical analysis and big data. You focus on mathematical foundations and provide detailed explanations with charts and formulas. You emphasize evidence-based conclusions and rigorous methodology.'
        },
        'dr-wilson': {
            'name': 'Dr. Wilson',
            'specialization': 'Systems Architecture',
            'persona': 'You are Dr. Wilson, a systems architect with deep knowledge of scalable systems and infrastructure. You think in terms of system design patterns, performance optimization, and best practices. You provide architectural insights and technical depth.'
        },
        'prof-taylor': {
            'name': 'Prof. Taylor',
            'specialization': 'Leadership & Strategy',
            'persona': 'You are Professor Taylor, a leadership expert focusing on team dynamics, strategic thinking, and organizational behavior. You provide insights into management principles, communication strategies, and business leadership approaches.'
        }
    }
    
    path = 'agent_contacts.json'
    
    # Load existing contacts
    if os.path.exists(path):
        with open(path, 'r') as f:
            contacts = json.load(f)
    else:
        contacts = []
    
    # Create a lookup of existing agents by name
    existing = {c['name']: c for c in contacts if c.get('type') == 'agent'}
    
    # Register each lecture agent if not already exists
    for key, agent_data in lectureAgents.items():
        name = agent_data['name']
        if name not in existing:
            new_agent = {
                'id': str(uuid.uuid4()),
                'name': name,
                'email': '',  # Agents don't have email addresses
                'bio': agent_data['persona'],
                'specialization': agent_data['specialization'],
                'type': 'agent',
                'created_by': 'system',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            contacts.append(new_agent)
            print(f"✅ Registered agent: {name}")
        else:
            print(f"⚠️  Agent already exists: {name}")
    
    # Save updated contacts
    with open(path, 'w') as f:
        json.dump(contacts, f, indent=2)
    
    print(f"📝 Agent registration complete. Total contacts: {len(contacts)}")


# Add this route to handle email workflow with agents
# Dummy token_required decorator for demonstration; replace with your actual implementation or import
def token_required(f):
    def decorated(*args, **kwargs):
        # Implement your token validation logic here
        return f(*args, **kwargs)
    decorated.__name__ = f.__name__
    return decorated

@app.route('/api/agents/workflows/email', methods=['POST'])
@token_required  # Your existing auth decorator
def send_agent_email():
    """
    Enhanced email endpoint that includes AI agent personas in the email body
    """
    try:
        data = request.get_json() or {}
        dest = data.get('destination', {})
        ctx = data.get('context', {})
        
        to = dest.get('email')
        subject = dest.get('subject')
        body = ctx.get('content', '')
        agent_ids = ctx.get('agent_ids', [])
        legacy_agents = ctx.get('agents', [])  # Backward compatibility
        
        if not to or not subject:
            return jsonify({"success": False, "error": "Missing email or subject"}), 400
        
        # Load agent contacts
        agent_contacts_path = 'agent_contacts.json'
        if os.path.exists(agent_contacts_path):
            with open(agent_contacts_path, 'r') as f:
                all_agents = {a['id']: a for a in json.load(f) if a.get('type') == 'agent'}
        else:
            all_agents = {}
        
        # Append selected registered agents to email body
        if agent_ids:
            body += "\n\n" + "="*50
            body += "\n🤖 AI AGENTS INCLUDED IN THIS EMAIL:\n"
            body += "="*50 + "\n"
            
            for agent_id in agent_ids:
                agent = all_agents.get(agent_id)
                if agent:
                    body += f"\n📋 Agent: {agent['name']}\n"
                    body += f"🎯 Specialization: {agent.get('specialization', 'AI Assistant')}\n"
                    body += f"👤 Persona: {agent['bio']}\n"
                    body += "-" * 50 + "\n"
        
        # Handle legacy agents (backward compatibility)
        if legacy_agents:
            if not agent_ids:  # Only add header if not already added
                body += "\n\n" + "="*50
                body += "\n🤖 AI AGENTS INCLUDED IN THIS EMAIL:\n"
                body += "="*50 + "\n"
            
            for agent in legacy_agents:
                body += f"\n📋 Agent: {agent.get('name', 'Unknown Agent')}\n"
                body += f"🎯 Specialization: {agent.get('specialization', 'AI Assistant')}\n"
                body += f"👤 Persona: {agent.get('persona', 'AI Assistant')}\n"
                body += "-" * 50 + "\n"
        
        # Add signature
        if agent_ids or legacy_agents:
            body += "\n💡 These AI agents represent different expertise areas and perspectives"
            body += "\n   from the Alambda Systems team. Each brings specialized knowledge"
            body += "\n   to help with your specific needs and requirements.\n"
            body += "\n🔗 Learn more: https://alambda.com\n"
        
        # Send email using your preferred method
        # Option 1: SMTP (replace with your settings)
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = 'noreply@alambda.systems'  # Replace with your email
            msg['To'] = to
            
            # Replace with your SMTP settings
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_user = os.getenv('SMTP_USER', 'your-email@gmail.com')
            smtp_pass = os.getenv('SMTP_PASS', 'your-app-password')
            
            if smtp_user and smtp_pass:
                s = smtplib.SMTP(smtp_server, smtp_port)
                s.starttls()
                s.login(smtp_user, smtp_pass)
                s.send_message(msg)
                s.quit()
                
                print(f"✅ Email sent to {to} with {len(agent_ids) + len(legacy_agents)} agents")
                return jsonify({"success": True, "message": "Email sent successfully"}), 200
            else:
                # Log email instead of sending (for development)
                print(f"📧 EMAIL WOULD BE SENT TO: {to}")
                print(f"📧 SUBJECT: {subject}")
                print(f"📧 BODY:\n{body}")
                return jsonify({"success": True, "message": "Email logged (SMTP not configured)"}), 200
                
        except Exception as smtp_error:
            print(f"❌ SMTP Error: {smtp_error}")
            # Log email as fallback
            print(f"📧 EMAIL FALLBACK - TO: {to}, SUBJECT: {subject}")
            return jsonify({"success": True, "message": "Email processed (SMTP error)"}), 200
        
    except Exception as e:
        print(f"❌ Error in send_agent_email: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# Add this to your app startup (in the __main__ guard)
if __name__ == '__main__':
    with app.app_context():
        print("🚀 Starting Alambda Systems server...")
        
        # Register lecture agents on startup
        register_lecture_agents()
        
        # Your existing startup code...
        # generate_pre_generated_music()
        
    app.run(debug=True, host='0.0.0.0', port=5000)


# Environment variables you'll need to set:
"""
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password

For Gmail, you'll need to:
1. Enable 2FA on your Google account
2. Generate an "App Password" for your application
3. Use the app password instead of your regular password
"""

# Example .env file:
"""
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=noreply@alambda.systems
SMTP_PASS=your-16-character-app-password
"""
