"""
Email Log Viewer - View logged emails instead of sending via SMTP
"""

import json
import os
from datetime import datetime
from flask import Flask, render_template_string, jsonify

def load_email_logs():
    """Load email logs from JSON file"""
    log_file = 'email_logs/outgoing_emails.json'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return json.load(f)
    return []

def view_emails():
    """Display emails in a simple HTML format"""
    emails = load_email_logs()
    
    template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Email Logs - Anthesis AI</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .email { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
            .email-header { background: #f5f5f5; padding: 10px; margin: -15px -15px 15px -15px; }
            .email-body { white-space: pre-wrap; background: #fafafa; padding: 10px; border-radius: 3px; }
            .timestamp { color: #666; font-size: 0.9em; }
            .agents { color: #0066cc; font-size: 0.9em; }
            h1 { color: #333; }
            .count { color: #666; }
        </style>
    </head>
    <body>
        <h1>📧 Email Logs</h1>
        <p class="count">Total emails: {{ email_count }}</p>
        
        {% for email in emails %}
        <div class="email">
            <div class="email-header">
                <strong>To:</strong> {{ email.to }}<br>
                <strong>Subject:</strong> {{ email.subject }}<br>
                <span class="timestamp">{{ email.timestamp }}</span>
                {% if email.agents %}
                <br><span class="agents">Agents: {{ email.agents|join(', ') }}</span>
                {% endif %}
            </div>
            <div class="email-body">{{ email.body }}</div>
        </div>
        {% endfor %}
        
        {% if email_count == 0 %}
        <p>No emails logged yet. Send some emails through the agent system!</p>
        {% endif %}
    </body>
    </html>
    '''
    
    from jinja2 import Template
    template = Template(template)
    return template.render(emails=emails, email_count=len(emails))

if __name__ == "__main__":
    # Simple standalone viewer
    print("📧 Email Log Viewer")
    print("=" * 30)
    
    emails = load_email_logs()
    print(f"Total emails: {len(emails)}")
    
    if emails:
        for i, email in enumerate(emails, 1):
            print(f"\n{i}. {email['subject']}")
            print(f"   To: {email['to']}")
            print(f"   Time: {email['timestamp']}")
            if email.get('agents'):
                print(f"   Agents: {', '.join(email['agents'])}")
            print(f"   Preview: {email['body'][:100]}...")
    else:
        print("No emails found.")
