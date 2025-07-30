"""
Agent Routes for Anthesis AI Agent System
"""

import uuid
import datetime
import json
import os
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timezone
from typing import Dict, Any

from .registry import get_agent_registry
from .auth import token_required
from .coordinator import CoordinatorAgent
from .document_analysis import DocumentAnalysisAgent
from .content_generation import ContentGenerationAgent

# Get blueprint from __init__.py to avoid circular import
from . import agents_bp

# Initialize coordinator agent
coordinator = CoordinatorAgent()
registry = get_agent_registry()
registry.register_agent(coordinator)

@agents_bp.route('/health', methods=['GET'])
def agent_health():
    """Health check for agent system"""
    registry = get_agent_registry()
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "registered_agents": registry.get_agent_count()
    })

@agents_bp.route('/', methods=['GET'])
def list_agents():
    """List all available agents"""
    try:
        registry = get_agent_registry()
        agents = registry.list_all_agents()
        return jsonify({
            "success": True,
            "data": agents,
            "count": len(agents),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@agents_bp.route('/contacts', methods=['GET'])
def list_contacts():
    """List all available contacts (agents + users) for chat"""
    try:
        registry = get_agent_registry()
        contacts = []
        
        # Get all agents
        for agent in registry.get_all_agents():
            agent_dict = agent.to_dict()
            contacts.append({
                "id": agent_dict["id"],
                "name": agent_dict["name"],
                "type": agent_dict.get("agent_type", "ai"),
                "description": agent_dict.get("description", ""),
                "icon": agent_dict.get("persona_icon", "🤖")
            })
        
        return jsonify(contacts)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@agents_bp.route('/<agent_id>', methods=['GET'])
def get_agent_info(agent_id: str):
    """Get information about a specific agent"""
    try:
        registry = get_agent_registry()
        agent = registry.get_agent(agent_id)
        if not agent:
            return jsonify({
                "success": False,
                "error": f"Agent {agent_id} not found",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 404
        
        agent_info = agent.to_dict()
        agent_info["available_functions"] = agent.get_available_functions()
        
        return jsonify({
            "success": True,
            "data": agent_info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@agents_bp.route('/<agent_id>/execute', methods=['POST'])
def execute_agent_function(agent_id: str):
    """Execute a function on a specific agent"""
    try:
        data = request.get_json() or {}
        function_name = data.get("function_name")
        parameters = data.get("parameters", {})
        
        if not function_name:
            return jsonify({
                "success": False,
                "error": "function_name is required",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 400
        
        # Use coordinator to execute the function
        result = coordinator.execute("execute_agent_function", {
            "agent_id": agent_id,
            "function_name": function_name,
            "parameters": parameters
        })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@agents_bp.route('/workflows/rfp', methods=['POST'])
def process_rfp_workflow():
    """Process RFP workflow (extract + generate response)"""
    try:
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 400

        # Get additional parameters
        form_data = request.form.to_dict()
        company_info_str = form_data.get('company_info', 'Alambda Systems')
        response_style = form_data.get('response_style', 'professional')
        
        # Parse company_info if it's JSON, otherwise create a basic dictionary
        try:
            import json
            company_info = json.loads(company_info_str) if company_info_str.startswith('{') else {
                'name': company_info_str,
                'focus': 'AI and software development',
                'why_hot': 'Experienced team'
            }
        except (json.JSONDecodeError, AttributeError):
            company_info = {
                'name': company_info_str if company_info_str else 'Alambda Systems',
                'focus': 'AI and software development', 
                'why_hot': 'Experienced team'
            }
        
        # Use coordinator to process RFP
        result = coordinator.execute("process_rfp_workflow", {
            "file_data": file.read(),
            "filename": file.filename,
            "company_info": company_info,
            "response_style": response_style
        })
        
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@agents_bp.route('/workflows/email', methods=['POST'])
@token_required
def email_workflow():
    """Generate and log email workflow"""
    try:
        data = request.get_json() or {}
        destination = data.get('destination', {})
        context = data.get('context', {})
        
        to_addr = destination.get('email', '')
        subject = destination.get('subject', 'No Subject')
        agent_ids = context.get('agent_ids', [])
        
        if not to_addr:
            return jsonify({
                "success": False,
                "error": "Email address is required",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 400
        
        # Generate email content using coordinator
        email_result = coordinator.execute("generate_email_workflow", {
            "recipient": to_addr,
            "subject": subject,
            "context": context.get('content', ''),
            "tone": context.get('tone', 'professional')
        })
        
        if not email_result.get('success'):
            return jsonify(email_result), 500
        
        # Log email instead of sending via SMTP
        log_dir = os.path.join(current_app.root_path, 'email_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'outgoing_emails.json')
        
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
            "to": to_addr,
            "subject": subject,
            "body": email_result.get('data', {}).get('email_content', ''),
            "agents": agent_ids,
            "workflow_result": email_result
        }
        
        # Load existing emails
        if os.path.exists(log_file):
            with open(log_file) as f:
                emails = json.load(f)
        else:
            emails = []
        
        emails.append(entry)
        
        # Save updated emails
        with open(log_file, 'w') as f:
            json.dump(emails, f, indent=2)
        
        return jsonify({
            "success": True,
            "message": "Email recorded",
            "email_id": entry["id"],
            "timestamp": entry["timestamp"]
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@agents_bp.route('/workflows/messaging', methods=['POST'])
def messaging_workflow():
    """Messaging workflow"""
    try:
        data = request.get_json() or {}
        # Implementation for messaging workflow
        return jsonify({
            "success": True,
            "message": "Messaging workflow not yet implemented",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@agents_bp.route('/workflows/analyze-document', methods=['POST'])
def analyze_document_workflow():
    """Analyze document workflow"""
    try:
        registry = get_agent_registry()
        
        # Get or create document analysis agent
        doc_agents = registry.get_agents_by_capability("document_analysis")
        if not doc_agents:
            doc_agent = DocumentAnalysisAgent()
            registry.register_agent(doc_agent)
        else:
            doc_agent = doc_agents[0]
        
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 400
        
        # Process document
        result = doc_agent.execute("process_rfp_upload", {
            "file_data": file.read(),
            "filename": file.filename
        })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@agents_bp.route('/capabilities', methods=['GET'])
def list_capabilities():
    """List all available capabilities across all agents"""
    try:
        registry = get_agent_registry()
        capabilities = {}
        
        for agent in registry.get_all_agents():
            for capability in agent.capabilities:
                if capability not in capabilities:
                    capabilities[capability] = []
                capabilities[capability].append({
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "functions": agent.get_available_functions()
                })
        
        return jsonify({
            "success": True,
            "data": capabilities,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500

@agents_bp.route('/initialize', methods=['POST'])
def initialize_agents():
    """Initialize default agents"""
    try:
        from .base import initialize_default_agents
        registry = initialize_default_agents()
        
        return jsonify({
            "success": True,
            "message": "Agents initialized successfully",
            "agent_count": registry.get_agent_count(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500
