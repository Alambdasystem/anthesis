#!/usr/bin/env python3
"""
Script to properly register lecture agents in both registry files
"""

import json
import uuid
from datetime import datetime, timezone
import os

def register_agents_to_registry():
    """Register lecture agents to agents_registry.json"""
    
    lecture_agents_data = {
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
    
    registry_path = 'agents_registry.json'
    
    # Load existing registry
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = []
    
    # Get existing agent names
    existing_names = {agent['name'] for agent in registry}
    
    # Add lecture agents if they don't exist
    for lecture_id, agent_data in lecture_agents_data.items():
        if agent_data['name'] not in existing_names:
            new_agent = {
                "id": str(uuid.uuid4()),
                "name": agent_data['name'],
                "description": f"Lecture specialist in {agent_data['specialization']}",
                "capabilities": [
                    "lecture_delivery",
                    "educational_content",
                    "persona_interaction"
                ],
                "persona_icon": "🎓",
                "persona_color": "#17a2b8",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_used": None,
                "usage_count": 0,
                "lecture_id": lecture_id,
                "specialization": agent_data['specialization'],
                "persona": agent_data['persona'],
                "agent_type": "lecture"
            }
            registry.append(new_agent)
            print(f"✅ Added {agent_data['name']} to registry")
        else:
            print(f"⚠️  {agent_data['name']} already exists in registry")
    
    # Save updated registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"📝 Registry updated. Total agents: {len(registry)}")
    
    # Verify the save worked
    with open(registry_path, 'r') as f:
        verification = json.load(f)
    print(f"🔍 Verification: File contains {len(verification)} agents")
    
    # Print agent names for confirmation
    agent_names = [agent['name'] for agent in verification]
    print(f"📋 Agent names in registry: {agent_names}")

if __name__ == "__main__":
    register_agents_to_registry()
