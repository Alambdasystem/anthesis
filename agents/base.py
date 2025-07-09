"""
Base Agent Class for Anthesis AI Agent System
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all agents in the Anthesis system"""
    
    def __init__(self, name: str, description: str, capabilities: List[str], 
                 persona_icon: str = "🤖", persona_color: str = "#6c757d"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.persona_icon = persona_icon
        self.persona_color = persona_color
        self.created_at = datetime.now(timezone.utc)
        self.last_used = None
        self.usage_count = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "persona_icon": self.persona_icon,
            "persona_color": self.persona_color,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count
        }
    
    def log_usage(self):
        """Log agent usage"""
        self.usage_count += 1
        self.last_used = datetime.now(timezone.utc)
    
    @abstractmethod
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the agent's primary function"""
        pass
    
    @abstractmethod
    def get_available_functions(self) -> List[Dict[str, Any]]:
        """Get list of available functions this agent can perform"""
        pass

class AgentResult:
    """Standard result format for agent operations"""
    
    def __init__(self, success: bool, data: Any = None, error: str = None, 
                 agent_id: str = None, function_used: str = None):
        self.success = success
        self.data = data
        self.error = error
        self.agent_id = agent_id
        self.function_used = function_used
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "agent_id": self.agent_id,
            "function_used": self.function_used,
            "timestamp": self.timestamp.isoformat()
        }

class AgentRegistry:
    """Registry for managing all agents - DEPRECATED: Use registry.py instead"""
    
    def __init__(self):
        print("WARNING: Using deprecated AgentRegistry from base.py. Use agents.registry.AgentRegistry instead")
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_file = "agents_registry.json"
        self.load_agents()
    
    def register_agent(self, agent: BaseAgent):
        """Register a new agent"""
        self.agents[agent.id] = agent
        self.save_agents()
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """Get all agents with a specific capability"""
        return [agent for agent in self.agents.values() 
                if capability in agent.capabilities]
    
    def list_all_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [agent.to_dict() for agent in self.agents.values()]
    
    def save_agents(self):
        """Save agent registry to file"""
        try:
            with open(self.agent_file, 'w') as f:
                json.dump(self.list_all_agents(), f, indent=2)
        except Exception as e:
            print(f"Error saving agents: {e}")
    
    def load_agents(self):
        """Load agent registry from file"""
        try:
            with open(self.agent_file, 'r') as f:
                # This would need to be implemented with proper agent reconstruction
                pass
        except FileNotFoundError:
            # Create empty registry
            pass

# Global agent registry instance - DEPRECATED
agent_registry = AgentRegistry()

def initialize_default_agents():
    """Initialize all default agents in the system"""
    from .registry import get_agent_registry
    from .content_generation import ContentGenerationAgent
    from .document_analysis import DocumentAnalysisAgent
    from .lecture_agents import LECTURE_AGENTS
    from .user_agent import UserAgent
    
    registry = get_agent_registry()
    
    # Clear existing agents to avoid duplicates
    registry.clear_agents()
    
    # Register AI agents
    try:
        registry.register_agent(ContentGenerationAgent())
        print("✅ Registered ContentGenerationAgent")
    except Exception as e:
        print(f"❌ Failed to register ContentGenerationAgent: {e}")
    
    try:
        registry.register_agent(DocumentAnalysisAgent())
        print("✅ Registered DocumentAnalysisAgent")
    except Exception as e:
        print(f"❌ Failed to register DocumentAnalysisAgent: {e}")
    
    # Register lecture agents
    for agent_id, agent in LECTURE_AGENTS.items():
        try:
            registry.register_agent(agent)
            print(f"✅ Registered LectureAgent: {agent.name}")
        except Exception as e:
            print(f"❌ Failed to register LectureAgent {agent_id}: {e}")
    
    # Register human users as agents
    cadet_file = os.path.join(os.getcwd(), 'enrollment.json')
    if os.path.exists(cadet_file):
        try:
            with open(cadet_file) as f:
                cadets = json.load(f)
            
            # Handle both old format (dict) and new format (list)
            if isinstance(cadets, dict):
                if 'enrollments' in cadets:
                    # New format with schema
                    enrollments = cadets['enrollments']
                    for cadet_id, cadet_data in enrollments.items():
                        if isinstance(cadet_data, dict):
                            name = f"{cadet_data.get('firstName', '')} {cadet_data.get('lastName', '')}"
                            user_agent = UserAgent(
                                user_id=cadet_id,
                                name=name.strip(),
                                email=cadet_data.get('email', ''),
                                role='cadet'
                            )
                            registry.register_agent(user_agent)
                            print(f"✅ Registered UserAgent: {name.strip()}")
                else:
                    # Old format - direct dict of users
                    for cadet_id, cadet_data in cadets.items():
                        if isinstance(cadet_data, dict):
                            name = f"{cadet_data.get('firstName', '')} {cadet_data.get('lastName', '')}"
                            user_agent = UserAgent(
                                user_id=cadet_id,
                                name=name.strip(),
                                email=cadet_data.get('email', ''),
                                role='cadet'
                            )
                            registry.register_agent(user_agent)
                            print(f"✅ Registered UserAgent: {name.strip()}")
            elif isinstance(cadets, list):
                # List format
                for cadet in cadets:
                    if isinstance(cadet, dict) and 'firstName' in cadet and 'lastName' in cadet:
                        name = f"{cadet['firstName']} {cadet['lastName']}"
                        user_agent = UserAgent(
                            user_id=cadet.get('id', str(uuid.uuid4())),
                            name=name,
                            email=cadet.get('email', ''),
                            role=cadet.get('role', 'cadet')
                        )
                        registry.register_agent(user_agent)
                        print(f"✅ Registered UserAgent: {name}")
        except Exception as e:
            print(f"❌ Failed to load cadet data: {e}")
    else:
        print("ℹ️ No enrollment.json found, skipping user agent registration")
    
    print(f"🎯 Agent initialization complete. Total agents: {registry.get_agent_count()}")
    return registry
