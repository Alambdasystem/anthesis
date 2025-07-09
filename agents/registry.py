"""
Agent Registry - Singleton pattern for managing all agents
"""

import json
import os
from typing import Dict, List, Optional
from .base import BaseAgent

class AgentRegistry:
    """Singleton registry for managing all agents"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if AgentRegistry._instance is not None:
            raise Exception("AgentRegistry is a singleton. Use get_instance()")
        
        self._agents: Dict[str, BaseAgent] = {}
        self.agent_file = "agents_registry.json"
    
    def register_agent(self, agent: BaseAgent):
        """Register a new agent"""
        self._agents[agent.id] = agent
        self.save_agents()
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self._agents.get(agent_id)
    
    def get_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """Get all agents with a specific capability"""
        return [agent for agent in self._agents.values() 
                if capability in agent.capabilities]
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Get all agents of a specific type"""
        return [agent for agent in self._agents.values() 
                if hasattr(agent, 'agent_type') and agent.agent_type == agent_type]
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents"""
        return list(self._agents.values())
    
    def list_all_agents(self) -> List[Dict]:
        """List all registered agents as dictionaries"""
        return [agent.to_dict() for agent in self._agents.values()]
    
    def save_agents(self):
        """Save agent registry to file"""
        try:
            with open(self.agent_file, 'w') as f:
                json.dump(self.list_all_agents(), f, indent=2)
        except Exception as e:
            print(f"Error saving agents: {e}")
    
    def load_agents(self):
        """Load agent registry from file (metadata only)"""
        try:
            with open(self.agent_file, 'r') as f:
                # This loads metadata only - agents need to be re-instantiated
                agent_data = json.load(f)
                print(f"Loaded {len(agent_data)} agent records from registry")
        except FileNotFoundError:
            print("No existing agent registry found, starting fresh")
        except Exception as e:
            print(f"Error loading agents: {e}")
    
    def clear_agents(self):
        """Clear all agents (useful for testing)"""
        self._agents.clear()
        self.save_agents()
    
    def get_agent_count(self) -> int:
        """Get total number of registered agents"""
        return len(self._agents)

# Global function to get registry instance
def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance"""
    return AgentRegistry.get_instance()
