"""
User Agent - Represents human users as agents in the system
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from .base import BaseAgent, AgentResult

class UserAgent(BaseAgent):
    """Agent that represents a human user in the system"""
    
    def __init__(self, user_id: str, name: str, email: str, role: str = "cadet"):
        super().__init__(
            name=name,
            description=f"Human user: {name} ({role})",
            capabilities=["human_interaction", "learning", "communication"],
            persona_icon="👤",
            persona_color="#28a745"
        )
        self.user_id = user_id
        self.email = email
        self.role = role
        self.agent_type = "human"
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute task (human users don't execute automated tasks)"""
        self.log_usage()
        
        return AgentResult(
            success=False,
            error="Human agents cannot execute automated tasks",
            agent_id=self.id,
            function_used=task
        ).to_dict()
    
    def get_available_functions(self) -> List[Dict[str, Any]]:
        """Get available functions for human agents"""
        return [
            {
                "name": "send_message",
                "description": "Send a message to this user",
                "parameters": {
                    "message": "Message content",
                    "sender": "Sender information"
                }
            },
            {
                "name": "get_profile",
                "description": "Get user profile information",
                "parameters": {}
            }
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary with user-specific fields"""
        base_dict = super().to_dict()
        base_dict.update({
            "user_id": self.user_id,
            "email": self.email,
            "role": self.role,
            "agent_type": "human"
        })
        return base_dict
    
    def send_message(self, message: str, sender: str = "System") -> Dict[str, Any]:
        """Send a message to this user (placeholder for actual implementation)"""
        return AgentResult(
            success=True,
            data={
                "message": f"Message queued for {self.name}: {message}",
                "recipient": self.email,
                "sender": sender,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            agent_id=self.id,
            function_used="send_message"
        ).to_dict()
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile information"""
        return AgentResult(
            success=True,
            data={
                "user_id": self.user_id,
                "name": self.name,
                "email": self.email,
                "role": self.role,
                "capabilities": self.capabilities,
                "created_at": self.created_at.isoformat(),
                "last_used": self.last_used.isoformat() if self.last_used else None,
                "usage_count": self.usage_count
            },
            agent_id=self.id,
            function_used="get_profile"
        ).to_dict()
