"""
Content Generation Agent for AI-powered content creation
"""

import json
import requests
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from .base import BaseAgent, AgentResult

class ContentGenerationAgent(BaseAgent):
    """Agent specialized in AI content generation using Ollama"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434/api/chat", default_model: str = "llama3.2"):
        super().__init__(
            name="Content Generation Agent",
            description="Generates AI-powered content including emails, RFP responses, and proposals",
            capabilities=["email_generation", "rfp_response", "proposal_writing", "content_creation"],
            persona_icon="✍️",
            persona_color="#00bfae"
        )
        self.ollama_url = ollama_url
        self.default_model = default_model
        self.session = requests.Session()
        self.agent_type = "ai"
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute content generation task"""
        self.log_usage()
        
        if context is None:
            context = {}
        
        try:
            if task == "generate_email":
                return self._generate_email(context)
            elif task == "generate_rfp_response":
                return self._generate_rfp_response(context)
            elif task == "generate_proposal":
                return self._generate_proposal(context)
            elif task == "generate_content":
                return self._generate_content(context)
            else:
                return AgentResult(
                    success=False,
                    error=f"Unknown task: {task}",
                    agent_id=self.id
                ).to_dict()
        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e),
                agent_id=self.id,
                function_used=task
            ).to_dict()
    
    def get_available_functions(self) -> List[Dict[str, Any]]:
        """Get available functions for this agent"""
        return [
            {
                "name": "generate_email",
                "description": "Generate a professional email draft",
                "parameters": {
                    "recipient": "Information about the recipient",
                    "subject": "Email subject",
                    "context": "Additional context for the email",
                    "tone": "Email tone (professional, friendly, formal)"
                }
            },
            {
                "name": "generate_rfp_response",
                "description": "Generate a response to an RFP document",
                "parameters": {
                    "rfp_text": "The RFP document text",
                    "company_info": "Information about the responding company",
                    "response_style": "Style of response (detailed, concise, technical)"
                }
            },
            {
                "name": "generate_proposal",
                "description": "Generate a business proposal",
                "parameters": {
                    "proposal_type": "Type of proposal",
                    "client_info": "Information about the client",
                    "requirements": "Project requirements"
                }
            },
            {
                "name": "generate_content",
                "description": "Generate general content based on a prompt",
                "parameters": {
                    "prompt": "Content generation prompt",
                    "max_tokens": "Maximum tokens to generate",
                    "temperature": "Generation temperature (0.0-1.0)"
                }
            }
        ]
    
    def _call_ollama(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Call Ollama API for content generation"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = self.session.post(self.ollama_url, json=payload, timeout=120)
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")
        
        result = response.json()
        return result.get("message", {}).get("content", "").strip()
    
    def _generate_email(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an email draft"""
        recipient = context.get("recipient", {})
        subject = context.get("subject", "")
        email_context = context.get("context", "")
        tone = context.get("tone", "professional")
        
        # Build prompt for email generation
        prompt = f"""Write a {tone} email with the following details:
        
Subject: {subject}
Recipient: {recipient.get('name', 'Unknown')} from {recipient.get('company', 'Unknown Company')}
Context: {email_context}

Generate a well-structured email that:
1. Has an appropriate greeting
2. Clearly states the purpose
3. Provides relevant information
4. Includes a professional closing
5. Matches the requested tone: {tone}

Email:"""
        
        system_prompt = "You are a professional email writing assistant. Create clear, concise, and effective business emails."
        
        try:
            generated_content = self._call_ollama(prompt, system_prompt)
            
            return AgentResult(
                success=True,
                data={
                    "subject": subject,
                    "content": generated_content,
                    "recipient": recipient,
                    "tone": tone,
                    "generated_at": datetime.now(timezone.utc).isoformat()
                },
                agent_id=self.id,
                function_used="generate_email"
            ).to_dict()
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Failed to generate email: {str(e)}",
                agent_id=self.id,
                function_used="generate_email"
            ).to_dict()
    
    def _generate_rfp_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an RFP response"""
        rfp_text = context.get("rfp_text", "")
        company_info = context.get("company_info", {})
        response_style = context.get("response_style", "detailed")
        
        if not rfp_text:
            return AgentResult(
                success=False,
                error="No RFP text provided",
                agent_id=self.id,
                function_used="generate_rfp_response"
            ).to_dict()
        
        # Build prompt for RFP response
        prompt = f"""You are a proposal writer. Here is the full RFP text:
{rfp_text}

Company Information:
- Name: {company_info.get('name', 'Alambda Systems')}
- Focus: {company_info.get('focus', 'AI and software development')}
- Why they're a good fit: {company_info.get('why_hot', 'Experienced team')}

Create a {response_style} RFP response that:
1. Extracts and lists the key proposal requirements
2. Addresses each requirement specifically
3. Highlights the company's relevant experience and capabilities
4. Provides a compelling case for why this company should be selected
5. Maintains a professional and confident tone

Structure the response with:
- Executive Summary
- Understanding of Requirements
- Proposed Solution
- Company Qualifications
- Conclusion

RFP Response:"""
        
        system_prompt = "You are an expert proposal writer specializing in RFP responses. Create comprehensive, professional, and compelling proposals."
        
        try:
            generated_content = self._call_ollama(prompt, system_prompt, temperature=0.8, max_tokens=3000)
            
            return AgentResult(
                success=True,
                data={
                    "response_content": generated_content,
                    "company_info": company_info,
                    "response_style": response_style,
                    "rfp_text_length": len(rfp_text),
                    "generated_at": datetime.now(timezone.utc).isoformat()
                },
                agent_id=self.id,
                function_used="generate_rfp_response"
            ).to_dict()
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Failed to generate RFP response: {str(e)}",
                agent_id=self.id,
                function_used="generate_rfp_response"
            ).to_dict()
    
    def _generate_proposal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a business proposal"""
        proposal_type = context.get("proposal_type", "general")
        client_info = context.get("client_info", {})
        requirements = context.get("requirements", "")
        
        prompt = f"""Create a {proposal_type} business proposal with the following details:
        
Client: {client_info.get('name', 'Unknown Client')}
Requirements: {requirements}

Generate a professional proposal that includes:
1. Executive Summary
2. Problem Understanding
3. Proposed Solution
4. Timeline and Deliverables
5. Investment and Terms

Proposal:"""
        
        system_prompt = "You are a business proposal specialist. Create compelling and professional proposals that win business."
        
        try:
            generated_content = self._call_ollama(prompt, system_prompt)
            
            return AgentResult(
                success=True,
                data={
                    "proposal_content": generated_content,
                    "proposal_type": proposal_type,
                    "client_info": client_info,
                    "generated_at": datetime.now(timezone.utc).isoformat()
                },
                agent_id=self.id,
                function_used="generate_proposal"
            ).to_dict()
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Failed to generate proposal: {str(e)}",
                agent_id=self.id,
                function_used="generate_proposal"
            ).to_dict()
    
    def _generate_content(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate general content"""
        prompt = context.get("prompt", "")
        max_tokens = context.get("max_tokens", 2000)
        temperature = context.get("temperature", 0.7)
        
        if not prompt:
            return AgentResult(
                success=False,
                error="No prompt provided",
                agent_id=self.id,
                function_used="generate_content"
            ).to_dict()
        
        try:
            generated_content = self._call_ollama(prompt, temperature=temperature, max_tokens=max_tokens)
            
            return AgentResult(
                success=True,
                data={
                    "content": generated_content,
                    "prompt": prompt,
                    "parameters": {
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    },
                    "generated_at": datetime.now(timezone.utc).isoformat()
                },
                agent_id=self.id,
                function_used="generate_content"
            ).to_dict()
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Failed to generate content: {str(e)}",
                agent_id=self.id,
                function_used="generate_content"
            ).to_dict()
