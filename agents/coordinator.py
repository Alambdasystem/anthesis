"""
Coordinator Agent for orchestrating multi-agent workflows
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union

from .base import BaseAgent, AgentResult, agent_registry
from .document_analysis import DocumentAnalysisAgent
from .content_generation import ContentGenerationAgent

class CoordinatorAgent(BaseAgent):
    """Agent that coordinates and orchestrates other agents"""
    
    def __init__(self):
        super().__init__(
            name="Coordinator Agent",
            description="Orchestrates multi-agent workflows and manages agent interactions",
            capabilities=["workflow_orchestration", "agent_coordination", "task_delegation", "result_aggregation"],
            persona_icon="🎯",
            persona_color="#0a84ff"
        )
        self.workflow_history: List[Dict[str, Any]] = []
        self.agent_type = "ai"
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute coordination task"""
        self.log_usage()
        
        if context is None:
            context = {}
        
        try:
            if task == "process_rfp_workflow":
                return self._process_rfp_workflow(context)
            elif task == "generate_email_workflow":
                return self._generate_email_workflow(context)
            elif task == "list_available_agents":
                return self._list_available_agents(context)
            elif task == "execute_agent_function":
                return self._execute_agent_function(context)
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
                "name": "process_rfp_workflow",
                "description": "Complete RFP processing workflow (extract text + generate response)",
                "parameters": {
                    "file_data": "RFP file data",
                    "filename": "RFP filename",
                    "company_info": "Company information for response",
                    "response_style": "Style of response"
                }
            },
            {
                "name": "generate_email_workflow",
                "description": "Generate email with context analysis",
                "parameters": {
                    "recipient": "Recipient information",
                    "subject": "Email subject",
                    "context": "Email context",
                    "tone": "Email tone"
                }
            },
            {
                "name": "list_available_agents",
                "description": "List all available agents and their capabilities",
                "parameters": {}
            },
            {
                "name": "execute_agent_function",
                "description": "Execute a specific function on a specific agent",
                "parameters": {
                    "agent_id": "ID of the agent to use",
                    "function_name": "Name of the function to execute",
                    "parameters": "Parameters for the function"
                }
            }
        ]
    
    def _get_or_create_agent(self, agent_type: str) -> Union[BaseAgent, None]:
        """Get or create an agent of the specified type"""
        # Check if agent already exists in registry
        agents = agent_registry.get_agents_by_capability(agent_type)
        if agents:
            return agents[0]  # Return first available agent
        
        # Create new agent based on type
        if agent_type == "document_analysis":
            agent = DocumentAnalysisAgent()
            agent_registry.register_agent(agent)
            return agent
        elif agent_type == "content_generation":
            agent = ContentGenerationAgent()
            agent_registry.register_agent(agent)
            return agent
        
        return None
    
    def _process_rfp_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process complete RFP workflow"""
        workflow_id = f"rfp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        workflow_log = {
            "workflow_id": workflow_id,
            "workflow_type": "rfp_processing",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "steps": [],
            "results": {},
            "agent_conversations": []
        }
        
        try:
            # Step 1: Document Analysis
            workflow_log["agent_conversations"].append({
                "agent_id": self.id,
                "agent_name": self.name,
                "message": "🤖 Starting RFP processing workflow...",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": "initialization"
            })
            
            doc_agent = self._get_or_create_agent("document_analysis")
            if not doc_agent:
                return AgentResult(
                    success=False,
                    error="Document analysis agent not available",
                    agent_id=self.id,
                    function_used="process_rfp_workflow"
                ).to_dict()
            
            workflow_log["agent_conversations"].append({
                "agent_id": doc_agent.id,
                "agent_name": doc_agent.name,
                "message": "📄 Analyzing uploaded document and extracting text...",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": "document_analysis"
            })
            
            # Extract text from RFP
            extraction_context = {
                "file_data": context.get("file_data"),
                "filename": context.get("filename", "rfp.pdf")
            }
            
            extraction_result = doc_agent.execute("process_rfp_upload", extraction_context)
            workflow_log["steps"].append({
                "step": "document_extraction",
                "agent_id": doc_agent.id,
                "agent_name": doc_agent.name,
                "success": extraction_result.get("success", False),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": f"Processed file: {extraction_context.get('filename')}"
            })
            
            if extraction_result.get("success"):
                extracted_text = extraction_result.get("data", {}).get("text", "")
                word_count = len(extracted_text.split())
                workflow_log["agent_conversations"].append({
                    "agent_id": doc_agent.id,
                    "agent_name": doc_agent.name,
                    "message": f"✅ Successfully extracted {word_count} words from document",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": "document_analysis"
                })
            else:
                workflow_log["agent_conversations"].append({
                    "agent_id": doc_agent.id,
                    "agent_name": doc_agent.name,
                    "message": f"❌ Failed to extract text: {extraction_result.get('error')}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": "document_analysis"
                })
            
            if not extraction_result.get("success"):
                workflow_log["completed_at"] = datetime.now(timezone.utc).isoformat()
                workflow_log["final_result"] = extraction_result
                return AgentResult(
                    success=False,
                    error=f"Document extraction failed: {extraction_result.get('error')}",
                    agent_id=self.id,
                    function_used="process_rfp_workflow",
                    data={"workflow_log": workflow_log}
                ).to_dict()
            
            # Step 2: Content Generation
            workflow_log["agent_conversations"].append({
                "agent_id": self.id,
                "agent_name": self.name,
                "message": "🔄 Handing off to Content Generation Agent...",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": "handoff"
            })
            
            content_agent = self._get_or_create_agent("content_generation")
            if not content_agent:
                return AgentResult(
                    success=False,
                    error="Content generation agent not available",
                    agent_id=self.id,
                    function_used="process_rfp_workflow"
                ).to_dict()
            
            workflow_log["agent_conversations"].append({
                "agent_id": content_agent.id,
                "agent_name": content_agent.name,
                "message": "✍️ Generating comprehensive RFP response...",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": "content_generation"
            })
            
            # Generate RFP response
            generation_context = {
                "rfp_text": extraction_result.get("data", {}).get("text", ""),
                "company_info": context.get("company_info", {}),
                "response_style": context.get("response_style", "detailed")
            }
            
            generation_result = content_agent.execute("generate_rfp_response", generation_context)
            workflow_log["steps"].append({
                "step": "content_generation",
                "agent_id": content_agent.id,
                "agent_name": content_agent.name,
                "success": generation_result.get("success", False),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": f"Generated {generation_context.get('response_style')} response"
            })
            
            if generation_result.get("success"):
                workflow_log["agent_conversations"].append({
                    "agent_id": content_agent.id,
                    "agent_name": content_agent.name,
                    "message": "✅ RFP response generated successfully",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": "content_generation"
                })
            else:
                workflow_log["agent_conversations"].append({
                    "agent_id": content_agent.id,
                    "agent_name": content_agent.name,
                    "message": f"❌ Failed to generate response: {generation_result.get('error')}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": "content_generation"
                })
            
            # Final coordination step
            workflow_log["agent_conversations"].append({
                "agent_id": self.id,
                "agent_name": self.name,
                "message": "🎯 Workflow completed successfully!",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": "completion"
            })
            
            # Compile final result
            workflow_log["completed_at"] = datetime.now(timezone.utc).isoformat()
            workflow_log["final_result"] = generation_result
            
            if generation_result.get("success"):
                final_data = {
                    "rfp_response": generation_result.get("data", {}),
                    "extraction_info": extraction_result.get("data", {}),
                    "workflow_log": workflow_log
                }
                
                return AgentResult(
                    success=True,
                    data=final_data,
                    agent_id=self.id,
                    function_used="process_rfp_workflow"
                ).to_dict()
            else:
                return AgentResult(
                    success=False,
                    error=f"Content generation failed: {generation_result.get('error')}",
                    agent_id=self.id,
                    function_used="process_rfp_workflow",
                    data={"workflow_log": workflow_log}
                ).to_dict()
            
        except Exception as e:
            workflow_log["completed_at"] = datetime.now(timezone.utc).isoformat()
            workflow_log["error"] = str(e)
            return AgentResult(
                success=False,
                error=f"Workflow failed: {str(e)}",
                agent_id=self.id,
                function_used="process_rfp_workflow",
                data={"workflow_log": workflow_log}
            ).to_dict()
    
    def _generate_email_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate email workflow"""
        workflow_id = f"email_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        workflow_log = {
            "workflow_id": workflow_id,
            "workflow_type": "email_generation",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "steps": [],
            "agent_conversations": []
        }
        
        try:
            workflow_log["agent_conversations"].append({
                "agent_id": self.id,
                "agent_name": self.name,
                "message": "🤖 Starting email generation workflow...",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": "initialization"
            })
            
            # Get content generation agent
            content_agent = self._get_or_create_agent("content_generation")
            if not content_agent:
                return AgentResult(
                    success=False,
                    error="Content generation agent not available",
                    agent_id=self.id,
                    function_used="generate_email_workflow"
                ).to_dict()
            
            recipient = context.get("recipient", {})
            subject = context.get("subject", "")
            tone = context.get("tone", "professional")
            
            workflow_log["agent_conversations"].append({
                "agent_id": content_agent.id,
                "agent_name": content_agent.name,
                "message": f"✍️ Generating {tone} email to {recipient.get('name', 'recipient')}...",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": "content_generation"
            })
            
            # Generate email
            result = content_agent.execute("generate_email", context)
            
            workflow_log["steps"].append({
                "step": "email_generation",
                "agent_id": content_agent.id,
                "agent_name": content_agent.name,
                "success": result.get("success", False),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": f"Generated {tone} email with subject: {subject}"
            })
            
            if result.get("success"):
                workflow_log["agent_conversations"].append({
                    "agent_id": content_agent.id,
                    "agent_name": content_agent.name,
                    "message": "✅ Email generated successfully",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": "content_generation"
                })
                
                workflow_log["agent_conversations"].append({
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "message": "🎯 Email workflow completed successfully!",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": "completion"
                })
                
                result["data"]["workflow_id"] = workflow_id
                result["data"]["workflow_log"] = workflow_log
            else:
                workflow_log["agent_conversations"].append({
                    "agent_id": content_agent.id,
                    "agent_name": content_agent.name,
                    "message": f"❌ Failed to generate email: {result.get('error')}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": "content_generation"
                })
            
            workflow_log["completed_at"] = datetime.now(timezone.utc).isoformat()
            return result
            
        except Exception as e:
            workflow_log["agent_conversations"].append({
                "agent_id": self.id,
                "agent_name": self.name,
                "message": f"❌ Workflow failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": "error"
            })
            
            workflow_log["completed_at"] = datetime.now(timezone.utc).isoformat()
            return AgentResult(
                success=False,
                error=f"Email workflow failed: {str(e)}",
                agent_id=self.id,
                function_used="generate_email_workflow",
                data={"workflow_log": workflow_log}
            ).to_dict()
    
    def _list_available_agents(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """List all available agents"""
        try:
            agents_info = []
            
            # Get all registered agents
            all_agents = agent_registry.list_all_agents()
            
            for agent_data in all_agents:
                # Try to get the actual agent to get its functions
                agent = agent_registry.get_agent(agent_data["id"])
                if agent:
                    agent_info = agent_data.copy()
                    agent_info["available_functions"] = agent.get_available_functions()
                    agents_info.append(agent_info)
            
            return AgentResult(
                success=True,
                data={
                    "agents": agents_info,
                    "total_agents": len(agents_info),
                    "queried_at": datetime.now(timezone.utc).isoformat()
                },
                agent_id=self.id,
                function_used="list_available_agents"
            ).to_dict()
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Failed to list agents: {str(e)}",
                agent_id=self.id,
                function_used="list_available_agents"
            ).to_dict()
    
    def _execute_agent_function(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function on a specific agent"""
        agent_id = context.get("agent_id")
        function_name = context.get("function_name")
        parameters = context.get("parameters", {})
        
        if not agent_id or not function_name:
            return AgentResult(
                success=False,
                error="agent_id and function_name are required",
                agent_id=self.id,
                function_used="execute_agent_function"
            ).to_dict()
        
        try:
            agent = agent_registry.get_agent(agent_id)
            if not agent:
                return AgentResult(
                    success=False,
                    error=f"Agent not found: {agent_id}",
                    agent_id=self.id,
                    function_used="execute_agent_function"
                ).to_dict()
            
            # Execute the function
            result = agent.execute(function_name, parameters)
            
            # Add coordination metadata
            if isinstance(result, dict) and "data" in result:
                result["data"]["coordinated_by"] = self.id
                result["data"]["coordination_timestamp"] = datetime.now(timezone.utc).isoformat()
            
            return result
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Function execution failed: {str(e)}",
                agent_id=self.id,
                function_used="execute_agent_function"
            ).to_dict()
