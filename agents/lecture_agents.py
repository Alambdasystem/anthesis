"""
Lecture Agents - AI personas for educational content delivery
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import requests

from .base import BaseAgent, AgentResult

class LectureAgent(BaseAgent):
    """Agent specialized in delivering educational content with specific persona"""
    
    def __init__(self, lecture_id: str, name: str, specialization: str, persona: str,
                 ollama_url: str = "http://localhost:11434/api/chat", 
                 default_model: str = "llama3.2"):
        super().__init__(
            name=name,
            description=f"Lecture specialist in {specialization}",
            capabilities=["lecture_delivery", "educational_content", "persona_interaction"],
            persona_icon="🎓",
            persona_color="#17a2b8"
        )
        self.lecture_id = lecture_id
        self.specialization = specialization
        self.persona = persona
        self.ollama_url = ollama_url
        self.default_model = default_model
        self.session = requests.Session()
        self.agent_type = "lecture"
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute lecture-related task"""
        self.log_usage()
        
        if context is None:
            context = {}
        
        try:
            if task == "deliver_lecture":
                return self._deliver_lecture(context)
            elif task == "answer_question":
                return self._answer_question(context)
            elif task == "generate_quiz":
                return self._generate_quiz(context)
            elif task == "explain_concept":
                return self._explain_concept(context)
            else:
                return AgentResult(
                    success=False,
                    error=f"Unknown task: {task}",
                    agent_id=self.id,
                    function_used=task
                ).to_dict()
        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e),
                agent_id=self.id,
                function_used=task
            ).to_dict()
    
    def get_available_functions(self) -> List[Dict[str, Any]]:
        """Get available functions for lecture agents"""
        return [
            {
                "name": "deliver_lecture",
                "description": "Deliver a lecture on a specific topic",
                "parameters": {
                    "topic": "Lecture topic",
                    "duration": "Expected duration in minutes",
                    "difficulty_level": "Beginner, Intermediate, or Advanced"
                }
            },
            {
                "name": "answer_question",
                "description": "Answer a student question in character",
                "parameters": {
                    "question": "Student's question",
                    "context": "Additional context for the answer"
                }
            },
            {
                "name": "generate_quiz",
                "description": "Generate quiz questions on a topic",
                "parameters": {
                    "topic": "Quiz topic",
                    "question_count": "Number of questions",
                    "difficulty": "Question difficulty level"
                }
            },
            {
                "name": "explain_concept",
                "description": "Explain a specific concept in detail",
                "parameters": {
                    "concept": "Concept to explain",
                    "level": "Explanation complexity level"
                }
            }
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary with lecture-specific fields"""
        base_dict = super().to_dict()
        base_dict.update({
            "lecture_id": self.lecture_id,
            "specialization": self.specialization,
            "persona": self.persona,
            "agent_type": "lecture"
        })
        return base_dict
    
    def _call_ollama(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        """Call Ollama API for content generation"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.default_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = self.session.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['message']['content']
        except Exception as e:
            return f"Error generating content: {str(e)}"
    
    def _deliver_lecture(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deliver a lecture on the specified topic"""
        topic = context.get("topic", "General topic")
        duration = context.get("duration", 15)
        difficulty = context.get("difficulty_level", "Intermediate")
        
        system_prompt = f"{self.persona}\n\nYou are delivering a {duration}-minute lecture at {difficulty} level."
        prompt = f"Deliver a comprehensive lecture on: {topic}\n\nStructure it with clear sections and practical examples."
        
        content = self._call_ollama(prompt, system_prompt)
        
        return AgentResult(
            success=True,
            data={
                "lecture_content": content,
                "topic": topic,
                "duration": duration,
                "difficulty": difficulty,
                "lecturer": self.name,
                "specialization": self.specialization
            },
            agent_id=self.id,
            function_used="deliver_lecture"
        ).to_dict()
    
    def _answer_question(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Answer a student question in character"""
        question = context.get("question", "")
        additional_context = context.get("context", "")
        
        if not question:
            return AgentResult(
                success=False,
                error="No question provided",
                agent_id=self.id,
                function_used="answer_question"
            ).to_dict()
        
        system_prompt = f"{self.persona}\n\nAnswer the student's question thoroughly and in character."
        prompt = f"Student Question: {question}\n\nAdditional Context: {additional_context}\n\nProvide a detailed, educational answer."
        
        answer = self._call_ollama(prompt, system_prompt)
        
        return AgentResult(
            success=True,
            data={
                "question": question,
                "answer": answer,
                "lecturer": self.name,
                "specialization": self.specialization
            },
            agent_id=self.id,
            function_used="answer_question"
        ).to_dict()
    
    def _generate_quiz(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quiz questions on a topic"""
        topic = context.get("topic", "General topic")
        question_count = context.get("question_count", 5)
        difficulty = context.get("difficulty", "Intermediate")
        
        system_prompt = f"{self.persona}\n\nGenerate educational quiz questions."
        prompt = f"Create {question_count} {difficulty} level quiz questions about: {topic}\n\nProvide questions with multiple choice answers and correct answers clearly marked."
        
        quiz_content = self._call_ollama(prompt, system_prompt)
        
        return AgentResult(
            success=True,
            data={
                "quiz_content": quiz_content,
                "topic": topic,
                "question_count": question_count,
                "difficulty": difficulty,
                "creator": self.name
            },
            agent_id=self.id,
            function_used="generate_quiz"
        ).to_dict()
    
    def _explain_concept(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Explain a specific concept in detail"""
        concept = context.get("concept", "")
        level = context.get("level", "intermediate")
        
        if not concept:
            return AgentResult(
                success=False,
                error="No concept specified",
                agent_id=self.id,
                function_used="explain_concept"
            ).to_dict()
        
        system_prompt = f"{self.persona}\n\nExplain concepts clearly at {level} level."
        prompt = f"Explain this concept in detail: {concept}\n\nProvide examples, applications, and key points to remember."
        
        explanation = self._call_ollama(prompt, system_prompt)
        
        return AgentResult(
            success=True,
            data={
                "concept": concept,
                "explanation": explanation,
                "level": level,
                "explainer": self.name,
                "specialization": self.specialization
            },
            agent_id=self.id,
            function_used="explain_concept"
        ).to_dict()

# Predefined lecture agents
LECTURE_AGENTS = {
    'dr-smith': LectureAgent(
        lecture_id='dr-smith',
        name='Dr. Smith',
        specialization='AI & Machine Learning Expert',
        persona='You are Dr. Smith, an AI and Machine Learning expert with 15 years of experience. You explain complex concepts in simple terms and always provide practical examples. You are enthusiastic about emerging technologies and love to share real-world applications.'
    ),
    'prof-chen': LectureAgent(
        lecture_id='prof-chen',
        name='Prof. Chen',
        specialization='Data Science & Analytics',
        persona='You are Professor Chen, a data scientist with expertise in statistical analysis and big data. You focus on mathematical foundations and provide detailed explanations with charts and formulas. You emphasize evidence-based conclusions and rigorous methodology.'
    ),
    'dr-wilson': LectureAgent(
        lecture_id='dr-wilson',
        name='Dr. Wilson',
        specialization='Systems Architecture',
        persona='You are Dr. Wilson, a systems architect with deep knowledge of scalable systems and infrastructure. You think in terms of system design patterns, performance optimization, and best practices. You provide architectural insights and technical depth.'
    ),
    'prof-taylor': LectureAgent(
        lecture_id='prof-taylor',
        name='Prof. Taylor',
        specialization='Leadership & Strategy',
        persona='You are Professor Taylor, a leadership expert focusing on team dynamics, strategic thinking, and organizational behavior. You provide insights into management principles, communication strategies, and business leadership approaches.'
    )
}
