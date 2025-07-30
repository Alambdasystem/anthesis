#!/usr/bin/env python3
"""
Direct test of quiz generation functionality
Tests the quiz generation from lecture content without needing the full Flask server
"""
import requests
import json
from datetime import datetime

# Configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2"

def test_direct_quiz_generation():
    """Test quiz generation directly using Ollama API"""
    print("🧪 Testing Direct Quiz Generation from Lecture Content")
    print("=" * 60)
    
    # Sample lecture content
    lecture_content = """
    Introduction to Network Security
    
    Today we'll explore the fundamentals of network security, including:
    
    1. Network Security Concepts
    - Firewalls and their types (packet filtering, stateful, application-level)
    - Intrusion Detection Systems (IDS) and Intrusion Prevention Systems (IPS)
    - Virtual Private Networks (VPNs) for secure remote access
    
    2. Common Network Threats
    - Man-in-the-middle attacks
    - DDoS (Distributed Denial of Service) attacks
    - Port scanning and vulnerability assessments
    - SQL injection through web applications
    
    3. Network Security Protocols
    - SSL/TLS for encrypted communications
    - IPSec for network layer security
    - WPA2/WPA3 for wireless network security
    
    4. Best Practices
    - Network segmentation to limit attack scope
    - Regular security audits and penetration testing
    - Keeping systems and software updated
    - Employee training on security awareness
    
    5. Practical Implementation
    Students will configure a basic firewall and analyze network traffic
    to identify potential security threats.
    """
    
    topic = "Network Security Fundamentals"
    week = 2
    lecturer = "Prof. Alex Martinez"
    
    print(f"📖 Topic: {topic}")
    print(f"👨‍🏫 Lecturer: {lecturer}")
    print(f"📅 Week: {week}")
    print(f"📄 Content Length: {len(lecture_content)} characters")
    print()
    
    # Truncate content if too long
    max_content_length = 2000
    if len(lecture_content) > max_content_length:
        lecture_content = lecture_content[:max_content_length] + "..."
        print(f"⚠️  Content truncated to {max_content_length} characters")
    
    # Create comprehensive quiz prompt
    prompt = f"""Based on this lecture content, create a comprehensive quiz with 5-8 multiple choice questions.

LECTURE: "{topic}" by {lecturer}
WEEK: {week}
CONTENT: {lecture_content}

Create questions that test:
1. Key concepts and definitions
2. Practical applications
3. Problem-solving scenarios
4. Critical thinking

Return ONLY a JSON object in this exact format:
{{
  "quiz_title": "Quiz: {topic}",
  "week": {week},
  "lecturer": "{lecturer}",
  "questions": [
    {{
      "question": "What is the main purpose of a firewall in network security?",
      "options": ["A) To speed up network traffic", "B) To filter and control network traffic", "C) To store user passwords", "D) To backup network data"],
      "correct_answer": "B",
      "explanation": "Firewalls are designed to filter and control network traffic based on security rules."
    }}
  ]
}}

Generate exactly 5-8 well-crafted questions based on the lecture content."""
    
    try:
        print("🔄 Sending request to Ollama...")
        
        # Call Ollama API
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are an expert quiz generator. Create comprehensive quizzes based on lecture content. Always return valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500,
            "stream": False
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Ollama API error: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        # Parse the response
        content = response.json()
        quiz_content = content.get("message", {}).get("content", "").strip()
        
        print(f"📝 Raw AI Response Length: {len(quiz_content)} characters")
        print()
        
        # Clean up the response to extract JSON
        print("🧹 Cleaning response...")
        if quiz_content.startswith("```json"):
            quiz_content = quiz_content[7:]
        elif quiz_content.startswith("```"):
            quiz_content = quiz_content[3:]
        
        if quiz_content.endswith("```"):
            quiz_content = quiz_content[:-3]
        
        quiz_content = quiz_content.strip()
        
        print(f"📋 Cleaned Response (first 200 chars): {quiz_content[:200]}...")
        print()
        
        try:
            quiz_data = json.loads(quiz_content)
            print("✅ JSON parsing successful!")
            print()
            
            # Validate the structure
            if "questions" not in quiz_data or not isinstance(quiz_data["questions"], list):
                print("❌ Invalid quiz structure: missing or invalid 'questions' field")
                return
            
            # Display the generated quiz
            print("📋 GENERATED QUIZ")
            print("=" * 30)
            print(f"Title: {quiz_data.get('quiz_title', 'Untitled Quiz')}")
            print(f"Week: {quiz_data.get('week', 'N/A')}")
            print(f"Lecturer: {quiz_data.get('lecturer', 'N/A')}")
            print()
            
            questions = quiz_data.get('questions', [])
            print(f"Number of Questions: {len(questions)}")
            print()
            
            # Display each question
            for i, question in enumerate(questions, 1):
                print(f"Question {i}: {question.get('question', 'No question text')}")
                
                options = question.get('options', [])
                for option in options:
                    print(f"  {option}")
                
                print(f"  ✓ Correct Answer: {question.get('correct_answer', 'N/A')}")
                
                explanation = question.get('explanation', '')
                if explanation:
                    print(f"  💡 Explanation: {explanation}")
                
                print()
            
            # Validation summary
            print("📊 VALIDATION SUMMARY")
            print("=" * 25)
            print(f"✅ Quiz generation: SUCCESS")
            print(f"✅ Questions generated: {len(questions)}")
            print(f"✅ Valid JSON structure: YES")
            print(f"✅ All questions have text: {all('question' in q and q['question'] for q in questions)}")
            print(f"✅ All questions have options: {all('options' in q and isinstance(q['options'], list) and len(q['options']) >= 2 for q in questions)}")
            print(f"✅ All questions have correct answers: {all('correct_answer' in q and q['correct_answer'] for q in questions)}")
            print(f"✅ All questions have explanations: {all('explanation' in q and q['explanation'] for q in questions)}")
            
            # Save the generated quiz
            filename = f"generated_quiz_{topic.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(quiz_data, f, indent=2)
            print(f"💾 Quiz saved to: {filename}")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw content: {quiz_content}")
            
            # Try to provide a fallback quiz
            fallback_quiz = {
                "quiz_title": f"Quiz: {topic}",
                "week": week,
                "lecturer": lecturer,
                "questions": [
                    {
                        "question": f"What is the primary focus of {topic}?",
                        "options": ["A) Basic networking concepts", "B) Advanced security measures", "C) Hardware configuration", "D) Software development"],
                        "correct_answer": "B",
                        "explanation": "The lecture focuses on advanced security measures and protection strategies."
                    },
                    {
                        "question": "Which of the following is a common network security threat?",
                        "options": ["A) Software updates", "B) Data backups", "C) DDoS attacks", "D) User training"],
                        "correct_answer": "C",
                        "explanation": "DDoS attacks are a significant threat to network availability and security."
                    }
                ]
            }
            
            print("🆘 Using fallback quiz:")
            print(json.dumps(fallback_quiz, indent=2))
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to Ollama")
        print("Make sure Ollama is running on http://localhost:11434")
    
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Ollama took too long to respond")
        print("Try reducing the content length or increasing timeout")
    
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")

def main():
    """Main test function"""
    print("🚀 Starting Direct Quiz Generation Test")
    print("=" * 60)
    
    test_direct_quiz_generation()
    
    print("\n🏁 Test Complete!")

if __name__ == "__main__":
    main()
