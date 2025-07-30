#!/usr/bin/env python3
"""
Test script for quiz generation from lecture content
Tests the /api/quizzes/generate-from-lecture endpoint
"""
import requests
import json
import jwt
from datetime import datetime, timezone, timedelta

# Configuration
BASE_URL = "http://localhost:5000"
SECRET_KEY = "your_secret_here"  # Should match the one in app.py

def create_test_token():
    """Create a test JWT token for authentication"""
    try:
        payload = {
            "username": "testuser",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        print(f"🔑 Token created successfully")
        return token
    except Exception as e:
        print(f"❌ Error creating token: {e}")
        return None

def test_quiz_generation():
    """Test the quiz generation from lecture content"""
    print("🧪 Testing Quiz Generation from Lecture Content")
    print("=" * 50)
    
    # Create test token
    token = create_test_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Sample lecture content for testing
    sample_lecture_content = """
    Introduction to Cybersecurity Fundamentals
    
    In today's digital age, cybersecurity has become one of the most critical aspects of technology infrastructure. This lecture covers the fundamental concepts that every cybersecurity professional should understand.
    
    Key Topics Covered:
    
    1. What is Cybersecurity?
    Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks. These cyberattacks are usually aimed at accessing, changing, or destroying sensitive information; extorting money from users; or interrupting normal business processes.
    
    2. The CIA Triad
    The CIA triad is a fundamental concept in cybersecurity:
    - Confidentiality: Ensuring that information is accessible only to authorized users
    - Integrity: Maintaining the accuracy and completeness of data
    - Availability: Ensuring that authorized users have access to information when needed
    
    3. Common Types of Cyber Threats
    - Malware: Malicious software including viruses, worms, and ransomware
    - Phishing: Fraudulent attempts to obtain sensitive information
    - Social Engineering: Manipulating people to divulge confidential information
    - DDoS Attacks: Distributed Denial of Service attacks that overwhelm systems
    
    4. Defense Strategies
    - Implement strong password policies
    - Use multi-factor authentication
    - Keep software and systems updated
    - Regular security awareness training
    - Network segmentation and monitoring
    
    5. Career Paths in Cybersecurity
    - Security Analyst
    - Penetration Tester
    - Security Architect
    - Incident Response Specialist
    - Compliance Auditor
    
    Practical Exercise:
    Students should identify potential vulnerabilities in a sample network diagram and propose appropriate security measures.
    
    Next Week Preview:
    We will dive deeper into network security protocols and hands-on lab exercises with security tools.
    """
    
    # Test data for quiz generation
    test_data = {
        "lecture_content": sample_lecture_content,
        "topic": "Cybersecurity Fundamentals",
        "week": 1,
        "lecturer": "Dr. Sarah Chen"
    }
    
    print(f"📖 Lecture Topic: {test_data['topic']}")
    print(f"👨‍🏫 Lecturer: {test_data['lecturer']}")
    print(f"📅 Week: {test_data['week']}")
    print(f"📄 Content Length: {len(sample_lecture_content)} characters")
    print()
    
    try:
        # Make request to the quiz generation endpoint
        print("🔄 Sending request to quiz generation endpoint...")
        print(f"🌐 URL: {BASE_URL}/api/quizzes/generate-from-lecture")
        
        response = requests.post(
            f"{BASE_URL}/api/quizzes/generate-from-lecture",
            headers=headers,
            json=test_data,
            timeout=120  # Extended timeout for AI generation
        )
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            quiz_data = response.json()
            print("✅ Quiz Generated Successfully!")
            print()
            
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
            
            # Test summary
            print("📊 TEST SUMMARY")
            print("=" * 20)
            print(f"✅ Quiz generation: SUCCESS")
            print(f"✅ Questions generated: {len(questions)}")
            print(f"✅ All questions have options: {all('options' in q for q in questions)}")
            print(f"✅ All questions have correct answers: {all('correct_answer' in q for q in questions)}")
            print(f"✅ All questions have explanations: {all('explanation' in q for q in questions)}")
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Response text: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to Flask server")
        print("Make sure the Flask app is running on http://localhost:5000")
    
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: The request took too long")
        print("This might indicate the AI model is taking too long to respond")
    
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")

def test_basic_quiz_generation():
    """Test the basic quiz generation endpoint"""
    print("\n🧪 Testing Basic Quiz Generation")
    print("=" * 40)
    
    token = create_test_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    test_data = {
        "track": "cybersecurity-1",
        "week": 1
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/quizzes",
            headers=headers,
            json=test_data,
            timeout=60
        )
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            quiz_data = response.json()
            print("✅ Basic Quiz Generated Successfully!")
            print(f"Quiz ID: {quiz_data.get('id', 'N/A')}")
            print(f"Track: {quiz_data.get('track', 'N/A')}")
            print(f"Week: {quiz_data.get('week', 'N/A')}")
            print(f"Questions: {len(quiz_data.get('questions', []))}")
        else:
            print(f"❌ Basic quiz generation failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Error in basic quiz test: {str(e)}")

def main():
    """Main test function"""
    print("🚀 Starting Quiz Generation Tests")
    print("=" * 60)
    
    # Test the quiz generation from lecture content
    test_quiz_generation()
    
    # Test basic quiz generation
    test_basic_quiz_generation()
    
    print("\n🏁 Tests Complete!")

if __name__ == "__main__":
    main()
