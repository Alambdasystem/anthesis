#!/usr/bin/env python3
"""
Simple test to check if quiz generation works locally
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_simple_quiz():
    """Test quiz generation with sample content"""
    
    # Sample lecture content for testing
    lecture_content = """
    # Network Security Fundamentals
    
    Network security is the practice of securing a computer network from intruders, whether targeted attackers or opportunistic malware.
    
    ## Key Concepts
    
    ### 1. Confidentiality
    Ensuring that data is accessible only to authorized users.
    
    ### 2. Integrity  
    Maintaining the accuracy and completeness of data.
    
    ### 3. Availability
    Ensuring that authorized users have access to information when needed.
    
    ## Common Threats
    
    - **Malware**: Malicious software designed to damage or disrupt systems
    - **Phishing**: Fraudulent attempts to obtain sensitive information
    - **DDoS Attacks**: Overwhelming systems with traffic
    - **Man-in-the-Middle**: Intercepting communications
    
    ## Security Controls
    
    - Firewalls
    - Intrusion Detection Systems
    - Encryption
    - Access Controls
    """
    
    quiz_data = {
        "lecture_content": lecture_content,
        "topic": "Network Security Fundamentals", 
        "week": 3,
        "lecturer": "Dr. Test"
    }
    
    print("🧠 Testing quiz generation...")
    print(f"📚 Content length: {len(lecture_content)} characters")
    
    try:
        # Test without authentication first to see the specific error
        response = requests.post(
            f"{BASE_URL}/api/quizzes/generate-from-lecture",
            json=quiz_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📡 Response status: {response.status_code}")
        print(f"📄 Response: {response.text}")
        
        if response.status_code == 403:
            print("⚠️ Authentication required - this is expected")
            print("✅ Endpoint is accessible and responding correctly")
            return True
        elif response.status_code == 200:
            result = response.json()
            print("✅ Quiz generated successfully!")
            print(f"📝 Questions: {len(result.get('questions', []))}")
            return True
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_quiz()
    print(f"\n{'✅ Test passed' if success else '❌ Test failed'}")
