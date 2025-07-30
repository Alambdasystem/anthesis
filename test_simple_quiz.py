#!/usr/bin/env python3
"""
Simple Quiz Generation Test
Tests the quiz generation without authentication.
"""

import json
import time
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:5000"

def test_quiz_generation_direct():
    """Test the quiz generation API directly"""
    try:
        quiz_data = {
            "lecture_content": """
# Introduction to Cybersecurity

Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks.

## Key Concepts:
- Confidentiality: Keeping data private and secure
- Integrity: Ensuring data accuracy and preventing tampering
- Availability: Making sure systems are accessible when needed

## Common Threats:
- Malware: Malicious software that can damage systems
- Phishing: Fraudulent emails that steal credentials  
- Social Engineering: Manipulating people to reveal information
- DDoS: Overwhelming systems with traffic to make them unavailable

## Security Controls:
- Firewalls: Network security barriers that control traffic
- Antivirus: Software to detect and remove malware
- Encryption: Protecting data by converting it to codes
- Access Controls: Limiting who can access what resources
            """,
            "topic": "Introduction to Cybersecurity Fundamentals",
            "week": 1,
            "lecturer": "Dr. Test Smith"
        }
        
        logger.info("🧠 Testing quiz generation API...")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/api/quizzes/generate-from-lecture", 
            json=quiz_data, 
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        generation_time = time.time() - start_time
        
        logger.info(f"Response Status: {response.status_code}")
        logger.info(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Quiz generated successfully in {generation_time:.2f} seconds")
            
            # Validate quiz structure
            if 'questions' in result and len(result['questions']) > 0:
                logger.info(f"📝 Quiz contains {len(result['questions'])} questions")
                
                # Show all questions
                for i, question in enumerate(result['questions'], 1):
                    logger.info(f"\n📋 Question {i}:")
                    logger.info(f"Q: {question.get('question', 'N/A')}")
                    logger.info(f"Options: {question.get('options', [])}")
                    logger.info(f"Correct: {question.get('correct_answer', 'N/A')}")
                    logger.info(f"Explanation: {question.get('explanation', 'N/A')}")
                
                return True
            else:
                logger.error("❌ Quiz has no questions")
                return False
        else:
            logger.error(f"❌ Quiz generation failed: {response.status_code}")
            logger.error(f"Response text: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Quiz generation error: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Testing Quiz Generation API")
    logger.info("=" * 50)
    
    success = test_quiz_generation_direct()
    
    if success:
        logger.info("\n✅ Quiz generation test passed!")
        logger.info("📊 The API is working and can be used by the frontend.")
    else:
        logger.error("\n❌ Quiz generation test failed!")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
