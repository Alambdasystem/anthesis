#!/usr/bin/env python3
"""
Complete workflow test: Generate lecture -> Generate quiz from lecture
Simulates the complete user experience for a week's learning
"""
import requests
import json
from datetime import datetime

# Configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2"

def generate_lecture_content(track, week, topic):
    """Generate lecture content using AI"""
    print(f"📚 Generating lecture content for {track}, Week {week}: {topic}")
    
    prompt = f"""Generate a comprehensive lecture for Week {week} of the {track} track on the topic: {topic}

Please create a detailed lecture that includes:
1. Learning objectives
2. Key concepts and definitions
3. Practical examples
4. Real-world applications
5. Best practices
6. Summary and next steps

Format the content as a structured lecture suitable for cybersecurity students.
The lecture should be informative, engaging, and approximately 1500-2000 words."""

    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are an expert cybersecurity instructor creating educational content."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": False
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        
        if response.status_code == 200:
            content = response.json().get("message", {}).get("content", "").strip()
            print(f"✅ Lecture generated ({len(content)} characters)")
            return content
        else:
            print(f"❌ Failed to generate lecture: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error generating lecture: {e}")
        return None

def generate_quiz_from_lecture(lecture_content, topic, week, lecturer):
    """Generate quiz from lecture content"""
    print(f"🧠 Generating quiz from lecture content...")
    
    # Truncate content if too long for the prompt
    max_content_length = 2500
    if len(lecture_content) > max_content_length:
        lecture_content = lecture_content[:max_content_length] + "..."
        print(f"⚠️  Content truncated to {max_content_length} characters for quiz generation")
    
    prompt = f"""Based on this lecture content, create a comprehensive quiz with 6-8 multiple choice questions.

LECTURE: "{topic}" by {lecturer}
WEEK: {week}
CONTENT: {lecture_content}

Create questions that test:
1. Key concepts and definitions from the lecture
2. Practical applications discussed
3. Understanding of security principles
4. Critical thinking about implementation

Return ONLY a JSON object in this exact format:
{{
  "quiz_title": "Week {week} Quiz: {topic}",
  "week": {week},
  "lecturer": "{lecturer}",
  "topic": "{topic}",
  "questions": [
    {{
      "question": "What is the primary purpose of...",
      "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
      "correct_answer": "B",
      "explanation": "Detailed explanation of why this is correct..."
    }}
  ]
}}

Generate exactly 6-8 well-crafted questions that test understanding of the lecture material."""

    try:
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are an expert quiz generator. Create comprehensive quizzes based on lecture content. Always return valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": False
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        
        if response.status_code == 200:
            content = response.json().get("message", {}).get("content", "").strip()
            
            # Clean up the response
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            try:
                quiz_data = json.loads(content)
                print(f"✅ Quiz generated with {len(quiz_data.get('questions', []))} questions")
                return quiz_data
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                return None
        else:
            print(f"❌ Failed to generate quiz: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error generating quiz: {e}")
        return None

def save_lecture_and_quiz(track, week, lecturer, topic, lecture_content, quiz_data):
    """Save lecture and quiz data to files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save lecture
    lecture_data = {
        "id": f"{track}-week-{week}-{lecturer.lower().replace(' ', '-')}",
        "track": track,
        "week": week,
        "professor": lecturer,
        "topic": topic,
        "content": lecture_content,
        "created_at": datetime.now().isoformat(),
        "word_count": len(lecture_content.split()),
        "character_count": len(lecture_content)
    }
    
    lecture_filename = f"lecture_{track}_week{week}_{timestamp}.json"
    with open(lecture_filename, 'w') as f:
        json.dump(lecture_data, f, indent=2)
    
    # Save quiz
    quiz_filename = f"quiz_{track}_week{week}_{timestamp}.json"
    with open(quiz_filename, 'w') as f:
        json.dump(quiz_data, f, indent=2)
    
    print(f"💾 Lecture saved to: {lecture_filename}")
    print(f"💾 Quiz saved to: {quiz_filename}")
    
    return lecture_filename, quiz_filename

def test_complete_workflow():
    """Test the complete workflow for a user's weekly learning"""
    print("🚀 Testing Complete Weekly Learning Workflow")
    print("=" * 60)
    
    # Test scenario
    track = "cybersecurity-1"
    week = 3
    topic = "Incident Response and Digital Forensics"
    lecturer = "Dr. Elena Rodriguez"
    
    print(f"📋 Test Scenario:")
    print(f"   Track: {track}")
    print(f"   Week: {week}")
    print(f"   Topic: {topic}")
    print(f"   Lecturer: {lecturer}")
    print()
    
    # Step 1: Generate lecture content
    print("=" * 40)
    print("STEP 1: GENERATE LECTURE CONTENT")
    print("=" * 40)
    
    lecture_content = generate_lecture_content(track, week, topic)
    if not lecture_content:
        print("❌ Failed to generate lecture content. Aborting test.")
        return
    
    print(f"📖 Lecture Content Preview (first 300 chars):")
    print(f"{lecture_content[:300]}...")
    print()
    
    # Step 2: Generate quiz from lecture
    print("=" * 40)
    print("STEP 2: GENERATE QUIZ FROM LECTURE")
    print("=" * 40)
    
    quiz_data = generate_quiz_from_lecture(lecture_content, topic, week, lecturer)
    if not quiz_data:
        print("❌ Failed to generate quiz. Aborting test.")
        return
    
    # Display quiz summary
    questions = quiz_data.get('questions', [])
    print(f"📋 Quiz Summary:")
    print(f"   Title: {quiz_data.get('quiz_title', 'N/A')}")
    print(f"   Questions: {len(questions)}")
    print(f"   Week: {quiz_data.get('week', 'N/A')}")
    print(f"   Lecturer: {quiz_data.get('lecturer', 'N/A')}")
    print()
    
    # Show first question as example
    if questions:
        first_q = questions[0]
        print(f"📝 Sample Question:")
        print(f"   Q: {first_q.get('question', 'N/A')}")
        print(f"   Options: {len(first_q.get('options', []))}")
        print(f"   Answer: {first_q.get('correct_answer', 'N/A')}")
        print()
    
    # Step 3: Save both lecture and quiz
    print("=" * 40)
    print("STEP 3: SAVE CONTENT")
    print("=" * 40)
    
    lecture_file, quiz_file = save_lecture_and_quiz(
        track, week, lecturer, topic, lecture_content, quiz_data
    )
    
    # Step 4: Validation and Summary
    print("=" * 40)
    print("STEP 4: VALIDATION & SUMMARY")
    print("=" * 40)
    
    print("✅ WORKFLOW VALIDATION:")
    print(f"   ✅ Lecture generated: YES ({len(lecture_content)} chars)")
    print(f"   ✅ Quiz generated: YES ({len(questions)} questions)")
    print(f"   ✅ Content saved: YES (2 files)")
    print()
    
    print("📊 QUIZ QUALITY CHECK:")
    print(f"   ✅ Valid JSON structure: YES")
    print(f"   ✅ All questions have text: {all('question' in q and q['question'] for q in questions)}")
    print(f"   ✅ All questions have options: {all('options' in q and isinstance(q['options'], list) and len(q['options']) >= 3 for q in questions)}")
    print(f"   ✅ All questions have answers: {all('correct_answer' in q and q['correct_answer'] for q in questions)}")
    print(f"   ✅ All questions have explanations: {all('explanation' in q and q['explanation'] for q in questions)}")
    print()
    
    print("🎯 USER EXPERIENCE SIMULATION:")
    print(f"   1. User enrolled in {track}")
    print(f"   2. Week {week} begins")
    print(f"   3. User accesses lecture: '{topic}'")
    print(f"   4. User completes lecture reading")
    print(f"   5. Quiz becomes available with {len(questions)} questions")
    print(f"   6. User can test understanding of key concepts")
    print()
    
    print("🏆 SUCCESS: Complete workflow executed successfully!")
    print(f"   📁 Files created: {lecture_file}, {quiz_file}")

def main():
    """Main test function"""
    test_complete_workflow()
    print("\n🏁 Complete Workflow Test Finished!")

if __name__ == "__main__":
    main()
