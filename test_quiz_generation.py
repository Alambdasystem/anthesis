#!/usr/bin/env python3
"""
Test script for quiz generation from lecture content
"""
import requests
import json
import time

BASE_URL = "https://api.alambda.com"

def test_quiz_generation():
    """Test the new quiz generation endpoint"""
    print("🧪 Testing Quiz Generation from Lecture Content")
    print("=" * 50)
    
    # Test data
    test_lecture = {
        "lecture_content": """
        Welcome to Week 3 of Python Programming Fundamentals. 
        Today we'll explore object-oriented programming concepts including classes, objects, and inheritance.
        
        A class is a blueprint for creating objects. Think of it as a template that defines the properties 
        and methods that objects of that class will have. For example, if we have a Car class, it might 
        have properties like color, make, model, and year, and methods like start(), stop(), and accelerate().
        
        Objects are instances of classes. When you create an object from a class, you're instantiating 
        that class. Each object has its own set of property values but shares the same methods defined 
        in the class.
        
        Inheritance allows us to create new classes based on existing classes. The new class (child class) 
        inherits properties and methods from the parent class, and can also have its own additional 
        properties and methods. This promotes code reuse and creates hierarchical relationships between classes.
        
        Let's look at a practical example:
        
        class Animal:
            def __init__(self, name, species):
                self.name = name
                self.species = species
            
            def make_sound(self):
                return "Some generic animal sound"
        
        class Dog(Animal):
            def __init__(self, name, breed):
                super().__init__(name, "Canine")
                self.breed = breed
            
            def make_sound(self):
                return "Woof!"
        
        Key concepts to remember:
        1. Encapsulation - bundling data and methods together
        2. Inheritance - creating new classes from existing ones
        3. Polymorphism - same method name, different implementations
        4. Abstraction - hiding complex implementation details
        """,
        "topic": "Week 3: Object-Oriented Programming in Python",
        "week": 3,
        "lecturer": "Dr. Smith"
    }
    
    try:
        # Step 1: Test quiz generation directly (skip auth for now)
        print("1. Testing quiz generation endpoint directly...")
        
        # Use a simple test without authentication first
        headers = {
            'Content-Type': 'application/json'
        }
        
        quiz_response = requests.post(
            f"{BASE_URL}/api/quizzes/generate-from-lecture",
            json=test_lecture,
            headers=headers,
            timeout=120  # Increased timeout for Ollama processing
        )
        
        print(f"Quiz generation response status: {quiz_response.status_code}")
        
        if quiz_response.status_code == 401:
            print("⚠️ Authentication required. Let's test with the browser approach...")
            # Try to use the dashboard approach
            print("2. Testing via browser interface...")
            print("Please open the browser and generate a quiz from a lecture to test the functionality.")
            return False
        
        if quiz_response.status_code != 200:
            print(f"❌ Quiz generation failed: {quiz_response.status_code}")
            print(f"Response: {quiz_response.text}")
            return False
        
        quiz_data = quiz_response.json()
        print("✅ Quiz generated successfully!")
        
        # Step 3: Display quiz results
        print("\n📋 Generated Quiz:")
        print("-" * 30)
        print(f"Title: {quiz_data.get('quiz_title', 'N/A')}")
        print(f"Lecturer: {quiz_data.get('lecturer', 'N/A')}")
        print(f"Week: {quiz_data.get('week', 'N/A')}")
        print(f"Number of questions: {len(quiz_data.get('questions', []))}")
        
        # Display first question as sample
        questions = quiz_data.get('questions', [])
        if questions:
            print(f"\n🔍 Sample Question:")
            first_q = questions[0]
            print(f"Q: {first_q.get('question', 'N/A')}")
            for option in first_q.get('options', []):
                print(f"   {option}")
            print(f"Correct Answer: {first_q.get('correct_answer', 'N/A')}")
            print(f"Explanation: {first_q.get('explanation', 'N/A')}")
        
        print(f"\n🎉 Test completed successfully!")
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out - Ollama may be processing slowly")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - Is the Flask server running?")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Starting quiz generation test...")
    success = test_quiz_generation()
    
    if success:
        print("\n✅ All tests passed! Quiz generation is working correctly.")
    else:
        print("\n❌ Test failed. Check the server logs for more details.")
