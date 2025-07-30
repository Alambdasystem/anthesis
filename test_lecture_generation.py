#!/usr/bin/env python3
"""
Test script for lecture generation functionality
"""

import requests
import json
import time

def test_lecture_generation():
    """Test the lecture generation functionality"""
    
    print("🎓 Testing Lecture Generation Functionality")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Check if Flask app is running
    try:
        response = requests.get(base_url, timeout=5)
        print(f"✅ Flask app running: {response.status_code == 200}")
    except:
        print("❌ Flask app not running")
        return
    
    # Test 2: Check agent execution endpoint
    try:
        test_payload = {
            "agent_id": "dr-smith",
            "function": "deliver_lecture",
            "context": {
                "topic": "Test Topic - Introduction to AI",
                "duration": 30,
                "difficulty_level": "Beginner"
            }
        }
        
        # Note: This will fail without auth token, but we can check if endpoint exists
        response = requests.post(f"{base_url}/api/agents/execute", 
                               json=test_payload, 
                               timeout=10)
        
        if response.status_code == 401:
            print("✅ Agent execution endpoint exists (auth required)")
        elif response.status_code == 200:
            result = response.json()
            print(f"✅ Agent execution successful: {result.get('success', False)}")
        else:
            print(f"⚠️ Agent execution endpoint status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing agent execution: {e}")
    
    # Test 3: Check lectures endpoint
    try:
        response = requests.get(f"{base_url}/api/lectures/cadet_1", timeout=5)
        print(f"✅ Lectures endpoint accessible: {response.status_code in [200, 401]}")
    except Exception as e:
        print(f"❌ Error testing lectures endpoint: {e}")
    
    # Test 4: Check PDF generation endpoint
    try:
        test_payload = {"track": "cadet_1", "week": 1}
        response = requests.post(f"{base_url}/lectures/generate", 
                               json=test_payload, 
                               timeout=10)
        
        if response.status_code == 401:
            print("✅ PDF generation endpoint exists (auth required)")
        elif response.status_code == 200:
            print("✅ PDF generation endpoint working")
        else:
            print(f"⚠️ PDF generation endpoint status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing PDF generation: {e}")
    
    # Test 5: Check if agents are importable
    try:
        import sys
        import os
        sys.path.append(os.getcwd())
        
        from agents.lecture_agents import LECTURE_AGENTS
        
        print(f"✅ Lecture agents loaded: {len(LECTURE_AGENTS)} agents")
        
        # Test local agent execution
        dr_smith = LECTURE_AGENTS.get('dr-smith')
        if dr_smith:
            print(f"✅ Dr. Smith agent available: {dr_smith.name}")
            
            # Test lecture delivery function
            try:
                result = dr_smith.execute('deliver_lecture', {
                    'topic': 'Test Lecture - Machine Learning Basics',
                    'duration': 15,
                    'difficulty_level': 'Beginner'
                })
                
                if result.get('success'):
                    print("✅ Local agent lecture generation successful")
                    print(f"   📝 Generated {len(result['data']['lecture_content'])} characters")
                else:
                    print(f"❌ Local agent execution failed: {result.get('error')}")
                    
            except Exception as e:
                print(f"❌ Error in local agent execution: {e}")
        else:
            print("❌ Dr. Smith agent not found")
            
    except ImportError as e:
        print(f"❌ Could not import agents: {e}")
    except Exception as e:
        print(f"❌ Error testing agents: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Lecture Generation Test Summary")
    print("=" * 50)
    print("✅ If all tests above show checkmarks, lecture generation should work!")
    print("🔧 To test in browser:")
    print("   1. Open cadet dashboard")
    print("   2. Go to Lectures tab")
    print("   3. Select a week (1-14)")
    print("   4. Choose a professor (Dr. Smith, Prof. Chen, etc.)")
    print("   5. Click 'Generate New Lecture'")
    print("   6. Check console for detailed logs")

def test_ollama_integration():
    """Test if Ollama is available for AI generation"""
    print("\n🤖 Testing Ollama Integration")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama running with {len(models)} models")
            
            for model in models:
                print(f"   📦 {model.get('name', 'Unknown')}")
            
            # Test generation
            test_payload = {
                "model": "llama3.2:latest",
                "messages": [{"role": "user", "content": "Write a brief introduction to machine learning in 2 sentences."}],
                "stream": False
            }
            
            response = requests.post("http://localhost:11434/api/chat", 
                                   json=test_payload, 
                                   timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('message', {}).get('content', '')
                print(f"✅ Ollama generation test successful")
                print(f"   📝 Generated: {content[:100]}...")
            else:
                print(f"⚠️ Ollama generation failed: {response.status_code}")
        else:
            print(f"❌ Ollama not responding: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Ollama not available: {e}")

if __name__ == "__main__":
    test_lecture_generation()
    test_ollama_integration()
