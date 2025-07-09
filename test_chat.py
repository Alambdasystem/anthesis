"""
Test Chat System - Verify chat functionality
"""

import requests
import json
import os

def test_chat_system():
    """Test the chat system endpoints"""
    print("🧪 Testing Chat System")
    print("=" * 50)
    
    base_url = "http://localhost:5000"  # Adjust as needed
    
    # Test 1: Get contacts list
    print("\n1. Testing contacts endpoint...")
    try:
        response = requests.get(f"{base_url}/api/agents/contacts")
        if response.status_code == 200:
            contacts = response.json()
            print(f"✅ Found {len(contacts)} contacts")
            for contact in contacts[:3]:  # Show first 3
                print(f"   - {contact.get('name')} ({contact.get('type')})")
        else:
            print(f"❌ Failed to get contacts: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing contacts: {e}")
    
    # Test 2: Check chat inbox file
    print("\n2. Testing chat inbox file...")
    chat_file = "data/chat_inbox.json"
    if os.path.exists(chat_file):
        with open(chat_file, 'r') as f:
            messages = json.load(f)
        print(f"✅ Chat inbox exists with {len(messages)} messages")
    else:
        print("❌ Chat inbox file not found")
    
    # Test 3: Check agent registry
    print("\n3. Testing agent registry...")
    try:
        response = requests.get(f"{base_url}/api/agents/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Agent system healthy with {health.get('registered_agents', 0)} agents")
        else:
            print(f"❌ Agent health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing agent health: {e}")
    
    print("\n✅ Chat system test completed!")

if __name__ == "__main__":
    test_chat_system()
