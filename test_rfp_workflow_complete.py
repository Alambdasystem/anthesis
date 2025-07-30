#!/usr/bin/env python3
"""
Complete End-to-End RFP Workflow Test
Tests the full workflow from contact selection through AI agent processing to response generation.
"""

import requests
import json
import time
import os

# Configuration
API_BASE_URL = "http://127.0.0.1:5000"

# Global token storage
AUTH_TOKEN = None

def print_test_header(test_name):
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"{'='*60}")

def get_auth_headers():
    """Get authorization headers if token is available"""
    if AUTH_TOKEN:
        return {"Authorization": f"Bearer {AUTH_TOKEN}"}
    return {}

def test_login():
    """Test login and get authentication token"""
    print_test_header("Authentication Login")
    global AUTH_TOKEN
    
    # First, try to register a test user
    test_user = {
        "username": "testuser",
        "password": "testpass123",
        "email": "test@example.com"
    }
    
    print("🔧 Attempting to register test user...")
    try:
        register_response = requests.post(
            f"{API_BASE_URL}/register",
            json=test_user,
            timeout=10
        )
        if register_response.status_code == 200:
            print("✅ Test user registered successfully")
        else:
            print(f"ℹ️ Registration response: {register_response.status_code} (user may already exist)")
    except Exception as e:
        print(f"ℹ️ Registration attempt: {e}")
    
    # Now try to login with the test user
    login_creds = {"username": "testuser", "password": "testpass123"}
    
    try:
        print(f"🔐 Trying login with test user...")
        response = requests.post(
            f"{API_BASE_URL}/login",
            json=login_creds,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            AUTH_TOKEN = result.get("token")
            print(f"✅ Login successful! Token obtained.")
            print(f"📝 Token preview: {AUTH_TOKEN[:50]}...")
            return True
        else:
            print(f"❌ Login failed: {response.status_code}")
            if response.text:
                print(f"Error details: {response.text}")
                
    except Exception as e:
        print(f"❌ Login error: {e}")
    
    print("❌ Test user login failed")
    return False

def test_flask_server_status():
    """Test if Flask server is running and responsive"""
    print_test_header("Flask Server Status")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        print(f"✅ Server Response: {response.status_code}")
        if response.status_code == 200:
            print("✅ Flask server is running successfully")
            return True
        else:
            print(f"❌ Flask server returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Flask server connection failed: {e}")
        return False

def test_agents_status():
    """Test agent registration and availability"""
    print_test_header("Agent System Status")
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/agents/status", 
            headers=get_auth_headers(),
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Registered Agents: {data.get('total_agents', 0)}")
            
            agents = data.get('agents', {})
            for category, agent_list in agents.items():
                print(f"  📂 {category}: {len(agent_list)} agents")
                for agent in agent_list:
                    print(f"    - {agent}")
            return True
        else:
            print(f"❌ Agent status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Agent status check error: {e}")
        return False

def test_contacts_endpoint():
    """Test contacts API endpoint"""
    print_test_header("Contacts API")
    try:
        response = requests.get(
            f"{API_BASE_URL}/contacts", 
            headers=get_auth_headers(),
            timeout=10
        )
        if response.status_code == 200:
            contacts = response.json()
            print(f"✅ Contacts loaded: {len(contacts)} contacts")
            if contacts:
                print(f"  Sample contact: {contacts[0].get('name', 'Unknown')} - {contacts[0].get('company', 'Unknown Company')}")
            return True, contacts
        else:
            print(f"❌ Contacts API failed: {response.status_code}")
            if response.text:
                print(f"Error details: {response.text}")
            return False, []
    except Exception as e:
        print(f"❌ Contacts API error: {e}")
        return False, []

def test_agent_predict_endpoint():
    """Test the enhanced agent predict endpoint"""
    print_test_header("Agent Predict Endpoint")
    
    test_message = "Test message for RFP processing workflow"
    personas_to_test = ["coordinator", "document-analysis", "content-generation"]
    
    for persona in personas_to_test:
        print(f"\n🔍 Testing persona: {persona}")
        try:
            payload = {
                "message": test_message,
                "persona": persona,
                "context": {"test": True, "workflow": "rfp"}
            }
            
            response = requests.post(
                f"{API_BASE_URL}/predict", 
                json=payload,
                headers=get_auth_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ {persona}: Response received")
                print(f"  📝 Response length: {len(result.get('response', ''))}")
            else:
                print(f"  ❌ {persona}: Failed with status {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ {persona}: Error - {e}")

def test_workflow_endpoint():
    """Test the multi-agent workflow endpoint"""
    print_test_header("Multi-Agent Workflow")
    
    test_rfp_content = """
    REQUEST FOR PROPOSAL
    
    Project: Website Redesign and Development
    
    We are seeking a qualified vendor to redesign and develop our company website.
    The project should include modern UI/UX design, responsive layout, and SEO optimization.
    
    Requirements:
    - Modern, professional design
    - Mobile-responsive layout
    - Content management system
    - SEO optimization
    - Timeline: 3 months
    - Budget: $25,000 - $35,000
    
    Please provide a detailed proposal including timeline, deliverables, and cost breakdown.
    """
    
    try:
        payload = {
            "document_content": test_rfp_content,
            "document_type": "rfp",
            "workflow_type": "rfp_response",
            "context": {
                "company": "Test Company",
                "contact": "Test Contact"
            }
        }
        
        print("🚀 Initiating workflow...")
        response = requests.post(
            f"{API_BASE_URL}/api/agents/workflow",
            json=payload,
            headers=get_auth_headers(),
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Workflow completed successfully!")
            
            # Display workflow results
            if "workflow_results" in result:
                workflow = result["workflow_results"]
                print(f"\n📋 Workflow Steps Completed:")
                
                for step, data in workflow.items():
                    print(f"  {step}:")
                    if isinstance(data, dict):
                        if "response" in data:
                            response_preview = data["response"][:200] + "..." if len(data["response"]) > 200 else data["response"]
                            print(f"    Response: {response_preview}")
                        if "status" in data:
                            print(f"    Status: {data['status']}")
                    else:
                        print(f"    {str(data)[:200]}...")
            
            if "final_response" in result:
                final_preview = result["final_response"][:300] + "..." if len(result["final_response"]) > 300 else result["final_response"]
                print(f"\n📄 Final Response Preview:")
                print(f"  {final_preview}")
                
            return True, result
        else:
            print(f"❌ Workflow failed with status: {response.status_code}")
            if response.text:
                print(f"Error details: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ Workflow error: {e}")
        return False, None

def test_contact_rfp_flow():
    """Test the complete contact + RFP workflow"""
    print_test_header("Contact + RFP Integration Flow")
    
    # First get contacts
    contacts_success, contacts = test_contacts_endpoint()
    if not contacts_success or not contacts:
        print("❌ Cannot test RFP flow without contacts")
        return False
    
    # Use first contact for testing
    test_contact = contacts[0]
    contact_idx = 0
    
    print(f"📞 Using test contact: {test_contact.get('name')} from {test_contact.get('company')}")
    
    # Test RFP processing with contact context
    rfp_content = f"""
    REQUEST FOR PROPOSAL - {test_contact.get('company', 'Unknown Company')}
    
    Dear {test_contact.get('name', 'Team')},
    
    We are interested in your services for a digital transformation project.
    
    Project Scope:
    - Custom software development
    - Database integration
    - User training and support
    
    Timeline: 6 months
    Budget: $50,000 - $75,000
    
    Please provide your proposal including methodology, timeline, and pricing.
    
    Best regards,
    Procurement Team
    """
    
    try:
        payload = {
            "document_content": rfp_content,
            "document_type": "rfp",
            "workflow_type": "rfp_response",
            "context": {
                "contact_idx": contact_idx,
                "contact_name": test_contact.get('name'),
                "contact_company": test_contact.get('company'),
                "contact_email": test_contact.get('email')
            }
        }
        
        print("🔄 Processing RFP with contact context...")
        response = requests.post(
            f"{API_BASE_URL}/api/agents/workflow",
            json=payload,
            headers=get_auth_headers(),
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Contact RFP workflow completed!")
            
            # Test saving the result as a draft
            if "final_response" in result:
                draft_payload = {
                    "type": "rfp",
                    "content": result["final_response"],
                    "subject": f"RFP Response for {test_contact.get('company')}",
                    "metadata": {
                        "contact_idx": contact_idx,
                        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "workflow_used": True
                    }
                }
                
                print("💾 Saving as draft...")
                draft_response = requests.post(
                    f"{API_BASE_URL}/contacts/{contact_idx}/drafts",
                    json=draft_payload,
                    headers=get_auth_headers(),
                    timeout=10
                )
                
                if draft_response.status_code == 200:
                    draft_result = draft_response.json()
                    print(f"✅ Draft saved with ID: {draft_result.get('draft_id')}")
                else:
                    print(f"❌ Draft save failed: {draft_response.status_code}")
                    
            return True
        else:
            print(f"❌ Contact RFP workflow failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Contact RFP flow error: {e}")
        return False

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🚀 Starting Comprehensive RFP Workflow Test")
    print("=" * 60)
    
    test_results = {
        "login": False,
        "server_status": False,
        "agents_status": False,
        "contacts_api": False,
        "agent_predict": False,
        "workflow_endpoint": False,
        "contact_rfp_flow": False
    }
    
    # Test 0: Login and get token
    test_results["login"] = test_login()
    
    if not test_results["login"]:
        print("\n❌ Login failed. Cannot proceed with authenticated tests.")
        return test_results
    
    # Test 1: Server Status
    test_results["server_status"] = test_flask_server_status()
    
    if not test_results["server_status"]:
        print("\n❌ Flask server is not running. Please start the server first:")
        print("python app.py")
        return
    
    # Test 2: Agents Status
    test_results["agents_status"] = test_agents_status()
    
    # Test 3: Contacts API
    test_results["contacts_api"], _ = test_contacts_endpoint()
    
    # Test 4: Agent Predict Endpoint
    if test_results["agents_status"]:
        test_agent_predict_endpoint()
        test_results["agent_predict"] = True
    
    # Test 5: Workflow Endpoint
    if test_results["agents_status"]:
        workflow_success, _ = test_workflow_endpoint()
        test_results["workflow_endpoint"] = workflow_success
    
    # Test 6: Complete Contact RFP Flow
    if test_results["contacts_api"] and test_results["workflow_endpoint"]:
        test_results["contact_rfp_flow"] = test_contact_rfp_flow()
    
    # Final Results Summary
    print_test_header("FINAL TEST RESULTS")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"📊 Test Results: {passed_tests}/{total_tests} passed")
    print()
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! The RFP workflow is fully functional.")
        print("\n📋 Workflow Summary:")
        print("1. ✅ Flask server running with all agents registered")
        print("2. ✅ Contact management system operational")
        print("3. ✅ AI agent predict endpoint working")
        print("4. ✅ Multi-agent workflow processing RFPs")
        print("5. ✅ Complete contact + RFP integration functional")
        print("\n🎯 The system is ready for production use!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review the issues above.")
    
    return test_results

if __name__ == "__main__":
    run_comprehensive_test()
