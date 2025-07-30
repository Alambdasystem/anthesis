#!/usr/bin/env python3
"""
Fixed Agent Testing Script - Bypasses registry issues
"""

import json
import requests
import os
from datetime import datetime

def test_agent_framework():
    """Final comprehensive test of the agent framework"""
    
    print("🧪 Starting Final Agent Framework Test")
    print("=" * 60)
    
    results = {
        "passed": 0,
        "failed": 0,
        "tests": []
    }
    
    def log_test(name, passed, details=""):
        status = "✅ PASS" if passed else "❌ FAIL"
        results["tests"].append({"name": name, "passed": passed, "details": details})
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
        print(f"{status}: {name}")
        if details:
            print(f"    {details}")
    
    # Test 1: Check if duplicate methods are fixed
    print("\n🔍 Testing Code Quality Fixes...")
    try:
        with open("cadet_dashboard.html", 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        getContextSummary_count = html_content.count('getContextSummary()')
        setContext_count = html_content.count('setContext(week, topic, context)')
        
        log_test("Duplicate Methods Fixed", 
                getContextSummary_count == 1 and setContext_count == 1,
                f"getContextSummary: {getContextSummary_count}, setContext: {setContext_count}")
    except Exception as e:
        log_test("Duplicate Methods Test", False, f"Error: {e}")
    
    # Test 2: Check agent contacts
    print("\n🔍 Testing Agent Contacts...")
    try:
        with open("agent_contacts.json", 'r') as f:
            contacts = json.load(f)
        
        agent_contacts = [c for c in contacts if c.get('type') == 'agent']
        expected_agents = ['Dr. Smith', 'Prof. Chen', 'Dr. Wilson', 'Prof. Taylor']
        
        all_found = all(any(c['name'] == name for c in agent_contacts) for name in expected_agents)
        log_test("All Agents in Contacts", all_found, 
                f"Found {len(agent_contacts)} agent contacts")
        
        for name in expected_agents:
            found = any(c['name'] == name for c in agent_contacts)
            log_test(f"Contact: {name}", found, "Agent contact present")
            
    except Exception as e:
        log_test("Agent Contacts Test", False, f"Error: {e}")
    
    # Test 3: Check UI elements
    print("\n🔍 Testing UI Elements...")
    try:
        with open("cadet_dashboard.html", 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for agent buttons
        agent_buttons = ['data-agent="dr-smith"', 'data-agent="prof-chen"', 
                        'data-agent="dr-wilson"', 'data-agent="prof-taylor"']
        
        for button in agent_buttons:
            found = button in html_content
            log_test(f"UI Button: {button}", found, "Button element present")
        
        # Check for chat elements
        chat_elements = ['id="chatInput"', 'id="mindPanelChat"', 'id="mindPanelInput"']
        for element in chat_elements:
            found = element in html_content
            log_test(f"Chat Element: {element}", found, "Chat element present")
        
        # Check for status indicators (our improvement)
        status_css = ".agent-status" in html_content
        log_test("Status Indicators Added", status_css, "CSS for agent status present")
        
        # Check for JavaScript enhancements
        status_js = "AgentStatusManager" in html_content
        log_test("Status Manager Added", status_js, "JavaScript status manager present")
        
    except Exception as e:
        log_test("UI Elements Test", False, f"Error: {e}")
    
    # Test 4: Check API endpoints
    print("\n🔍 Testing API Integration...")
    endpoints = [
        "/api/agents",
        "/api/agents/list", 
        "/api/contacts"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"http://localhost:5000{endpoint}", timeout=5)
            log_test(f"Endpoint: {endpoint}", response.status_code == 200,
                    f"Status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            log_test(f"Endpoint: {endpoint}", False, "Connection refused")
        except Exception as e:
            log_test(f"Endpoint: {endpoint}", False, f"Error: {e}")
    
    # Test 5: Check Flask app
    print("\n🔍 Testing Flask Application...")
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        log_test("Flask App Running", response.status_code == 200,
                f"Status: {response.status_code}")
    except Exception as e:
        log_test("Flask App Running", False, f"Error: {e}")
    
    # Test 6: Check Ollama integration
    print("\n🔍 Testing Ollama Integration...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        log_test("Ollama API", response.status_code == 200,
                f"Status: {response.status_code}")
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            has_llama = any('llama' in model.get('name', '').lower() for model in models)
            log_test("Llama Model Available", has_llama,
                    f"Models: {[m.get('name') for m in models]}")
    except Exception as e:
        log_test("Ollama Integration", False, f"Error: {e}")
    
    # Test 7: Check improvements files
    print("\n🔍 Testing Improvement Files...")
    
    improvement_files = [
        "agent_metrics.json",
        "TESTING_REPORT.md"
    ]
    
    for file in improvement_files:
        exists = os.path.exists(file)
        log_test(f"File: {file}", exists, f"File {'exists' if exists else 'missing'}")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    
    total = results["passed"] + results["failed"]
    success_rate = (results["passed"] / total * 100) if total > 0 else 0
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if results["failed"] > 0:
        print("\n❌ FAILED TESTS:")
        for test in results["tests"]:
            if not test["passed"]:
                print(f"  - {test['name']}: {test['details']}")
    
    # Overall assessment
    print("\n🎯 OVERALL ASSESSMENT:")
    if success_rate >= 90:
        print("🟢 EXCELLENT: Agent framework is fully operational!")
    elif success_rate >= 80:
        print("🟡 GOOD: Agent framework is mostly working with minor issues")
    elif success_rate >= 70:
        print("🟠 FAIR: Agent framework has some issues that need attention")
    else:
        print("🔴 POOR: Agent framework needs significant work")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"final_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "success_rate": success_rate,
            "total_tests": total,
            "passed": results["passed"],
            "failed": results["failed"],
            "tests": results["tests"]
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    return success_rate

if __name__ == "__main__":
    test_agent_framework()
