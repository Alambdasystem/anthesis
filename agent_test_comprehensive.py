#!/usr/bin/env python3
"""
Comprehensive Agent Framework Testing Script
Based on agent_test_plan.md
"""

import json
import requests
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agents.lecture_agents import LectureAgent, LECTURE_AGENTS
    from agents.base import BaseAgent, AgentResult
    AGENT_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import agent modules: {e}")
    AGENT_IMPORTS_AVAILABLE = False

class AgentTester:
    """Comprehensive testing framework for agent system"""
    
    def __init__(self):
        self.test_results = []
        self.errors = []
        self.warnings = []
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        print(f"{status}: {test_name}")
        if details:
            print(f"    {details}")
    
    def log_warning(self, message: str):
        """Log warning"""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")
    
    def log_error(self, message: str):
        """Log error"""
        self.errors.append(message)
        print(f"❌ ERROR: {message}")

    # PHASE 1: Core Agent Functions Tests
    
    def test_lecture_agent_class(self):
        """Test LectureAgent class functionality"""
        print("\n🔍 Testing LectureAgent Class...")
        
        if not AGENT_IMPORTS_AVAILABLE:
            self.log_test("LectureAgent Import", False, "Could not import agent modules")
            return
            
        try:
            # Test agent creation
            agent = LectureAgent(
                lecture_id="test-agent",
                name="Test Agent", 
                specialization="Testing",
                persona="Test persona for testing purposes"
            )
            self.log_test("LectureAgent Creation", True, f"Created agent: {agent.name}")
            
            # Test agent properties
            self.log_test("Agent Properties", 
                         agent.name == "Test Agent" and agent.specialization == "Testing",
                         f"Name: {agent.name}, Specialization: {agent.specialization}")
            
            # Test available functions
            functions = agent.get_available_functions()
            expected_functions = ["deliver_lecture", "answer_question", "generate_quiz", "explain_concept"]
            has_all_functions = all(any(f["name"] == func for f in functions) for func in expected_functions)
            self.log_test("Available Functions", has_all_functions, 
                         f"Functions: {[f['name'] for f in functions]}")
            
        except Exception as e:
            self.log_test("LectureAgent Class", False, f"Exception: {str(e)}")
    
    def test_predefined_agents(self):
        """Test predefined lecture agents"""
        print("\n🔍 Testing Predefined Agents...")
        
        if not AGENT_IMPORTS_AVAILABLE:
            self.log_test("Predefined Agents", False, "Could not import agent modules")
            return
            
        try:
            expected_agents = ['dr-smith', 'prof-chen', 'dr-wilson', 'prof-taylor']
            
            for agent_id in expected_agents:
                if agent_id in LECTURE_AGENTS:
                    agent = LECTURE_AGENTS[agent_id]
                    self.log_test(f"Agent {agent_id} exists", True, 
                                f"Name: {agent.name}, Specialization: {agent.specialization}")
                else:
                    self.log_test(f"Agent {agent_id} exists", False, "Agent not found in LECTURE_AGENTS")
                    
        except Exception as e:
            self.log_test("Predefined Agents", False, f"Exception: {str(e)}")
    
    def test_agent_registry_file(self):
        """Test agent registry JSON file"""
        print("\n🔍 Testing Agent Registry File...")
        
        registry_path = "agents_registry.json"
        if not os.path.exists(registry_path):
            self.log_test("Registry File Exists", False, f"File not found: {registry_path}")
            return
            
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            self.log_test("Registry File Load", True, f"Loaded {len(registry)} agents")
            
            # Check for lecture agents
            lecture_agents = [agent for agent in registry if agent.get('agent_type') == 'lecture']
            expected_names = ['Dr. Smith', 'Prof. Chen', 'Dr. Wilson', 'Prof. Taylor']
            
            for name in expected_names:
                found = any(agent['name'] == name for agent in lecture_agents)
                self.log_test(f"Registry contains {name}", found, 
                            f"Agent {'found' if found else 'not found'} in registry")
                            
        except Exception as e:
            self.log_test("Registry File", False, f"Exception: {str(e)}")
    
    def test_agent_contacts_file(self):
        """Test agent contacts JSON file"""
        print("\n🔍 Testing Agent Contacts File...")
        
        contacts_path = "agent_contacts.json"
        if not os.path.exists(contacts_path):
            self.log_test("Contacts File Exists", False, f"File not found: {contacts_path}")
            return
            
        try:
            with open(contacts_path, 'r') as f:
                contacts = json.load(f)
            
            self.log_test("Contacts File Load", True, f"Loaded {len(contacts)} contacts")
            
            # Check for agent contacts
            agent_contacts = [contact for contact in contacts if contact.get('type') == 'agent']
            expected_names = ['Dr. Smith', 'Prof. Chen', 'Dr. Wilson', 'Prof. Taylor']
            
            for name in expected_names:
                found = any(contact['name'] == name for contact in agent_contacts)
                self.log_test(f"Contacts contains {name}", found,
                            f"Contact {'found' if found else 'not found'} in contacts")
                            
        except Exception as e:
            self.log_test("Contacts File", False, f"Exception: {str(e)}")

    # PHASE 2: UI/UX Component Tests
    
    def test_html_structure(self):
        """Test HTML structure for agent UI elements"""
        print("\n🔍 Testing HTML Structure...")
        
        html_path = "cadet_dashboard.html"
        if not os.path.exists(html_path):
            self.log_test("HTML File Exists", False, f"File not found: {html_path}")
            return
            
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check for agent selector buttons
            agent_buttons = ['data-agent="dr-smith"', 'data-agent="prof-chen"', 
                           'data-agent="dr-wilson"', 'data-agent="prof-taylor"']
            
            for button in agent_buttons:
                found = button in html_content
                self.log_test(f"Agent button {button}", found, 
                            f"Button {'found' if found else 'not found'} in HTML")
            
            # Check for chat elements
            chat_elements = ['id="chatInput"', 'id="mindPanelChat"', 'id="mindPanelInput"']
            for element in chat_elements:
                found = element in html_content
                self.log_test(f"Chat element {element}", found,
                            f"Element {'found' if found else 'not found'} in HTML")
            
            # Check for duplicate methods (from test plan)
            getContextSummary_count = html_content.count('getContextSummary()')
            setContext_count = html_content.count('setContext(week, topic, context)')
            
            self.log_test("Duplicate getContextSummary", getContextSummary_count <= 1,
                        f"Found {getContextSummary_count} instances (should be 1)")
            self.log_test("Duplicate setContext", setContext_count <= 1,
                        f"Found {setContext_count} instances (should be 1)")
                        
        except Exception as e:
            self.log_test("HTML Structure", False, f"Exception: {str(e)}")
    
    def test_javascript_errors(self):
        """Test for common JavaScript errors in HTML"""
        print("\n🔍 Testing JavaScript Structure...")
        
        html_path = "cadet_dashboard.html"
        if not os.path.exists(html_path):
            return
            
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check for LectureAgent class definition
            lecture_agent_class = "class LectureAgent" in html_content
            self.log_test("LectureAgent JS Class", lecture_agent_class,
                        f"JavaScript class {'found' if lecture_agent_class else 'not found'}")
            
            # Check for agent initialization
            agents_init = "lectureAgents" in html_content
            self.log_test("Agents Initialization", agents_init,
                        f"Agent initialization {'found' if agents_init else 'not found'}")
            
            # Check for error handling patterns
            try_catch_count = html_content.count('try {')
            error_handling_count = html_content.count('catch')
            
            self.log_test("Error Handling Present", try_catch_count > 0 and error_handling_count > 0,
                        f"Found {try_catch_count} try blocks and {error_handling_count} catch blocks")
                        
        except Exception as e:
            self.log_test("JavaScript Structure", False, f"Exception: {str(e)}")

    # PHASE 3: Integration Tests
    
    def test_flask_app_running(self):
        """Test if Flask app is running"""
        print("\n🔍 Testing Flask App Integration...")
        
        try:
            response = requests.get("http://localhost:5000", timeout=5)
            self.log_test("Flask App Running", response.status_code == 200,
                        f"Status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.log_test("Flask App Running", False, "Connection refused - app not running")
        except Exception as e:
            self.log_test("Flask App Running", False, f"Exception: {str(e)}")
    
    def test_agent_endpoints(self):
        """Test agent-related API endpoints"""
        print("\n🔍 Testing Agent API Endpoints...")
        
        endpoints = [
            "/api/agents",
            "/api/agents/list",
            "/api/contacts"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"http://localhost:5000{endpoint}", timeout=5)
                self.log_test(f"Endpoint {endpoint}", response.status_code == 200,
                            f"Status: {response.status_code}")
            except requests.exceptions.ConnectionError:
                self.log_test(f"Endpoint {endpoint}", False, "Connection refused")
            except Exception as e:
                self.log_test(f"Endpoint {endpoint}", False, f"Exception: {str(e)}")
    
    def test_ollama_integration(self):
        """Test Ollama API integration"""
        print("\n🔍 Testing Ollama Integration...")
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            self.log_test("Ollama API Running", response.status_code == 200,
                        f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                has_llama = any('llama' in model.get('name', '').lower() for model in models)
                self.log_test("Llama Model Available", has_llama,
                            f"Models: {[m.get('name') for m in models]}")
                            
        except requests.exceptions.ConnectionError:
            self.log_test("Ollama API Running", False, "Connection refused - Ollama not running")
        except Exception as e:
            self.log_test("Ollama Integration", False, f"Exception: {str(e)}")

    # Test Execution Methods
    
    def run_phase1_tests(self):
        """Run Phase 1: Core Agent Functions"""
        print("=" * 60)
        print("🚀 PHASE 1: Core Agent Functions Testing")
        print("=" * 60)
        
        self.test_lecture_agent_class()
        self.test_predefined_agents()
        self.test_agent_registry_file()
        self.test_agent_contacts_file()
    
    def run_phase2_tests(self):
        """Run Phase 2: UI/UX Components"""
        print("\n" + "=" * 60)
        print("🎨 PHASE 2: UI/UX Components Testing")
        print("=" * 60)
        
        self.test_html_structure()
        self.test_javascript_errors()
    
    def run_phase3_tests(self):
        """Run Phase 3: Integration Testing"""
        print("\n" + "=" * 60)
        print("🔗 PHASE 3: Integration Testing")
        print("=" * 60)
        
        self.test_flask_app_running()
        self.test_agent_endpoints()
        self.test_ollama_integration()
    
    def run_all_tests(self):
        """Run all test phases"""
        print("🧪 Starting Comprehensive Agent Framework Testing")
        print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.run_phase1_tests()
        self.run_phase2_tests()
        self.run_phase3_tests()
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results if test['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"🔴 Errors: {len(self.errors)}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for test in self.test_results:
                if not test['passed']:
                    print(f"  - {test['test']}: {test['details']}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.errors:
            print("\n🔴 ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        # Save results to file
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'total': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'success_rate': success_rate,
                    'timestamp': datetime.now().isoformat()
                },
                'tests': self.test_results,
                'warnings': self.warnings,
                'errors': self.errors
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")


def main():
    """Main test execution"""
    tester = AgentTester()
    
    if len(sys.argv) > 1:
        phase = sys.argv[1].lower()
        if phase == "phase1":
            tester.run_phase1_tests()
        elif phase == "phase2":
            tester.run_phase2_tests()
        elif phase == "phase3":
            tester.run_phase3_tests()
        else:
            print(f"Unknown phase: {phase}")
            print("Usage: python agent_test_comprehensive.py [phase1|phase2|phase3]")
    else:
        tester.run_all_tests()
    
    tester.print_summary()


if __name__ == "__main__":
    main()
