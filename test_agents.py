"""
Test script for the new agent system
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.base import initialize_default_agents
from agents.registry import get_agent_registry

def test_agent_system():
    print("🧪 Testing Agent System")
    print("=" * 50)
    
    # Initialize agents
    print("\n1. Initializing agents...")
    try:
        registry = initialize_default_agents()
        print(f"✅ Agents initialized. Total count: {registry.get_agent_count()}")
    except Exception as e:
        print(f"❌ Failed to initialize agents: {e}")
        return
    
    # List all agents
    print("\n2. Listing all agents...")
    agents = registry.list_all_agents()
    for i, agent in enumerate(agents, 1):
        print(f"   {i}. {agent['name']} ({agent.get('agent_type', 'AI')})")
        print(f"      - ID: {agent['id']}")
        print(f"      - Capabilities: {', '.join(agent['capabilities'])}")
        print()
    
    # Test capabilities
    print("3. Testing capabilities...")
    capabilities = {}
    for agent in registry.get_all_agents():
        for capability in agent.capabilities:
            if capability not in capabilities:
                capabilities[capability] = []
            capabilities[capability].append(agent.name)
    
    for capability, agent_names in capabilities.items():
        print(f"   📋 {capability}: {', '.join(agent_names)}")
    
    # Test specific agent types
    print("\n4. Testing agent types...")
    lecture_agents = registry.get_agents_by_type("lecture")
    human_agents = registry.get_agents_by_type("human")
    ai_agents = [a for a in registry.get_all_agents() if not hasattr(a, 'agent_type')]
    
    print(f"   🎓 Lecture agents: {len(lecture_agents)}")
    print(f"   👥 Human agents: {len(human_agents)}")
    print(f"   🤖 AI agents: {len(ai_agents)}")
    
    # Test agent execution
    print("\n5. Testing agent execution...")
    if lecture_agents:
        agent = lecture_agents[0]
        print(f"   Testing {agent.name}...")
        result = agent.execute("explain_concept", {
            "concept": "Machine Learning",
            "level": "beginner"
        })
        print(f"   Result success: {result.get('success', False)}")
        if result.get('success'):
            print(f"   Content preview: {result.get('data', {}).get('explanation', 'No content')[:100]}...")
    
    print("\n✅ Agent system test completed!")

if __name__ == "__main__":
    test_agent_system()
