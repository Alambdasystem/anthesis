// Test script to verify agent fixes
console.log('🧪 Testing agent system fixes...');

// Test 1: Verify agent ID generation
const testAgent = new LectureAgent('Prof. Chen', 'Data Science', 'Test persona');
console.log('✓ Agent ID for Prof. Chen:', testAgent.lecture_id);
console.log('✓ Expected: prof-chen, Got:', testAgent.lecture_id);

// Test 2: Verify all required methods exist
const requiredMethods = ['getName', 'getStatus', 'getUsageStats', 'getContext', 'setContext', 'reset'];
const missingMethods = [];

requiredMethods.forEach(method => {
  if (typeof testAgent[method] !== 'function') {
    missingMethods.push(method);
  }
});

if (missingMethods.length === 0) {
  console.log('✅ All required methods are present');
} else {
  console.log('❌ Missing methods:', missingMethods);
}

// Test 3: Verify lectureAgents object
const expectedAgents = ['dr-smith', 'prof-chen', 'dr-wilson', 'prof-taylor'];
const missingAgents = [];

expectedAgents.forEach(agentId => {
  if (!lectureAgents[agentId]) {
    missingAgents.push(agentId);
  }
});

if (missingAgents.length === 0) {
  console.log('✅ All expected agents are defined');
} else {
  console.log('❌ Missing agents:', missingAgents);
}

// Test 4: Verify getContext method works
testAgent.setContext('Test context');
const context = testAgent.getContext();
console.log('✓ Context test - Set: "Test context", Got:', context);

console.log('🎯 Test completed. Check console for any errors.');
