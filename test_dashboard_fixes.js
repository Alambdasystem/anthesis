console.log("🔧 TESTING DASHBOARD FIXES");

// Test 1: Check if week dropdowns are populated
const weekSelect = document.getElementById('weekSelect');
const quizWeek = document.getElementById('quizWeek');

console.log("📋 Week dropdown tests:");
if (weekSelect && weekSelect.options.length > 1) {
    console.log(`✅ weekSelect: ${weekSelect.options.length - 1} weeks loaded`);
} else {
    console.log("❌ weekSelect: Not populated or missing");
}

if (quizWeek && quizWeek.options.length > 1) {
    console.log(`✅ quizWeek: ${quizWeek.options.length - 1} weeks loaded`);
} else {
    console.log("❌ quizWeek: Not populated or missing");
}

// Test 2: Check if lectureAgents is defined
console.log("📋 Lecture agents test:");
if (typeof lectureAgents !== 'undefined') {
    console.log(`✅ lectureAgents: ${Object.keys(lectureAgents).length} agents defined`);
    
    // Test agent methods
    const testAgent = lectureAgents['dr-smith'];
    if (testAgent) {
        console.log("📋 Agent methods test:");
        
        try {
            const stats = testAgent.getUsageStats();
            console.log(`✅ getUsageStats: ${stats.totalLectures} lectures, ${stats.avgRating} rating`);
        } catch(e) {
            console.log(`❌ getUsageStats: ${e.message}`);
        }
        
        try {
            const status = testAgent.getStatus();
            console.log(`✅ getStatus: ${status}`);
        } catch(e) {
            console.log(`❌ getStatus: ${e.message}`);
        }
        
        try {
            const name = testAgent.getName();
            console.log(`✅ getName: ${name}`);
        } catch(e) {
            console.log(`❌ getName: ${e.message}`);
        }
    }
} else {
    console.log("❌ lectureAgents: Not defined");
}

// Test 3: Check if currentAgent is defined
console.log("📋 Current agent test:");
if (typeof currentAgent !== 'undefined') {
    console.log(`✅ currentAgent: ${currentAgent.name}`);
} else {
    console.log("❌ currentAgent: Not defined");
}

console.log("🎯 Run this script in browser console to check fixes!");
