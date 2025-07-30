// Dashboard Count Display Test
console.log('📊 Testing Dashboard Count Display...');

// Test function to manually refresh dashboard counts
function testDashboardCounts() {
  console.log('🔄 Manually refreshing dashboard counts...');
  
  // Get professor lecture history
  const professorLectureHistory = JSON.parse(localStorage.getItem('professorLectureHistory') || '{}');
  let totalLectures = 0;
  
  console.log('📚 Professor Lecture History:', professorLectureHistory);
  
  Object.keys(professorLectureHistory).forEach(professorId => {
    const count = professorLectureHistory[professorId].length;
    totalLectures += count;
    console.log(`  ${professorId}: ${count} lectures`);
  });
  
  console.log(`📈 Total Lectures: ${totalLectures}`);
  
  // Update display elements
  const lecturesEl = document.getElementById('lecturesCountDisplay');
  const modulesEl = document.getElementById('modulesCountDisplay');
  const quizzesEl = document.getElementById('quizzesCountDisplay');
  const completedEl = document.getElementById('completedCountDisplay');
  
  console.log('🎯 Found Elements:', {
    lectures: !!lecturesEl,
    modules: !!modulesEl,
    quizzes: !!quizzesEl,
    completed: !!completedEl
  });
  
  if (lecturesEl) {
    lecturesEl.innerText = totalLectures;
    console.log(`✅ Updated lectures count to: ${totalLectures}`);
  }
  
  if (modulesEl) {
    modulesEl.innerText = 8;
    console.log('✅ Updated modules count to: 8');
  }
  
  if (quizzesEl) {
    quizzesEl.innerText = 5;
    console.log('✅ Updated quizzes count to: 5');
  }
  
  if (completedEl) {
    completedEl.innerText = 13;
    console.log('✅ Updated completed count to: 13');
  }
  
  console.log('🎉 Dashboard counts updated!');
}

// Test after page loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', testDashboardCounts);
} else {
  testDashboardCounts();
}

// Expose function globally for manual testing
window.testDashboardCounts = testDashboardCounts;

console.log('💡 Run testDashboardCounts() to manually refresh counts');
console.log('📊 Dashboard should now show actual lecture counts instead of 0 0 0');
