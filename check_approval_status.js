console.log("🔍 APPROVAL STATUS CHECK");
console.log("=" * 30);

// Check if agreement was already accepted
const agreementAccepted = localStorage.getItem('agreementAccepted');
const agreementDate = localStorage.getItem('agreementDate');

if (agreementAccepted === 'true') {
    console.log("✅ Agreement: APPROVED");
    console.log(`📅 Date: ${agreementDate}`);
} else {
    console.log("❌ Agreement: NOT APPROVED");
    console.log("💡 To approve: Click '🎉 Accept the Challenge' button");
}

// Check enrollment status
const enrollmentStatus = localStorage.getItem('enrollmentStatus');
console.log(`📋 Enrollment Status: ${enrollmentStatus || 'Not Set'}`);

// Show how to approve
console.log("\n🎯 HOW TO APPROVE:");
console.log("1. Click '🎉 Accept the Challenge' button");
console.log("2. Check the agreement checkbox");
console.log("3. Click 'Accept Agreement' button");
console.log("4. Enjoy the confetti! 🎉");
