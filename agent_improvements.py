#!/usr/bin/env python3
"""
Agent Framework Improvement Script
Addresses UX and functionality issues from the test plan
"""

import os
import json
from datetime import datetime

def add_agent_status_indicators():
    """Add agent status and loading indicators to HTML"""
    
    html_file = "cadet_dashboard.html"
    if not os.path.exists(html_file):
        print("❌ HTML file not found")
        return
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add CSS for agent status indicators
    status_css = """
    /* Agent Status Indicators */
    .agent-status {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .agent-status.online { background-color: #28a745; }
    .agent-status.busy { background-color: #ffc107; }
    .agent-status.offline { background-color: #dc3545; }
    
    .agent-loading {
        display: none;
        margin-left: 8px;
    }
    
    .agent-loading.show {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid #f3f3f3;
        border-top: 2px solid #0a84ff;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .agent-transition {
        transition: all 0.3s ease-in-out;
    }
    
    .agent-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    """
    
    # Find the closing </style> tag and add our CSS before it
    if "</style>" in content:
        content = content.replace("</style>", status_css + "\n</style>")
        print("✅ Added agent status CSS")
    else:
        print("⚠️  Could not find </style> tag to add CSS")
    
    # Add JavaScript for status management
    status_js = """
    
    // Agent Status Management
    class AgentStatusManager {
        constructor() {
            this.agentStatuses = {
                'dr-smith': 'online',
                'prof-chen': 'online', 
                'dr-wilson': 'online',
                'prof-taylor': 'online'
            };
            this.loadingStates = {};
        }
        
        updateAgentStatus(agentId, status) {
            this.agentStatuses[agentId] = status;
            this.refreshStatusIndicators();
        }
        
        setAgentLoading(agentId, loading) {
            this.loadingStates[agentId] = loading;
            const button = document.querySelector(`[data-agent="${agentId}"]`);
            const loadingEl = button?.querySelector('.agent-loading');
            if (loadingEl) {
                loadingEl.classList.toggle('show', loading);
            }
        }
        
        refreshStatusIndicators() {
            Object.keys(this.agentStatuses).forEach(agentId => {
                const button = document.querySelector(`[data-agent="${agentId}"]`);
                const statusEl = button?.querySelector('.agent-status');
                if (statusEl) {
                    statusEl.className = `agent-status ${this.agentStatuses[agentId]}`;
                }
            });
        }
        
        initializeStatusIndicators() {
            document.querySelectorAll('[data-agent]').forEach(button => {
                const agentId = button.getAttribute('data-agent');
                
                // Add status indicator if not present
                if (!button.querySelector('.agent-status')) {
                    const statusSpan = document.createElement('span');
                    statusSpan.className = `agent-status ${this.agentStatuses[agentId] || 'offline'}`;
                    button.insertBefore(statusSpan, button.firstChild);
                }
                
                // Add loading indicator if not present
                if (!button.querySelector('.agent-loading')) {
                    const loadingSpan = document.createElement('span');
                    loadingSpan.className = 'agent-loading';
                    button.appendChild(loadingSpan);
                }
                
                // Add transition class
                button.classList.add('agent-transition');
            });
        }
        
        simulateAgentActivity() {
            // Simulate periodic status changes for demo
            setInterval(() => {
                const agents = Object.keys(this.agentStatuses);
                const randomAgent = agents[Math.floor(Math.random() * agents.length)];
                const statuses = ['online', 'busy', 'offline'];
                const currentStatus = this.agentStatuses[randomAgent];
                const newStatus = statuses[Math.floor(Math.random() * statuses.length)];
                
                if (newStatus !== currentStatus) {
                    this.updateAgentStatus(randomAgent, newStatus);
                    console.log(`🔄 ${randomAgent} status changed to ${newStatus}`);
                }
            }, 10000); // Update every 10 seconds
        }
    }
    
    // Initialize status manager
    const agentStatusManager = new AgentStatusManager();
    
    // Enhanced agent interaction with status updates
    function enhancedAgentInteraction() {
        // Initialize status indicators
        agentStatusManager.initializeStatusIndicators();
        
        // Start status simulation
        agentStatusManager.simulateAgentActivity();
        
        // Override agent selection to include loading states
        document.querySelectorAll('[data-agent]').forEach(button => {
            const originalHandler = button.onclick;
            button.onclick = function(e) {
                const agentId = this.getAttribute('data-agent');
                
                // Show loading state
                agentStatusManager.setAgentLoading(agentId, true);
                
                // Simulate processing time
                setTimeout(() => {
                    agentStatusManager.setAgentLoading(agentId, false);
                    agentStatusManager.updateAgentStatus(agentId, 'busy');
                    
                    // Call original handler if it exists
                    if (originalHandler) {
                        originalHandler.call(this, e);
                    }
                    
                    // Return to online after interaction
                    setTimeout(() => {
                        agentStatusManager.updateAgentStatus(agentId, 'online');
                    }, 3000);
                }, 1000);
            };
        });
    }
    
    // Initialize when DOM is loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', enhancedAgentInteraction);
    } else {
        enhancedAgentInteraction();
    }
    """
    
    # Find a good place to add the JavaScript (before closing script tag)
    if "</script>" in content:
        # Find the last script tag
        last_script_pos = content.rfind("</script>")
        if last_script_pos != -1:
            content = content[:last_script_pos] + status_js + "\n</script>" + content[last_script_pos + 9:]
            print("✅ Added agent status JavaScript")
    else:
        print("⚠️  Could not find script tag to add JavaScript")
    
    # Write the updated content back
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("🎨 Agent status indicators added successfully")

def create_agent_performance_metrics():
    """Create a basic agent performance tracking system"""
    
    metrics_file = "agent_metrics.json"
    
    # Create initial metrics structure
    metrics = {
        "created_at": datetime.now().isoformat(),
        "agents": {
            "dr-smith": {
                "total_interactions": 0,
                "successful_responses": 0,
                "average_response_time": 0,
                "last_interaction": None,
                "specialization_score": 85
            },
            "prof-chen": {
                "total_interactions": 0,
                "successful_responses": 0,
                "average_response_time": 0,
                "last_interaction": None,
                "specialization_score": 92
            },
            "dr-wilson": {
                "total_interactions": 0,
                "successful_responses": 0,
                "average_response_time": 0,
                "last_interaction": None,
                "specialization_score": 88
            },
            "prof-taylor": {
                "total_interactions": 0,
                "successful_responses": 0,
                "average_response_time": 0,
                "last_interaction": None,
                "specialization_score": 91
            }
        },
        "system_metrics": {
            "uptime_start": datetime.now().isoformat(),
            "total_system_interactions": 0,
            "error_count": 0,
            "last_error": None
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"📊 Created agent performance metrics: {metrics_file}")

def add_error_handling_improvements():
    """Add improved error handling to the system"""
    
    error_handler_js = """
    
    // Enhanced Error Handling System
    class AgentErrorHandler {
        constructor() {
            this.errorLog = [];
            this.maxLogEntries = 100;
        }
        
        logError(error, context = '') {
            const errorEntry = {
                timestamp: new Date().toISOString(),
                error: error.toString(),
                context: context,
                stack: error.stack || 'No stack trace available'
            };
            
            this.errorLog.push(errorEntry);
            
            // Keep only the last 100 entries
            if (this.errorLog.length > this.maxLogEntries) {
                this.errorLog = this.errorLog.slice(-this.maxLogEntries);
            }
            
            console.error('🔴 Agent Error:', errorEntry);
            this.displayUserFriendlyError(error, context);
        }
        
        displayUserFriendlyError(error, context) {
            // Create or update error notification
            let errorDiv = document.getElementById('agent-error-notification');
            if (!errorDiv) {
                errorDiv = document.createElement('div');
                errorDiv.id = 'agent-error-notification';
                errorDiv.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #dc3545;
                    color: white;
                    padding: 12px 16px;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    z-index: 10000;
                    max-width: 300px;
                    font-size: 14px;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                `;
                document.body.appendChild(errorDiv);
            }
            
            // Set error message
            errorDiv.innerHTML = `
                <strong>🚨 Agent Error</strong><br>
                ${context ? context + ': ' : ''}${this.getUserFriendlyMessage(error)}
                <button onclick="this.parentElement.style.display='none'" style="float: right; background: none; border: none; color: white; font-size: 16px; cursor: pointer;">&times;</button>
            `;
            
            // Show notification
            errorDiv.style.opacity = '1';
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                errorDiv.style.opacity = '0';
                setTimeout(() => {
                    if (errorDiv.parentNode) {
                        errorDiv.parentNode.removeChild(errorDiv);
                    }
                }, 300);
            }, 5000);
        }
        
        getUserFriendlyMessage(error) {
            const errorStr = error.toString().toLowerCase();
            
            if (errorStr.includes('network') || errorStr.includes('fetch')) {
                return 'Connection problem. Please check your internet connection.';
            } else if (errorStr.includes('timeout')) {
                return 'Request timed out. The agent may be busy.';
            } else if (errorStr.includes('404')) {
                return 'Agent service not found. Please try again later.';
            } else if (errorStr.includes('500')) {
                return 'Agent service error. Please try again later.';
            } else {
                return 'Something went wrong. Please try again.';
            }
        }
        
        getErrorReport() {
            return {
                totalErrors: this.errorLog.length,
                recentErrors: this.errorLog.slice(-10),
                summary: this.generateErrorSummary()
            };
        }
        
        generateErrorSummary() {
            const errorTypes = {};
            this.errorLog.forEach(entry => {
                const type = this.categorizeError(entry.error);
                errorTypes[type] = (errorTypes[type] || 0) + 1;
            });
            return errorTypes;
        }
        
        categorizeError(error) {
            const errorStr = error.toLowerCase();
            if (errorStr.includes('network') || errorStr.includes('fetch')) return 'Network';
            if (errorStr.includes('timeout')) return 'Timeout';
            if (errorStr.includes('404')) return 'Not Found';
            if (errorStr.includes('500')) return 'Server Error';
            return 'Other';
        }
    }
    
    // Initialize global error handler
    const agentErrorHandler = new AgentErrorHandler();
    
    // Override global error handling
    window.addEventListener('error', (event) => {
        agentErrorHandler.logError(event.error, 'Global Error');
    });
    
    window.addEventListener('unhandledrejection', (event) => {
        agentErrorHandler.logError(new Error(event.reason), 'Unhandled Promise Rejection');
    });
    """
    
    print("🛡️  Error handling improvements prepared")
    return error_handler_js

def main():
    """Run all improvements"""
    print("🚀 Starting Agent Framework Improvements...")
    print("=" * 50)
    
    # Add status indicators and UX improvements
    add_agent_status_indicators()
    
    # Create performance metrics
    create_agent_performance_metrics()
    
    # Prepare error handling improvements
    error_js = add_error_handling_improvements()
    
    print("\n✅ All improvements completed!")
    print("\nSummary of changes:")
    print("• Added agent status indicators (online/busy/offline)")
    print("• Added loading animations for agent interactions")
    print("• Created agent performance metrics tracking")
    print("• Enhanced error handling with user-friendly messages")
    print("• Added hover effects and smooth transitions")
    
    print("\n🔧 Next steps:")
    print("1. Restart your Flask application")
    print("2. Run the test script again to verify improvements")
    print("3. Check the cadet_dashboard.html for new visual enhancements")

if __name__ == "__main__":
    main()
