"""
Startup script for Anthesis AI Agent System
Run this script to set up and start the application
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        try:
            with open('.env.example', 'r') as example:
                content = example.read()
            with open('.env', 'w') as env_file:
                env_file.write(content)
            print("✅ .env file created from template")
            print("⚠️  Please edit .env file with your actual configuration values")
        except FileNotFoundError:
            print("❌ .env.example not found")
    else:
        print("✅ .env file already exists")

def create_directories():
    """Create necessary directories"""
    directories = [
        'email_logs',
        'uploads',
        'static/uploads',
        'data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def test_system():
    """Test the agent system"""
    print("\n🧪 Testing agent system...")
    try:
        from test_agents import test_agent_system
        test_agent_system()
        return True
    except Exception as e:
        print(f"❌ Agent system test failed: {e}")
        return False

def main():
    print("🚀 Anthesis AI Agent System Startup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Create .env file
    create_env_file()
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Test system
    if not test_system():
        print("❌ Setup completed but system test failed")
        print("💡 You may need to configure Ollama or other dependencies")
    
    print("\n✅ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("1. Edit .env file with your configuration")
    print("2. Make sure Ollama is running (for AI agents)")
    print("3. Run: python app.py")
    print("\n📚 Available endpoints:")
    print("- GET  /api/agents/           - List all agents")
    print("- GET  /api/agents/health     - System health check")
    print("- POST /api/agents/initialize - Initialize agents")
    print("- POST /api/agents/workflows/email - Email workflow")

if __name__ == "__main__":
    main()
