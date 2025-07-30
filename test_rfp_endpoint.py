#!/usr/bin/env python3
"""Test script to verify RFP response endpoints work correctly."""

import jwt
import requests
import json
from datetime import datetime, timedelta

# Same secret key as in app.py
SECRET_KEY = 'your_secret_key_here_change_this'

def create_test_token():
    """Create a valid JWT token for testing."""
    payload = {
        'username': 'test_user',
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def test_rfp_endpoint():
    """Test the RFP response endpoint."""
    # Create a valid token
    token = create_test_token()
    print(f"Created test token: {token[:50]}...")
    
    # Test the view endpoint
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    print("\n1. Testing RFP response view endpoint...")
    try:
        response = requests.get('http://localhost:5000/api/rfp-response/2', headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n2. Testing RFP response download endpoint...")
    try:
        response = requests.get('http://localhost:5000/api/rfp-response/2/download?format=txt', headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        if response.status_code == 200:
            print("Download successful!")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_rfp_endpoint()
