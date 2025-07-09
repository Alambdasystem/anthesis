"""
Authentication module for agents system
"""

import jwt
import os
from functools import wraps
from flask import request, jsonify, current_app
from datetime import datetime, timedelta, timezone

def validate_token(token: str) -> bool:
    """Validate JWT token"""
    try:
        # Use a proper secret key from environment or config
        secret_key = current_app.config.get('SECRET_KEY', 'your_secret_here')
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        return True
    except jwt.ExpiredSignatureError:
        return False
    except jwt.InvalidTokenError:
        return False
    except Exception:
        return False

def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'success': False, 'error': 'No token provided'}), 401
        
        if not validate_token(token):
            return jsonify({'success': False, 'error': 'Invalid or expired token'}), 401
        
        return f(*args, **kwargs)
    return decorated

def generate_token(user_data: dict) -> str:
    """Generate JWT token for user"""
    secret_key = current_app.config.get('SECRET_KEY', 'your_secret_here')
    payload = {
        'user_data': user_data,
        'exp': datetime.now(timezone.utc) + timedelta(hours=24),
        'iat': datetime.now(timezone.utc)
    }
    return jwt.encode(payload, secret_key, algorithm='HS256')
