from flask import Blueprint

# Create the agents blueprint
agents_bp = Blueprint('agents', __name__, url_prefix='/api/agents')

# Import routes after blueprint creation to avoid circular imports
from .routes import *