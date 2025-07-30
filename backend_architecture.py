"""
Anthesis Backend Architecture Plan
This file outlines what should be moved from frontend to backend for better architecture
"""

from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import json
import datetime
from typing import Dict, List, Optional
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///anthesis.db'
app.config['SECRET_KEY'] = 'your-secret-key-here'
db = SQLAlchemy(app)
CORS(app)

# ============================================================================
# DATABASE MODELS (Move all data storage to backend)
# ============================================================================

class Company(db.Model):
    """Company profile data - currently hardcoded in frontend"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    tagline = db.Column(db.String(500))
    founded_year = db.Column(db.String(4))
    headquarters = db.Column(db.String(200))
    employees = db.Column(db.String(50))
    annual_revenue = db.Column(db.String(50))
    certifications = db.Column(db.Text)  # JSON string
    industries_served = db.Column(db.Text)  # JSON string
    core_capabilities = db.Column(db.Text)  # JSON string
    key_technologies = db.Column(db.Text)  # JSON string
    team_leads = db.Column(db.Text)  # JSON string
    recent_projects = db.Column(db.Text)  # JSON string
    differentiators = db.Column(db.Text)  # JSON string
    awards_recognition = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Template(db.Model):
    """All template data - currently stored in frontend memory"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(100), nullable=False)  # rfp, sow, email, component
    template_type = db.Column(db.String(100))  # executive_summary, methodology, etc.
    content = db.Column(db.Text, nullable=False)
    variables = db.Column(db.Text)  # JSON string of required variables
    style_template = db.Column(db.Text)  # CSS/styling information
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class RFP(db.Model):
    """RFP data - currently in frontend AppState"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    client_name = db.Column(db.String(200))
    description = db.Column(db.Text)
    requirements = db.Column(db.Text)  # JSON string
    value = db.Column(db.Numeric(15, 2))
    deadline = db.Column(db.DateTime)
    status = db.Column(db.String(50), default='active')
    response_content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class SOW(db.Model):
    """Statement of Work - linked to RFPs"""
    id = db.Column(db.Integer, primary_key=True)
    rfp_id = db.Column(db.Integer, db.ForeignKey('rfp.id'))
    title = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text)
    status = db.Column(db.String(50), default='draft')
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    rfp = db.relationship('RFP', backref=db.backref('sows', lazy=True))

class Contact(db.Model):
    """Contact management - currently in frontend"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200))
    company = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    title = db.Column(db.String(200))
    notes = db.Column(db.Text)
    tags = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# ============================================================================
# BUSINESS LOGIC SERVICES (Move complex logic from frontend)
# ============================================================================

class TemplateService:
    """Handle all template operations - currently in TemplateManager class"""
    
    @staticmethod
    def get_company_profile() -> Dict:
        """Get company profile data from database"""
        company = Company.query.first()
        if not company:
            return TemplateService.get_default_company_profile()
        
        return {
            'name': company.name,
            'tagline': company.tagline,
            'founded_year': company.founded_year,
            'headquarters': company.headquarters,
            'employees': company.employees,
            'annual_revenue': company.annual_revenue,
            'certifications': json.loads(company.certifications or '[]'),
            'industries_served': json.loads(company.industries_served or '[]'),
            'core_capabilities': json.loads(company.core_capabilities or '[]'),
            'key_technologies': json.loads(company.key_technologies or '{}'),
            'team_leads': json.loads(company.team_leads or '[]'),
            'recent_projects': json.loads(company.recent_projects or '[]'),
            'differentiators': json.loads(company.differentiators or '[]'),
            'awards_recognition': json.loads(company.awards_recognition or '[]')
        }
    
    @staticmethod
    def get_default_company_profile() -> Dict:
        """Default company profile - same as frontend but in backend"""
        return {
            'name': 'Anthesis Technologies',
            'tagline': 'Innovative Solutions for Digital Transformation',
            'founded_year': '2018',
            'headquarters': 'Seattle, WA',
            'employees': '150+',
            'annual_revenue': '$25M+',
            'certifications': ['ISO 9001', 'SOC 2 Type II', 'AWS Partner', 'Microsoft Gold Partner'],
            'industries_served': ['Healthcare', 'Financial Services', 'Government', 'Education', 'Manufacturing'],
            'core_capabilities': [
                'Cloud Migration & Architecture',
                'Custom Software Development',
                'AI/ML Implementation',
                'Cybersecurity Solutions',
                'DevOps & Infrastructure',
                'Data Analytics & BI',
                'Mobile App Development',
                'UI/UX Design'
            ],
            'key_technologies': {
                'Cloud Platforms': ['AWS', 'Azure', 'Google Cloud'],
                'Programming Languages': ['Python', 'JavaScript', 'Java', 'C#', '.NET'],
                'Frameworks': ['React', 'Angular', 'Node.js', 'Django', 'Spring Boot'],
                'Databases': ['PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch'],
                'DevOps': ['Docker', 'Kubernetes', 'Jenkins', 'Terraform', 'Ansible']
            },
            'team_leads': [
                {
                    'name': 'Sarah Chen',
                    'title': 'Chief Technology Officer',
                    'experience': '15+ years',
                    'specialties': ['Cloud Architecture', 'AI/ML', 'Team Leadership'],
                    'certifications': ['AWS Solutions Architect', 'PMP', 'Certified Scrum Master']
                },
                {
                    'name': 'Michael Rodriguez',
                    'title': 'Lead Software Architect',
                    'experience': '12+ years',
                    'specialties': ['Microservices', 'Security', 'Performance Optimization'],
                    'certifications': ['Azure Solutions Architect', 'CISSP', 'Oracle Certified Professional']
                },
                {
                    'name': 'Emily Johnson',
                    'title': 'Project Manager',
                    'experience': '10+ years',
                    'specialties': ['Agile Methodologies', 'Stakeholder Management', 'Risk Assessment'],
                    'certifications': ['PMP', 'CSM', 'PRINCE2']
                }
            ],
            'recent_projects': [
                {
                    'client': 'Regional Healthcare System',
                    'project': 'Cloud Migration & EHR Integration',
                    'value': '$2.5M',
                    'duration': '18 months',
                    'technologies': ['AWS', 'HL7 FHIR', 'React', 'Node.js'],
                    'outcome': '40% reduction in infrastructure costs, 99.9% uptime'
                },
                {
                    'client': 'Financial Services Company',
                    'project': 'AI-Powered Fraud Detection System',
                    'value': '$1.8M',
                    'duration': '12 months',
                    'technologies': ['Python', 'TensorFlow', 'Kubernetes', 'PostgreSQL'],
                    'outcome': '85% improvement in fraud detection accuracy'
                },
                {
                    'client': 'Government Agency',
                    'project': 'Secure Document Management Platform',
                    'value': '$3.2M',
                    'duration': '24 months',
                    'technologies': ['Azure', 'C#', '.NET Core', 'SQL Server'],
                    'outcome': 'FedRAMP compliance achieved, 60% faster document processing'
                }
            ],
            'differentiators': [
                'Proven track record with 200+ successful projects',
                'Industry-leading security practices and compliance expertise',
                'Agile development methodology with continuous client collaboration',
                'Dedicated post-deployment support and maintenance',
                '24/7 monitoring and incident response capabilities'
            ],
            'awards_recognition': [
                'AWS Partner of the Year 2023',
                'Microsoft Solution Partner Excellence Award',
                'Inc. 5000 Fastest Growing Companies (3 consecutive years)',
                'Best Places to Work in Tech - Seattle Business Journal'
            ]
        }
    
    @staticmethod
    def generate_component_content(component_id: str, rfp_context: Dict = None) -> Dict:
        """Generate component content - move complex logic from frontend"""
        template = Template.query.filter_by(
            category='component',
            template_type=component_id,
            is_active=True
        ).first()
        
        if not template:
            return {'error': 'Component template not found'}
        
        company_data = TemplateService.get_company_profile()
        rfp_context = rfp_context or {}
        
        # Build replacement variables
        replacements = {
            # Company basics
            'company_name': company_data['name'],
            'company_tagline': company_data['tagline'],
            'founded_year': company_data['founded_year'],
            'headquarters': company_data['headquarters'],
            'employees': company_data['employees'],
            'annual_revenue': company_data['annual_revenue'],
            
            # Dynamic content
            'differentiators': TemplateService._generate_list_html(company_data['differentiators']),
            'certifications_list': TemplateService._generate_badges_html(company_data['certifications']),
            'core_capabilities_cards': TemplateService._generate_capability_cards(company_data['core_capabilities']),
            
            # RFP context
            'rfp_objective': rfp_context.get('objective', 'innovative technology solutions'),
            'industry': rfp_context.get('industry', 'your industry'),
            
            # Calculations
            'industries_count': str(len(company_data['industries_served'])),
            'core_focus': ' and '.join(company_data['core_capabilities'][:2])
        }
        
        # Process template
        content = template.content
        for key, value in replacements.items():
            content = content.replace(f'{{{key}}}', str(value))
        
        return {
            'content': content,
            'style': template.style_template,
            'name': template.name
        }
    
    @staticmethod
    def _generate_list_html(items: List[str]) -> str:
        return ''.join([f'<li>{item}</li>' for item in items])
    
    @staticmethod
    def _generate_badges_html(items: List[str]) -> str:
        return ''.join([f'<span class="cert-badge">{item}</span>' for item in items])
    
    @staticmethod
    def _generate_capability_cards(capabilities: List[str]) -> str:
        return ''.join([f'<div class="capability-card"><h4>{cap}</h4></div>' for cap in capabilities])

class RFPService:
    """Handle RFP operations"""
    
    @staticmethod
    def create_rfp(data: Dict) -> Dict:
        """Create new RFP"""
        rfp = RFP(
            title=data['title'],
            client_name=data.get('client_name'),
            description=data.get('description'),
            requirements=json.dumps(data.get('requirements', [])),
            value=data.get('value'),
            deadline=datetime.datetime.fromisoformat(data['deadline']) if data.get('deadline') else None
        )
        db.session.add(rfp)
        db.session.commit()
        
        return {
            'id': rfp.id,
            'title': rfp.title,
            'client_name': rfp.client_name,
            'description': rfp.description,
            'value': float(rfp.value) if rfp.value else None,
            'deadline': rfp.deadline.isoformat() if rfp.deadline else None,
            'status': rfp.status,
            'created_at': rfp.created_at.isoformat()
        }
    
    @staticmethod
    def generate_sow_from_rfp(rfp_id: int, sow_data: Dict) -> Dict:
        """Generate SOW from RFP - move complex logic from frontend"""
        rfp = RFP.query.get(rfp_id)
        if not rfp:
            return {'error': 'RFP not found'}
        
        # Get SOW template
        template = Template.query.filter_by(
            category='sow',
            template_type=sow_data.get('template_type', 'standard'),
            is_active=True
        ).first()
        
        if not template:
            return {'error': 'SOW template not found'}
        
        # Process template with RFP data
        variables = {
            'project_title': sow_data.get('title', rfp.title),
            'client_name': sow_data.get('client_name', rfp.client_name),
            'contractor_name': 'Anthesis Technologies',
            'contract_date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'project_objectives': sow_data.get('scope', rfp.description),
            'contract_value': f"${rfp.value:,.2f}" if rfp.value else 'TBD'
        }
        
        content = template.content
        for key, value in variables.items():
            content = content.replace(f'{{{key}}}', str(value))
        
        # Create SOW record
        sow = SOW(
            rfp_id=rfp_id,
            title=f"SOW - {rfp.title}",
            content=content
        )
        db.session.add(sow)
        db.session.commit()
        
        return {
            'id': sow.id,
            'title': sow.title,
            'content': sow.content,
            'rfp_id': sow.rfp_id,
            'created_at': sow.created_at.isoformat()
        }

# ============================================================================
# API ENDPOINTS (Replace frontend API calls)
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.datetime.utcnow().isoformat()})

@app.route('/api/company/profile', methods=['GET'])
def get_company_profile():
    """Get company profile data"""
    profile = TemplateService.get_company_profile()
    return jsonify({'success': True, 'data': profile})

@app.route('/api/company/profile', methods=['PUT'])
def update_company_profile():
    """Update company profile"""
    data = request.json
    
    company = Company.query.first()
    if not company:
        company = Company()
        db.session.add(company)
    
    company.name = data.get('name', company.name)
    company.tagline = data.get('tagline', company.tagline)
    company.certifications = json.dumps(data.get('certifications', []))
    # ... update other fields
    
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Profile updated'})

@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Get all templates by category"""
    category = request.args.get('category')
    query = Template.query.filter_by(is_active=True)
    if category:
        query = query.filter_by(category=category)
    
    templates = query.all()
    result = []
    
    for template in templates:
        result.append({
            'id': template.id,
            'name': template.name,
            'category': template.category,
            'template_type': template.template_type,
            'content': template.content,
            'variables': json.loads(template.variables or '[]'),
            'created_at': template.created_at.isoformat()
        })
    
    return jsonify({'success': True, 'data': result})

@app.route('/api/components/<component_id>/generate', methods=['POST'])
def generate_component(component_id):
    """Generate component content with company data"""
    rfp_context = request.json.get('rfp_context', {})
    result = TemplateService.generate_component_content(component_id, rfp_context)
    
    if 'error' in result:
        return jsonify({'success': False, 'error': result['error']}), 404
    
    return jsonify({'success': True, 'data': result})

@app.route('/api/rfps', methods=['GET'])
def get_rfps():
    """Get all RFPs"""
    rfps = RFP.query.all()
    result = []
    
    for rfp in rfps:
        result.append({
            'id': rfp.id,
            'title': rfp.title,
            'client_name': rfp.client_name,
            'description': rfp.description,
            'value': float(rfp.value) if rfp.value else None,
            'deadline': rfp.deadline.isoformat() if rfp.deadline else None,
            'status': rfp.status,
            'created_at': rfp.created_at.isoformat()
        })
    
    return jsonify({'success': True, 'data': result})

@app.route('/api/rfps', methods=['POST'])
def create_rfp():
    """Create new RFP"""
    data = request.json
    rfp_data = RFPService.create_rfp(data)
    return jsonify({'success': True, 'data': rfp_data})

@app.route('/api/sow/generate', methods=['POST'])
def generate_sow():
    """Generate SOW from RFP"""
    data = request.json
    rfp_id = data.get('rfp_id')
    
    if not rfp_id:
        return jsonify({'success': False, 'error': 'RFP ID required'}), 400
    
    result = RFPService.generate_sow_from_rfp(rfp_id, data)
    
    if 'error' in result:
        return jsonify({'success': False, 'error': result['error']}), 404
    
    return jsonify({'success': True, 'data': result})

@app.route('/api/contacts', methods=['GET'])
def get_contacts():
    """Get all contacts"""
    contacts = Contact.query.all()
    result = []
    
    for contact in contacts:
        result.append({
            'id': contact.id,
            'name': contact.name,
            'email': contact.email,
            'company': contact.company,
            'phone': contact.phone,
            'title': contact.title,
            'notes': contact.notes,
            'tags': json.loads(contact.tags or '[]'),
            'created_at': contact.created_at.isoformat()
        })
    
    return jsonify({'success': True, 'data': result})

@app.route('/api/contacts', methods=['POST'])
def create_contact():
    """Create new contact"""
    data = request.json
    
    contact = Contact(
        name=data['name'],
        email=data.get('email'),
        company=data.get('company'),
        phone=data.get('phone'),
        title=data.get('title'),
        notes=data.get('notes'),
        tags=json.dumps(data.get('tags', []))
    )
    
    db.session.add(contact)
    db.session.commit()
    
    return jsonify({
        'success': True, 
        'data': {
            'id': contact.id,
            'name': contact.name,
            'email': contact.email,
            'company': contact.company
        }
    })

# ============================================================================
# INITIALIZATION
# ============================================================================

def init_database():
    """Initialize database with default data"""
    db.create_all()
    
    # Create default company profile if none exists
    if not Company.query.first():
        default_profile = TemplateService.get_default_company_profile()
        company = Company(
            name=default_profile['name'],
            tagline=default_profile['tagline'],
            founded_year=default_profile['founded_year'],
            headquarters=default_profile['headquarters'],
            employees=default_profile['employees'],
            annual_revenue=default_profile['annual_revenue'],
            certifications=json.dumps(default_profile['certifications']),
            industries_served=json.dumps(default_profile['industries_served']),
            core_capabilities=json.dumps(default_profile['core_capabilities']),
            key_technologies=json.dumps(default_profile['key_technologies']),
            team_leads=json.dumps(default_profile['team_leads']),
            recent_projects=json.dumps(default_profile['recent_projects']),
            differentiators=json.dumps(default_profile['differentiators']),
            awards_recognition=json.dumps(default_profile['awards_recognition'])
        )
        db.session.add(company)
        db.session.commit()
        print("Default company profile created")

if __name__ == '__main__':
    init_database()
    app.run(debug=True, port=5000)

# ============================================================================
# MIGRATION PLAN
# ============================================================================

"""
WHAT SHOULD BE MOVED TO BACKEND:

1. DATA STORAGE & MANAGEMENT:
   ✅ Company profile data (currently hardcoded in frontend)
   ✅ All template storage (RFP, SOW, email, component templates)
   ✅ RFP data and management
   ✅ SOW generation and storage
   ✅ Contact management
   ✅ User authentication and sessions

2. BUSINESS LOGIC:
   ✅ Template processing and variable replacement
   ✅ Component content generation with company data
   ✅ SOW generation from RFP data
   ✅ Document style extraction and processing
   ✅ Complex template operations

3. API OPERATIONS:
   ✅ All CRUD operations for data
   ✅ Template management endpoints
   ✅ Component generation endpoints
   ✅ File upload and processing
   ✅ Document generation

WHAT CAN STAY IN FRONTEND:

1. UI LOGIC:
   - View management and navigation
   - Form handling and validation
   - Modal management
   - Notifications and user feedback

2. SIMPLE OPERATIONS:
   - Display formatting
   - Client-side validation
   - UI state management
   - Basic user interactions

3. PRESENTATION LAYER:
   - CSS styling
   - Component rendering
   - Layout management
   - Responsive design

BENEFITS OF BACKEND MIGRATION:
- Centralized data management
- Better security
- Easier testing and maintenance
- Scalability improvements
- Data persistence
- Multi-user support
- API-first architecture
"""
