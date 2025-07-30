"""
Anthesis Backend - Simplified Implementation
Run this with: python anthesis_backend.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sqlite3
import datetime
import os
from typing import Dict, List, Optional

app = Flask(__name__)
CORS(app)

# Database initialization
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('anthesis.db')
    cursor = conn.cursor()
    
    # Company profile table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS company_profile (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            tagline TEXT,
            founded_year TEXT,
            headquarters TEXT,
            employees TEXT,
            annual_revenue TEXT,
            certifications TEXT,
            industries_served TEXT,
            core_capabilities TEXT,
            key_technologies TEXT,
            team_leads TEXT,
            recent_projects TEXT,
            differentiators TEXT,
            awards_recognition TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Templates table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            template_type TEXT,
            content TEXT NOT NULL,
            variables TEXT,
            style_template TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # RFPs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rfps (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            client_name TEXT,
            description TEXT,
            requirements TEXT,
            value REAL,
            deadline TEXT,
            status TEXT DEFAULT 'active',
            response_content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # SOWs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sows (
            id INTEGER PRIMARY KEY,
            rfp_id INTEGER,
            title TEXT NOT NULL,
            content TEXT,
            status TEXT DEFAULT 'draft',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (rfp_id) REFERENCES rfps (id)
        )
    ''')
    
    # Contacts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            company TEXT,
            phone TEXT,
            title TEXT,
            notes TEXT,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    
    # Insert default company profile if none exists
    insert_default_data()

def insert_default_data():
    """Insert default company profile and templates"""
    conn = sqlite3.connect('anthesis.db')
    cursor = conn.cursor()
    
    # Check if company profile exists
    cursor.execute('SELECT COUNT(*) FROM company_profile')
    if cursor.fetchone()[0] == 0:
        default_company = {
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
                }
            ],
            'differentiators': [
                'Proven track record with 200+ successful projects',
                'Industry-leading security practices and compliance expertise',
                'Agile development methodology with continuous client collaboration'
            ],
            'awards_recognition': [
                'AWS Partner of the Year 2023',
                'Microsoft Solution Partner Excellence Award'
            ]
        }
        
        cursor.execute('''
            INSERT INTO company_profile 
            (name, tagline, founded_year, headquarters, employees, annual_revenue,
             certifications, industries_served, core_capabilities, key_technologies,
             team_leads, recent_projects, differentiators, awards_recognition)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            default_company['name'],
            default_company['tagline'],
            default_company['founded_year'],
            default_company['headquarters'],
            default_company['employees'],
            default_company['annual_revenue'],
            json.dumps(default_company['certifications']),
            json.dumps(default_company['industries_served']),
            json.dumps(default_company['core_capabilities']),
            json.dumps(default_company['key_technologies']),
            json.dumps(default_company['team_leads']),
            json.dumps(default_company['recent_projects']),
            json.dumps(default_company['differentiators']),
            json.dumps(default_company['awards_recognition'])
        ))
        
        print("Default company profile created")
    
    conn.commit()
    conn.close()

# API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.datetime.now().isoformat(),
        'service': 'Anthesis Backend'
    })

@app.route('/api/company/profile', methods=['GET'])
def get_company_profile():
    """Get company profile data"""
    conn = sqlite3.connect('anthesis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT name, tagline, founded_year, headquarters, employees, annual_revenue,
               certifications, industries_served, core_capabilities, key_technologies,
               team_leads, recent_projects, differentiators, awards_recognition
        FROM company_profile 
        ORDER BY id DESC 
        LIMIT 1
    ''')
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        profile = {
            'name': row[0],
            'tagline': row[1],
            'founded_year': row[2],
            'headquarters': row[3],
            'employees': row[4],
            'annual_revenue': row[5],
            'certifications': json.loads(row[6] or '[]'),
            'industries_served': json.loads(row[7] or '[]'),
            'core_capabilities': json.loads(row[8] or '[]'),
            'key_technologies': json.loads(row[9] or '{}'),
            'team_leads': json.loads(row[10] or '[]'),
            'recent_projects': json.loads(row[11] or '[]'),
            'differentiators': json.loads(row[12] or '[]'),
            'awards_recognition': json.loads(row[13] or '[]')
        }
        return jsonify({'success': True, 'data': profile})
    else:
        return jsonify({'success': False, 'error': 'Profile not found'}), 404

@app.route('/api/components/<component_id>/generate', methods=['POST'])
def generate_component(component_id):
    """Generate component content with company data"""
    try:
        # Get company profile
        conn = sqlite3.connect('anthesis.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, tagline, founded_year, headquarters, employees, annual_revenue,
                   certifications, core_capabilities, differentiators, awards_recognition
            FROM company_profile 
            ORDER BY id DESC 
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return jsonify({'success': False, 'error': 'Company profile not found'}), 404
        
        company_data = {
            'name': row[0],
            'tagline': row[1],
            'founded_year': row[2],
            'headquarters': row[3],
            'employees': row[4],
            'annual_revenue': row[5],
            'certifications': json.loads(row[6] or '[]'),
            'core_capabilities': json.loads(row[7] or '[]'),
            'differentiators': json.loads(row[8] or '[]'),
            'awards_recognition': json.loads(row[9] or '[]')
        }
        
        # Get RFP context from request
        rfp_context = request.json.get('rfp_context', {}) if request.json else {}
        
        # Generate component based on ID
        content = generate_component_content(component_id, company_data, rfp_context)
        
        return jsonify({'success': True, 'data': content})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def generate_component_content(component_id: str, company_data: dict, rfp_context: dict) -> dict:
    """Generate specific component content"""
    
    # Component templates
    templates = {
        'executive_summary': {
            'name': 'Executive Summary',
            'template': f'''
                <div class="component-section executive-summary">
                  <h2 class="section-header">Executive Summary</h2>
                  <div class="company-intro">
                    <h3>{company_data['name']}</h3>
                    <p class="tagline">{company_data['tagline']}</p>
                    <div class="company-stats">
                      <span class="stat">Founded: {company_data['founded_year']}</span>
                      <span class="stat">Employees: {company_data['employees']}</span>
                      <span class="stat">Revenue: {company_data['annual_revenue']}</span>
                    </div>
                  </div>
                  <div class="value-proposition">
                    <p>We understand your need for {rfp_context.get('objective', 'innovative technology solutions')} and are uniquely positioned to deliver exceptional results through our proven expertise in {', '.join(company_data['core_capabilities'][:3])}.</p>
                    <p>Our approach combines {rfp_context.get('methodology', 'Agile development practices')} with deep industry knowledge, ensuring maximum value and successful project delivery.</p>
                  </div>
                  <div class="key-highlights">
                    <h4>Why Choose {company_data['name']}</h4>
                    <ul class="highlight-list">
                      {''.join([f'<li>{diff}</li>' for diff in company_data['differentiators']])}
                    </ul>
                  </div>
                </div>
            ''',
            'style': '''
                .executive-summary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; }
                .executive-summary .section-header { color: white; font-size: 2.2em; margin-bottom: 20px; }
                .company-intro h3 { font-size: 1.8em; margin-bottom: 5px; }
                .tagline { font-style: italic; opacity: 0.9; margin-bottom: 15px; }
                .company-stats { display: flex; gap: 20px; margin-bottom: 20px; }
                .stat { background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 20px; font-size: 0.9em; }
                .value-proposition p { margin-bottom: 15px; line-height: 1.6; }
                .key-highlights h4 { margin-bottom: 15px; font-size: 1.3em; }
                .highlight-list { list-style: none; padding: 0; }
                .highlight-list li { padding: 8px 0; border-left: 3px solid white; padding-left: 15px; margin-bottom: 10px; }
            '''
        },
        'technical_capabilities': {
            'name': 'Technical Capabilities',
            'template': f'''
                <div class="component-section technical-capabilities">
                  <h2 class="section-header">Technical Capabilities</h2>
                  <div class="capabilities-grid">
                    <div class="core-capabilities">
                      <h3>Core Capabilities</h3>
                      <div class="capability-cards">
                        {''.join([f'<div class="capability-card"><h4>{cap}</h4></div>' for cap in company_data['core_capabilities']])}
                      </div>
                    </div>
                    <div class="certifications">
                      <h3>Certifications & Partnerships</h3>
                      <div class="cert-grid">
                        {''.join([f'<span class="cert-badge">{cert}</span>' for cert in company_data['certifications']])}
                      </div>
                    </div>
                  </div>
                </div>
            ''',
            'style': '''
                .technical-capabilities { padding: 30px; background: white; border: 1px solid #e0e0e0; border-radius: 12px; margin-bottom: 30px; }
                .capabilities-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 20px 0; }
                .capability-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
                .capability-card { background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
                .cert-grid { display: flex; flex-wrap: wrap; gap: 10px; }
                .cert-badge { background: #e3f2fd; color: #1976d2; padding: 8px 15px; border-radius: 20px; font-size: 0.9em; border: 1px solid #bbdefb; }
            '''
        }
    }
    
    if component_id in templates:
        return templates[component_id]
    else:
        return {
            'name': 'Component Not Found',
            'template': '<div class="error">Component template not found</div>',
            'style': '.error { color: red; padding: 20px; }'
        }

@app.route('/api/rfps', methods=['GET'])
def get_rfps():
    """Get all RFPs"""
    conn = sqlite3.connect('anthesis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, title, client_name, description, value, deadline, status, created_at
        FROM rfps 
        ORDER BY created_at DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    rfps = []
    for row in rows:
        rfps.append({
            'id': row[0],
            'title': row[1],
            'client_name': row[2],
            'description': row[3],
            'value': row[4],
            'deadline': row[5],
            'status': row[6],
            'created_at': row[7]
        })
    
    return jsonify({'success': True, 'data': rfps})

@app.route('/api/rfps', methods=['POST'])
def create_rfp():
    """Create new RFP"""
    data = request.json
    
    conn = sqlite3.connect('anthesis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO rfps (title, client_name, description, value, deadline)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        data.get('title'),
        data.get('client_name'),
        data.get('description'),
        data.get('value'),
        data.get('deadline')
    ))
    
    rfp_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True, 
        'data': {
            'id': rfp_id,
            'title': data.get('title'),
            'client_name': data.get('client_name'),
            'message': 'RFP created successfully'
        }
    })

@app.route('/api/sow/generate', methods=['POST'])
def generate_sow():
    """Generate SOW from RFP"""
    data = request.json
    rfp_id = data.get('rfp_id')
    
    if not rfp_id:
        return jsonify({'success': False, 'error': 'RFP ID required'}), 400
    
    # Get RFP data
    conn = sqlite3.connect('anthesis.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT title, client_name, description, value FROM rfps WHERE id = ?', (rfp_id,))
    rfp_row = cursor.fetchone()
    
    if not rfp_row:
        conn.close()
        return jsonify({'success': False, 'error': 'RFP not found'}), 404
    
    # Generate SOW content
    sow_content = f"""
# Statement of Work

**Project**: {rfp_row[0]}
**Client**: {rfp_row[1]}
**Contractor**: Anthesis Technologies
**Date**: {datetime.datetime.now().strftime('%Y-%m-%d')}

## 1. PROJECT SCOPE

### 1.1 Objectives
{rfp_row[2] or 'Project objectives to be defined based on client requirements.'}

### 1.2 Project Value
Total Contract Value: ${rfp_row[3]:,.2f if rfp_row[3] else 0}

## 2. DELIVERABLES
- Comprehensive project deliverables as outlined in the RFP
- Regular progress reports and milestone updates
- Final project documentation and handover

## 3. TIMELINE
Project timeline will be established based on detailed requirements analysis.

## 4. TERMS AND CONDITIONS
Standard terms and conditions apply as per Anthesis Technologies service agreement.
"""
    
    # Save SOW
    cursor.execute('''
        INSERT INTO sows (rfp_id, title, content)
        VALUES (?, ?, ?)
    ''', (rfp_id, f"SOW - {rfp_row[0]}", sow_content))
    
    sow_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True, 
        'data': {
            'id': sow_id,
            'title': f"SOW - {rfp_row[0]}",
            'content': sow_content,
            'rfp_id': rfp_id
        }
    })

@app.route('/api/contacts', methods=['GET'])
def get_contacts():
    """Get all contacts"""
    conn = sqlite3.connect('anthesis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, name, email, company, phone, title, notes, tags, created_at
        FROM contacts 
        ORDER BY name
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    contacts = []
    for row in rows:
        contacts.append({
            'id': row[0],
            'name': row[1],
            'email': row[2],
            'company': row[3],
            'phone': row[4],
            'title': row[5],
            'notes': row[6],
            'tags': json.loads(row[7] or '[]'),
            'created_at': row[8]
        })
    
    return jsonify({'success': True, 'data': contacts})

@app.route('/api/contacts', methods=['POST'])
def create_contact():
    """Create new contact"""
    data = request.json
    
    conn = sqlite3.connect('anthesis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO contacts (name, email, company, phone, title, notes, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('name'),
        data.get('email'),
        data.get('company'),
        data.get('phone'),
        data.get('title'),
        data.get('notes'),
        json.dumps(data.get('tags', []))
    ))
    
    contact_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True, 
        'data': {
            'id': contact_id,
            'name': data.get('name'),
            'email': data.get('email'),
            'company': data.get('company')
        }
    })

if __name__ == '__main__':
    # Initialize database
    init_db()
    print("Database initialized")
    print("Starting Anthesis Backend on http://localhost:5000")
    print("Health check: http://localhost:5000/api/health")
    
    # Run the app
    app.run(debug=True, port=5000, host='0.0.0.0')
