# Anthesis CRM System - Technical Specification Sheet

## Project Overview
**System Name:** Anthesis CRM & AI Agent Platform  
**Version:** 2.0  
**Date:** July 22, 2025  
**Type:** Web-based Customer Relationship Management System with AI Integration  
**Technology Stack:** Flask (Python), HTML5, CSS3, JavaScript, jQuery, Bootstrap 4.6.2  

## System Architecture

### Backend Components
- **Framework:** Flask (Python)
- **Authentication:** JWT Token-based authentication
- **Database:** JSON file-based storage (contacts.json, users.json, agent_contacts.json)
- **AI Integration:** Multi-agent system with 34 specialized AI agents
- **API Endpoints:** RESTful API with Bearer token authentication

### Frontend Components
- **Framework:** HTML5 + jQuery 3.6.0 + Bootstrap 4.6.2
- **UI Components:** Responsive modal-based interface
- **File Processing:** PapaParse 5.4.1 for CSV handling
- **Icons:** Font Awesome icons

## Core Features

### 1. Authentication System
- **Registration:** Username, email, password with validation
- **Login:** JWT token-based authentication
- **Session Management:** Automatic token validation and refresh
- **Logout:** Clean session termination with token removal

### 2. Contact Management
- **Contact Storage:** JSON-based contact database
- **Contact Display:** Interactive table with search and filter
- **Contact Selection:** Click-to-select with visual feedback
- **Contact Enrichment:** AI-powered data enhancement

### 3. AI Agent System (34 Agents)
#### Agent Categories:
- **Content Generation Agents:** Email, proposals, presentations, blog posts, social media, whitepapers
- **Document Analysis Agents:** Document summary, key point extraction, sentiment analysis, action items, risk assessment
- **Lecture Agents:** Educational content generation with customizable duration and audience
- **User Agent Assistants:** Research, follow-up planning, meeting preparation, engagement strategy
- **Specialized Agents:** Dr. Smith, Prof. Chen, Dr. Wilson, Prof. Taylor + 26 User Agents

### 4. Contact Action Buttons (9 Actions per Contact)
1. **🔍 Enrich** - Contact data enrichment with LinkedIn, company size, industry data
2. **✉️ Email** - Email composition and delivery system
3. **📋 RFP** - RFP response generation using multi-agent workflow
4. **📞 Call** - Call scheduling with date/time picker and purpose selection
5. **📝 Content** - Content generation for multiple formats (email, proposal, presentation, etc.)
6. **📄 Analysis** - Document analysis with multiple analysis types
7. **🎓 Lecture** - Lecture content generation with audience targeting
8. **👤 Assistant** - User agent assistance for various tasks
9. **🤖 AI Chat** - Interactive AI chat with contact context

## Technical Specifications

### API Endpoints
```
POST /register          - User registration
POST /login            - User authentication
GET  /health           - Server health check
GET  /profile          - User profile information
GET  /contacts         - Contact list retrieval
POST /contacts         - Contact creation
GET  /contacts/{id}/drafts - Draft retrieval
POST /contacts/{id}/drafts - Draft creation
PUT  /contacts/{id}/drafts/{draft_id} - Draft update
DELETE /contacts/{id}/drafts/{draft_id} - Draft deletion
POST /predict          - AI agent prediction
POST /api/agents/workflow - Multi-agent workflow processing
POST /api/agents/workflows/email - Email workflow processing
GET  /email-templates  - Email template management
```

### Authentication Flow
1. User submits credentials via login form
2. Server validates credentials against users.json
3. JWT token generated with 24-hour expiration
4. Token stored in localStorage
5. All API requests include Bearer token in Authorization header
6. Token validated on each request
7. Automatic logout on token expiration

### Multi-Agent Workflow System
- **Coordinator Agent:** Orchestrates workflow processes
- **Document Analysis Agent:** Processes and analyzes documents
- **Content Generation Agent:** Creates tailored content
- **Workflow Pipeline:** coordinator → document-analysis → content-generation
- **Context Passing:** Contact information flows through entire workflow

## Modal System Architecture

### Primary Modals
1. **Agent Modal** (`#agentModal`) - Main AI agent interface with tabbed layout
2. **Email Delivery Modal** (`#emailDeliveryModal`) - Email composition and sending
3. **RFP Response Modal** (`#rfpResponseModal`) - RFP document processing and response generation
4. **Email Draft Modal** (`#emailDraftModal`) - Email draft editing interface

### Dynamic Modals (Created on Demand)
1. **Call Modal** - Call scheduling interface
2. **Content Agent Modal** - Content generation interface
3. **Document Analysis Modal** - Document analysis interface
4. **Lecture Agent Modal** - Lecture content generation
5. **User Agent Modal** - User assistance interface

## Data Models

### Contact Model
```javascript
[
  "Contact Name",        // [0] - String
  "email@domain.com",    // [1] - Email
  "Company Name",        // [2] - String
  "Focus Area",          // [3] - String
  "Why Hot/Reason"       // [4] - String
]
```

### User Model
```json
{
  "username": {
    "email": "user@domain.com",
    "password": "hashed_password"
  }
}
```

### Draft Model
```json
{
  "id": "number",
  "subject": "string",
  "body": "string",
  "type": "email|rfp",
  "workflow_log": "object",
  "created_at": "timestamp"
}
```

## Security Features
- **Password Hashing:** Werkzeug security for password hashing
- **JWT Tokens:** Secure token-based authentication with expiration
- **CORS Configuration:** Cross-origin resource sharing setup
- **Input Validation:** Client and server-side validation
- **XSS Protection:** Content escaping and sanitization

## File Upload Capabilities
- **Supported Formats:** PDF, DOC, DOCX, TXT, CSV
- **CSV Processing:** PapaParse integration for contact imports
- **File Validation:** Client-side file type and size validation
- **Processing Pipeline:** File upload → AI analysis → Response generation

## Responsive Design
- **Mobile Support:** Bootstrap responsive grid system
- **Viewport Optimization:** Mobile-first design approach
- **Touch-Friendly:** Large buttons and touch targets
- **Cross-Browser:** Chrome, Firefox, Safari, Edge compatibility

## Performance Optimizations
- **Lazy Loading:** Modals created on demand
- **Event Delegation:** Efficient event handling for dynamic content
- **Local Storage:** Client-side token and preference storage
- **Async Operations:** Non-blocking API calls with loading indicators

## Error Handling
- **Authentication Errors:** Automatic logout on token expiration
- **Network Errors:** User-friendly error messages
- **Validation Errors:** Real-time form validation feedback
- **API Errors:** Graceful error handling with retry mechanisms

## Integration Points
- **AI Agent API:** Flask backend integration for all agent operations
- **Email System:** SMTP integration for email delivery
- **File Processing:** Server-side document processing capabilities
- **Workflow Engine:** Multi-agent coordination system

## Development Environment
- **Local Development:** http://localhost:5000 or http://127.0.0.1:5000
- **Production API:** https://api.alambda.com
- **Environment Detection:** Automatic API base URL detection
- **Debug Mode:** Console logging and error reporting

## Deployment Requirements
- **Python 3.7+** with Flask framework
- **Web Server:** Nginx or Apache recommended
- **SSL Certificate:** HTTPS required for production
- **File Permissions:** Read/write access to JSON data files
- **Memory:** Minimum 512MB RAM for AI agent operations

## Testing Coverage
- **Authentication Flow:** Login, logout, token validation
- **Contact Management:** CRUD operations
- **Agent System:** All 34 agents functional
- **Multi-Agent Workflow:** End-to-end RFP processing
- **Email System:** Composition, delivery, draft management
- **File Upload:** Document processing pipeline

## Browser Compatibility
- **Chrome:** 90+
- **Firefox:** 88+
- **Safari:** 14+
- **Edge:** 90+
- **Mobile Safari:** iOS 14+
- **Chrome Mobile:** Android 8+

## Configuration Variables
```javascript
const API_BASE_URL = window.location.origin.includes('localhost') || 
                     window.location.origin.includes('127.0.0.1') || 
                     window.location.origin.includes('file:') 
  ? 'http://localhost:5000' 
  : 'https://api.alambda.com';
```

## Maintenance Requirements
- **Log Rotation:** Workflow logs and error logs
- **Database Backup:** Regular JSON file backups
- **Token Cleanup:** Expired token management
- **Cache Management:** Browser cache optimization
- **Security Updates:** Regular dependency updates

## Future Enhancements
- **Database Migration:** Move from JSON to PostgreSQL/MongoDB
- **Real-time Updates:** WebSocket integration for live updates
- **Advanced Analytics:** Contact interaction analytics
- **Mobile App:** Native iOS/Android applications
- **SSO Integration:** Single sign-on with enterprise systems

---

**Document Version:** 1.0  
**Last Updated:** July 22, 2025  
**Prepared By:** AI Development Team  
**Review Status:** Ready for Production
