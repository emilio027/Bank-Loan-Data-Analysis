# 🚀 Quick Start Installation Guide
## Enterprise Credit Risk Intelligence Platform

> **Get up and running in under 10 minutes**

---

## 📋 **Prerequisites**

### **Minimum Requirements:**
- 💻 **Python 3.9+** 
- 🐳 **Docker** (recommended)
- 💾 **8GB RAM** minimum
- 🗄️ **PostgreSQL** (or use Docker setup)

### **Optional but Recommended:**
- 📊 **Power BI Desktop** (for dashboard features)
- ☁️ **Cloud account** (AWS/Azure for production)

---

## ⚡ **Option 1: Quick Docker Setup (Recommended)**

### **1. Clone and Setup**
```bash
# Clone the repository
git clone <repository-url>
cd Enterprise-Credit-Risk-Intelligence-Platform

# Copy environment template
cp Resources/configuration/.env.example .env
```

### **2. Configure Environment**
Edit `.env` file with your settings:
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=credit_risk_db
DB_USER=your_username
DB_PASSWORD=your_password

# API Configuration  
API_PORT=8080
DEBUG_MODE=true

# Power BI Integration (optional)
PBI_TENANT_ID=your_tenant_id
PBI_CLIENT_ID=your_client_id
PBI_CLIENT_SECRET=your_secret
```

### **3. Launch Platform**
```bash
# Start all services
docker-compose up -d

# Initialize database
docker-compose exec app python Technical/Source_Code/data_manager.py --init-db

# Access the platform
# Web Interface: http://localhost:8080
# API Docs: http://localhost:8080/api/docs
```

---

## 🐍 **Option 2: Python Installation**

### **1. Environment Setup**
```bash
# Create virtual environment
python -m venv credit_risk_env
source credit_risk_env/bin/activate  # On Windows: credit_risk_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Database Setup**
```bash
# Install PostgreSQL
# Ubuntu/Debian: sudo apt install postgresql
# macOS: brew install postgresql
# Windows: Download from postgresql.org

# Create database
createdb credit_risk_db
```

### **3. Application Setup**
```bash
# Initialize database
python Technical/Source_Code/data_manager.py --init-db

# Start the application
python Technical/Source_Code/credit_risk_main.py

# Or start interactive analysis
streamlit run Enterprise_Credit_Risk_Interactive_Analysis.py
```

---

## ✅ **Verification Steps**

### **1. Check Web Interface**
Visit `http://localhost:8080` and verify:
- ✅ Dashboard loads successfully
- ✅ Sample data is displayed
- ✅ Navigation works properly

### **2. Test API Endpoints**
```bash
# Health check
curl http://localhost:8080/api/health

# Sample risk assessment
curl -X POST http://localhost:8080/api/risk-assessment \
  -H "Content-Type: application/json" \
  -d '{"loan_amount": 100000, "credit_score": 750, "debt_to_income": 0.3}'
```

### **3. Verify ML Models**
```bash
# Test model predictions
python -c "
from Technical.Source_Code.ml_models import MLModelManager
manager = MLModelManager()
print('✅ ML models loaded successfully')
"
```

---

## 🎯 **Next Steps**

### **1. Explore the Platform**
- 📊 **Interactive Demo**: Open `interactive_demo.html` in your browser
- 📈 **Dashboard**: Access the main dashboard at localhost:8080
- 🧠 **ML Models**: Test predictions via the API or web interface

### **2. Load Your Data**
- 📁 **Data Format**: See `Technical/Documentation/data_format.md`
- 🔄 **Data Import**: Use the web interface or API endpoints
- 🗄️ **Database Schema**: Review `Technical/Source_Code/data_manager.py`

### **3. Customize Configuration**
- ⚙️ **Settings**: Modify `.env` file for your environment
- 🎨 **Branding**: Update logos and colors in `Resources/configuration/`
- 🔗 **Integrations**: Configure Power BI and external APIs

---

## 🆘 **Troubleshooting**

### **Common Issues:**

#### **Port Already in Use**
```bash
# Check what's using port 8080
lsof -i :8080

# Use different port
export API_PORT=8081
```

#### **Database Connection Failed**
```bash
# Check PostgreSQL is running
sudo service postgresql status

# Reset database
dropdb credit_risk_db
createdb credit_risk_db
python Technical/Source_Code/data_manager.py --init-db
```

#### **Docker Issues**
```bash
# Reset Docker containers
docker-compose down -v
docker-compose up -d --build
```

---

## 📞 **Support**

### **Getting Help:**
- 📖 **Documentation**: Check `Technical/Documentation/`
- 🐛 **Issues**: Report problems via GitHub issues
- 💬 **Support**: Contact technical support team

### **Resources:**
- 🎬 **Video Tutorials**: See `Demo/` folder
- 📋 **API Documentation**: Available at `/api/docs` when running
- 🧪 **Example Code**: Check `Technical/Source_Code/examples/`

---

**🎉 Congratulations! Your Enterprise Credit Risk Intelligence Platform is ready to use.**

*Next: Try the [Interactive Demo](../interactive_demo.html) to explore platform capabilities.*