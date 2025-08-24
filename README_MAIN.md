# Main Application Launcher

This `main.py` file provides a unified interface to run both Flask applications in the F&B Process Anomaly Detection System.

## 🚀 Quick Start

### Interactive Mode (Recommended)
```bash
python main.py
```
This will show an interactive menu where you can choose which application to run.

### Command Line Options

#### Run Original App Only (Module 1: Train-on-demand)
```bash
python main.py --app1
```
- **Port**: http://localhost:5000
- **Features**: Real-time process monitoring, data upload, model training, quality prediction, anomaly detection

#### Run Dual-Module App Only (Both modules)
```bash
python main.py --app2
```
- **Port**: http://localhost:5000
- **Features**: 
  - Module 1: Train-on-demand (same as original)
  - Module 2: Pre-trained prediction with instant analysis

#### Run Both Apps Simultaneously
```bash
python main.py --both
```
- **Original App**: http://localhost:5000
- **Dual-Module App**: http://localhost:5001

#### Check System Status
```bash
python main.py --status
```
Shows system status, dependencies, and available features.

## 📊 Application Comparison

### Original App (app.py)
- **Purpose**: Module 1 - Train-on-demand functionality
- **Features**:
  - Real-time process monitoring dashboard
  - Data upload and processing
  - Model training and evaluation
  - Quality prediction API
  - Anomaly detection alerts
  - Comprehensive reporting
  - Process parameter monitoring
  - Trend data visualization

### Dual-Module App (app_v2.py)
- **Purpose**: Both Module 1 and Module 2 functionality
- **Module 1**: Same as original app
- **Module 2**: Pre-trained prediction system
  - Instant anomaly prediction using pre-trained models
  - Multiple ML models (XGBoost, Random Forest, Neural Network)
  - Comprehensive analysis with graphs
  - JSON report generation
  - File upload and processing
  - Anomaly detection warnings

## 🔧 System Requirements

### Dependencies
The launcher checks for the following dependencies:
- Flask
- Pandas
- NumPy
- Joblib
- Matplotlib
- Seaborn
- XGBoost
- Scikit-learn

### Directory Structure
```
├── main.py                 # Main launcher
├── app/
│   ├── app.py             # Original application
│   ├── app_v2.py          # Dual-module application
│   ├── templates/         # HTML templates
│   └── static/            # CSS/JS files
├── src/                   # Source code
├── data/                  # Data files
│   └── model_module2/     # Pre-trained models
└── reports/               # Generated reports
```

## 🎯 Usage Examples

### 1. First Time Setup
```bash
# Check if everything is ready
python main.py --status

# If dependencies are missing, install them
pip install -r requirements.txt
```

### 2. Development Workflow
```bash
# Start with dual-module app for full functionality
python main.py --app2

# Or run both apps for comparison
python main.py --both
```

### 3. Production Deployment
```bash
# Run only the dual-module app (recommended for production)
python main.py --app2
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Port Already in Use
If you get a port conflict error:
```bash
# Kill processes using the port
lsof -ti:5000 | xargs kill -9
lsof -ti:5001 | xargs kill -9

# Or use different ports via environment variables
PORT=5002 python main.py --app2
```

#### 2. Missing Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Or install individually
pip install flask pandas numpy joblib matplotlib seaborn xgboost scikit-learn
```

#### 3. Model Files Not Found
If Module 2 models are missing:
```bash
# Check if models exist
python main.py --status

# If models are missing, train them first
cd notebooks
jupyter notebook 05_module2_model_training.ipynb
```

### Error Messages

- **"No module named 'flask'"**: Install Flask with `pip install flask`
- **"Model directory not found"**: Ensure `data/model_module2/` exists
- **"Port already in use"**: Kill existing processes or use different ports
- **"Permission denied"**: Run with appropriate permissions

## 📝 Configuration

### Environment Variables
You can configure the applications using environment variables:

```bash
# Set port for app_v2.py
export PORT=5001
python main.py --app2

# Set host and debug mode
export HOST=0.0.0.0
export DEBUG=False
python main.py --app2
```

### Available Environment Variables
- `PORT`: Port number (default: 5000)
- `HOST`: Host address (default: 0.0.0.0)
- `DEBUG`: Debug mode (default: True)
- `SECRET_KEY`: Flask secret key

## 🧪 Testing

Run the test script to verify main.py functionality:
```bash
python test_main.py
```

This will test:
- Import functionality
- Function availability
- Dependency checking
- Model file verification

## 📚 Additional Resources

- **Original App Documentation**: See `README.md` for detailed app functionality
- **Module 2 Training**: See `notebooks/05_module2_model_training.ipynb`
- **API Documentation**: Check the Flask routes in both app files
- **Configuration**: See `src/config.py` for all configuration options

## 🤝 Support

If you encounter issues:
1. Run `python main.py --status` to check system status
2. Check the troubleshooting section above
3. Review the error logs in the terminal
4. Ensure all dependencies are installed correctly

---

**Happy Coding! 🚀**
