# AutoML Time Series Studio

An intelligent web-based tool for automatic time series forecasting. Upload your CSV, and the system automatically selects the best model (XGBoost/Prophet/RandomForest) and generates forecasts.

## 🚀 Features

- **AutoML Model Selection**: Automatically chooses optimal model based on data characteristics
- **Interactive Dashboard**: Clean web interface for data upload and visualization
- **Real-time Forecasting**: Get predictions instantly
- **Model Insights**: View seasonality scores, anomaly detection, and model metrics
- **CSV Support**: Easy data upload with automatic column detection

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python
- **ML Models**: XGBoost, Prophet, RandomForest
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Data Processing**: Pandas, NumPy, Scikit-learn

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/AutoML_TS_Studio.git
cd AutoML_TS_Studio

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m backend.app
```

## 🎯 Usage

Open browser at http://localhost:8000
Upload your time series CSV file (must have date and value columns)
Set forecast horizon (1-365 days)
Click "Generate Forecast"
View predictions with automatic model selection

## 📊 Sample CSV Format

```csv
date,value
2024-01-01,100
2024-01-02,102
2024-01-03,101
```

## 🤖 How AutoML Works

< 30 data points: Uses XGBoost for short-term patterns
30-100 points with seasonality: Uses Prophet for seasonal patterns
Default: RandomForest for balanced performance

## 🗂️ Project Structure

```text
AutoML_TS_Studio/
├── backend/
│   ├── automl/          # ML models and AutoML logic
│   ├── preprocessing/   # Data cleaning and scaling
│   ├── routes/          # API endpoints
│   └── app.py           # FastAPI main application
├── frontend/
│   ├── css/             # Stylesheets
│   ├── js/              # Frontend logic
│   └── index.html       # Main dashboard
├── temp_uploads/        # Temporary CSV storage
├── requirements.txt     # Python dependencies
└── README.md
```
## 📝 License

MIT License

## 📧 Contact

Your Name - [your.email@example.com]