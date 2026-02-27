# 🏠 House Price Prediction - California Housing Dataset

A complete **end-to-end machine learning project** for predicting median house prices in California using **XGBoost Regressor**. Built with scikit-learn, pandas, matplotlib, seaborn, and XGBoost.

---

## 📋 Project Overview

This project demonstrates a full ML pipeline:

- Data loading & exploration
- Basic cleaning & EDA
- Model training with **XGBoost**
- Hyperparameter tuning using `RandomizedSearchCV`
- Model evaluation & visualization
- Model saving (both joblib and native XGBoost format)
- Model loading & inference

**Dataset**: California Housing (1990 U.S. Census) – 20,640 samples
**Task**: Regression
**Target**: Median House Value (`Price` in $100,000 units)

---

## 🛠️ Features

| Feature        | Description                              |
| -------------- | ---------------------------------------- |
| `MedInc`     | Median income in block group             |
| `HouseAge`   | Median house age in block group          |
| `AveRooms`   | Average number of rooms per household    |
| `AveBedrms`  | Average number of bedrooms per household |
| `Population` | Block group population                   |
| `AveOccup`   | Average number of household members      |
| `Latitude`   | Block group latitude                     |
| `Longitude`  | Block group longitude                    |

---

## 📊 Results

- **Train R²**: 0.9468
- **Test R²**: 0.8486

Visualizations included:

- Residual Distribution
- XGBoost Learning Curve (RMSE)

---

## 📁 Project Structure

```
house-price-prediction/
├── data/
│   └── raw/
│       └── house_prices_dataset.csv
├── models/
│   ├── xgb_house_price_model.pkl
│   └── xgb_house_price_model.json
├── notebooks/
│   └── house_price_prediction.ipynb
├── src/                  # (future scripts)
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd house-price-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `notebooks/house_price_prediction.ipynb` in Jupyter Notebook / VS Code / JupyterLab.

---

## 📌 Key Sections in the Notebook

1. **Imports & Libraries**
2. **Data Loading** (sklearn California Housing)
3. **Data Exploration & Saving**
4. **Data Cleaning & EDA**
5. **Model Training** (XGBoost)
6. **Hyperparameter Tuning**
7. **Model Evaluation** (R², Residuals, Learning Curve)
8. **Model Saving** (joblib + native `.json`)
9. **Model Loading & Prediction**

---

## 💾 Model Saving & Loading

The model is saved in **two formats**:

```python
# Joblib (recommended for scikit-learn style)
joblib.dump(model, '../models/xgb_house_price_model.pkl')

# Native XGBoost format
model.save_model('../models/xgb_house_price_model.json')
```

**Loading example**:

```python
# Joblib
model = joblib.load('../models/xgb_house_price_model.pkl')

# Native XGBoost
model = XGBRegressor()
model.load_model('../models/xgb_house_price_model.json')
```

---

## 🔮 Making Predictions

```python
import joblib
import numpy as np

model = joblib.load('../models/xgb_house_price_model.pkl')

# Example input (MedInc, HouseAge, AveRooms, ...)
sample = np.array([[8.3252, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23]])
prediction = model.predict(sample)
print(f"Predicted House Price: ${prediction[0]*100_000:,.0f}")
```

---

## 📈 Future Improvements

- Add more advanced feature engineering (e.g., distance to ocean, income buckets)
- Try other models (LightGBM, CatBoost, Neural Networks)
- Deploy as a web app (Streamlit / FastAPI)
- Dockerize the project
- Add CI/CD pipeline

---

## 📄 License

This project is for educational and demonstration purposes.
Made by: Ahmed Akram Amer | The Triple AI

---

**Made with ❤️ using XGBoost & scikit-learn**

Feel free to ⭐ the repo if you found it helpful!

```

```
