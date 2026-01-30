# 🏠 Housing Price Prediction

An end-to-end **Machine Learning web application** that predicts housing prices based on location and demographic features.  
The project demonstrates the complete ML lifecycle — from data preprocessing and model training to API development, frontend integration, and Docker-based deployment.

---

## 📌 Project Overview

This project uses a **Random Forest Regressor** trained on housing data to predict the **median house value**.  
The trained model is exposed via a **FastAPI REST API**, documented with **Swagger UI**, and connected to a **simple HTML frontend** for user interaction.  
The entire application is **Dockerized**, making it easy to run anywhere.

---

## 🚀 Features

- ✅ End-to-end Machine Learning pipeline  
- ✅ Data preprocessing using `ColumnTransformer`
- ✅ Random Forest regression model
- ✅ FastAPI backend with `/predict` endpoint
- ✅ Interactive Swagger documentation (`/docs`)
- ✅ Simple HTML frontend for predictions
- ✅ Dockerized application for easy deployment
- ✅ Clean project structure suitable for portfolio & internships

---

## 🛠 Tech Stack

**Backend & ML**
- Python
- scikit-learn
- Pandas & NumPy

**API**
- FastAPI
- Uvicorn

**Frontend**
- HTML
- CSS
- JavaScript

**DevOps**
- Docker
- Git & GitHub

---

## 📂 Project Structure

Housing-Price-Prediction/
│
├── api.py # FastAPI application
├── main.py # Model training & pipeline logic
├── index.html # Frontend UI
├── Dockerfile # Docker configuration
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Ignored files

---

## ▶️ Run the Project Locally (Docker Recommended)

### 1️⃣ Build Docker Image
```bash
docker build -t housing-api .
docker run -p 8000:8000 housing-api
🌐 Access the Application

Swagger API Docs:
👉 http://localhost:8000/docs

API Root:
👉 http://localhost:8000

Frontend UI:
👉 Open index.html in your browser
(or serve it using Live Server / any static server)

📤 Example API Request

Endpoint

POST /predict


Sample JSON Input

{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.32,
  "ocean_proximity": "NEAR BAY"
}


Sample Response

{
  "predicted_median_house_value": 484940.94
}

🧠 Machine Learning Details

Model: Random Forest Regressor

Preprocessing:

Numerical features: Median imputation + Standard Scaling

Categorical feature: One-Hot Encoding (ocean_proximity)

Evaluation Metric: RMSE

Hyperparameter Tuning: GridSearchCV<img width="954" height="869" alt="image" src="https://github.com/user-attachments/assets/d462dc8a-ace2-462c-9431-f357d0612486" />
