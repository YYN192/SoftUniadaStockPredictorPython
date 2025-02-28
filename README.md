# SoftUniada Stock Predictor (Python)

## 📌 About  
SoftUniada Stock Predictor is a Python-based stock forecasting tool that uses historical stock data from Yahoo Finance to predict future stock prices using **Linear Regression**.

The project is designed for educational purposes and demonstrates key concepts in **data preprocessing, machine learning, and financial forecasting**.

---

## 📑 Table of Contents  
- [Features](#-features)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Roadmap](#-roadmap)  

---

## ✨ Features  
✔️ Fetches real-time stock data using `yfinance`  
✔️ Processes and extracts key stock indicators (`HL_PCT`, `PCT_change`, `Volume`)  
✔️ Trains a **Linear Regression** model to predict stock prices  
✔️ Saves and loads trained models using `pickle`  
✔️ Generates and saves forecasted stock price graphs using `matplotlib`  

---

## 📥 Installation  
### Prerequisites  
Ensure you have **Python 3.10+** installed. You can install the required dependencies using:

```bash
pip install numpy pandas yfinance scikit-learn matplotlib
```
---


### 🚀 Usage

```bash
python main.py
```
Expected Output

-    Fetches stock data
-    Trains a Linear Regression model
-    Predicts stock prices
-    Generates and saves a forecast graph
  ![GOOGL_forecast](https://github.com/user-attachments/assets/ad89292d-c0f9-4b6e-9859-7741d93c033f)


### 🛠️ Roadmap
✅ Implement basic stock price prediction
✅ Add model saving/loading
🔲 Enhance prediction with more ML models (LSTMs, Random Forests)
🔲 Create a web UI for stock prediction visualization




