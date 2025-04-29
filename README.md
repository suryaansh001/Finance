**Financial Planner Project
Overview**

This project is a FastAPI-based web application for generating personalized financial plans. It processes user financial data, provides AI-driven predictions and suggestions, and generates PDF reports with charts and actionable advice. The application uses DistilRoBERTa for sentiment analysis and supports portfolio and bank statement analysis.
Features

User input form for financial details (salary, expenses, investments, goals, etc.).
Portfolio analysis with sector allocation and volatility metrics.
Bank statement analysis for spending categorization.
AI predictions for equity allocation, insurance needs, and savings rate.
Detailed suggestions for SIPs, insurance, spending cuts, and goal-specific plans.
PDF report generation with financial charts.
Chatbot for interactive financial advice.
SQLite database for user data and chat history.
**
Prerequisites**

Python 3.8+
**Dependencies** : pip install fastapi uvicorn pydantic sqlite3 reportlab matplotlib numpy pandas pdfplumber transformers torch


**
Setup**

Clone the repository:git clone https://github.com/suryaansh001/Finance/
cd Finance


Install dependencies

Run the application:uvicorn main:app --reload


Access the app at http://127.0.0.1:8000.

Usage

Generate Financial Plan:
Open the web interface.
Fill in user details (e.g., name, age, salary, expenses, goals).
Optionally upload a portfolio CSV (stock_name,quantity,purchase_price,current_price,sector) and bank statement PDF.
Submit to download a PDF report.


Chatbot:
Use the chat interface to ask financial queries (e.g., “What’s my insurance plan?”).
Responses are based on the latest user data.



Testing

Sample Input (Voldemort’s case):name: Voldemort
age: 40
salary: 100000
expenses: 60000
investments: 11340
goals: house,car
insurance: 100000
loans: 0
portfolio_file: (CSV with ₹11,340 value, volatility 0.04)


Portfolio CSV:stock_name,quantity,purchase_price,current_price,sector
RELIANCE,5,2000,2268,Energy


Expected Output:
PDF report with:
AI Predictions: Equity allocation (50%), insurance need (₹500,000), savings rate (₹20,000/month).
Suggestions: SIP (₹10,000/month), insurance increase (₹400,000), spending cuts (₹6,000).
Charts: Financial Overview, Income Breakdown, Portfolio Allocation.


Chatbot response (e.g., “Current insurance is ₹100,000, but ₹500,000 is recommended...”).


Database Check:sqlite3 financial_data.db "SELECT * FROM users"


**
Project Structure**

financial-planner/
├── financial_planner.py  # Main application code
├── reports/              # Generated PDF reports
├── financial_data.db     # SQLite database
├── README.md             # Project documentation
└── requirements.txt      # Dependencies

Notes

The application uses DistilRoBERTa for sentiment-based predictions. For generative output, consider integrating FinGPT (https://github.com/AI4Finance-Foundation/FinGPT).
The HLV calculation is capped at 30 years with input validation.
For production, deploy with Docker or a WSGI server (e.g., Gunicorn).

Contact
For issues or enhancements, contact the project maintainer.
