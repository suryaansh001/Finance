import fastapi
from fastapi import FastAPI, Form, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import sqlite3
from datetime import datetime
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import io
import os
import pandas as pd
import pdfplumber
import re
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

# Initialize DistilRoBERTa model
try:
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    print(f"Error loading DistilRoBERTa: {e}. Using dummy output.")
    tokenizer = None
    model = None

# Initialize database
def init_db():
    conn = sqlite3.connect("financial_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, age INTEGER, salary REAL, expenses REAL, investments REAL,
        goals TEXT, insurance REAL, loans REAL, portfolio_details TEXT,
        spending_details TEXT, report_path TEXT, created_at TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, message TEXT,
        response TEXT, timestamp TEXT, FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    c.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in c.fetchall()]
    if 'portfolio_details' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN portfolio_details TEXT")
    if 'spending_details' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN spending_details TEXT")
    conn.commit()
    conn.close()

init_db()

# Pydantic model for user input
class UserInput(BaseModel):
    name: str
    age: int
    salary: float
    expenses: float
    investments: float
    goals: str
    insurance: float
    loans: float
    portfolio_details: str = ""
    spending_details: str = ""

# Financial calculators
def calculate_future_value(principal, rate, years):
    return principal * (1 + rate) ** years

def calculate_sip(sip_amount, rate, years):
    months = years * 12
    monthly_rate = rate / 12
    return sip_amount * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)

def calculate_emi(principal, rate, years):
    months = years * 12
    monthly_rate = rate / 12
    return (principal * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1) if principal > 0 else 0

def calculate_hlv(age, salary, expenses):
    years_to_retire = min(max(60 - age, 0), 30)
    if salary <= expenses or age < 18 or age > 100:
        return 0
    return (salary - expenses) * 12 * years_to_retire

# Analyze portfolio
def analyze_portfolio(portfolio_df):
    total_value = (portfolio_df['quantity'] * portfolio_df['current_price']).sum()
    portfolio_df['value'] = portfolio_df['quantity'] * portfolio_df['current_price']
    portfolio_df['weight'] = portfolio_df['value'] / total_value
    sector_allocation = portfolio_df.groupby('sector')['value'].sum() / total_value if 'sector' in portfolio_df.columns else None
    portfolio_df['returns'] = (portfolio_df['current_price'] - portfolio_df['purchase_price']) / portfolio_df['purchase_price']
    volatility = portfolio_df['returns'].std() if len(portfolio_df) > 1 else 0.1
    suggestions = []
    if sector_allocation is not None and any(sector_allocation > 0.5):
        dominant_sector = sector_allocation.idxmax()
        suggestions.append(f"Portfolio is overweight in {dominant_sector}. Consider diversifying.")
    if volatility > 0.2:
        suggestions.append("High volatility detected. Add stable stocks or debt instruments.")
    return {
        "total_value": total_value,
        "sector_allocation": sector_allocation.to_dict() if sector_allocation is not None else {},
        "volatility": volatility,
        "suggestions": suggestions
    }

# Analyze bank statement
def analyze_bank_statement(pdf_file, salary):
    transactions = []
    categories = {
        'Food': ['restaurant', 'cafe', 'zomato', 'swiggy', 'food'],
        'Shopping': ['amazon', 'flipkart', 'myntra', 'shop', 'mall'],
        'Travel': ['uber', 'ola', 'flight', 'train', 'hotel'],
        'Housing': ['rent', 'mortgage', 'electricity', 'water'],
        'Entertainment': ['netflix', 'prime', 'cinema', 'concert']
    }
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                for line in lines:
                    match = re.match(r'(\d{2}/\d{2}/\d{4})\s+(.+?)\s+-?(\d+\.?\d*)', line)
                    if match:
                        date, description, amount = match.groups()
                        amount = float(amount)
                        if amount > 0:
                            category = 'Others'
                            description_lower = description.lower()
                            for cat, keywords in categories.items():
                                if any(keyword in description_lower for keyword in keywords):
                                    category = cat
                                    break
                            transactions.append({'date': date, 'description': description, 'amount': amount, 'category': category})
    spending_df = pd.DataFrame(transactions)
    if spending_df.empty:
        return {'total_spending': 0, 'category_spending': {}, 'overspending': [], 'ideal_spending': {}}
    total_spending = spending_df['amount'].sum()
    category_spending = spending_df.groupby('category')['amount'].sum().to_dict()
    income_range = '50K-1L' if 50000 <= salary <= 100000 else '1L+'
    avg_spending = {
        '50K-1L': {'Food': 15000, 'Shopping': 10000, 'Travel': 8000, 'Housing': 20000, 'Entertainment': 5000, 'Others': 10000},
        '1L+': {'Food': 25000, 'Shopping': 15000, 'Travel': 12000, 'Housing': 35000, 'Entertainment': 8000, 'Others': 15000}
    }
    ideal_percentages = {'Food': 0.2, 'Shopping': 0.1, 'Travel': 0.1, 'Housing': 0.3, 'Entertainment': 0.05, 'Others': 0.15}
    overspending = []
    ideal_spending = {cat: salary * perc for cat, perc in ideal_percentages.items()}
    for cat in category_spending:
        user_spending = category_spending.get(cat, 0)
        avg = avg_spending[income_range].get(cat, 10000)
        if user_spending > avg:
            overspending.append(f"Spending on {cat} (₹{user_spending:,.0f}) exceeds average (₹{avg:,.0f}) for your income range ({income_range}). Reduce by ₹{user_spending - avg:,.0f}.")
    return {
        'total_spending': total_spending,
        'category_spending': category_spending,
        'overspending': overspending,
        'ideal_spending': ideal_spending
    }

# Generate model predictions
def get_model_predictions(user_data, portfolio_analysis, spending_analysis):
    if tokenizer is None or model is None:
        return {
            "predictions": [
                f"Equity Allocation: 60% in {'equity funds' if user_data['age'] < 40 else 'balanced funds'}",
                f"Insurance Need: ₹{min(user_data['salary'] * 12 * 0.05, 500000):,.0f}",
                f"Savings Rate: ₹{min(user_data['salary'] * 0.2, 20000):,.0f}/month"
            ],
            "suggestions": [
                f"Start a ₹{min(user_data['salary'] * 0.1, 10000):,.0f}/month SIP in {'equity funds' if user_data['age'] < 40 else 'balanced funds'}. Expected 12% return over 10 years.",
                f"Current insurance is insufficient. Increase by ₹{(min(user_data['salary'] * 12 * 0.05, 500000) - user_data['insurance']):,.0f}.",
                f"Expenses are ₹{user_data['expenses']:,.0f}. Reduce non-essential spending by ₹{user_data['expenses'] * 0.1:,.0f}.",
                f"For goals ({user_data['goals']}), invest ₹{min(user_data['salary'] * 0.15, 15000):,.0f}/month in diversified funds."
            ] + spending_analysis['overspending']
        }
    input_text = f"""
    User: {user_data['name']}, Age: {user_data['age']}, Salary: ₹{user_data['salary']:,.0f}/month, 
    Expenses: ₹{user_data['expenses']:,.0f}/month, Investments: ₹{user_data['investments']:,.0f}, 
    Goals: {user_data['goals']}, Insurance: ₹{user_data['insurance']:,.0f}, Loans: ₹{user_data['loans']:,.0f}, 
    Portfolio Value: ₹{portfolio_analysis['total_value']:,.0f}, Volatility: {portfolio_analysis['volatility']:.2f}, 
    Spending: {json.dumps(spending_analysis['category_spending'])}.
    Suggest a detailed financial plan with equity allocation, insurance needs, savings rate, and actionable advice.
    """
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    sentiment = outputs.logits.softmax(dim=-1).argmax().item()
    predictions = []
    suggestions = []
    equity_pct = 60 if user_data['age'] < 40 else 50
    fund_type = 'equity funds' if user_data['age'] < 40 else 'balanced funds'
    if sentiment == 2:
        equity_pct += 10
        suggestions.append("Market sentiment is bullish. Increase equity exposure for higher returns.")
    elif sentiment == 0:
        equity_pct -= 10
        suggestions.append("Market sentiment is bearish. Shift to debt or balanced funds for stability.")
    predictions.append(f"Equity Allocation: {equity_pct}% in {fund_type} (based on {'bullish' if sentiment == 2 else 'neutral' if sentiment == 1 else 'bearish'} market sentiment and age).")
    ideal_insurance = min(user_data['salary'] * 12 * 0.05, 500000)
    hlv = calculate_hlv(user_data['age'], user_data['salary'], user_data['expenses'])
    predictions.append(f"Insurance Need: ₹{ideal_insurance:,.0f} (aligned with HLV of ₹{hlv:,.0f} to support goals: {user_data['goals']}).")
    if user_data['insurance'] < ideal_insurance:
        suggestions.append(f"Current insurance (₹{user_data['insurance']:,.0f}) is inadequate. Increase by ₹{(ideal_insurance - user_data['insurance']):,.0f} to secure family and goals ({user_data['goals']}).")
    savings = user_data['salary'] - user_data['expenses']
    ideal_savings = user_data['salary'] * 0.2
    predictions.append(f"Savings Rate: ₹{min(ideal_savings, savings if savings > 0 else 0):,.0f}/month ({'20%' if savings > 0 else '0%'} of salary to fund goals: {user_data['goals']}).")
    if savings < ideal_savings:
        suggestions.append(f"Current savings (₹{savings:,.0f}/month) are below target (₹{ideal_savings:,.0f}). Reduce non-essential expenses by ₹{(ideal_savings - savings):,.0f}.")
    sip_amount = min(user_data['salary'] * 0.1, 10000)
    suggestions.append(f"Start a ₹{sip_amount:,.0f}/month SIP in {fund_type} (e.g., HDFC Balanced Advantage Fund). Expected 12% return over 10 years, yielding ₹{calculate_sip(sip_amount, 0.12, 10):,.0f} for goals: {user_data['goals']}.")
    if spending_analysis['total_spending'] > 0:
        suggestions.append(f"Total spending is ₹{spending_analysis['total_spending']:,.0f}. Reduce discretionary spending by ₹{spending_analysis['total_spending'] * 0.1:,.0f} to increase savings.")
    else:
        suggestions.append(f"Expenses are ₹{user_data['expenses']:,.0f}. Reduce non-essential spending by ₹{user_data['expenses'] * 0.1:,.0f} to boost savings.")
    if portfolio_analysis['volatility'] < 0.1:
        suggestions.append(f"Portfolio volatility is low ({portfolio_analysis['volatility']:.2f}). Add small-cap funds for growth, maintaining 20% in debt funds.")
    elif portfolio_analysis['volatility'] > 0.2:
        suggestions.append(f"Portfolio volatility is high ({portfolio_analysis['volatility']:.2f}). Reduce equity exposure and allocate 30% to debt funds.")
    goals = user_data['goals'].split(',')
    for goal in goals:
        goal = goal.strip().lower()
        if goal in ['house', 'home']:
            suggestions.append(f"For a house, invest ₹{sip_amount * 1.5:,.0f}/month in a SIP for 15 years. Expected value: ₹{calculate_sip(sip_amount * 1.5, 0.12, 15):,.0f} for down payment.")
        elif goal in ['car']:
            suggestions.append(f"For a car, invest ₹{sip_amount * 0.5:,.0f}/month in a SIP for 5 years. Expected value: ₹{calculate_sip(sip_amount * 0.5, 0.12, 5):,.0f}.")
    suggestions.extend(spending_analysis['overspending'])
    return {
        "predictions": predictions,
        "suggestions": suggestions if suggestions else ["Your financial plan is well-structured."]
    }

# Generate graphs
def generate_graphs(user_data, portfolio_analysis, spending_analysis):
    plt.figure(figsize=(6, 4))
    labels = ['Salary', 'Expenses', 'Investments']
    values = [user_data['salary'], user_data['expenses'], user_data['investments']]
    plt.bar(labels, values, color=['blue', 'red', 'green'])
    plt.title("Financial Overview")
    plt.tight_layout()
    img_buffer1 = io.BytesIO()
    plt.savefig(img_buffer1, format='png')
    plt.close()
    img_buffer1.seek(0)
    img_buffer2 = None
    if portfolio_analysis['sector_allocation']:
        plt.figure(figsize=(6, 4))
        sectors = list(portfolio_analysis['sector_allocation'].keys())
        weights = list(portfolio_analysis['sector_allocation'].values())
        plt.pie(weights, labels=sectors, autopct='%1.1f%%')
        plt.title("Portfolio Sector Allocation")
        plt.tight_layout()
        img_buffer2 = io.BytesIO()
        plt.savefig(img_buffer2, format='png')
        plt.close()
        img_buffer2.seek(0)
    plt.figure(figsize=(6, 4))
    labels = ['Your Insurance', 'Average for Your Income']
    avg_insurance = min(user_data['salary'] * 12 * 0.05, 500000)
    values = [user_data['insurance'], avg_insurance]
    plt.bar(labels, values, color=['orange', 'grey'])
    plt.title("Health Insurance Comparison")
    plt.tight_layout()
    img_buffer3 = io.BytesIO()
    plt.savefig(img_buffer3, format='png')
    plt.close()
    img_buffer3.seek(0)
    img_buffer4 = None
    if spending_analysis['category_spending']:
        plt.figure(figsize=(8, 4))
        categories = list(spending_analysis['category_spending'].keys())
        user_spending = [spending_analysis['category_spending'].get(cat, 0) for cat in categories]
        ideal_spending = [spending_analysis['ideal_spending'].get(cat, 10000) for cat in categories]
        x = np.arange(len(categories))
        plt.bar(x - 0.2, user_spending, 0.4, label='Your Spending', color='red')
        plt.bar(x + 0.2, ideal_spending, 0.4, label='Ideal Spending', color='blue')
        plt.xticks(x, categories, rotation=45)
        plt.title("Spending Analysis")
        plt.legend()
        plt.tight_layout()
        img_buffer4 = io.BytesIO()
        plt.savefig(img_buffer4, format='png')
        plt.close()
        img_buffer4.seek(0)
    plt.figure(figsize=(6, 4))
    labels = ['Expenses', 'Savings', 'Investments']
    savings = user_data['salary'] - user_data['expenses']
    values = [user_data['expenses'], savings if savings > 0 else 0, user_data['investments']]
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['red', 'green', 'blue'])
    plt.title("Income Breakdown")
    plt.tight_layout()
    img_buffer5 = io.BytesIO()
    plt.savefig(img_buffer5, format='png')
    plt.close()
    img_buffer5.seek(0)
    return img_buffer1, img_buffer2, img_buffer3, img_buffer4, img_buffer5

# Generate PDF report
def generate_pdf_report(user_data, rule_based_results, model_results, portfolio_analysis, spending_analysis):
    report_path = f"reports/{user_data['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    os.makedirs("reports", exist_ok=True)
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph(f"Financial Plan for {user_data['name']}", styles['Title']))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
    elements.append(Spacer(1, 12))
    img_buffer1, img_buffer2, img_buffer3, img_buffer4, img_buffer5 = generate_graphs(user_data, portfolio_analysis, spending_analysis)
    elements.append(Paragraph("Financial Overview Chart", styles['Heading2']))
    elements.append(Image(img_buffer1, width=300, height=200))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Income Breakdown", styles['Heading2']))
    elements.append(Image(img_buffer5, width=300, height=200))
    elements.append(Spacer(1, 12))
    if img_buffer2:
        elements.append(Paragraph("Portfolio Sector Allocation", styles['Heading2']))
        elements.append(Image(img_buffer2, width=300, height=200))
        elements.append(Spacer(1, 12))
    elements.append(Paragraph("Health Insurance Comparison", styles['Heading2']))
    elements.append(Image(img_buffer3, width=300, height=200))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Spending Analysis", styles['Heading1']))
    if spending_analysis['category_spending']:
        data = [["Category", "Your Spending", "Ideal Spending"]]
        for cat in spending_analysis['category_spending']:
            data.append([cat, f"₹{spending_analysis['category_spending'][cat]:,.0f}", f"₹{spending_analysis['ideal_spending'].get(cat, 10000):,.0f}"])
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Image(img_buffer4, width=350, height=200))
    else:
        elements.append(Paragraph("No bank statement uploaded. Spending based on manual expenses input.", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Portfolio Analysis", styles['Heading1']))
    if portfolio_analysis['total_value'] > 0 and portfolio_analysis['sector_allocation']:
        elements.append(Paragraph(f"Total Portfolio Value: ₹{portfolio_analysis['total_value']:,.2f}", styles['Normal']))
        elements.append(Paragraph(f"Volatility (Risk): {portfolio_analysis['volatility']:.2f}", styles['Normal']))
        if portfolio_analysis['suggestions']:
            elements.append(Paragraph("Suggestions:", styles['Heading2']))
            for suggestion in portfolio_analysis['suggestions']:
                elements.append(Paragraph(suggestion, styles['Normal']))
    else:
        elements.append(Paragraph("No portfolio data provided. Investments based on manual input.", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("AI Predictions and Suggestions", styles['Heading1']))
    elements.append(Paragraph("Predictions:", styles['Heading2']))
    for pred in model_results['predictions']:
        elements.append(Paragraph(pred, styles['Normal']))
    elements.append(Paragraph("Suggestions:", styles['Heading2']))
    for sug in model_results['suggestions']:
        elements.append(Paragraph(sug, styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Standard Calculated Plan", styles['Heading1']))
    elements.append(Paragraph(f"Future Value of Investments: ₹{rule_based_results['fv']:,.2f}", styles['Normal']))
    elements.append(Paragraph(f"SIP Growth: ₹{rule_based_results['sip']:,.2f}", styles['Normal']))
    elements.append(Paragraph(f"EMI for Loans: ₹{rule_based_results['emi']:,.2f}", styles['Normal']))
    elements.append(Paragraph(f"Insurance Need (HLV): ₹{rule_based_results['hlv']:,.2f}", styles['Normal']))
    elements.append(Spacer(1, 12))
    doc.build(elements)
    return report_path

# Chatbot response
def chatbot_response(message, user_data, portfolio_analysis, spending_analysis, user_id):
    message = message.lower().strip()
    conn = sqlite3.connect("financial_data.db")
    c = conn.cursor()
    c.execute("SELECT message, response FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 3", (user_id,))
    history = c.fetchall()
    conn.close()
    if tokenizer is None or model is None:
        return "Consider investing in mutual funds. Start a ₹5,000/month SIP."
    context = " ".join([f"User: {h[0]} Bot: {h[1]}" for h in history])
    input_text = f"""
    User: {user_data['name']}, Age: {user_data['age']}, Salary: ₹{user_data['salary']:,.0f}/month, 
    Expenses: ₹{user_data['expenses']:,.0f}/month, Investments: ₹{user_data['investments']:,.0f}, 
    Goals: {user_data['goals']}, Insurance: ₹{user_data['insurance']:,.0f}, Loans: ₹{user_data['loans']:,.0f}, 
    Portfolio Value: ₹{portfolio_analysis['total_value']:,.0f}, Volatility: {portfolio_analysis['volatility']:.2f}, 
    Spending: {json.dumps(spending_analysis['category_spending'])}.
    Chat History: {context}
    Query: {message}
    Provide detailed financial advice in a professional tone.
    """
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    sentiment = outputs.logits.softmax(dim=-1).argmax().item()
    savings = user_data['salary'] - user_data['expenses']
    sip_amount = min(user_data['salary'] * 0.1, 10000)
    ideal_insurance = min(user_data['salary'] * 12 * 0.05, 500000)
    fund_type = 'equity funds' if user_data['age'] < 40 else 'balanced funds'
    if "invest" in message:
        response = f"{'Market is bullish, consider higher equity exposure.' if sentiment == 2 else 'Market is neutral, maintain steady investments.' if sentiment == 1 else 'Market is bearish, focus on safer funds.'} Start a ₹{sip_amount:,.0f}/month SIP in {fund_type} (e.g., SBI Equity Hybrid Fund). Expected 12% return over 10 years, yielding ₹{calculate_sip(sip_amount, 0.12, 10):,.0f} for goals: {user_data['goals']}."
    elif "insurance" in message:
        hlv = calculate_hlv(user_data['age'], user_data['salary'], user_data['expenses'])
        response = f"Current insurance is ₹{user_data['insurance']:,.0f}, but ₹{ideal_insurance:,.0f} is recommended (HLV: ₹{hlv:,.0f}). {'Market is bullish, secure a term plan now.' if sentiment == 2 else 'Market is neutral, plan carefully.' if sentiment == 1 else 'Market is bearish, but insurance is essential.'} Increase by ₹{(ideal_insurance - user_data['insurance']):,.0f} to support goals: {user_data['goals']}."
    elif "goals" in message:
        response = f"Your goals are: {user_data['goals']}. {'Market is bullish, increase SIP contributions.' if sentiment == 2 else 'Market is neutral, maintain steady SIPs.' if sentiment == 1 else 'Market is bearish, add debt funds.'} Start a ₹{sip_amount:,.0f}/month SIP in {fund_type}, expected to yield ₹{calculate_sip(sip_amount, 0.12, 10):,.0f} in 10 years."
    elif "spending" in message:
        response = f"{'Upload a bank statement to analyze spending.' if not spending_analysis['category_spending'] else 'Spending analysis: ' + '; '.join(spending_analysis['overspending']) + '. '}Expenses are ₹{user_data['expenses']:,.0f}. {'Market is bullish, save more.' if sentiment == 2 else 'Market is neutral, spending is acceptable.' if sentiment == 1 else 'Market is bearish, reduce expenses.'} Cut ₹{user_data['expenses'] * 0.1:,.0f} from non-essential spending."
    else:
        response = f"{'Market is bullish, focus on SIPs and insurance.' if sentiment == 2 else 'Market is neutral, maintain a steady plan.' if sentiment == 1 else 'Market is bearish, prioritize safe investments.'} Your portfolio (₹{portfolio_analysis['total_value']:,.0f}, volatility {portfolio_analysis['volatility']:.2f}) is stable. Start a ₹{sip_amount:,.0f}/month SIP and increase insurance by ₹{(ideal_insurance - user_data['insurance']):,.0f} for goals: {user_data['goals']}."
    return response

# API endpoint for generating plan
@app.post("/generate_plan")
async def generate_plan(
    name: str = Form(...),
    age: int = Form(...),
    salary: float = Form(...),
    expenses: float = Form(...),
    investments: float = Form(...),
    goals: str = Form(...),
    insurance: float = Form(...),
    loans: float = Form(...),
    portfolio_file: UploadFile = File(default=None),
    bank_statement: UploadFile = File(default=None)
):
    if age < 18 or age > 100:
        raise HTTPException(status_code=400, detail="Age must be between 18 and 100")
    if salary < 0 or expenses < 0 or investments < 0 or insurance < 0 or loans < 0:
        raise HTTPException(status_code=400, detail="Financial values cannot be negative")
    portfolio_analysis = {"total_value": investments, "sector_allocation": {}, "volatility": 0.1, "suggestions": []}
    portfolio_details = ""
    if portfolio_file and portfolio_file.filename:
        try:
            portfolio_df = pd.read_csv(portfolio_file.file)
            required_columns = ['stock_name', 'quantity', 'purchase_price', 'current_price']
            if not all(col in portfolio_df.columns for col in required_columns):
                raise HTTPException(status_code=400, detail="Portfolio CSV must include stock_name, quantity, purchase_price, current_price")
            portfolio_analysis = analyze_portfolio(portfolio_df)
            investments = portfolio_analysis['total_value']
            portfolio_details = portfolio_df.to_json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")
    spending_analysis = {'total_spending': 0, 'category_spending': {}, 'overspending': [], 'ideal_spending': {}}
    spending_details = ""
    if bank_statement and bank_statement.filename:
        try:
            with open("temp_statement.pdf", "wb") as f:
                f.write(await bank_statement.read())
            spending_analysis = analyze_bank_statement("temp_statement.pdf", salary)
            spending_details = json.dumps(spending_analysis)
            os.remove("temp_statement.pdf")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing bank statement: {str(e)}")
    user_data = {
        "name": name,
        "age": age,
        "salary": salary,
        "expenses": expenses,
        "investments": investments,
        "goals": goals,
        "insurance": insurance,
        "loans": loans
    }
    rule_based_results = {
        "fv": calculate_future_value(user_data["investments"], 0.08, 10),
        "sip": calculate_sip(5000, 0.12, 10),
        "emi": calculate_emi(user_data["loans"], 0.1, 5),
        "hlv": calculate_hlv(user_data["age"], user_data["salary"], user_data["expenses"])
    }
    model_results = get_model_predictions(user_data, portfolio_analysis, spending_analysis)
    report_path = generate_pdf_report(user_data, rule_based_results, model_results, portfolio_analysis, spending_analysis)
    conn = sqlite3.connect("financial_data.db")
    c = conn.cursor()
    c.execute('''INSERT INTO users (name, age, salary, expenses, investments, goals, insurance, loans, portfolio_details, spending_details, report_path, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (name, age, salary, expenses, investments, goals, insurance, loans, portfolio_details, spending_details, report_path, datetime.now().isoformat()))
    user_id = c.lastrowid
    conn.commit()
    conn.close()
    return FileResponse(report_path, media_type="application/pdf", filename=f"{name}_financial_plan.pdf")

# API endpoint for chatbot
@app.post("/chat")
async def chat(message: str = Form(...)):
    conn = sqlite3.connect("financial_data.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users ORDER BY created_at DESC LIMIT 1")
    user = c.fetchone()
    if user:
        user_id = user[0]
        user_data = {
            "name": user[1], "age": user[2], "salary": user[3], "expenses": user[4],
            "investments": user[5], "goals": user[6], "insurance": user[7], "loans": user[8]
        }
        portfolio_details = user[9]
        spending_details = user[10]
        portfolio_analysis = {"total_value": user_data["investments"], "sector_allocation": {}, "volatility": 0.1, "suggestions": []}
        if portfolio_details:
            portfolio_df = pd.read_json(portfolio_details)
            portfolio_analysis = analyze_portfolio(portfolio_df)
        spending_analysis = json.loads(spending_details) if spending_details else {'total_spending': 0, 'category_spending': {}, 'overspending': [], 'ideal_spending': {}}
        response = chatbot_response(message, user_data, portfolio_analysis, spending_analysis, user_id)
        c.execute('''INSERT INTO chat_history (user_id, message, response, timestamp)
                     VALUES (?, ?, ?, ?)''',
                  (user_id, message, response, datetime.now().isoformat()))
        conn.commit()
    else:
        response = "Please provide your financial details to receive personalized advice."
    conn.close()
    return {"response": response}

# Frontend
@app.get("/")
async def get_form():
    html_content = """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FinPT - Financial Planner</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white font-sans p-8">
    <h1 class="text-4xl font-bold mb-6 text-green-400">FinPT - Financial Planner</h1>
    <div class="flex flex-col lg:flex-row gap-8">
        <!-- Form Section -->
        <div class="lg:w-2/3 space-y-4">
            <form action="/generate_plan" method="post" enctype="multipart/form-data" class="space-y-4">
                <div>
                    <label class="block mb-1">Name:</label>
                    <input type="text" name="name" required class="w-full p-2 bg-gray-800 border border-gray-600 rounded">
                </div>
                <div>
                    <label class="block mb-1">Age:</label>
                    <input type="number" name="age" min="18" max="100" required class="w-full p-2 bg-gray-800 border border-gray-600 rounded">
                </div>
                <div>
                    <label class="block mb-1">Monthly Salary (₹):</label>
                    <input type="number" name="salary" step="0.01" min="0" required class="w-full p-2 bg-gray-800 border border-gray-600 rounded">
                </div>
                <div>
                    <label class="block mb-1">Monthly Expenses (₹):</label>
                    <input type="number" name="expenses" step="0.01" min="0" required class="w-full p-2 bg-gray-800 border border-gray-600 rounded">
                </div>
                <div>
                    <label class="block mb-1">Current Investments (₹):</label>
                    <input type="number" name="investments" step="0.01" min="0" required class="w-full p-2 bg-gray-800 border border-gray-600 rounded">
                </div>
                <p class="text-sm text-gray-400">Enter manually or upload portfolio CSV to auto-fill.</p>
                <div>
                    <label class="block mb-1">Financial Goals (comma-separated):</label>
                    <textarea name="goals" required class="w-full p-2 bg-gray-800 border border-gray-600 rounded"></textarea>
                </div>
                <div>
                    <label class="block mb-1">Current Insurance (₹):</label>
                    <input type="number" name="insurance" step="0.01" min="0" required class="w-full p-2 bg-gray-800 border border-gray-600 rounded">
                </div>
                <div>
                    <label class="block mb-1">Current Loans (₹):</label>
                    <input type="number" name="loans" step="0.01" min="0" required class="w-full p-2 bg-gray-800 border border-gray-600 rounded">
                </div>
                <div>
                    <label class="block mb-1">Upload Portfolio (CSV):</label>
                    <input type="file" name="portfolio_file" accept=".csv" class="w-full p-2 bg-gray-800 border border-gray-600 rounded">
                </div>
                <div>
                    <label class="block mb-1">Upload Bank Statement (PDF):</label>
                    <input type="file" name="bank_statement" accept=".pdf" class="w-full p-2 bg-gray-800 border border-gray-600 rounded">
                </div>
                <button type="submit" class="w-full bg-green-500 hover:bg-green-600 text-white p-2 rounded font-bold transition">Generate Financial Plan</button>
            </form>
        </div>

        <!-- Chat Section -->
        <div class="lg:w-1/3 space-y-4">
            <h2 class="text-2xl font-semibold text-green-300">Financial Assistant</h2>
            <div id="chatbox" class="h-96 bg-gray-800 border border-gray-600 p-4 rounded overflow-y-auto space-y-2"></div>
            <input type="text" id="chat_input" placeholder="Enter your financial query" class="w-full p-2 bg-gray-700 border border-gray-500 rounded text-white">
            <button onclick="sendMessage()" class="w-full bg-blue-500 hover:bg-blue-600 p-2 rounded text-white font-semibold transition">Send</button>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('chat_input');
            const chatbox = document.getElementById('chatbox');
            const message = input.value.trim();
            if (!message) return;

            // Add user message
            const userDiv = document.createElement("div");
            userDiv.textContent = message;
            userDiv.className = "message bg-blue-100 text-blue-800 text-right p-2 rounded fade-in self-end";
            chatbox.appendChild(userDiv);

            chatbox.scrollTop = chatbox.scrollHeight;
            input.value = '';

            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: `message=${encodeURIComponent(message)}`
            });

            const data = await response.json();
            const botDiv = document.createElement("div");
            botDiv.textContent = data.response;
            botDiv.className = "message bg-green-100 text-green-900 text-left p-2 rounded fade-in";
            chatbox.appendChild(botDiv);

            chatbox.scrollTop = chatbox.scrollHeight;
        }

        // Allow Enter key to send
        document.getElementById('chat_input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        // Basic fade-in animation
        const style = document.createElement('style');
        style.innerHTML = `
            .fade-in {
                animation: fadeIn 0.5s ease-in-out;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>

    """
    return HTMLResponse(content=html_content)
