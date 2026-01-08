
import mysql.connector
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression


db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Vaishnavi@123',
     database='phonepay_analytics'  
)

cursor = db.cursor()
print("✅ Connected to phonepay_analytics")


cursor.execute("SHOW TABLES")
tables = [t[0] for t in cursor.fetchall()]
print("\n Tables Found:")
for t in tables:
    print(" -", t)



# Folder Setup

os.makedirs("eda_outputs", exist_ok=True)
# Helper Function to run SQL queries
def run_query(query, filename=None):
    cursor.execute(query)
    data = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(data, columns=cols)
    if filename:
        df.to_csv(f"eda_outputs/{filename}.csv", index=False)
        print(f"💾 Saved: eda_outputs/{filename}.csv")
    return df

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


print("\n===== BASIC DATABASE METRICS =====")
metrics = {
    " Users": "SELECT COUNT(*) FROM users;",
    " Transactions": "SELECT COUNT(*) FROM transactions;",
    " Recharges": "SELECT COUNT(*) FROM Recharge_Bills;",
    " Loans": "SELECT COUNT(*) FROM loans;",
    " Insurance": "SELECT COUNT(*) FROM insurance;",
    " Money Transfers": "SELECT COUNT(*) FROM Money_Transfer;"
}
for name, q in metrics.items():
    cursor.execute(q)
    print(f"{name}: {cursor.fetchone()[0]}")



#  EXPLORATORY DATA ANALYSIS (EDA)


print("\n===== EXPLORATORY DATA ANALYSIS (EDA) =====")

users = run_query("SELECT * FROM users;", filename="users_data")
transactions = run_query("SELECT * FROM transactions;", filename="transactions_data")
insurance = run_query("SELECT * FROM insurance;", filename="insurance_data")
loans = run_query("SELECT * FROM loans;", filename="loans_data")
money_transfer = run_query("SELECT * FROM Money_Transfer;", filename="money_transfer_data")
recharge_bills = run_query("SELECT * FROM Recharge_Bills;", filename="recharge_bills_data")

print("\n Data Loaded Successfully from MySQL!")


#  Basic Info and Missing Value Summary

datasets = {
    "Users": users,
    "Transactions": transactions,
    "Insurance": insurance,
    "Loans": loans,
    "Money_Transfer": money_transfer,
    "Recharge_Bills": recharge_bills
}

summary = {}
for name, df in datasets.items():
    summary[name] = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum()
    }

summary_df = pd.DataFrame(summary).T
print("\n===== DATASET SUMMARY =====")
print(summary_df)







#  TRANSACTION ANALYSIS

query = """
SELECT 
    DATE(Date) AS txn_day,
    COUNT(Transaction_ID) AS total_txns,
    SUM(Amount) AS total_amount
FROM transactions
GROUP BY txn_day
ORDER BY txn_day;
"""
txn_trend = run_query(query, "transactions_trend")

sns.lineplot(data=txn_trend, x="txn_day", y="total_amount", label="Revenue")
sns.lineplot(data=txn_trend, x="txn_day", y="total_txns", label="Transactions")
plt.title("Transaction Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/transactions_trend.png")
plt.show()


# Service Type Performance
query = """
SELECT 
    `Service_Type` AS service_type,
    COUNT(*) AS total_count,
    SUM(Amount) AS total_value
FROM transactions
GROUP BY service_type
ORDER BY total_value DESC;
"""
service_perf = run_query(query, "service_type_performance")

sns.barplot(data=service_perf, x="service_type", y="total_value", palette="Blues_d")
plt.title("Transaction Value by Service Type")
plt.xlabel("Service Type")
plt.ylabel("Total Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/service_type.png")
plt.show()

# RECHARGE ANALYSIS

query = """
SELECT 
    Recharge_Type,
    COUNT(*) AS total_recharges,
    SUM(Amount) AS total_value
FROM Recharge_Bills
GROUP BY Recharge_Type
ORDER BY total_value DESC;
"""
recharge = run_query(query, "recharge_summary")

sns.barplot(data=recharge, x="Recharge_Type", y="total_value", palette="coolwarm")
plt.title("Recharge Value by Type")
plt.xlabel("Recharge Type")
plt.ylabel("Total Value")
plt.tight_layout()
plt.savefig("eda_outputs/recharge_type.png")
plt.show()


#  LOAN ANALYSIS

query = """
SELECT 
    Loan_Type,
    COUNT(*) AS total_loans,
    SUM(Loan_Amount) AS total_amount
FROM loans
GROUP BY Loan_Type
ORDER BY total_amount DESC;
"""
loans = run_query(query, "loan_summary")

sns.barplot(data=loans, x="Loan_Type", y="total_amount", palette="crest")
plt.title(" Loan Amount by Type")
plt.xlabel("Loan Type")
plt.ylabel("Total Loan Amount")
plt.tight_layout()
plt.savefig("eda_outputs/loan_type.png")
plt.show()


# INSURANCE ANALYSIS

query = """
SELECT 
    Payment_Status,
    COUNT(*) AS total_policies,
    SUM(Premium) AS total_premium
FROM insurance
GROUP BY Payment_Status;
"""
insurance = run_query(query, "insurance_summary")

sns.barplot(data=insurance, x="Payment_Status", y="total_premium", palette="magma")
plt.title("Insurance Premiums by Payment Status")
plt.xlabel("Payment Status")
plt.ylabel("Total Premium")
plt.tight_layout()
plt.savefig("eda_outputs/insurance_status.png")
plt.show()


# MONEY TRANSFER ANALYSIS

query = """
SELECT 
    Reason,
    COUNT(*) AS total_transfers,
    SUM(Amount) AS total_amount
FROM Money_Transfer
GROUP BY Reason
ORDER BY total_amount DESC;
"""
transfer = run_query(query, "transfer_summary")

sns.barplot(data=transfer, x="Reason", y="total_amount", palette="viridis")
plt.title(" Money Transfers by Reason")
plt.xlabel("Reason")
plt.ylabel("Total Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/money_transfer.png")
plt.show()


# USER ACTIVITY INSIGHT (JOIN + TRANSACTIONS)

query = """
SELECT 
    u.User_ID,
    u.Join_Date,
    COUNT(t.Transaction_ID) AS total_transactions,
    SUM(t.Amount) AS total_spent
FROM users u
LEFT JOIN transactions t ON u.User_ID = t.User_ID
GROUP BY u.User_ID, u.Join_Date
ORDER BY total_spent DESC;
"""
user_activity = run_query(query, "user_activity")

sns.scatterplot(data=user_activity, x="total_transactions", y="total_spent")
plt.title("👥 User Activity: Transactions vs Total Spend")
plt.xlabel("Total Transactions")
plt.ylabel("Total Spend")
plt.tight_layout()
plt.savefig("eda_outputs/user_activity.png")
plt.show()





#  Visualization: Dataset Overview

# Create bar plot
plt.figure(figsize=(10, 5))
ax = sns.barplot(x=summary_df.index, y="Rows", data=summary_df, palette="viridis")

# Add value annotations on top of each bar
for i, v in enumerate(summary_df["Rows"]):
    ax.text(i, v, f'{int(v):,}', 
            horizontalalignment='center',
            verticalalignment='bottom')

plt.title("Number of Records in Each Dataset")
plt.xlabel("Dataset")
plt.ylabel("Row Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/record_count.png")
plt.show()


#  UNIVARIATE ANALYSIS (Distribution of Numeric Columns)

for name, df in datasets.items():
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        print(f"\n📘 Univariate Analysis: {name}")
        for col in numeric_cols[:3]:  # Plot first 3 numeric columns for each dataset
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f"Distribution of {col} - {name}")
            plt.tight_layout()
            plt.savefig(f"eda_outputs/{name}_{col}_distribution.png")
            plt.show()











# PREDICTIVE ANALYSIS USING MACHINE LEARNING


print("\n===== STEP 4: PREDICTIVE ANALYSIS =====")

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler


# DATA PREPARATION


# Example use-case: Predict Transaction Amount using Linear Regression
# (You can replace with your own dependent & independent variables)

transactions = run_query("SELECT * FROM transactions;", filename="transactions_data")
print("\n Transaction data loaded for predictive analysis")

# Drop missing values and duplicates
transactions.dropna(inplace=True)
transactions.drop_duplicates(inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in transactions.select_dtypes(include='object').columns:
    transactions[col] = le.fit_transform(transactions[col].astype(str))

# Display correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(transactions.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix - Transactions")
plt.tight_layout()
plt.savefig("eda_outputs/Transactions_Correlation_Predictive.png")
plt.show()


#  LINEAR REGRESSION MODEL


# Predict 'Amount' based on other numeric/categorical features
if "Amount" in transactions.columns:
    X = transactions.drop("Amount", axis=1)
    y = transactions["Amount"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Predictions
    y_pred = lr_model.predict(X_test)

    # Evaluation
    print("\n===== LINEAR REGRESSION RESULTS =====")
    print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

    # Scatter Plot - Predicted vs Actual
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual Amount")
    plt.ylabel("Predicted Amount")
    plt.title("Actual vs Predicted Transaction Amounts")
    plt.tight_layout()
    plt.savefig("eda_outputs/LinearRegression_Prediction.png")
    plt.show()


# LOGISTIC REGRESSION MODEL


# Example use-case: Predict if a Loan is Approved (binary classification)
loans = run_query("SELECT * FROM loans;", filename="loans_data")
print("\n Loan data loaded for logistic regression")

# Drop missing & duplicates
loans.dropna(inplace=True)
loans.drop_duplicates(inplace=True)

# Encode categorical columns
for col in loans.select_dtypes(include='object').columns:
    loans[col] = le.fit_transform(loans[col].astype(str))

# Check if there's a binary target column
target_cols = [col for col in loans.columns if loans[col].nunique() == 2]
if target_cols:
    target = target_cols[0]  # Pick the first binary column
    print(f"\n Target variable selected for classification: {target}")

    X = loans.drop(target, axis=1)
    y = loans[target]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and fit logistic regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    # Predictions
    y_pred = log_model.predict(X_test)

    # Evaluation
    print("\n===== LOGISTIC REGRESSION RESULTS =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Visualization - Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues")
    plt.title("Loan Approval Prediction - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("eda_outputs/LogisticRegression_ConfusionMatrix.png")
    plt.show()

else:
    print(" No binary target variable found in Loans dataset for logistic regression.")


