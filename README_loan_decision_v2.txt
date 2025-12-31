Loan Decision Engine v2

Files:
- loan_decision_v2.py : main script
- sample_applicants.txt : applicants to score (CSV format)
- sample_historical_loans.txt : labeled historical loans for training (CSV format, includes defaulted_12m)

Quick start:
1) Train a model on historical data and score applicants (select top 8)
   python loan_decision_v2.py sample_applicants.txt 8 --train-file sample_historical_loans.txt --output results.csv

2) Score using built-in fallback model (no training)
   python loan_decision_v2.py sample_applicants.txt 8 --output results.csv

Notes:
- By default, previous_default=1 is a hard reject (policy).
  To allow it (NOT recommended for a typical simple demo), use:
    --allow-previous-default

- Input can be .txt but must be CSV with header.

Columns expected in applicants:
id,name,age,monthly_income,monthly_debt,employment_years,credit_history_months,late_payments_12m,previous_default,loan_amount,loan_term_months
Optional extra columns (recommended):
collateral_value,months_with_bank,avg_balance_3m,num_bounced_checks,num_active_loans

Training file needs all above + defaulted_12m (0/1).
