#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Loan Decision Dashboard (Streamlit) - v2

Improvements:
- More readable "Monthly Payment" (no scientific notation; optional scaling)
- Interest rate input accepts either percent (18) or decimal (0.18). The engine also normalizes it.

Run:
  pip install streamlit pandas numpy scikit-learn
  streamlit run loan_web_app_v2.py
"""

from __future__ import annotations

import importlib
import os
import tempfile
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


def import_engine():
    """Import the decision engine from a sibling python file."""
    for name in (
        # Prefer patched versions if present
        "loan_decision_v2_ratefix",
        "loan_decision_v2_fixed_ratefix",
        # Fall back
        "loan_decision_v2",
        "loan_decision_v2_fixed",
    ):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


ENGINE = import_engine()


# -------------------------
# English mappings
# -------------------------
REASON_CODE_TO_EN: Dict[str, str] = {
    # Data quality
    "DQ_AGE_NONPOS": "Invalid age (<= 0)",
    "DQ_INCOME_NEG": "Monthly income is negative",
    "DQ_DEBT_NEG": "Monthly debt is negative",
    "DQ_LOAN_NONPOS": "Requested loan amount must be positive",
    "DQ_TERM_NONPOS": "Loan term must be positive",
    "DQ_EMP_NEG": "Employment years is negative",
    "DQ_CH_NEG": "Credit history months is negative",
    "DQ_LATE_NEG": "Late payments (12m) is negative",
    "DQ_DEFAULT_NOT01": "previous_default must be 0 or 1",
    "DQ_COLL_NEG": "Collateral value is negative",
    "DQ_BANKMONTH_NEG": "months_with_bank is negative",
    "DQ_BAL_NEG": "avg_balance_3m is negative",
    "DQ_BOUNCE_NEG": "num_bounced_checks is negative",
    "DQ_ACTIVELOAN_NEG": "num_active_loans is negative",
    # Data quality -> manual review
    "DQ_AGE_GT100": "Age is unusually high (manual review)",
    "DQ_EMP_GT_AGE": "Employment years inconsistent with age (manual review)",
    "DQ_CH_GT_AGE": "Credit history inconsistent with age (manual review)",
    "DQ_BANKMONTHS_SUSPECT": "months_with_bank looks suspicious (manual review)",
    "DQ_LATE_GT30": "Late payments (12m) unusually high (manual review)",
    # Policy
    "POL_AGE_MIN": "Below minimum allowed age",
    "POL_AGE_MAX": "Above maximum allowed age",
    "POL_INCOME_ZERO": "Monthly income is zero",
    "POL_INCOME_MIN": "Below minimum monthly income",
    "POL_DTI_MAX": "Debt-to-income (DTI) above allowed maximum",
    "POL_PTI_MAX": "Payment-to-income (PTI) above allowed maximum",
    "POL_PREV_DEFAULT": "Previous default on record",
    # Eligibility / capacity
    "ELIG_GRADE_NOT_ALLOWED": "Grade is not allowed for this product",
    "ELIG_NOT_SELECTED_CAPACITY": "Eligible but not selected (capacity limit)",
}

TOP_FACTOR_LABEL_TO_EN: Dict[str, str] = {
    "PTI (قسط/درآمد)": "PTI (payment-to-income)",
    "DTI (تعهدات/درآمد)": "DTI (debt-to-income)",
    "تاخیرهای ۱۲ ماه": "Late payments (12 months)",
    "نسبت وام به درآمد سالانه": "Loan-to-annual-income ratio",
    "مدت وام (سال)": "Loan term (years)",
    "سابقه کار (سال)": "Employment length (years)",
    "سابقه اعتباری (سال)": "Credit history length (years)",
    "پوشش وثیقه": "Collateral coverage",
    "سابقه با بانک (سال)": "Relationship with bank (years)",
    "موجودی ۳ماهه نسبت به درآمد": "Avg 3m balance / income",
    "تعداد چک برگشتی/تراکنش ناموفق": "Bounced checks / failed transactions",
    "تعداد وام‌های فعال": "Active loans count",
    "سن": "Age",
}


def translate_top_risk_factors(factors: List[str]) -> str:
    out: List[str] = []
    for item in factors or []:
        if ":" in item:
            label, rest = item.split(":", 1)
            label = label.strip()
            label_en = TOP_FACTOR_LABEL_TO_EN.get(label, label)
            out.append(f"{label_en}:{rest}")
        else:
            out.append(TOP_FACTOR_LABEL_TO_EN.get(item, item))
    return " | ".join(out)


def reason_messages_from_codes(codes: List[str]) -> str:
    msgs = [REASON_CODE_TO_EN.get(c, c) for c in (codes or [])]
    return " | ".join(msgs)


def save_uploaded_to_tmp(uploaded_file, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def fmt_amount(x: Optional[float], scale: float) -> str:
    """Format an amount nicely, avoiding scientific notation."""
    if x is None:
        return ""
    try:
        v = float(x) / float(scale)
    except Exception:
        return ""
    # No decimals; add thousands separators
    return f"{v:,.0f}"


def df_from_results(apps, results, amount_scale: float) -> pd.DataFrame:
    rows = []
    for a in apps:
        r = results[a.id]
        pay_raw = r.features.get("payment", None)
        rows.append({
            "ID": a.id,
            "Name": a.name,
            "Status": r.decision,  # SELECTED / WAITLIST / REVIEW / REJECT
            "Grade": r.grade,
            "PD": float(r.pd),
            "PD (%)": float(r.pd) * 100.0,
            "Reason Codes": " | ".join(r.reason_codes or []),
            "Reason Messages": reason_messages_from_codes(r.reason_codes),
            "Top Risk Drivers": translate_top_risk_factors(r.top_risk_factors),
            "DTI": r.features.get("dti", None),
            "PTI": r.features.get("pti", None),
            # Display-friendly payment
            "Monthly Payment": fmt_amount(pay_raw, amount_scale),
        })

    df = pd.DataFrame(rows)

    # Stable sorting
    order = {"SELECTED": 0, "WAITLIST": 1, "REVIEW": 2, "REJECT": 3}
    df["_ord"] = df["Status"].map(lambda x: order.get(x, 99))
    df = df.sort_values(by=["_ord", "PD"], ascending=[True, True]).drop(columns=["_ord"])
    return df


def decision_counts(df: pd.DataFrame) -> pd.DataFrame:
    order = ["SELECTED", "WAITLIST", "REVIEW", "REJECT"]
    counts = df["Status"].value_counts().reindex(order).fillna(0).astype(int)
    return counts.rename_axis("Status").reset_index(name="Count")


def main():
    st.set_page_config(page_title="Loan Decision Dashboard", layout="wide")

    st.title("Loan Decision Dashboard")
    st.caption("Upload a CSV/TXT file, set capacity (N), and get results in 4 states: SELECTED / WAITLIST / REVIEW / REJECT.")

    if ENGINE is None:
        st.error(
            "Decision engine module was not found. Place this file next to 'loan_decision_v2.py' (or the patched '*_ratefix.py')."
        )
        st.stop()

    # -------------------------
    # Main inputs (simple)
    # -------------------------
    left, right = st.columns([2, 1], gap="large")

    with left:
        applicants_file = st.file_uploader("Applicants file (CSV or TXT with header)", type=["csv", "txt"])
        n_award = st.number_input(
            "Capacity: number of loans to approve (N)",
            min_value=0,
            value=10,
            step=1,
        )

    # -------------------------
    # Settings (sidebar)
    # -------------------------
    with st.sidebar:
        st.header("Settings")

        st.subheader("Payment calculation")
        annual_rate = st.number_input(
            "Annual interest rate (enter 18 for 18% OR 0.18 for 18%)",
            min_value=0.0,
            max_value=100.0,
            value=18.0,
            step=0.5,
        )

        st.subheader("Display")
        amount_scale = st.number_input(
            "Amount display scale (divide amounts by this factor for display only)",
            min_value=1.0,
            value=1.0,
            step=1.0,
            help="Example: if your file uses Rial but you prefer Toman, set this to 10.",
        )

        allowed_grades_str = st.text_input("Allowed grades (e.g., A,B,C)", value="A,B,C")
        max_dti = st.number_input("Max DTI", min_value=0.0, max_value=2.0, value=0.60, step=0.01)
        max_pti = st.number_input("Max PTI", min_value=0.0, max_value=2.0, value=0.40, step=0.01)
        min_age = st.number_input("Min age", min_value=0, max_value=120, value=18, step=1)
        max_age = st.number_input("Max age", min_value=0, max_value=120, value=70, step=1)
        min_income = st.number_input("Min monthly income", min_value=0.0, value=0.0, step=100000.0)

        reject_prev = st.checkbox("Hard reject when previous_default = 1", value=True)

        st.divider()
        st.subheader("Optional: Train a model")
        st.caption("Upload a labeled historical file (must include 'defaulted_12m') to train Logistic Regression.")
        train_file = st.file_uploader("Historical training file (CSV/TXT)", type=["csv", "txt"], key="train")
        use_training = st.checkbox("Use training file", value=False)

        model_in = st.text_input("Load a saved model (.pkl) (optional)", value="")
        model_out = st.text_input("Save trained model as (.pkl) (optional)", value="")

    # Run
    run = st.button("Run", type="primary", disabled=(applicants_file is None))

    if not run:
        st.info("Step 1: Upload the applicants file • Step 2: Set capacity (N) • Step 3: Click Run")
        st.stop()

    # -------------------------
    # Compute
    # -------------------------
    try:
        applicants_path = save_uploaded_to_tmp(applicants_file, suffix=os.path.splitext(applicants_file.name)[1] or ".csv")
        apps, _missing_optional = ENGINE.read_applicants(applicants_path)

        allowed_grades = [x.strip().upper() for x in allowed_grades_str.split(",") if x.strip()]

        # Pick model: load -> train -> built-in
        if model_in.strip():
            model_obj = ENGINE.TrainedLogitModel.load(model_in.strip())
            st.success(f"Loaded model: {model_in.strip()}")

        elif use_training and train_file is not None:
            train_path = save_uploaded_to_tmp(train_file, suffix=os.path.splitext(train_file.name)[1] or ".csv")
            train_apps, train_labels, _train_fields = ENGINE.read_training_set(train_path)

            model_obj = ENGINE.train_model(
                train_apps=train_apps,
                train_labels=train_labels,
                annual_rate=float(annual_rate),
                max_dti=float(max_dti),
                max_pti=float(max_pti),
                min_age=int(min_age),
                max_age=int(max_age),
                min_income=float(min_income),
                reject_on_previous_default=bool(reject_prev),
            )
            st.success("Model trained from historical data.")

            if model_out.strip():
                model_obj.save(model_out.strip())
                st.success(f"Saved model: {model_out.strip()}")

        else:
            model_obj = ENGINE.BuiltInModel()
            st.info("Using the built-in scoring model (no training file provided).")

        results = ENGINE.decide(
            applicants=apps,
            n_award=int(n_award),
            annual_rate=float(annual_rate),
            allowed_grades=allowed_grades,
            max_dti=float(max_dti),
            max_pti=float(max_pti),
            min_age=int(min_age),
            max_age=int(max_age),
            min_income=float(min_income),
            model_obj=model_obj,
            reject_on_previous_default=bool(reject_prev),
        )

        df = df_from_results(apps, results, amount_scale=float(amount_scale))

        # -------------------------
        # Overview: metrics + chart
        # -------------------------
        counts_df = decision_counts(df)
        counts = {row["Status"]: int(row["Count"]) for _, row in counts_df.iterrows()}

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total applicants", len(df))
        m2.metric("SELECTED", counts.get("SELECTED", 0))
        m3.metric("WAITLIST", counts.get("WAITLIST", 0))
        m4.metric("REVIEW", counts.get("REVIEW", 0))
        m5.metric("REJECT", counts.get("REJECT", 0))

        st.subheader("Overview")
        st.bar_chart(counts_df.set_index("Status"))

        with st.expander("Show grade distribution"):
            grade_counts = df["Grade"].value_counts().sort_index()
            st.bar_chart(grade_counts)

        # -------------------------
        # 4-state output tabs
        # -------------------------
        st.subheader("Results")
        tab_sel, tab_wait, tab_rev, tab_rej = st.tabs(["SELECTED", "WAITLIST", "REVIEW", "REJECT"])

        with tab_sel:
            st.dataframe(df[df["Status"] == "SELECTED"], use_container_width=True, hide_index=True)
        with tab_wait:
            st.dataframe(df[df["Status"] == "WAITLIST"], use_container_width=True, hide_index=True)
        with tab_rev:
            st.dataframe(df[df["Status"] == "REVIEW"], use_container_width=True, hide_index=True)
        with tab_rej:
            st.dataframe(df[df["Status"] == "REJECT"], use_container_width=True, hide_index=True)

        with st.expander("Show ALL results"):
            st.dataframe(df, use_container_width=True, hide_index=True)

        # -------------------------
        # Downloads
        # -------------------------
        st.divider()
        st.subheader("Downloads")

        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        out_tmp.close()
        ENGINE.write_results_csv(out_tmp.name, apps, results)
        with open(out_tmp.name, "rb") as f:
            st.download_button(
                "Download full results.csv",
                data=f.read(),
                file_name="results.csv",
                mime="text/csv",
            )

        def dl_button(label: str, status: str, filename: str):
            st.download_button(
                label,
                data=df[df["Status"] == status].to_csv(index=False).encode("utf-8"),
                file_name=filename,
                mime="text/csv",
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dl_button("Download SELECTED.csv", "SELECTED", "selected.csv")
        with c2:
            dl_button("Download WAITLIST.csv", "WAITLIST", "waitlist.csv")
        with c3:
            dl_button("Download REVIEW.csv", "REVIEW", "review.csv")
        with c4:
            dl_button("Download REJECT.csv", "REJECT", "rejected.csv")

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
