#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loan Decision Engine v2
- (الف) Explainability: top risk contributions + standardized reasons
- (ج) Real model option: train Logistic Regression on labeled historical data
- (د) Data quality controls: reject / manual-review on inconsistent or missing data
- Extra features added:
    months_with_bank, avg_balance_3m, num_bounced_checks, num_active_loans
Input: CSV text file (can be .txt) with header.
"""

import argparse
import csv
import io
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional sklearn (used for training). If missing, script still works with built-in model.
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    LogisticRegression = None
    StandardScaler = None


# -------------------------
# Utilities
# -------------------------
def sigmoid(x: float) -> float:
    # Numerically stable sigmoid
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)

def normalize_annual_rate(annual_rate: float) -> float:
    """Normalize annual interest rate.

    This engine accepts interest rate in either:
      - Decimal form: 0.18 means 18%
      - Percent form: 18 means 18%

    To reduce common input mistakes, if annual_rate > 2.0 we assume it's a percent
    and convert it by dividing by 100.

    Examples:
      0.18 -> 0.18
      18   -> 0.18
    """
    if annual_rate is None:
        return 0.0
    try:
        ar = float(annual_rate)
    except Exception:
        return 0.0
    if ar > 2.0:
        return ar / 100.0
    return ar




def amortized_monthly_payment(principal: float, annual_rate: float, months: int) -> float:
    """
    قسط ماهانه با فرمول اقساطی (amortized).
    annual_rate مثلا 0.18 یعنی 18%
    """
    if months <= 0:
        return float("inf")
    if principal <= 0:
        return 0.0

    annual_rate = normalize_annual_rate(annual_rate)
    r = annual_rate / 12.0
    if abs(r) < 1e-12:
        return principal / months

    pow_ = (1.0 + r) ** months
    return principal * (r * pow_) / (pow_ - 1.0)


def safe_float(v: str, field: str) -> float:
    if v is None:
        raise ValueError(f"فیلد '{field}' خالی است")
    v = str(v).strip()
    if v == "":
        raise ValueError(f"فیلد '{field}' خالی است")
    try:
        return float(v)
    except Exception as e:
        raise ValueError(f"فیلد '{field}' عدد معتبر نیست: {v!r}") from e


def safe_int(v: str, field: str) -> int:
    if v is None:
        raise ValueError(f"فیلد '{field}' خالی است")
    v = str(v).strip()
    if v == "":
        raise ValueError(f"فیلد '{field}' خالی است")
    try:
        return int(float(v))
    except Exception as e:
        raise ValueError(f"فیلد '{field}' عدد صحیح معتبر نیست: {v!r}") from e


def detect_delimiter(text_sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ","


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -------------------------
# Data models
# -------------------------
@dataclass
class Applicant:
    id: str
    name: str
    age: int
    monthly_income: float
    monthly_debt: float
    employment_years: float
    credit_history_months: int
    late_payments_12m: int
    previous_default: int
    loan_amount: float
    loan_term_months: int

    collateral_value: float = 0.0
    months_with_bank: int = 0
    avg_balance_3m: float = 0.0
    num_bounced_checks: int = 0
    num_active_loans: int = 0


@dataclass
class Result:
    pd: float
    grade: str
    decision: str  # SELECTED / WAITLIST / REJECT / REVIEW
    policy_ok: bool
    eligible: bool
    selected: bool
    reasons: List[str]                # human readable
    reason_codes: List[str]           # standardized
    top_risk_factors: List[str]       # explainability
    features: Dict[str, float]


# -------------------------
# Policy + Data quality checks
# -------------------------
def data_quality_and_policy(
    app: Applicant,
    annual_rate: float,
    max_dti: float,
    max_pti: float,
    min_age: int,
    max_age: int,
    min_income: float,
    reject_on_previous_default: bool = True,
) -> Tuple[str, List[str], List[str], Dict[str, float]]:
    """
    Returns:
      status: "OK" | "REJECT" | "REVIEW"
      reasons (human)
      reason_codes (stable)
      derived features (dti, pti, payment, ...)
    """

    reasons: List[str] = []
    codes: List[str] = []
    review_flags: List[str] = []
    review_codes: List[str] = []

    # Basic validity (hard reject)
    def hard(cond: bool, code: str, msg: str):
        if cond:
            codes.append(code)
            reasons.append(msg)

    def review(cond: bool, code: str, msg: str):
        if cond:
            review_codes.append(code)
            review_flags.append(msg)

    hard(app.age <= 0, "DQ_AGE_NONPOS", "سن نامعتبر (<=0)")
    hard(app.monthly_income < 0, "DQ_INCOME_NEG", "درآمد ماهانه منفی است")
    hard(app.monthly_debt < 0, "DQ_DEBT_NEG", "بدهی/تعهدات ماهانه منفی است")
    hard(app.loan_amount <= 0, "DQ_LOAN_NONPOS", "مبلغ وام باید مثبت باشد")
    hard(app.loan_term_months <= 0, "DQ_TERM_NONPOS", "مدت وام باید مثبت باشد")
    hard(app.employment_years < 0, "DQ_EMP_NEG", "سابقه کار منفی است")
    hard(app.credit_history_months < 0, "DQ_CH_NEG", "سابقه اعتباری منفی است")
    hard(app.late_payments_12m < 0, "DQ_LATE_NEG", "تعداد تاخیر منفی است")
    hard(app.previous_default not in (0, 1), "DQ_DEFAULT_NOT01", "previous_default باید 0 یا 1 باشد")
    hard(app.collateral_value < 0, "DQ_COLL_NEG", "ارزش وثیقه منفی است")
    hard(app.months_with_bank < 0, "DQ_BANKMONTH_NEG", "months_with_bank منفی است")
    hard(app.avg_balance_3m < 0, "DQ_BAL_NEG", "avg_balance_3m منفی است")
    hard(app.num_bounced_checks < 0, "DQ_BOUNCE_NEG", "num_bounced_checks منفی است")
    hard(app.num_active_loans < 0, "DQ_ACTIVELOAN_NEG", "num_active_loans منفی است")

    if reasons:
        return "REJECT", reasons, codes, {}

    # Soft sanity checks (manual review)
    review(app.age > 100, "DQ_AGE_GT100", "سن خیلی بالا است (نیاز به بررسی دستی)")
    review(app.employment_years > max(0, app.age - 14), "DQ_EMP_GT_AGE", "سابقه کار با سن همخوان نیست (نیاز به بررسی دستی)")
    review(app.credit_history_months > app.age * 12, "DQ_CH_GT_AGE", "سابقه اعتباری با سن همخوان نیست (نیاز به بررسی دستی)")
    review(app.months_with_bank > app.credit_history_months + 120, "DQ_BANKMONTHS_SUSPECT", "months_with_bank غیرعادی است (نیاز به بررسی دستی)")
    review(app.late_payments_12m > 30, "DQ_LATE_GT30", "تاخیرهای ۱۲ ماه خیلی زیاد/غیرعادی است (نیاز به بررسی دستی)")

    # Policy checks
    hard(app.age < min_age, "POL_AGE_MIN", f"سن کمتر از حد مجاز ({min_age})")
    hard(app.age > max_age, "POL_AGE_MAX", f"سن بیشتر از حد مجاز ({max_age})")

    if app.monthly_income == 0:
        hard(True, "POL_INCOME_ZERO", "درآمد ماهانه صفر است")
    else:
        hard(app.monthly_income < min_income, "POL_INCOME_MIN", f"درآمد ماهانه کمتر از حداقل تعیین‌شده ({min_income})")

    payment = amortized_monthly_payment(app.loan_amount, annual_rate, app.loan_term_months) if app.monthly_income > 0 else float("inf")
    dti = (app.monthly_debt / app.monthly_income) if app.monthly_income > 0 else 1.0
    pti = (payment / app.monthly_income) if app.monthly_income > 0 else 1.0

    hard(dti > max_dti, "POL_DTI_MAX", f"DTI بالا (تعهدات/درآمد) = {dti:.2f} > {max_dti:.2f}")
    hard(pti > max_pti, "POL_PTI_MAX", f"PTI بالا (قسط/درآمد) = {pti:.2f} > {max_pti:.2f}")

    if reject_on_previous_default:
        hard(app.previous_default == 1, "POL_PREV_DEFAULT", "سابقه نکول/بدحسابی جدی (previous_default=1)")

    derived = {
        "payment": payment,
        "dti": dti,
        "pti": pti,
    }

    if reasons:
        return "REJECT", reasons, codes, derived

    if review_flags:
        # Review beats OK: do not auto-approve
        return "REVIEW", review_flags, review_codes, derived

    return "OK", [], [], derived


# -------------------------
# Feature engineering
# -------------------------
FEATURE_NAMES = [
    "pti",
    "dti",
    "late_payments_12m",
    "loan_to_annual_income",
    "loan_term_years",
    "employment_years",
    "credit_history_years",
    "collateral_coverage",
    "months_with_bank_years",
    "balance_to_income",
    "num_bounced_checks",
    "num_active_loans",
    "age",
]


def build_features(app: Applicant, derived: Dict[str, float]) -> Dict[str, float]:
    monthly_income = max(1.0, app.monthly_income)
    annual_income = monthly_income * 12.0

    pti = clip(float(derived.get("pti", 1.0)), 0.0, 2.0)
    dti = clip(float(derived.get("dti", 1.0)), 0.0, 2.0)

    credit_history_years = clip(app.credit_history_months / 12.0, 0.0, 30.0)
    months_with_bank_years = clip(app.months_with_bank / 12.0, 0.0, 30.0)

    loan_to_annual_income = clip(app.loan_amount / annual_income, 0.0, 20.0)
    loan_term_years = clip(app.loan_term_months / 12.0, 0.1, 15.0)

    collateral_coverage = clip((app.collateral_value / app.loan_amount) if app.loan_amount > 0 else 0.0, 0.0, 5.0)

    balance_to_income = clip(app.avg_balance_3m / monthly_income, 0.0, 50.0)

    feats = {
        "pti": pti,
        "dti": dti,
        "late_payments_12m": clip(float(app.late_payments_12m), 0.0, 60.0),
        "loan_to_annual_income": loan_to_annual_income,
        "loan_term_years": loan_term_years,
        "employment_years": clip(float(app.employment_years), 0.0, 50.0),
        "credit_history_years": credit_history_years,
        "collateral_coverage": collateral_coverage,
        "months_with_bank_years": months_with_bank_years,
        "balance_to_income": balance_to_income,
        "num_bounced_checks": clip(float(app.num_bounced_checks), 0.0, 50.0),
        "num_active_loans": clip(float(app.num_active_loans), 0.0, 50.0),
        "age": clip(float(app.age), 0.0, 120.0),
    }
    return feats


def feats_to_vector(feats: Dict[str, float]) -> np.ndarray:
    return np.array([float(feats[n]) for n in FEATURE_NAMES], dtype=float)


# -------------------------
# Models
# -------------------------
class BuiltInModel:
    """
    مدل داخلیِ دستی (fallback) برای وقتی داده آموزشی نداریم یا sklearn در دسترس نیست.
    """
    def __init__(self):
        # Coefs aligned with FEATURE_NAMES
        self.intercept = -2.0
        self.coefs = np.array([
            3.0,   # pti
            1.8,   # dti
            0.25,  # late_payments_12m
            0.9,   # loan_to_annual_income (raw)
            0.10,  # loan_term_years
            -0.12, # employment_years
            -0.10, # credit_history_years
            -0.70, # collateral_coverage (raw)
            -0.10, # months_with_bank_years
            -0.25, # balance_to_income
            0.35,  # bounced checks
            0.12,  # active loans
            0.01,  # age
        ], dtype=float)

    def predict_pd_and_explain(self, feats: Dict[str, float]) -> Tuple[float, float, List[Tuple[str, float]]]:
        x = feats_to_vector(feats)
        # Use log1p transforms for some heavy-tailed variables to be more stable
        x_t = x.copy()
        # loan_to_annual_income, collateral_coverage, balance_to_income
        idx_loan = FEATURE_NAMES.index("loan_to_annual_income")
        idx_coll = FEATURE_NAMES.index("collateral_coverage")
        idx_bal = FEATURE_NAMES.index("balance_to_income")
        x_t[idx_loan] = math.log1p(x_t[idx_loan])
        x_t[idx_coll] = math.log1p(x_t[idx_coll])
        x_t[idx_bal] = math.log1p(x_t[idx_bal])

        # Apply coefficients (keeping coefs same; treat transformed)
        logit = self.intercept + float(np.dot(self.coefs, x_t))
        pd = sigmoid(logit)

        contribs = [(FEATURE_NAMES[i], float(self.coefs[i] * x_t[i])) for i in range(len(FEATURE_NAMES))]
        contribs.sort(key=lambda t: t[1], reverse=True)
        return pd, logit, contribs


class TrainedLogitModel:
    """
    Logistic Regression + StandardScaler model wrapper.
    Stores:
      scaler mean/std
      coefficients / intercept
    """
    def __init__(self, scaler: StandardScaler, clf: LogisticRegression):
        self.scaler = scaler
        self.clf = clf

    def predict_pd_and_explain(self, feats: Dict[str, float]) -> Tuple[float, float, List[Tuple[str, float]]]:
        x = feats_to_vector(feats).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        logit = float(self.clf.intercept_[0] + np.dot(x_scaled, self.clf.coef_[0])[0])
        pd = float(self.clf.predict_proba(x_scaled)[0, 1])

        contribs = [(FEATURE_NAMES[i], float(self.clf.coef_[0][i] * x_scaled[0, i])) for i in range(len(FEATURE_NAMES))]
        contribs.sort(key=lambda t: t[1], reverse=True)
        return pd, logit, contribs

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "TrainedLogitModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, TrainedLogitModel):
            raise ValueError("فایل مدل معتبر نیست.")
        return obj


def train_model(
    train_apps: List[Applicant],
    train_labels: List[int],
    annual_rate: float,
    max_dti: float,
    max_pti: float,
    min_age: int,
    max_age: int,
    min_income: float,
    reject_on_previous_default: bool,
) -> TrainedLogitModel:
    if LogisticRegression is None or StandardScaler is None:
        raise RuntimeError("برای آموزش مدل، نصب scikit-learn لازم است. (pip install scikit-learn)")

    X: List[np.ndarray] = []
    y: List[int] = []

    # Use only records that pass hard rejects; "REVIEW" can still be used for training (optional)
    dropped = 0
    for app, label in zip(train_apps, train_labels):
        status, reasons, codes, derived = data_quality_and_policy(
            app, annual_rate, max_dti, max_pti, min_age, max_age, min_income, reject_on_previous_default
        )
        if status == "REJECT":
            dropped += 1
            continue
        feats = build_features(app, derived)
        X.append(feats_to_vector(feats))
        y.append(int(label))

    if len(X) < 20:
        raise ValueError(f"داده آموزشی کافی نیست. پس از حذف رکوردهای نامعتبر فقط {len(X)} رکورد باقی مانده.")

    X_mat = np.vstack(X)
    y_vec = np.array(y, dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mat)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_scaled, y_vec)

    return TrainedLogitModel(scaler, clf)


# -------------------------
# Grading + Decisions
# -------------------------
def grade_from_pd(pd: float) -> str:
    # Sample mapping; tune later with real portfolio stats
    if pd < 0.02:
        return "A"
    if pd < 0.05:
        return "B"
    if pd < 0.10:
        return "C"
    if pd < 0.20:
        return "D"
    return "E"


def pretty_feature_name(name: str) -> str:
    mapping = {
        "pti": "PTI (قسط/درآمد)",
        "dti": "DTI (تعهدات/درآمد)",
        "late_payments_12m": "تاخیرهای ۱۲ ماه",
        "loan_to_annual_income": "نسبت وام به درآمد سالانه",
        "loan_term_years": "مدت وام (سال)",
        "employment_years": "سابقه کار (سال)",
        "credit_history_years": "سابقه اعتباری (سال)",
        "collateral_coverage": "پوشش وثیقه",
        "months_with_bank_years": "سابقه با بانک (سال)",
        "balance_to_income": "موجودی ۳ماهه نسبت به درآمد",
        "num_bounced_checks": "تعداد چک برگشتی/تراکنش ناموفق",
        "num_active_loans": "تعداد وام‌های فعال",
        "age": "سن",
    }
    return mapping.get(name, name)


def top_risk_factors(contribs: List[Tuple[str, float]], k: int = 4) -> List[str]:
    # Only positive contributions are "risk increasing"
    positives = [(n, v) for n, v in contribs if v > 0]
    positives.sort(key=lambda t: t[1], reverse=True)
    out = []
    for n, v in positives[:k]:
        out.append(f"{pretty_feature_name(n)}: {v:+.2f}")
    return out


def decide(
    applicants: List[Applicant],
    n_award: int,
    annual_rate: float,
    allowed_grades: List[str],
    max_dti: float,
    max_pti: float,
    min_age: int,
    max_age: int,
    min_income: float,
    model_obj,
    reject_on_previous_default: bool,
) -> Dict[str, Result]:
    results: Dict[str, Result] = {}
    eligible_pool: List[Tuple[str, float]] = []  # (id, pd)

    for app in applicants:
        status, reasons, codes, derived = data_quality_and_policy(
            app, annual_rate, max_dti, max_pti, min_age, max_age, min_income, reject_on_previous_default
        )

        if status == "REJECT":
            results[app.id] = Result(
                pd=1.0,
                grade="E",
                decision="REJECT",
                policy_ok=False,
                eligible=False,
                selected=False,
                reasons=reasons,
                reason_codes=codes,
                top_risk_factors=[],
                features=derived,
            )
            continue

        if status == "REVIEW":
            # Do not auto approve
            results[app.id] = Result(
                pd=1.0,
                grade="E",
                decision="REVIEW",
                policy_ok=False,
                eligible=False,
                selected=False,
                reasons=reasons,
                reason_codes=codes,
                top_risk_factors=[],
                features=derived,
            )
            continue

        feats = build_features(app, derived)
        pd, logit, contribs = model_obj.predict_pd_and_explain(feats)
        grade = grade_from_pd(pd)

        decision_reasons: List[str] = []
        decision_codes: List[str] = []

        eligible = grade in allowed_grades
        if not eligible:
            decision_codes.append("ELIG_GRADE_NOT_ALLOWED")
            decision_reasons.append(f"گرید نامناسب برای این محصول (grade={grade}, allowed={allowed_grades})")

        res = Result(
            pd=pd,
            grade=grade,
            decision="WAITLIST" if eligible else "REJECT",
            policy_ok=True,
            eligible=eligible,
            selected=False,
            reasons=decision_reasons,
            reason_codes=decision_codes,
            top_risk_factors=top_risk_factors(contribs, k=4),
            features={**feats, **derived, "model_logit": logit},
        )
        results[app.id] = res

        if eligible:
            eligible_pool.append((app.id, pd))

    # Select top N by lowest PD
    eligible_pool.sort(key=lambda t: t[1])
    winners = set([app_id for app_id, _ in eligible_pool[:max(0, n_award)]])

    for app_id, _ in eligible_pool:
        if app_id in winners:
            results[app_id].selected = True
            results[app_id].decision = "SELECTED"
        else:
            # Remain WAITLIST
            results[app_id].decision = "WAITLIST"
            results[app_id].reasons.append("واجد شرایط بود ولی به دلیل محدودیت ظرفیت انتخاب نشد")
            results[app_id].reason_codes.append("ELIG_NOT_SELECTED_CAPACITY")

    return results


# -------------------------
# IO
# -------------------------
BASE_REQUIRED_FIELDS = [
    "id", "name", "age", "monthly_income", "monthly_debt", "employment_years",
    "credit_history_months", "late_payments_12m", "previous_default",
    "loan_amount", "loan_term_months",
]

OPTIONAL_FIELDS = [
    "collateral_value", "months_with_bank", "avg_balance_3m", "num_bounced_checks", "num_active_loans"
]


def read_csv_records(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        raw_lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not raw_lines:
        raise ValueError("فایل ورودی خالی است یا فقط کامنت دارد.")

    sample = "".join(raw_lines[:30])
    delim = detect_delimiter(sample)
    reader = csv.DictReader(io.StringIO("".join(raw_lines)), delimiter=delim)

    if reader.fieldnames is None:
        raise ValueError("هدر (ستون‌ها) در فایل پیدا نشد.")

    records = list(reader)
    return records, list(reader.fieldnames)


def parse_applicant(row: Dict[str, str], line_no: int) -> Applicant:
    def get(field: str, default: Optional[str] = None) -> Optional[str]:
        v = row.get(field, default)
        if v is None:
            return None
        return str(v).strip()

    # Required
    try:
        app = Applicant(
            id=get("id") or "",
            name=get("name") or "",
            age=safe_int(get("age"), "age"),
            monthly_income=safe_float(get("monthly_income"), "monthly_income"),
            monthly_debt=safe_float(get("monthly_debt"), "monthly_debt"),
            employment_years=safe_float(get("employment_years"), "employment_years"),
            credit_history_months=safe_int(get("credit_history_months"), "credit_history_months"),
            late_payments_12m=safe_int(get("late_payments_12m"), "late_payments_12m"),
            previous_default=safe_int(get("previous_default"), "previous_default"),
            loan_amount=safe_float(get("loan_amount"), "loan_amount"),
            loan_term_months=safe_int(get("loan_term_months"), "loan_term_months"),
            # Optional (defaults)
            collateral_value=float(get("collateral_value", "0") or 0),
            months_with_bank=int(float(get("months_with_bank", "0") or 0)),
            avg_balance_3m=float(get("avg_balance_3m", "0") or 0),
            num_bounced_checks=int(float(get("num_bounced_checks", "0") or 0)),
            num_active_loans=int(float(get("num_active_loans", "0") or 0)),
        )
    except Exception as e:
        raise ValueError(f"خط {line_no}: خطا در خواندن رکورد: {e}") from e

    if not app.id:
        raise ValueError(f"خط {line_no}: ستون id خالی است")
    if not app.name:
        app.name = f"Applicant_{app.id}"

    return app


def read_applicants(path: str) -> Tuple[List[Applicant], List[str]]:
    records, fieldnames = read_csv_records(path)

    missing = [c for c in BASE_REQUIRED_FIELDS if c not in fieldnames]
    if missing:
        raise ValueError(f"ستون‌های لازم موجود نیستند: {missing}. ستون‌های موجود: {fieldnames}")

    apps: List[Applicant] = []
    for idx, row in enumerate(records, start=2):
        apps.append(parse_applicant(row, idx))
    return apps, fieldnames


def read_training_set(path: str) -> Tuple[List[Applicant], List[int], List[str]]:
    records, fieldnames = read_csv_records(path)
    if "defaulted_12m" not in fieldnames:
        raise ValueError("برای فایل آموزشی ستون defaulted_12m لازم است (0/1).")

    missing = [c for c in BASE_REQUIRED_FIELDS if c not in fieldnames]
    if missing:
        raise ValueError(f"ستون‌های لازم موجود نیستند: {missing}. ستون‌های موجود: {fieldnames}")

    apps: List[Applicant] = []
    labels: List[int] = []
    for idx, row in enumerate(records, start=2):
        apps.append(parse_applicant(row, idx))
        labels.append(safe_int(row.get("defaulted_12m"), "defaulted_12m"))

    return apps, labels, fieldnames


def write_results_csv(path: str, apps: List[Applicant], results: Dict[str, Result]) -> None:
    fieldnames = [
        "id", "name",
        "decision", "selected", "eligible",
        "pd", "grade",
        "policy_ok",
        "reason_codes",
        "reasons",
        "top_risk_factors",
        "pti", "dti", "payment",
        "loan_to_annual_income", "loan_term_years",
        "balance_to_income", "collateral_coverage",
        "model_logit",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for app in apps:
            r = results[app.id]
            w.writerow({
                "id": app.id,
                "name": app.name,
                "decision": r.decision,
                "selected": int(r.selected),
                "eligible": int(r.eligible),
                "pd": f"{r.pd:.6f}",
                "grade": r.grade,
                "policy_ok": int(r.policy_ok),
                "reason_codes": " | ".join(r.reason_codes),
                "reasons": " | ".join(r.reasons),
                "top_risk_factors": " | ".join(r.top_risk_factors),
                "pti": f"{r.features.get('pti', float('nan')):.4f}",
                "dti": f"{r.features.get('dti', float('nan')):.4f}",
                "payment": f"{r.features.get('payment', float('nan')):.2f}",
                "loan_to_annual_income": f"{r.features.get('loan_to_annual_income', float('nan')):.4f}",
                "loan_term_years": f"{r.features.get('loan_term_years', float('nan')):.4f}",
                "balance_to_income": f"{r.features.get('balance_to_income', float('nan')):.4f}",
                "collateral_coverage": f"{r.features.get('collateral_coverage', float('nan')):.4f}",
                "model_logit": f"{r.features.get('model_logit', float('nan')):.4f}",
            })


# -------------------------
# Reporting
# -------------------------
def print_report(applicants: List[Applicant], results: Dict[str, Result], n_award: int) -> None:
    total = len(applicants)
    counts = {"SELECTED": 0, "WAITLIST": 0, "REJECT": 0, "REVIEW": 0}
    grade_counts: Dict[str, int] = {}

    for a in applicants:
        r = results[a.id]
        counts[r.decision] = counts.get(r.decision, 0) + 1
        grade_counts[r.grade] = grade_counts.get(r.grade, 0) + 1

    scored_pds = [results[a.id].pd for a in applicants if results[a.id].policy_ok]
    avg_pd = (sum(scored_pds) / len(scored_pds)) if scored_pds else None

    print("\n====================")
    print("گزارش کلی")
    print("====================")
    print(f"تعداد کل متقاضیان: {total}")
    print(f"ظرفیت اعطا (n_award): {n_award}")
    print(f"SELECTED: {counts.get('SELECTED',0)} | WAITLIST: {counts.get('WAITLIST',0)} | REJECT: {counts.get('REJECT',0)} | REVIEW: {counts.get('REVIEW',0)}")
    if avg_pd is not None:
        print(f"میانگین PD (فقط رکوردهای score شده): {avg_pd*100:.2f}%")
    print("توزیع گریدها:", " | ".join([f"{k}:{v}" for k, v in sorted(grade_counts.items())]))

    winners = [a for a in applicants if results[a.id].decision == "SELECTED"]
    winners.sort(key=lambda a: results[a.id].pd)

    print("\n====================")
    print("افرادی که وام می‌گیرند (SELECTED)")
    print("====================")
    if not winners:
        print("هیچ متقاضی‌ای انتخاب نشد.")
    else:
        for a in winners:
            r = results[a.id]
            print(f"[SELECTED] {a.id} | {a.name} | PD={r.pd*100:.2f}% | Grade={r.grade} | "
                  f"DTI={r.features.get('dti', float('nan')):.2f} | PTI={r.features.get('pti', float('nan')):.2f}")

    # Review
    reviews = [a for a in applicants if results[a.id].decision == "REVIEW"]
    if reviews:
        print("\n====================")
        print("نیازمند بررسی دستی (REVIEW)")
        print("====================")
        for a in reviews:
            r = results[a.id]
            why = " | ".join(r.reasons) if r.reasons else "-"
            print(f"[REVIEW] {a.id} | {a.name} | دلیل: {why}")

    rejects = [a for a in applicants if results[a.id].decision == "REJECT"]
    if rejects:
        print("\n====================")
        print("رد شده‌ها (REJECT)")
        print("====================")
        for a in rejects[:50]:
            r = results[a.id]
            why = " | ".join(r.reasons) if r.reasons else "-"
            print(f"[REJECT] {a.id} | {a.name} | Grade={r.grade} | PD={r.pd*100:.2f}% | دلیل: {why}")
        if len(rejects) > 50:
            print(f"... ({len(rejects)-50} مورد دیگر در فایل خروجی قابل مشاهده است)")

    waitlist = [a for a in applicants if results[a.id].decision == "WAITLIST"]
    if waitlist:
        print("\n====================")
        print("واجد شرایط ولی انتخاب نشده (WAITLIST)")
        print("====================")
        waitlist.sort(key=lambda a: results[a.id].pd)
        for a in waitlist[:50]:
            r = results[a.id]
            print(f"[WAITLIST] {a.id} | {a.name} | PD={r.pd*100:.2f}% | Grade={r.grade}")
        if len(waitlist) > 50:
            print(f"... ({len(waitlist)-50} مورد دیگر در فایل خروجی قابل مشاهده است)")


def main():
    parser = argparse.ArgumentParser(
        description="Loan Decision Engine v2 (Policy + Data Quality + PD model + Grade A-E + Select top N)",
    )
    parser.add_argument("input_file", help="فایل متقاضیان (CSV در قالب txt هم قابل قبول است)")
    parser.add_argument("n_award", type=int, help="ظرفیت اعطای وام (چند نفر انتخاب شوند)")

    parser.add_argument("--annual-rate", type=float, default=0.18, help="نرخ سود سالانه برای محاسبه قسط (پیش‌فرض 0.18)")
    parser.add_argument("--allowed-grades", type=str, default="A,B,C", help="گریدهای قابل قبول (مثلا A,B,C)")
    parser.add_argument("--max-dti", type=float, default=0.60, help="حداکثر DTI (پیش‌فرض 0.60)")
    parser.add_argument("--max-pti", type=float, default=0.40, help="حداکثر PTI (پیش‌فرض 0.40)")
    parser.add_argument("--min-age", type=int, default=18, help="حداقل سن (پیش‌فرض 18)")
    parser.add_argument("--max-age", type=int, default=70, help="حداکثر سن (پیش‌فرض 70)")
    parser.add_argument("--min-income", type=float, default=0.0, help="حداقل درآمد ماهانه (پیش‌فرض 0)")

    # Training / model persistence
    parser.add_argument("--train-file", type=str, default="", help="فایل تاریخی برچسب‌دار برای آموزش (ستون defaulted_12m)")
    parser.add_argument("--model-in", type=str, default="", help="مدل ذخیره‌شده (pickle) را بارگذاری کن")
    parser.add_argument("--model-out", type=str, default="", help="اگر آموزش دادی، مدل را ذخیره کن (pickle)")

    parser.add_argument("--output", type=str, default="", help="اگر بدهی، نتایج در CSV ذخیره می‌شود (مثلا results.csv)")
    parser.add_argument("--allow-previous-default", action="store_true", help="اگر فعال باشد، previous_default=1 رد قطعی نمی‌شود (در غیر این صورت رد می‌شود)")
    args = parser.parse_args()

    allowed_grades = [x.strip().upper() for x in args.allowed_grades.split(",") if x.strip()]

    applicants, in_fields = read_applicants(args.input_file)

    # Warn if optional columns missing (not fatal)
    missing_optional = [c for c in OPTIONAL_FIELDS if c not in in_fields]
    if missing_optional:
        print("⚠️  هشدار: بعضی ستون‌های تکمیلی در فایل ورودی وجود ندارند و با 0 پر می‌شوند:", missing_optional)

    reject_prev = (not bool(args.allow_previous_default))

    model_obj = None
    if args.model_in:
        model_obj = TrainedLogitModel.load(args.model_in)
        print(f"✅ مدل از فایل بارگذاری شد: {args.model_in}")
    elif args.train_file:
        train_apps, train_labels, train_fields = read_training_set(args.train_file)
        model_obj = train_model(
            train_apps=train_apps,
            train_labels=train_labels,
            annual_rate=args.annual_rate,
            max_dti=args.max_dti,
            max_pti=args.max_pti,
            min_age=args.min_age,
            max_age=args.max_age,
            min_income=args.min_income,
            reject_on_previous_default=reject_prev,
        )
        print(f"✅ مدل Logistic Regression با داده تاریخی آموزش داده شد: {args.train_file}")
        if args.model_out:
            model_obj.save(args.model_out)
            print(f"✅ مدل ذخیره شد: {args.model_out}")
    else:
        model_obj = BuiltInModel()
        print("ℹ️  مدل داخلی (Built-in) استفاده شد. برای مدل واقعی‌تر، --train-file بده یا --model-in بارگذاری کن.")

    results = decide(
        applicants=applicants,
        n_award=args.n_award,
        annual_rate=args.annual_rate,
        allowed_grades=allowed_grades,
        max_dti=args.max_dti,
        max_pti=args.max_pti,
        min_age=args.min_age,
        max_age=args.max_age,
        min_income=args.min_income,
        model_obj=model_obj,
        reject_on_previous_default=reject_prev,
    )

    print_report(applicants, results, args.n_award)

    if args.output:
        write_results_csv(args.output, applicants, results)
        print(f"\n✅ فایل خروجی ذخیره شد: {args.output}")


if __name__ == "__main__":
    main()
