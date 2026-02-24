import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Recruitment Predictor", page_icon="🧠", layout="wide")

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_recruitment_model.joblib"
SUMMARY_PATH = ARTIFACTS_DIR / "model_summary.json"
DATA_DIR = Path("data")
RETRAINING_DATA_PATH = DATA_DIR / "retraining_data.csv"

REQUIRED_BASE_COLUMNS = [
    "Age",
    "Gender",
    "EducationLevel",
    "ExperienceYears",
    "PreviousCompanies",
    "DistanceFromCompany",
    "InterviewScore",
    "SkillScore",
    "PersonalityScore",
    "RecruitmentStrategy",
]

GENDER_TO_CODE = {
    "Female": 0,
    "Male": 1,
    "Other": 2,
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_summary():
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ExperienceBucket"] = pd.cut(
        out["ExperienceYears"],
        bins=[-1, 2, 5, 10, float("inf")],
        labels=["Entry", "Junior", "Mid", "Senior"],
    )
    out["ExperienceBucketCode"] = out["ExperienceBucket"].cat.codes

    out["TechnicalStageScore"] = 0.6 * out["SkillScore"] + 0.4 * out["InterviewScore"]
    out["BehavioralStageScore"] = 0.7 * out["PersonalityScore"] + 0.3 * out["InterviewScore"]
    out["OverallCompositeScore"] = (
        0.4 * out["InterviewScore"]
        + 0.35 * out["SkillScore"]
        + 0.25 * out["PersonalityScore"]
    )

    return out


def validate_base_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    return missing


def normalize_gender(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Gender" not in out.columns:
        return out

    if pd.api.types.is_numeric_dtype(out["Gender"]):
        out["Gender"] = pd.to_numeric(out["Gender"], errors="coerce").fillna(1).astype(int)
        return out

    gender_map_ci = {k.lower(): v for k, v in GENDER_TO_CODE.items()}
    out["Gender"] = (
        out["Gender"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(gender_map_ci)
        .fillna(1)
        .astype(int)
    )
    return out


def predict_dataframe(model, df_base: pd.DataFrame) -> pd.DataFrame:
    df_clean = normalize_gender(df_base)
    df_features = compute_derived_features(df_clean)

    preds = model.predict(df_features)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df_features)[:, 1]
    else:
        probs = [None] * len(df_features)

    result = df_clean.copy()
    result["PredictedHiringDecision"] = preds
    result["HiringProbability"] = probs
    return result


def append_retraining_data(df_pred: pd.DataFrame, source: str) -> int:
    DATA_DIR.mkdir(exist_ok=True)
    out = df_pred.copy()
    out["DataSource"] = source
    out["CollectedAtUTC"] = datetime.utcnow().isoformat(timespec="seconds")
    out["ActualHiringDecision"] = pd.NA

    header = not RETRAINING_DATA_PATH.exists()
    out.to_csv(RETRAINING_DATA_PATH, mode="a", header=header, index=False)
    return len(out)


st.title("Recruitment Predictor")
st.caption("Predict hiring decision using the trained model from this project.")

if not MODEL_PATH.exists():
    st.error(f"Model not found at: {MODEL_PATH}")
    st.stop()

model = load_model()
summary = load_summary()

if "single_output_df" not in st.session_state:
    st.session_state["single_output_df"] = None
if "batch_output_df" not in st.session_state:
    st.session_state["batch_output_df"] = None

with st.expander("Model Details", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Selected Model", summary.get("selected_model_label", "N/A"))
    col_b.metric("Selection Metric", summary.get("selection_metric", "N/A"))
    col_c.metric("Test F1", summary.get("test_f1", "N/A"))

st.markdown("---")

single_tab, batch_tab = st.tabs(["Single Candidate", "Batch CSV"])

with single_tab:
    st.subheader("Single Candidate Prediction")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
        gender_label = st.selectbox("Gender", options=list(GENDER_TO_CODE.keys()), index=1)
        education = st.selectbox("EducationLevel (encoded)", options=[1, 2, 3, 4], index=1)
        exp_years = st.number_input("ExperienceYears", min_value=0, max_value=40, value=5, step=1)

    with c2:
        prev_companies = st.number_input("PreviousCompanies", min_value=0, max_value=20, value=2, step=1)
        distance = st.number_input("DistanceFromCompany", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        interview = st.number_input("InterviewScore", min_value=0, max_value=100, value=70, step=1)

    with c3:
        skill = st.number_input("SkillScore", min_value=0, max_value=100, value=75, step=1)
        personality = st.number_input("PersonalityScore", min_value=0, max_value=100, value=72, step=1)
        strategy = st.selectbox("RecruitmentStrategy (encoded)", options=[1, 2, 3], index=0)

    if st.button("Predict Candidate", type="primary"):
        base_df = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": GENDER_TO_CODE[gender_label],
                    "EducationLevel": education,
                    "ExperienceYears": exp_years,
                    "PreviousCompanies": prev_companies,
                    "DistanceFromCompany": distance,
                    "InterviewScore": interview,
                    "SkillScore": skill,
                    "PersonalityScore": personality,
                    "RecruitmentStrategy": strategy,
                }
            ]
        )
        st.session_state["single_output_df"] = predict_dataframe(model, base_df)

    if st.session_state["single_output_df"] is not None:
        out = st.session_state["single_output_df"]
        pred = int(out.loc[0, "PredictedHiringDecision"])
        prob = float(out.loc[0, "HiringProbability"])

        st.success(f"Predicted HiringDecision: {pred}")
        st.metric("Hiring Probability", f"{prob:.3f}")
        st.dataframe(out, use_container_width=True)

        if st.button("Save Single Record for Retraining", key="save_single"):
            saved_rows = append_retraining_data(out, source="single_form")
            st.success(f"Saved {saved_rows} row to {RETRAINING_DATA_PATH}")

with batch_tab:
    st.subheader("Batch Prediction from CSV")
    st.write("Upload a CSV with the base columns used in training.")
    st.code(
        ", ".join(REQUIRED_BASE_COLUMNS),
        language="text",
    )

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        input_df = pd.read_csv(file)
        missing_cols = validate_base_columns(input_df)

        if missing_cols:
            st.error("Missing required columns: " + ", ".join(missing_cols))
        else:
            st.session_state["batch_output_df"] = predict_dataframe(model, input_df[REQUIRED_BASE_COLUMNS])

    if st.session_state["batch_output_df"] is not None:
        output_df = st.session_state["batch_output_df"]
        st.success(f"Predictions completed for {len(output_df)} rows.")
        st.dataframe(output_df.head(50), use_container_width=True)

        csv_bytes = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions CSV",
            data=csv_bytes,
            file_name="recruitment_predictions.csv",
            mime="text/csv",
        )

        if st.button("Save Batch Records for Retraining", key="save_batch"):
            saved_rows = append_retraining_data(output_df, source="batch_csv")
            st.success(f"Saved {saved_rows} rows to {RETRAINING_DATA_PATH}")

st.markdown("---")
st.caption("Tip: run with `streamlit run app.py`")
st.caption(f"Retraining data target: {RETRAINING_DATA_PATH.resolve()}")
if RETRAINING_DATA_PATH.exists():
    retrain_df = pd.read_csv(RETRAINING_DATA_PATH)
    st.success(f"Retraining data available: {len(retrain_df)} rows")
    st.dataframe(retrain_df.tail(20), use_container_width=True)
    st.download_button(
        label="Download Retraining Data CSV",
        data=retrain_df.to_csv(index=False).encode("utf-8"),
        file_name="retraining_data.csv",
        mime="text/csv",
    )
else:
    st.warning("No retraining_data.csv yet. Save a single or batch prediction to create it.")
