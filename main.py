import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_curve, auc,
    matthews_corrcoef
)

st.set_page_config(page_title="Fraud Detector", layout="centered")
st.title("💳 Credit Card Fraud Detection Using Machine Learning")
st.write("Detect fraud using multiple ML models with visual evaluation, live alerts, and explanations.")

@st.cache_resource
def train_models():
    df = pd.read_csv("creditcard.csv")

    # Balance the dataset
    fraud_df = df[df['Class'] == 1]
    non_fraud_df = df[df['Class'] == 0].sample(n=len(fraud_df), random_state=42)
    balanced_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)
    balanced_df.rename(columns={"Amount": "amt", "Class": "is_fraud"}, inplace=True)

    X = balanced_df.drop('is_fraud', axis=1)
    y = balanced_df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_results = {}

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    lr_metrics = {
        "accuracy": accuracy_score(y_test, lr_pred),
        "confusion": confusion_matrix(y_test, lr_pred),
        "report": classification_report(y_test, lr_pred, output_dict=True),
        "roc": roc_curve(y_test, lr_probs),
        "auc": auc(*roc_curve(y_test, lr_probs)[:2]),
        "mcc": matthews_corrcoef(y_test, lr_pred),
        "model": lr_model
    }

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = {
        "accuracy": accuracy_score(y_test, rf_pred),
        "confusion": confusion_matrix(y_test, rf_pred),
        "report": classification_report(y_test, rf_pred, output_dict=True),
        "roc": roc_curve(y_test, rf_probs),
        "auc": auc(*roc_curve(y_test, rf_probs)[:2]),
        "mcc": matthews_corrcoef(y_test, rf_pred),
        "model": rf_model,
        "feature_importance": pd.Series(rf_model.feature_importances_, index=X.columns)
    }

    # Ensemble
    ensemble_model = VotingClassifier(
        estimators=[('lr', lr_model), ('rf', rf_model)],
        voting='soft'
    )
    ensemble_model.fit(X_train, y_train)
    ensemble_pred = ensemble_model.predict(X_test)
    ensemble_probs = ensemble_model.predict_proba(X_test)[:, 1]
    ensemble_metrics = {
        "accuracy": accuracy_score(y_test, ensemble_pred),
        "confusion": confusion_matrix(y_test, ensemble_pred),
        "report": classification_report(y_test, ensemble_pred, output_dict=True),
        "roc": roc_curve(y_test, ensemble_probs),
        "auc": auc(*roc_curve(y_test, ensemble_probs)[:2]),
        "mcc": matthews_corrcoef(y_test, ensemble_pred),
        "model": ensemble_model
    }

    return {
        "Logistic Regression": lr_metrics,
        "Random Forest": rf_metrics,
        "Ensemble (Voting)": ensemble_metrics
    }, X.columns, X_test, y_test

# Load models and features once (cached)
models, feature_columns, X_test, y_test = train_models()

uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    # Summary table
    comparison_data = []
    for name, data in models.items():
        cm = data['confusion']
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        report = data["report"]
        comparison_data.append({
            "Model": name,
            "Accuracy": data["accuracy"],
            "Precision": report["1"]["precision"],
            "Recall": report["1"]["recall"],
            "F1-Score": report["1"]["f1-score"],
            "Specificity": specificity,
            "MCC": data["mcc"],
            "AUC": data["auc"]
        })

    comparison_df = pd.DataFrame(comparison_data).set_index("Model")
    st.markdown("### 📊 Model Comparison Summary")
    st.dataframe(comparison_df.style.format("{:.2f}"))

    # Visuals
    for name, data in models.items():
        st.markdown(f"#### 🔎 {name}")
        st.markdown(f"**Accuracy**: `{data['accuracy']:.2f}`")

        # Confusion Matrix
        fig, ax = plt.subplots()
        sns.heatmap(data["confusion"], annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        with st.expander("📊 Confusion Matrix"):
            st.pyplot(fig)

        # ROC Curve
        fpr, tpr = data["roc"][0], data["roc"][1]
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {data['auc']:.2f}", color='darkorange')
        ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend(loc="lower right")
        with st.expander("📈 ROC Curve"):
            st.pyplot(fig2)

        if name == "Random Forest":
            st.markdown("#### 🧠 Top Features (Random Forest)")
            imp = data["feature_importance"].sort_values(ascending=False).head(10)
            st.bar_chart(imp)

    # Prediction Interface
    selected_model_name = st.selectbox("🧪 Choose model for prediction:", list(models.keys()))
    selected_model = models[selected_model_name]["model"]
    threshold = st.slider("🎯 Set prediction threshold", 0.0, 1.0, 0.5, step=0.01)

    try:
        input_df = pd.read_csv(uploaded_file)
        input_df.rename(columns={"Amount": "amt", "Class": "is_fraud"}, inplace=True)

        # Ensure column alignment with training features
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Predict
        probs = selected_model.predict_proba(input_df)[:, 1]
        preds = (probs >= threshold).astype(int)

        result_df = input_df.copy()
        result_df["fraud_probability"] = probs.round(2)
        result_df["is_fraud_prediction"] = preds

        if "amt" not in result_df.columns:
            result_df["amt"] = 0.0  # fallback if amount is missing

        st.subheader(f"🔍 Prediction Results ({selected_model_name})")
        st.dataframe(result_df[["amt", "is_fraud_prediction", "fraud_probability"]].head(10))

        # Alerts
        st.subheader("🚨 Fraud Alerts")
        filter_option = st.selectbox(
            "Filter by risk level",
            ["All", "🔴 High Risk (≥ 0.80)", "🟠 Medium Risk (0.50–0.79)", "🟢 Low Risk (< 0.50)"]
        )

        filtered_df = result_df.copy()
        if filter_option == "🔴 High Risk (≥ 0.80)":
            filtered_df = filtered_df[filtered_df["fraud_probability"] >= 0.80]
        elif filter_option == "🟠 Medium Risk (0.50–0.79)":
            filtered_df = filtered_df[(filtered_df["fraud_probability"] >= 0.50) & (filtered_df["fraud_probability"] < 0.80)]
        elif filter_option == "🟢 Low Risk (< 0.50)":
            filtered_df = filtered_df[filtered_df["fraud_probability"] < 0.50]

        if "fraud_display_limit" not in st.session_state or st.session_state.get("last_filter") != filter_option:
            st.session_state.fraud_display_limit = 5
            st.session_state.last_filter = filter_option

        display_limit = st.session_state.fraud_display_limit
        alerts_to_display = filtered_df.head(display_limit)

        for i, row in alerts_to_display.iterrows():
            tx_id = f"Txn #{i + 1}"
            prob = row["fraud_probability"]
            pred = row["is_fraud_prediction"]

            if prob >= 0.80:
                st.error(f"🔴 {tx_id}: **High Risk Fraud** (Prob: `{prob:.2f}`)")
            elif prob >= 0.50:
                st.warning(f"🟠 {tx_id}: **Medium Risk Fraud** (Prob: `{prob:.2f}`)")
            else:
                if pred == 1:
                    st.info(f"🟢 {tx_id}: **Low Risk Fraud** (Prob: `{prob:.2f}`)")
                else:
                    st.success(f"🟢 {tx_id}: Legitimate Transaction (Prob: `{prob:.2f}`)")

        if display_limit < len(filtered_df):
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔽 Show More Alerts"):
                    st.session_state.fraud_display_limit += 5
            with col2:
                if st.button("🔁 Show All Alerts"):
                    st.session_state.fraud_display_limit = len(filtered_df)

        st.markdown(f"**🧾 Total predictions shown:** `{len(filtered_df)}`  &nbsp;&nbsp;&nbsp; 🚨 **Total frauds detected:** `{filtered_df['is_fraud_prediction'].sum()}`")

    except Exception as e:
        st.error(f"❌ Error processing file or predictions: {e}")
 
