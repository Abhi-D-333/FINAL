import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

st.set_page_config(page_title="Supervised Customer Segmentation", layout="wide")
st.title("🎯 Supervised Customer Segmentation Dashboard")

# Load model and transformers
@st.cache_resource
def load_assets():
    required_files = [
        "WATER/final_segment_model.pkl",
        "WATER/final_tfidf_vectorizer.pkl",
        "WATER/final_pca_projection.pkl"
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        st.error(f"❌ The following file(s) are missing: {', '.join(missing)}")
        st.stop()

    model = joblib.load("WATER/final_segment_model.pkl")
    tfidf = joblib.load("WATER/final_tfidf_vectorizer.pkl")
    pca = joblib.load("WATER/final_pca_projection.pkl")
    return model, tfidf, pca

model, tfidf, pca = load_assets()

# Features used
features = ["Age", "Income", "SpendingScore", "PurchaseFrequency"]

# File uploader
st.header("📁 Upload New Customer Data (without Segment)")
test_file = st.file_uploader("Upload test CSV file", type=["csv"])

if test_file:
    df_test = pd.read_csv(test_file)
    st.write(df_test.head())

    required_cols = features + ["Review"]
    if all(col in df_test.columns for col in required_cols):
        # TF-IDF + numeric features
        review_features = tfidf.transform(df_test["Review"].fillna("")).toarray()
        review_df = pd.DataFrame(review_features, columns=[f"review_tfidf_{i}" for i in range(review_features.shape[1])])
        numeric_df = df_test[features].fillna(0)
        X_new = pd.concat([review_df, numeric_df.reset_index(drop=True)], axis=1)

        # Prediction
        df_test["PredictedSegment"] = model.predict(X_new)

        # PCA Projection
        components = pca.transform(X_new)
        df_test["PC1"] = components[:, 0]
        df_test["PC2"] = components[:, 1]

        # Results
        st.subheader("📊 Predicted Segments")
        st.write(df_test[features + ["Review", "PredictedSegment"]].head())

        st.subheader("🌀 Segment Visualization (PCA)")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_test["PC1"], df_test["PC2"], c=df_test["PredictedSegment"], cmap="tab10", s=60)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Customer Segments (PCA Projection)")
        st.pyplot(fig)
    else:
        st.error(f"CSV must contain columns: {required_cols}")
