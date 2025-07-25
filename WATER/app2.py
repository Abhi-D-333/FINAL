import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Supervised Customer Segmentation", layout="wide")
st.title("🎯 Supervised Customer Segmentation Dashboard")

# -----------------------
# 📦 Load pretrained model, vectorizer, and PCA
# -----------------------
def load_assets():
    required_files = [
        "final_segment_model.pkl",
        "final_tfidf_vectorizer.pkl",
        "final_pca_projection.pkl"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"❌ The following file(s) are missing: {', '.join(missing_files)}")
        st.stop()

    model = joblib.load("final_segment_model.pkl")
    tfidf = joblib.load("final_tfidf_vectorizer.pkl")
    pca = joblib.load("final_pca_projection.pkl")
    return model, tfidf, pca

model, tfidf, pca = load_assets()

# -----------------------
# 🔍 Define features used during training
# -----------------------
features = ["Age", "Income", "SpendingScore", "PurchaseFrequency"]

# -----------------------
# 📁 Upload new test data
# -----------------------
st.header("📁 Upload New Customer Data (No Segment Column Required)")
test_file = st.file_uploader("Upload test CSV file", type=["csv"])

if test_file:
    df_test = pd.read_csv(test_file)
    st.write(df_test.head())

    required_cols = features + ["Review"]
    if all(col in df_test.columns for col in required_cols):
        # Preprocess reviews
        review_features = tfidf.transform(df_test["Review"].fillna("")).toarray()
        review_df = pd.DataFrame(review_features, columns=[f"review_tfidf_{i}" for i in range(review_features.shape[1])])

        # Preprocess numeric data
        numeric_df = df_test[features].fillna(0)
        X_new = pd.concat([review_df, numeric_df.reset_index(drop=True)], axis=1)

        # Predict segments
        predictions = model.predict(X_new)
        df_test["PredictedSegment"] = predictions

        # Visualize via PCA
        components = pca.transform(X_new)
        df_test["PC1"] = components[:, 0]
        df_test["PC2"] = components[:, 1]

        # Display prediction results
        st.subheader("📊 Predicted Segments")
        st.write(df_test[features + ["Review", "PredictedSegment"]].head())

        # Plot PCA projection
        st.subheader("🌀 PCA Cluster Visualization")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_test["PC1"], df_test["PC2"], c=df_test["PredictedSegment"], cmap="tab10", s=60)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Customer Segments (PCA View)")
        st.pyplot(fig)
    else:
        st.error(f"❌ CSV must contain the following columns: {required_cols}")
