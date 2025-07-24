import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Supervised Customer Segmentation", layout="wide")
st.title("🎯 Supervised Customer Segmentation Dashboard")

# Load pretrained model components
@st.cache_resource
def load_assets():
    model = joblib.load("final_segment_model.pkl")
    tfidf = joblib.load("final_tfidf_vectorizer.pkl")
    pca = joblib.load("final_pca_projection.pkl")
    return model, tfidf, pca

model, tfidf, pca = load_assets()
features = ["Age", "Income", "SpendingScore", "PurchaseFrequency"]

# Upload new data
st.header("🔍 Upload New Customer Data")
test_file = st.file_uploader("Upload new customer CSV (no Segment column)", type=["csv"])

if test_file:
    df_test = pd.read_csv(test_file)
    st.subheader("📄 Uploaded Data Preview")
    st.write(df_test.head())

    if all(col in df_test.columns for col in features + ["Review"]):
        # Transform reviews
        review_features = tfidf.transform(df_test["Review"].fillna("")).toarray()
        review_df = pd.DataFrame(review_features, columns=[f"review_tfidf_{i}" for i in range(review_features.shape[1])])
        
        # Combine features
        numeric_features = df_test[features].fillna(0)
        X_new = pd.concat([review_df, numeric_features.reset_index(drop=True)], axis=1)

        # Predict segments
        df_test["PredictedSegment"] = model.predict(X_new)

        # PCA for visualization
        components = pca.transform(X_new)
        df_test["PC1"] = components[:, 0]
        df_test["PC2"] = components[:, 1]

        st.subheader("📊 Predicted Segments")
        st.write(df_test[features + ["Review", "PredictedSegment"]].head())

        st.subheader("🌀 PCA Visualization")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_test["PC1"], df_test["PC2"], c=df_test["PredictedSegment"], cmap="tab10", s=60)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Predicted Customer Segments")
        st.pyplot(fig)
    else:
        st.error(f"CSV must contain: {features + ['Review']}")
