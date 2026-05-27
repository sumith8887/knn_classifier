import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Iris KNN Classifier", page_icon="🌸", layout="wide")

st.title("🌸 Iris Classification using KNN")
st.caption("A simple Streamlit app for the Iris dataset using K-Nearest Neighbors.")

@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["target_name"] = df["target"].map({i: name for i, name in enumerate(iris.target_names)})
    return df, iris

df, iris = load_data()

feature_names = iris.feature_names
target_names = iris.target_names

X = df[feature_names]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)
cm = confusion_matrix(y_test, y_pred)

col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    st.subheader("Model Performance")
    st.metric("Accuracy", f"{accuracy:.2%}")
    st.text(report)
    st.write("Confusion Matrix")
    st.dataframe(
        pd.DataFrame(cm, index=target_names, columns=target_names),
        use_container_width=True
    )

st.divider()
st.subheader("Make a Prediction")

c1, c2 = st.columns(2)

with c1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5, step=0.1)

with c2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1)

input_data = pd.DataFrame([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]], columns=feature_names)

if st.button("Predict Species"):
    pred_class = model.predict(input_data)[0]
    pred_name = target_names[pred_class]
    pred_proba = model.predict_proba(input_data)[0]

    st.success(f"Predicted Iris Species: **{pred_name}**")

    proba_df = pd.DataFrame({
        "Species": target_names,
        "Probability": pred_proba
    })
    st.bar_chart(proba_df.set_index("Species"))

st.caption("Educational demo only.")