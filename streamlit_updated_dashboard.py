
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Title
st.set_page_config(page_title="AI-Based Intrusion Detection", layout="wide")
st.title("AI-Based Intrusion Detection Dashboard")
st.write("Analyze and detect anomalies in network traffic.")

# File Upload Section
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read uploaded dataset
    data = pd.read_csv(uploaded_file)
    st.sidebar.write("Dataset Preview:")
    st.sidebar.dataframe(data.head())

    # Check for label column
    potential_labels = ['Label', 'Target', 'Class']
    label_column = None
    for col in potential_labels:
        if col in data.columns:
            label_column = col
            break

    if label_column is None:
        st.sidebar.error("No label column detected! Please select one.")
        label_column = st.sidebar.selectbox("Select the label column:", data.columns)

    if label_column:
        st.sidebar.success(f"Using '{label_column}' as the label column.")

        # Display Summary
        st.header("Dataset Summary")
        st.write(f"**Total Records**: {len(data)}")
        st.write("**Column Information**:")
        st.write(data.describe())

        # Visualize Distribution of Labels
        st.header("Label Distribution")
        label_dist = data[label_column].value_counts()
        fig = px.pie(values=label_dist, names=label_dist.index, title="Label Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Train a RandomForest Model (Demo)
        X = data.drop(columns=[label_column], errors='ignore')
        y = data[label_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Simulated model (replace with your trained model)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Show Results in a Table
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': predictions
        })
        st.header("Prediction Results")
        st.dataframe(results_df.head())

        # Confusion Matrix Visualization
        from sklearn.metrics import confusion_matrix
        import plotly.figure_factory as ff

        cm = confusion_matrix(y_test, predictions)
        cm_fig = ff.create_annotated_heatmap(
            z=cm, x=['Normal', 'Malicious'], y=['Normal', 'Malicious'], colorscale="Viridis"
        )
        st.header("Confusion Matrix")
        st.plotly_chart(cm_fig, use_container_width=True)
    else:
        st.error("No valid label column selected. Please upload a dataset with labels.")
else:
    st.info("Please upload a dataset to start.")
