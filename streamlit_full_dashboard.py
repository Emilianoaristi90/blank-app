
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import plotly.figure_factory as ff

# Page Configuration
st.set_page_config(page_title="Advanced Intrusion Detection Dashboard", layout="wide")
st.title("Advanced Intrusion Detection Dashboard")
st.write("A comprehensive tool for analyzing and detecting anomalies in network traffic.")

# Sidebar: File Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read uploaded dataset
    data = pd.read_csv(uploaded_file)
    st.sidebar.write("Dataset Preview:")
    st.sidebar.dataframe(data.head())

    # Detect or select label column
    potential_labels = ['Label', 'Target', 'Class', 'attack_cat']
    label_column = None
    for col in potential_labels:
        if col in data.columns:
            label_column = col
            break

    if label_column is None:
        st.sidebar.error("No label column detected! Please select one.")
        label_column = st.sidebar.selectbox("Select the label column:", data.columns)
    else:
        st.sidebar.success(f"Using '{label_column}' as the label column.")

    if label_column:
        # Dataset Summary
        st.header("Dataset Summary")
        st.write(f"**Total Records**: {len(data)}")
        st.write("**Column Information**:")
        st.write(data.describe())

        # Visualize Label Distribution
        st.header("Label Distribution")
        label_dist = data[label_column].value_counts()
        fig = px.pie(values=label_dist, names=label_dist.index, title="Label Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Advanced Visualization: Correlation Heatmap
        st.header("Feature Correlations")
        corr = data.select_dtypes(include='number').corr()
        corr_fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(corr_fig, use_container_width=True)

        # Train-Test Split and Model Training
        X = data.drop(columns=[label_column], errors='ignore')
        y = data[label_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Show Results
        st.header("Model Predictions")
        st.write("Classification Report:")
        report = classification_report(y_test, predictions, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion Matrix
        st.header("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        cm_fig = ff.create_annotated_heatmap(
            z=cm, x=['Normal', 'Malicious'], y=['Normal', 'Malicious'], colorscale="Viridis"
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        # Simulated Real-Time Data Visualization
        st.header("Real-Time Traffic Monitoring")
        simulated_traffic = pd.DataFrame({
            'Timestamp': pd.date_range(start='2024-01-01', periods=len(X_test), freq='S'),
            'Predicted': predictions
        })
        fig = px.scatter(simulated_traffic, x='Timestamp', y='Predicted',
                         title="Simulated Real-Time Predictions",
                         labels={'Predicted': 'Traffic Type (0=Normal, 1=Malicious)'})
        st.plotly_chart(fig, use_container_width=True)

        # Generate Report Option
        st.header("Generate Report")
        if st.button("Download Report"):
            report_df = pd.DataFrame(report).transpose()
            report_csv = report_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download CSV Report",
                data=report_csv,
                file_name="classification_report.csv",
                mime="text/csv",
            )
else:
    st.info("Please upload a dataset to start.")
