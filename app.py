import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Define a dictionary of evaluation metrics for regression and classification
regression_metrics = {
    "Mean Squared Error": mean_squared_error,
    "R-squared (R2)": r2_score,
    "Root Mean Squared Error (RMSE)": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "Mean Absolute Error (MAE)": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
    "Explained Variance Score": lambda y_true, y_pred: r2_score(y_true, y_pred),
}

classification_metrics = {
    "Accuracy": accuracy_score,
    "F1 Score": f1_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "AUC-ROC Score": roc_auc_score,
}

# Add colorful styles
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f5f5;
        animation: fadeIn 2s;
    }
    .sidebar .sidebar-content {
        background-color: #e0f2f1;
    }
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f5f5;
        animation: fadeIn 2s;
    }
    .sidebar .sidebar-content {
        background-color: #e0f2f1;
    }
    .contact-info {
        position: absolute;
        top: 10px;
        right: 10px;
        display: flex;
        flex-direction: row;
    }
    .contact-info a {
        margin-left: 10px;
        text-decoration: none;
    }
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add emojis and styling to the title
st.title("🚀 Streamlit EDA and Machine Learning App By Ali Shan📊🤖")

def main():
     # Contact Info
    st.markdown(
        """
        <div class="contact-info">
            <a href="mailto:shan42691@gmail.com" target="_blank">
                <img src="https://logos-world.net/wp-content/uploads/2020/11/Gmail-Logo.png" alt="Email" width="30" height="30">
            </a>
            <a href="https://github.com/Ali1-Shan" target="_blank">
                <img src="https://th.bing.com/th/id/R.7a864f07681f187fb572468bfc949977?rik=3fUik6Pc6xTrHQ&pid=ImgRaw&r=0" alt="GitHub" width="30" height="30">
            </a>
            <a href="https://www.facebook.com/profile.php?id=100016971300759" target="_blank">
                <img src="https://th.bing.com/th/id/OIP.PbFI2-A9F5qEJPkicFC-ZwHaHa?pid=ImgDet&w=600&h=600&rs=1" alt="Facebook" width="30" height="30">
            </a>
            <a href="https://www.linkedin.com/in/ali-shan-176871267/" target="_blank">
                <img src="https://th.bing.com/th/id/OIP.Cverxr-lN_3QjtMAqJFQYwHaEK?pid=ImgDet&rs=1" alt="LinkedIn" width="30" height="30">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
   

    # Initialize data and X
    data = None
    X = None

    # Upload dataset or select sample data
    data_source = st.sidebar.radio("Select Data Source", ["Upload Dataset", "Use Sample Data"])
    if data_source == "Upload Dataset":
        uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
    else:
        sample_dataset_name = st.sidebar.selectbox("Select a sample dataset", ["tips", "iris"])
        data = sns.load_dataset(sample_dataset_name)

    if data is not None:

        # EDA Section
        if st.sidebar.checkbox("Do you want to perform basic EDA analysis?"):
            st.write("### Dataset Overview:")
            st.write(data.head())
            st.write("### Basic Statistics:")
            st.write(data.describe())
            if len(data.columns) < 5:
                st.write("### Pairplot:")
                fig = sns.pairplot(data)
                st.pyplot(fig)
            else:
                st.write("### Correlation Matrix:")
                numeric_data = data.select_dtypes(include=[np.number])
                corr_matrix = numeric_data.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, ax=ax, cmap="coolwarm")
                st.pyplot(fig)

     # Plotting Section
    if st.sidebar.checkbox("Do you want to plot the data?"):
        st.sidebar.subheader("Chart Type")
        chart_type = st.sidebar.selectbox("Select chart type", ["Scatter Plot", "Bar Chart", "Box Plot", "Violin Plot", "Histogram"])

        if chart_type == "Scatter Plot":
            st.sidebar.subheader("Select Columns")
            num_of_cols = st.sidebar.selectbox("Select number of columns to plot", [1, 2, 3, "you can only draw upto 3 in Box plot According to Andrew Abela Guide"])
            
            if num_of_cols == 1:
                single_col = st.sidebar.selectbox("Select a column to plot", data.columns)
                st.warning("Andrew Abela's Chart Suggestion: Use a Single Variable Plot")
                if single_col:
                    fig = px.scatter(data, x=single_col, title=f"Scatter Plot of {single_col}")
                    st.plotly_chart(fig)
            elif num_of_cols == 2:
                col1 = st.sidebar.selectbox("Select the first column", data.columns)
                col2 = st.sidebar.selectbox("Select the second column", data.columns)
                st.warning("Andrew Abela's Chart Suggestion: Use a Scatter Plot")
                if col1 and col2:
                    fig = px.scatter(data, x=col1, y=col2, title=f"Scatter Plot of {col1} vs {col2}")
                    st.plotly_chart(fig)
            elif num_of_cols == 3:
                col1 = st.sidebar.selectbox("Select the first column", data.columns)
                col2 = st.sidebar.selectbox("Select the second column", data.columns)
                col3 = st.sidebar.selectbox("Select the third column", data.columns)
                st.warning("Andrew Abela's Chart Suggestion: Consider 3D Scatter Plot")
                if col1 and col2 and col3:
                    fig = px.scatter_3d(data, x=col1, y=col2, z=col3, title=f"3D Scatter Plot of {col1}, {col2}, {col3}")
                    st.plotly_chart(fig)
            else:
                st.warning("Please select an appropriate chart type.")
        
        elif chart_type == "Bar Chart":
            st.sidebar.subheader("Select Columns")
            num_of_cols = st.sidebar.selectbox("Select number of columns to plot", [1, 2, "you can only draw upto 2 in Bar  plot According to Andrew Abela Guide"])
            
            if num_of_cols == 1:
                single_col = st.sidebar.selectbox("Select a column to plot", data.columns)
                st.warning("Andrew Abela's Chart Suggestion: Use a Bar Chart")
                if single_col:
                    fig = px.bar(data, x=single_col, title=f"Bar Chart of {single_col}")
                    st.plotly_chart(fig)
            elif num_of_cols == 2:
                col1 = st.sidebar.selectbox("Select the first column", data.columns)
                col2 = st.sidebar.selectbox("Select the second column", data.columns)
                st.warning("Andrew Abela's Chart Suggestion: Use a Grouped Bar Chart")
                if col1 and col2:
                    fig = px.bar(data, x=col1, y=col2, title=f"Bar Chart of {col1} vs {col2}")
                    st.plotly_chart(fig)
            else:
                st.warning("Please select an appropriate chart type.")

        elif chart_type == "Box Plot":
            st.sidebar.subheader("Select Columns")
            num_of_cols = st.sidebar.selectbox("Select number of columns to plot", [1, "you can only draw upto 1 in Box plot According to Andrew Abela Guide"])
            
            if num_of_cols == 1:
                single_col = st.sidebar.selectbox("Select a column to plot", data.columns)
                if single_col is not None:  # Check if a column is selected
                    st.warning("Andrew Abela's Chart Suggestion: Use a Box Plot")
                    fig = px.box(data, x=single_col, title=f"Box Plot of {single_col}")
                    st.plotly_chart(fig)

        elif chart_type == "Violin Plot":
            st.sidebar.subheader("Select Columns")
            num_of_cols = st.sidebar.selectbox("Select number of columns to plot", [1, "you can only draw upto 1  in Violin plot According to Andrew Abela Guide"])
            
            if num_of_cols == 1:
                single_col = st.sidebar.selectbox("Select a column to plot", data.columns)
                st.warning("Andrew Abela's Chart Suggestion: Use a Violin Plot")
                if single_col:
                    fig = px.violin(data, x=single_col, title=f"Violin Plot of {single_col}")
                    st.plotly_chart(fig)
            else:
                st.warning("Please select a chart type for more than 1 column.")

        elif chart_type == "Histogram":
            st.sidebar.subheader("Select Columns")
            num_of_cols = st.sidebar.selectbox("Select number of columns to plot", [1, "you can only draw upto 1 in Histogram According to Andrew Abela Guide"])
            
            if num_of_cols == 1:
                single_col = st.sidebar.selectbox("Select a column to plot", data.columns)
                st.warning("Andrew Abela's Chart Suggestion: Use a Histogram")
                if single_col:
                    fig = px.histogram(data, x=single_col, title=f"Histogram of {single_col}")
                    st.plotly_chart(fig)
            else:
                st.warning("Please select an appropriate chart type.")
    

    # ML Task Selection
    if st.sidebar.checkbox("Do you want to perform a machine learning task?"):
        st.sidebar.subheader("Machine Learning Tasks")
        X_columns = st.sidebar.multiselect("Select feature columns (X)", data.columns)
        y_column = st.sidebar.selectbox("Select target column (y)", data.columns)

        # Encoding Section
        st.sidebar.subheader("Data Encoding")
        st.write(data.dtypes)
        categorical_cols = data[X_columns].select_dtypes(include=['object', 'category']).columns.tolist()
        encoding_method = st.sidebar.selectbox("Select an encoding method", ["None", "Label Encoding", "One-Hot Encoding"])

        if encoding_method == "Label Encoding":
            label_encoder = LabelEncoder()
            for col in categorical_cols:
                data[col] = label_encoder.fit_transform(data[col])
            st.write("Data after Label Encoding:")
            st.write(data.head())
        elif encoding_method == "One-Hot Encoding":
            data = pd.get_dummies(data, columns=categorical_cols)
            st.write("Data after One-Hot Encoding:")
            st.write(data.head())

        # Train-test split ratio
        split_ratio = st.sidebar.slider("Select train-test split ratio (%)", 10, 90, 80)
        st.sidebar.text(f"Train set size: {split_ratio}%")
        st.sidebar.text(f"Test set size: {100-split_ratio}%")

        if y_column in data.columns:
            task_type = st.sidebar.radio("Select Task Type", ["Regression", "Classification"])
            X = data[X_columns]  # Assign X here
            y = data[y_column]

            # Scaling
            if X is not None and X.shape[0] > 0:  # Check if X is not None and not empty
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            # Splitting data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_ratio) / 100.0, random_state=42)

            # Model selection and evaluation
            if task_type == "Regression":
                st.sidebar.warning("This is a Regression Problem!")

                # Regression models selection
                regression_models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge Regression": Ridge(),
                    "Lasso Regression": Lasso(),
                    "Support Vector Regression": SVR(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "Gradient Boosting Regressor": GradientBoostingRegressor(),
                }

                selected_regression_models = st.sidebar.multiselect("Select Regression Models", list(regression_models.keys()))

                if selected_regression_models:
                    regression_results = {}
                    best_regression_model = None
                    worst_regression_model = None
                    best_score = float('-inf')  # Initialize with negative infinity
                    worst_score = float('inf')  # Initialize with positive infinity

                    # Allow the user to select evaluation metrics
                    selected_metrics = st.sidebar.multiselect(
                        "Select Regression Evaluation Metrics (at least 3)",
                        list(regression_metrics.keys()),
                        default=list(regression_metrics.keys())[:3],  # Default to the first 3 metrics
                    )

                    for model_name in selected_regression_models:
                        model = regression_models[model_name]
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        results = {}

                        # Calculate selected evaluation metrics
                        for metric_name in selected_metrics:
                            metric_func = regression_metrics[metric_name]
                            results[metric_name] = metric_func(y_test, y_pred)

                        regression_results[model_name] = results

                        # Track the best and worst models based on the selected metric
                        score = results[selected_metrics[0]]  # You can choose any metric as the default score
                        if score > best_score:
                            best_score = score
                            best_regression_model = model_name
                        if score < worst_score:
                            worst_score = score
                            worst_regression_model = model_name

                        # Residual Analysis
                        residuals = y_test - y_pred
                        st.write(f"### Residual Analysis for {model_name}")
                        st.write("Residual Plot:")
                        fig, ax = plt.subplots()
                        ax.scatter(y_pred, residuals)
                        ax.axhline(y=0, color='k', linestyle='--')
                        ax.set_xlabel("Predicted Values")
                        ax.set_ylabel("Residuals")
                        st.pyplot(fig)

                    # Display selected evaluation metrics for each model
                    for model_name, results in regression_results.items():
                        st.write(f"### {model_name} Evaluation Metrics:")
                        for metric_name, metric_value in results.items():
                            st.write(f"{metric_name}: {metric_value}")

                    # Display the best and worst regression models based on the selected metric
                    st.write(f"Best Regression Model based on {selected_metrics[0]}: {best_regression_model} ({best_score})")
                    st.write(f"Worst Regression Model based on {selected_metrics[0]}: {worst_regression_model} ({worst_score})")

                                        
                                        
                            

       
            elif task_type == "Classification":
                st.sidebar.warning("This is a Classification Problem!")

                # Encoding the target column if it's categorical
                le = LabelEncoder()
                y = le.fit_transform(y)

                # Classification models selection
                classification_models = {
                    "Logistic Regression": LogisticRegression(),
                    "Random Forest Classifier": RandomForestClassifier(),
                    "Gradient Boosting Classifier": GradientBoostingClassifier(),
                    "Support Vector Classifier": SVC(probability=True),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Decision Tree Classifier": DecisionTreeClassifier()
                }

                selected_classification_models = st.sidebar.multiselect("Select Classification Models", list(classification_models.keys()))

                if selected_classification_models:
                    classification_results = {}
                    for model_name in selected_classification_models:
                        model = classification_models[model_name]
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted')  # assuming multiclass, change if needed
                        classification_results[model_name] = [acc, f1]

                        # Confusion Matrix
                        cm = confusion_matrix(y_test, y_pred)
                        st.write(f"### Confusion Matrix for {model_name}")
                        st.write("Confusion Matrix:")
                        st.write(pd.DataFrame(cm, columns=le.classes_, index=le.classes_))
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        if len(np.unique(y)) == 2:  # Check if it's a binary classification problem
                            y_prob = model.predict_proba(X_test)[:, 1]
                            fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
                            auc = roc_auc_score(y_test, y_prob)
                            st.write(f"### ROC Curve for {model_name}")
                            fig, ax = plt.subplots()
                            ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                            ax.plot([0, 1], [0, 1], 'k--')
                            ax.set_xlabel("False Positive Rate")
                            ax.set_ylabel("True Positive Rate")
                            ax.legend(loc='lower right')
                            st.pyplot(fig)

   

        


                    best_classification_model_name = max(classification_results, key=lambda k: classification_results[k][0])
                    worst_classification_model_name = min(classification_results, key=lambda k: classification_results[k][0])

                    st.write(f"### Best Classification Model: {best_classification_model_name}")
                    st.write(f"Accuracy: {classification_results[best_classification_model_name][0]}")
                    st.write(f"F1-Score: {classification_results[best_classification_model_name][1]}")

                    st.write(f"### Worst Classification Model: {worst_classification_model_name}")
                    st.write(f"Accuracy: {classification_results[worst_classification_model_name][0]}")
                    st.write(f"F1-Score: {classification_results[worst_classification_model_name][1]}")

    
if __name__ == "__main__":
    main()

               
