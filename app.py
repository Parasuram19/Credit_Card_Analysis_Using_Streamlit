import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import pickle
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'saved_models' not in st.session_state:
    st.session_state.saved_models = {}

# Define functions for data processing and model loading
def load_data():
    """
    Load the German credit data.
    In a production environment, this would load from a database or file storage.
    For this demo, we'll simulate loading the data.
    """
    try:
        # Look for data file in current directory
        if os.path.exists('german_credit.csv'):
            df = pd.read_csv('german_credit.csv')
            return df
        else:
            # Create a sample dataset for demonstration
            data = {
                'Age': [67, 22, 49, 45, 53, 35, 53, 35, 61, 28, 25, 24, 22, 60, 28, 32, 53, 25, 44, 42, 
                        31, 40, 38, 23, 57, 55, 50, 48, 45, 35],
                'Sex': ['male', 'female', 'male', 'male', 'male', 'male', 'male', 'male', 'male', 'male',
                        'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'male', 'female',
                        'female', 'male', 'male', 'female', 'male', 'male', 'male', 'female', 'male', 'female'],
                'Job': [2, 2, 1, 2, 2, 1, 2, 3, 1, 3, 1, 2, 2, 3, 2, 1, 2, 2, 3, 2, 1, 1, 3, 2, 2, 3, 2, 1, 2, 1],
                'Housing': ['own', 'own', 'own', 'free', 'free', 'free', 'own', 'rent', 'own', 'own',
                            'rent', 'rent', 'own', 'own', 'rent', 'own', 'free', 'rent', 'own', 'own',
                            'own', 'rent', 'own', 'free', 'own', 'own', 'rent', 'own', 'free', 'own'],
                'Saving accounts': [np.nan, 'little', 'little', 'little', 'little', np.nan, 'quite rich', 
                                    'little', 'rich', 'little', 'little', 'moderate', 'moderate', 'quite rich',
                                    np.nan, 'moderate', 'quite rich', 'little', np.nan, 'little', 'moderate',
                                    'little', np.nan, 'quite rich', 'rich', 'little', 'rich', 'little', 'moderate', 'little'],
                'Checking account': ['little', 'moderate', np.nan, 'little', 'little', np.nan, np.nan, 
                                    'moderate', np.nan, 'moderate', 'little', 'moderate', 'little', np.nan,
                                    'moderate', 'moderate', 'little', 'moderate', 'little', np.nan, 'little',
                                    np.nan, 'rich', 'little', 'moderate', 'little', np.nan, 'moderate', 'little', 'moderate'],
                'Credit amount': [1169, 5951, 2096, 7882, 4870, 9055, 2835, 6948, 3059, 5234, 1295, 4308, 1567, 1199,
                                3398, 1361, 1203, 4573, 1877, 2012, 2622, 7511, 1289, 1882, 4788, 6615, 1287, 4151, 2012, 1402],
                'Duration': [6, 48, 12, 42, 24, 36, 24, 36, 12, 30, 12, 48, 12, 12, 30, 6, 36, 24, 12, 30,
                            24, 36, 12, 12, 30, 60, 24, 30, 12, 6],
                'Purpose': ['radio/TV', 'radio/TV', 'education', 'furniture/equipment', 'car', 'education',
                            'furniture/equipment', 'car', 'radio/TV', 'car', 'car', 'furniture/equipment',
                            'radio/TV', 'car', 'radio/TV', 'radio/TV', 'education', 'furniture/equipment',
                            'car', 'car', 'furniture/equipment', 'education', 'car', 'radio/TV',
                            'furniture/equipment', 'business', 'car', 'business', 'radio/TV', 'car'],
                'Risk': ['good', 'bad', 'good', 'good', 'bad', 'good', 'good', 'good', 'good', 'bad',
                        'good', 'bad', 'good', 'good', 'bad', 'good', 'good', 'bad', 'good', 'good',
                        'bad', 'bad', 'good', 'good', 'good', 'bad', 'good', 'bad', 'good', 'good']
            }
            
            df = pd.DataFrame(data)
            return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """
    Preprocess the data for modeling.
    """
    # Make a copy of the dataframe
    df_processed = df.copy()
    
    # Feature engineering
    # Create age groups
    df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 
                                      bins=[0, 25, 35, 45, 55, 100], 
                                      labels=['<25', '25-35', '35-45', '45-55', '55+'])
    
    # Create credit amount groups - handle case when all values are the same
    if df_processed['Credit amount'].nunique() > 1:
        df_processed['CreditAmountGroup'] = pd.qcut(df_processed['Credit amount'], 
                                                   q=4, 
                                                   labels=['Low', 'Medium', 'High', 'Very High'])
    else:
        # If only one unique value, assign a fixed group
        df_processed['CreditAmountGroup'] = 'Medium'  # Default to Medium when only one value
    
    # Calculate credit amount to duration ratio
    df_processed['CreditPerMonth'] = df_processed['Credit amount'] / df_processed['Duration']
    
    # Create a binary target variable (if not already)
    if 'Risk' in df_processed.columns:
        df_processed['Target'] = (df_processed['Risk'] == 'bad').astype(int)
    else:
        # For new data without risk labels
        df_processed['Target'] = 0  # Placeholder, will be predicted
    
    return df_processed

def prepare_features_targets(df):
    """
    Prepare features and targets for modeling.
    """
    # Define feature columns
    categorical_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'AgeGroup', 'CreditAmountGroup']
    numerical_features = ['Age', 'Job', 'Credit amount', 'Duration', 'CreditPerMonth']
    
    # Define target
    target = 'Target'
    
    # Split into features and target
    X = df[categorical_features + numerical_features]
    if target in df.columns:
        y = df[target]
    else:
        y = None
    
    return X, y

def load_model(model_path):
    """
    Load a trained model from disk.
    """
    try:
        # Try different locations
        if os.path.exists(model_path):
            full_path = model_path
        elif os.path.exists(os.path.join('models', model_path)):
            full_path = os.path.join('models', model_path)
        else:
            return None, f"Model file not found at {model_path} or models/{model_path}"
        
        with open(full_path, 'rb') as f:
            model = pickle.load(f)
        
        return model, None
    except Exception as e:
        return None, str(e)

def find_available_models():
    """
    Find all available model files.
    """
    model_files = []
    
    # Check models directory
    if os.path.exists('models'):
        model_files.extend([os.path.join('models', f) for f in os.listdir('models') if f.endswith('.pkl')])
    
    # Check current directory
    model_files.extend([f for f in os.listdir() if f.endswith('.pkl')])
    
    return model_files

def predict_risk(model, applicant_data):
    """
    Predict credit risk for a new applicant.
    """
    # Convert applicant data to DataFrame
    applicant_df = pd.DataFrame([applicant_data])
    
    # Preprocess applicant data
    applicant_processed = preprocess_data(applicant_df)
    
    # Get features
    X_applicant, _ = prepare_features_targets(applicant_processed)
    
    # Make prediction
    risk_proba = model.predict_proba(X_applicant)[0, 1]
    risk_class = 'Bad Credit Risk' if risk_proba > 0.5 else 'Good Credit Risk'
    
    return risk_class, risk_proba

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model on test data.
    """
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate precision, recall, and F1 score
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Return evaluation results
    return {
        'Accuracy': accuracy,
        'Confusion Matrix': cm,
        'Classification Report': report,
        'ROC AUC': roc_auc,
        'FPR': fpr,
        'TPR': tpr,
        'Precision': precision,
        'Recall': recall,
        'Average Precision': avg_precision
    }

def feature_importance(model):
    """
    Extract feature importance from a model if available.
    """
    # Check if the model has feature importances
    if hasattr(model['model'], 'feature_importances_'):
        importances = model['model'].feature_importances_
        return importances
    elif hasattr(model['model'], 'coef_'):
        importances = np.abs(model['model'].coef_[0])
        return importances
    else:
        return None

# Streamlit UI Components
def main():
    # Load data
    df = load_data()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Data Exploration", "Model Comparison", "Prediction"])
    
    # Page content
    if page == "Dashboard":
        dashboard_page(df)
    elif page == "Data Exploration":
        data_exploration_page(df)
    elif page == "Model Comparison":
        model_comparison_page(df)
    elif page == "Prediction":
        # elif page == "Prediction":
        prediction_page(df)

def dashboard_page(df):
    """
    Display dashboard with key metrics and visualizations.
    """
    st.title("Credit Risk Prediction Dashboard 💰")
    
    # Display brief overview
    st.markdown("""
    This dashboard provides credit risk insights using machine learning models. You can:
    - Explore the credit data
    - Compare different prediction models
    - Make risk predictions for new applicants
    """)
    
    # Data summary metrics
    if df is not None:
        # Process data for display
        df_processed = preprocess_data(df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            if 'Risk' in df.columns:
                bad_rate = (df['Risk'] == 'bad').mean() * 100
                st.metric("Bad Credit Rate", f"{bad_rate:.1f}%")
            else:
                st.metric("Bad Credit Rate", "N/A")
        
        with col3:
            avg_credit = df['Credit amount'].mean()
            st.metric("Avg. Credit Amount", f"${avg_credit:,.0f}")
        
        with col4:
            avg_duration = df['Duration'].mean()
            st.metric("Avg. Duration (months)", f"{avg_duration:.1f}")
        
        # Main charts
        st.subheader("Credit Risk Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Risk' in df.columns:
                fig = px.pie(df, names='Risk', title='Risk Distribution', 
                             color_discrete_map={'good': '#3498db', 'bad': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Risk data not available for visualization")
        
        with col2:
            fig = px.histogram(df, x='Credit amount', nbins=20, title='Credit Amount Distribution',
                              opacity=0.7, color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Credit amount vs duration with risk
        st.subheader("Credit Amount vs. Duration")
        if 'Risk' in df.columns:
            fig = px.scatter(df, x='Duration', y='Credit amount', color='Risk', size='Age',
                            color_discrete_map={'good': '#3498db', 'bad': '#e74c3c'},
                            opacity=0.7, title='Credit Amount vs. Duration by Risk Level')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(df, x='Duration', y='Credit amount', size='Age',
                            opacity=0.7, title='Credit Amount vs. Duration')
            st.plotly_chart(fig, use_container_width=True)
        
        # Show available models
        model_files = find_available_models()
        if model_files:
            st.subheader("Available Models")
            st.write(f"Found {len(model_files)} prediction models. Go to the Model Comparison page for details.")
        else:
            st.warning("No saved models found. Please upload trained models to the 'models' directory.")
    else:
        st.error("Failed to load data. Please check data source.")

def data_exploration_page(df):
    """
    Page for exploring and visualizing the data.
    """
    st.title("Credit Data Exploration 🔍")
    
    if df is not None:
        # Process data
        df_processed = preprocess_data(df)
        
        # Show raw data
        st.subheader("Raw Credit Data")
        show_raw = st.checkbox("Show raw data")
        if show_raw:
            st.dataframe(df, height=300)
        
        # Show processed data
        st.subheader("Processed Credit Data")
        show_processed = st.checkbox("Show processed data")
        if show_processed:
            st.dataframe(df_processed, height=300)
        
        # Feature exploration
        st.subheader("Feature Exploration")
        
        # Feature selection
        feature_options = df.columns.tolist()
        
        # Select feature visualization type
        viz_type = st.selectbox(
            "Select visualization type",
            ["Distribution", "Relationship with Risk", "Correlation"]
        )
        
        if viz_type == "Distribution":
            feature = st.selectbox("Select feature to visualize", feature_options)
            
            if pd.api.types.is_numeric_dtype(df[feature]):
                # Histogram for numeric features
                fig = px.histogram(df, x=feature, nbins=20, 
                                  color='Risk' if 'Risk' in df.columns else None,
                                  marginal="box", 
                                  opacity=0.7,
                                  color_discrete_map={'good': '#3498db', 'bad': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Bar chart for categorical features
                if 'Risk' in df.columns:
                    # Grouped bar chart with risk
                    risk_counts = df.groupby([feature, 'Risk']).size().reset_index(name='count')
                    fig = px.bar(risk_counts, x=feature, y='count', color='Risk',
                                barmode='group', opacity=0.8,
                                color_discrete_map={'good': '#3498db', 'bad': '#e74c3c'})
                else:
                    # Simple bar chart
                    counts = df[feature].value_counts().reset_index()
                    counts.columns = [feature, 'count']
                    fig = px.bar(counts, x=feature, y='count', opacity=0.8)
                    
                st.plotly_chart(fig, use_container_width=True)
                
        elif viz_type == "Relationship with Risk" and 'Risk' in df.columns:
            # Select features to compare with risk
            feature1 = st.selectbox("Select first feature", feature_options)
            feature2 = st.selectbox("Select second feature", 
                                   [f for f in feature_options if f != feature1])
            
            # Create scatter plot
            if pd.api.types.is_numeric_dtype(df[feature1]) and pd.api.types.is_numeric_dtype(df[feature2]):
                fig = px.scatter(df, x=feature1, y=feature2, color='Risk',
                                opacity=0.7, 
                                color_discrete_map={'good': '#3498db', 'bad': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
            elif pd.api.types.is_numeric_dtype(df[feature1]) and not pd.api.types.is_numeric_dtype(df[feature2]):
                # Box plot
                fig = px.box(df, x=feature2, y=feature1, color='Risk',
                           color_discrete_map={'good': '#3498db', 'bad': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
            elif not pd.api.types.is_numeric_dtype(df[feature1]) and pd.api.types.is_numeric_dtype(df[feature2]):
                # Box plot with axes flipped
                fig = px.box(df, x=feature1, y=feature2, color='Risk',
                           color_discrete_map={'good': '#3498db', 'bad': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Heatmap for two categorical features
                pivot = pd.crosstab(df[feature1], df[feature2], values=df['Risk'].map({'good': 0, 'bad': 1}), aggfunc='mean')
                fig = px.imshow(pivot, text_auto=True, color_continuous_scale='RdBu_r',
                              title=f'Bad Credit Rate by {feature1} and {feature2}')
                st.plotly_chart(fig, use_container_width=True)
                
        elif viz_type == "Correlation":
            # Select numeric columns for correlation
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 1:
                # Calculate correlation matrix
                corr_matrix = df[numeric_cols].corr()
                
                # Plot heatmap
                fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                              title='Feature Correlation Matrix')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric features for correlation analysis.")
    else:
        st.error("Data could not be loaded. Please check the data source.")

def model_comparison_page(df):
    """
    Page for comparing different models.
    """
    st.title("Model Comparison 📊")
    
    # Check if data is available
    if df is None:
        st.error("Data could not be loaded. Please check the data source.")
        return
    
    # Find available models
    model_files = find_available_models()
    
    if not model_files:
        st.warning("No models found. Please add model files (.pkl) to the models directory.")
        return
    
    # Process data for evaluation
    df_processed = preprocess_data(df)
    X, y = prepare_features_targets(df_processed)
    
    # Only proceed if target is available
    if y is None:
        st.warning("Target variable not available in the data. Cannot evaluate models.")
        return
    
    # Split data for evaluation
    test_size = st.slider("Test set size", 0.1, 0.5, 0.3, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Allow user to select models to compare
    st.subheader("Select Models to Compare")
    
    selected_models = {}
    for model_file in model_files:
        model_name = os.path.basename(model_file)
        if st.checkbox(f"Use {model_name}", value=True):
            # Load model if not already in session state
            if model_file not in st.session_state.saved_models:
                with st.spinner(f"Loading {model_name}..."):
                    model, error = load_model(model_file)
                    if error:
                        st.error(f"Error loading {model_name}: {error}")
                        continue
                    
                    # Save to session state
                    st.session_state.saved_models[model_file] = {
                        'name': model_name,
                        'model': model
                    }
            
            selected_models[model_file] = st.session_state.saved_models[model_file]
    
    if not selected_models:
        st.warning("Please select at least one model to evaluate.")
        return
    
    # Evaluate selected models
    if st.button("Compare Models"):
        st.subheader("Model Performance Comparison")
        
        model_results = {}
        
        with st.spinner("Evaluating models..."):
            for model_file, model_info in selected_models.items():
                model_name = model_info['name']
                model = model_info['model']
                
                # Evaluate model
                results = evaluate_model(model, X_test, y_test)
                model_results[model_name] = results
            
        # Display comparison metrics
        metrics_df = pd.DataFrame({
            'Model': [],
            'Accuracy': [],
            'Precision (Bad)': [],
            'Recall (Bad)': [],
            'F1-Score (Bad)': [],
            'ROC AUC': []
        })
        
        for model_name, results in model_results.items():
            new_row = pd.DataFrame({
                'Model': [model_name],
                'Accuracy': [results['Accuracy']],
                'Precision (Bad)': [results['Classification Report']['1']['precision']],
                'Recall (Bad)': [results['Classification Report']['1']['recall']],
                'F1-Score (Bad)': [results['Classification Report']['1']['f1-score']],
                'ROC AUC': [results['ROC AUC']]
            })
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        
        # Style and display metrics table
        st.dataframe(metrics_df.style.highlight_max(subset=['Accuracy', 'Precision (Bad)', 
                                                           'Recall (Bad)', 'F1-Score (Bad)', 
                                                           'ROC AUC'], color='#a8d08d'),
                    height=len(metrics_df) * 35 + 38)
        
        # Plot ROC curves
        st.subheader("ROC Curves")
        fig = go.Figure()
        
        for model_name, results in model_results.items():
            fig.add_trace(go.Scatter(
                x=results['FPR'],
                y=results['TPR'],
                mode='lines',
                name=f"{model_name} (AUC = {results['ROC AUC']:.3f})"
            ))
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            height=500,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot confusion matrices
        st.subheader("Confusion Matrices")
        
        cols = st.columns(min(3, len(model_results)))
        col_idx = 0
        
        for model_name, results in model_results.items():
            with cols[col_idx % len(cols)]:
                cm = results['Confusion Matrix']
                
                # Create confusion matrix heatmap
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Good', 'Bad'],
                    y=['Good', 'Bad'],
                    text_auto=True,
                    color_continuous_scale='Blues',
                    title=f"{model_name}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display additional metrics
                tn, fp, fn, tp = cm.ravel()
                total = tn + fp + fn + tp
                
                metrics = {
                    "True Negative": f"{tn} ({tn/total:.1%})",
                    "False Positive": f"{fp} ({fp/total:.1%})",
                    "False Negative": f"{fn} ({fn/total:.1%})",
                    "True Positive": f"{tp} ({tp/total:.1%})"
                }
                
                for metric_name, value in metrics.items():
                    st.text(f"{metric_name}: {value}")
                
                col_idx += 1

def prediction_page(df):
    """
    Page for making predictions for new applicants.
    """
    st.title("Credit Risk Prediction 🔮")
    
    # Find available models
    model_files = find_available_models()
    
    if not model_files:
        st.warning("No models found. Please add model files (.pkl) to the models directory.")
        return
    
    # Select model for prediction
    st.subheader("Select Model")
    
    model_options = [os.path.basename(model_file) for model_file in model_files]
    selected_model_name = st.selectbox("Choose a model for prediction", model_options)
    selected_model_path = model_files[model_options.index(selected_model_name)]
    
    # Load selected model
    if selected_model_path not in st.session_state.saved_models:
        with st.spinner(f"Loading {selected_model_name}..."):
            model, error = load_model(selected_model_path)
            if error:
                st.error(f"Error loading model: {error}")
                return
            
            # Save to session state
            st.session_state.saved_models[selected_model_path] = {
                'name': selected_model_name,
                'model': model
            }
    
    model_info = st.session_state.saved_models[selected_model_path]
    
    # Input form for applicant data
    st.subheader("Enter Applicant Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.selectbox("Job", [1, 2, 3, 4], 
                          format_func=lambda x: {1: "Unskilled", 2: "Skilled", 3: "Management", 4: "Self-employed"}[x])
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich", "unknown"])
        checking_account = st.selectbox("Checking account", ["little", "moderate", "rich", "unknown"])
    
    with col2:
        credit_amount = st.number_input("Credit amount ($)", min_value=100, max_value=20000, value=5000)
        duration = st.slider("Duration (months)", min_value=6, max_value=72, value=24, step=6)
        purpose = st.selectbox("Purpose", ["car", "furniture/equipment", "radio/TV", "domestic appliances", 
                                         "repairs", "education", "business", "vacation", "other"])
    
    # Prepare applicant data
    applicant_data = {
        'Age': age,
        'Sex': sex,
        'Job': job,
        'Housing': housing,
        'Saving accounts': None if saving_accounts == "unknown" else saving_accounts,
        'Checking account': None if checking_account == "unknown" else checking_account,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Purpose': purpose
    }
    
    # Make prediction
    if st.button("Predict Risk"):
        with st.spinner("Analyzing risk..."):
            # Add small delay for better UX
            time.sleep(0.5)
            
            risk_class, risk_proba = predict_risk(model_info['model'], applicant_data)
            
            # Display prediction
            st.subheader("Risk Assessment")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if risk_class == "Good Credit Risk":
                    st.success(f"Prediction: {risk_class}")
                else:
                    st.error(f"Prediction: {risk_class}")
            
            with col2:
                # Create gauge for risk probability
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_proba * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Score"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#636EFA"},
                        'steps': [
                            {'range': [0, 50], 'color': "#3D9970"},
                            {'range': [50, 100], 'color': "#FF4136"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Show interpretation
            st.subheader("Risk Factors")
            
            # Check if model has feature importances
            importances = feature_importance(model_info)
            
            if importances is not None:
                # Get feature names
                X, _ = prepare_features_targets(preprocess_data(pd.DataFrame([applicant_data])))
                features = X.columns
                
                # Create dataframe of feature importances
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                })
                
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Plot feature importances
                fig = px.bar(importance_df.head(10), x='Importance', y='Feature', orientation='h',
                            title='Top Features Influencing Risk',
                            color='Importance',
                            color_continuous_scale='Viridis')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show risk reduction suggestions
                st.subheader("Risk Reduction Suggestions")
                st.write("Based on the model, consider these changes to potentially reduce risk:")
                
                suggestions = []
                
                if 'Duration' in importance_df['Feature'].values and duration > 24:
                    suggestions.append("- Reduce the loan duration if possible")
                
                if 'Credit amount' in importance_df['Feature'].values and credit_amount > 5000:
                    suggestions.append("- Request a smaller loan amount")
                
                if 'Saving accounts' in importance_df['Feature'].values and saving_accounts in ["little", "unknown"]:
                    suggestions.append("- Increase savings before applying")
                
                if suggestions:
                    for suggestion in suggestions:
                        st.markdown(suggestion)
                else:
                    st.write("No specific suggestions available for this applicant.")
            else:
                st.info("This model doesn't provide feature importance information for interpretation.")

if __name__ == "__main__":
    main()