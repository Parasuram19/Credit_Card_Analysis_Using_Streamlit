import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pickle
import os
import time
import warnings
warnings.filterwarnings("ignore")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Define functions for data processing and model training
def load_data(file_path=None):
    """
    Load the German credit data.
    """
    try:
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Data loaded from {file_path}")
            return df
        else:
            # Create a sample dataset for demonstration
            print("Creating sample dataset for demonstration...")
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
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the data for modeling.
    """
    # Make a copy of the dataframe
    df_processed = df.copy()
    
    # Drop index column if it exists
    if 'Unnamed: 0' in df_processed.columns:
        df_processed = df_processed.drop('Unnamed: 0', axis=1)
    
    # If Risk column doesn't exist, we'll create a synthetic one for demo purposes
    if 'Risk' not in df_processed.columns:
        print("Warning: 'Risk' column not found. Creating synthetic risk labels for demonstration.")
        # Use a simple rule for demonstration: higher credit amounts have higher risk
        median_credit = df_processed['Credit amount'].median()
        df_processed['Risk'] = np.where(df_processed['Credit amount'] > median_credit, 'bad', 'good')
    
    # Feature engineering
    # Create age groups
    df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 
                                      bins=[0, 25, 35, 45, 55, 100], 
                                      labels=['<25', '25-35', '35-45', '45-55', '55+'])
    
    # Create credit amount groups
    df_processed['CreditAmountGroup'] = pd.qcut(df_processed['Credit amount'], 
                                               q=4, 
                                               labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Calculate credit amount to duration ratio
    df_processed['CreditPerMonth'] = df_processed['Credit amount'] / df_processed['Duration']
    
    # Create a binary target variable
    df_processed['Target'] = (df_processed['Risk'] == 'bad').astype(int)
    
    # Check class balance and print
    target_counts = df_processed['Target'].value_counts()
    print("\nTarget class distribution:")
    print(f"  Class 0 (good risk): {target_counts.get(0, 0)} samples")
    print(f"  Class 1 (bad risk): {target_counts.get(1, 0)} samples")
    
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

def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline for numerical and categorical features.
    """
    # Define categorical and numerical features
    categorical_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'AgeGroup', 'CreditAmountGroup']
    numerical_features = ['Age', 'Job', 'Credit amount', 'Duration', 'CreditPerMonth']
    
    # Define preprocessing steps for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing steps for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_features),
            ('categorical', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def train_models(X_train, y_train, models_to_train=None):
    """
    Train multiple machine learning models.
    """
    # First, check if we have both classes in the training set
    class_counts = np.bincount(y_train)
    print(f"\nClass distribution in training set: {class_counts}")
    
    if len(class_counts) < 2 or 0 in class_counts:
        print("Error: Training data does not contain examples from both classes.")
        print("This will cause problems with classification algorithms.")
        print("Consider using stratified sampling or generating synthetic data.")
        return {}, {}
    
    preprocessor = create_preprocessing_pipeline()
    
    # Define default models to train if none specified
    if models_to_train is None:
        models_to_train = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier()
        }
    
    # Train each model
    trained_models = {}
    model_metrics = {}
    
    for name, model in models_to_train.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        try:
            # Create a pipeline with preprocessing and model
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Calculate cross-validation score
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            mean_cv_score = np.mean(cv_scores)
            
            # Store model and metrics
            trained_models[name] = pipeline
            model_metrics[name] = {
                'Training Time': training_time,
                'Cross-Validation Score': mean_cv_score
            }
            
            print(f"  Training time: {training_time:.2f} seconds")
            print(f"  Cross-validation score: {mean_cv_score:.4f}")
            
        except Exception as e:
            print(f"  Error training {name}: {e}")
    
    return trained_models, model_metrics

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data.
    """
    evaluation_results = {}
    
    # Check if we have any samples in the test set
    if len(y_test) == 0:
        print("Error: Empty test set. Cannot evaluate models.")
        return {}
    
    # Check if we have both classes in the test set
    class_counts = np.bincount(y_test)
    print(f"Class distribution in test set: {class_counts}")
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        try:
            # Make predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test)
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate precision, recall, and F1 score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Print results
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    TN: {tn}, FP: {fp}")
            print(f"    FN: {fn}, TP: {tp}")
            
            # Store evaluation results
            evaluation_results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc,
                'Confusion Matrix': cm
            }
        except Exception as e:
            print(f"  Error evaluating {name}: {e}")
    
    return evaluation_results

def save_model(model, filename):
    """
    Save a trained model to disk.
    """
    try:
        # Save model to models directory
        filepath = os.path.join('models', filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def plot_results(evaluation_results):
    """
    Plot evaluation results for visual comparison.
    """
    try:
        # Create a directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Extract metrics for plotting
        models = list(evaluation_results.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        
        # Create a dataframe for plotting
        plot_data = []
        for model in models:
            for metric in metrics:
                plot_data.append({
                    'Model': model,
                    'Metric': metric,
                    'Value': evaluation_results[model][metric]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Plot performance comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y='Value', hue='Metric', data=plot_df)
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png')
        print("Performance comparison plot saved to 'plots/model_comparison.png'")
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        for model_name in models:
            # Get the model and make predictions
            model = evaluation_results[model_name]
            auc_score = model['ROC AUC']
            plt.plot(None, None, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig('plots/roc_curves.png')
        print("ROC curves plot saved to 'plots/roc_curves.png'")
        
    except Exception as e:
        print(f"Error plotting results: {e}")

def main():
    print("Credit Risk Prediction - Model Training")
    print("======================================")
    
    # Get file path
    data_file = input("Enter path to data file (leave blank for sample data): ").strip()
    
    # Load data
    df = load_data(data_file if data_file else None)
    if df is None:
        print("Error: Failed to load data. Exiting.")
        return
    
    print(f"\nData loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFeature summary:")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique()} unique values, {df[col].isnull().sum()} missing values")
    
    # Preprocess data
    print("\nPreprocessing data...")
    df_processed = preprocess_data(df)
    print("Data preprocessing complete.")
    
    # Prepare features and target
    X, y = prepare_features_targets(df_processed)
    print(f"\nFeatures prepared: {X.shape[1]} features")
    
    # Check class distribution in full dataset
    if y is not None:
        class_counts = np.bincount(y)
        print(f"Class distribution in full dataset: {class_counts}")
    
    # Split data with stratification to ensure both classes are in train and test sets
    test_size = 0.2
    random_state = 42
    
    print(f"\nSplitting data with test_size={test_size}, random_state={random_state}, stratify=y")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Select models to train
    print("\nSelect models to train:")
    print("1. Logistic Regression")
    print("2. Random Forest")
    print("3. Gradient Boosting")
    print("4. XGBoost")
    print("5. SVM")
    print("6. KNN")
    print("7. All models")
    
    selection = input("Enter model numbers to train (comma-separated) or 7 for all: ").strip()
    
    models_to_train = {}
    if selection == "7" or selection.lower() == "all":
        models_to_train = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier()
        }
    else:
        model_options = {
            "1": ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
            "2": ('Random Forest', RandomForestClassifier(random_state=42)),
            "3": ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
            "4": ('XGBoost', XGBClassifier(random_state=42)),
            "5": ('SVM', SVC(probability=True, random_state=42)),
            "6": ('KNN', KNeighborsClassifier())
        }
        
        for model_num in selection.split(','):
            model_num = model_num.strip()
            if model_num in model_options:
                name, model = model_options[model_num]
                models_to_train[name] = model
    
    if not models_to_train:
        print("No models selected. Exiting.")
        return
    
    # Train models
    print("\nTraining models...")
    trained_models, model_metrics = train_models(X_train, y_train, models_to_train)
    
    if not trained_models:
        print("No models were successfully trained. Exiting.")
        return
    
    # Evaluate models
    print("\nEvaluating models...")
    evaluation_results = evaluate_models(trained_models, X_test, y_test)
    
    if evaluation_results:
        # Plot results
        print("\nPlotting results...")
        plot_results(evaluation_results)
    
    # Save models
    print("\nSaving models...")
    for name, model in trained_models.items():
        filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        if save_model(model, filename):
            print(f"  {name} saved as {filename}")
    
    print("\nTraining and evaluation complete. Models saved to 'models' directory.")

if __name__ == "__main__":
    main()