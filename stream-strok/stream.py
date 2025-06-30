# -*- coding: utf-8 -*-
"""
Enhanced Stroke Prediction ML Analysis - Streamlit Dashboard
Author: Enhanced Interactive Version
Purpose: Interactive web dashboard for stroke prediction analysis
Run with: streamlit run stroke_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
from datetime import datetime
import io
import base64

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# =============================
# 🎨 Page Configuration
# =============================
st.set_page_config(
    page_title="🏥 Stroke Prediction ML Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# 🎨 Custom CSS
# =============================
st.markdown("""
<style>
    body, .stApp {
        background: linear-gradient(135deg, #181c3a 0%, #23244d 100%) !important;
        color: #f4f6fa;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .main-header {
        font-size: 2.8rem;
        color: #fff;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-shadow: 0 2px 16px #2e3192, 0 1px 0 #fff;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    .main-header img {
        border-radius: 12px;
        box-shadow: 0 4px 32px #2e319288;
        background: #23244d;
        padding: 0.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #23244d 0%, #2e3192 100%);
        padding: 1.2rem;
        border-radius: 18px;
        border-left: 6px solid #6c63ff;
        margin: 0.7rem 0;
        box-shadow: 0 2px 16px #2e319288;
        transition: box-shadow 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 32px #6c63ff88;
    }
    .success-box, .warning-box, .info-box {
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
        box-shadow: 0 2px 16px #2e319288;
    }
    .success-box {
        background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        color: #181c3a;
        border: none;
    }
    .warning-box {
        background: linear-gradient(90deg, #fffde4 0%, #f9d423 100%);
        color: #181c3a;
        border: none;
    }
    .info-box {
        background: linear-gradient(90deg, #6c63ff 0%, #38f9d7 100%);
        color: #fff;
        border: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #23244d;
        border-radius: 14px;
        padding: 0.3rem 0.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 16px #2e319288;
    }
    .stTabs [data-baseweb="tab"] {
        color: #f4f6fa;
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 8px 8px 0 0;
        margin-right: 0.5rem;
        transition: background 0.2s, color 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #6c63ff 0%, #38f9d7 100%);
        color: #181c3a;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6c63ff 0%, #38f9d7 100%);
        color: #fff;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 2px 16px #6c63ff44;
        transition: background 0.2s, color 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #38f9d7 0%, #6c63ff 100%);
        color: #fff;
    }
    .stSidebar {
        background: #23244d !important;
        color: #fff !important;
        border-radius: 0 18px 18px 0;
        box-shadow: 2px 0 24px #2e319288;
    }
    .stSidebar .stMarkdown, .stSidebar .stInfo {
        color: #fff !important;
    }
    .stDataFrame, .stTable {
        background: #23244d !important;
        color: #f4f6fa !important;
        border-radius: 12px;
        box-shadow: 0 2px 16px #2e319288;
    }
    .stMetric {
        background: #23244d;
        color: #38f9d7;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 2px 16px #6c63ff44;
    }
    hr {
        border: none;
        border-top: 2px solid #6c63ff;
        margin: 2rem 0 1rem 0;
    }
    @media (max-width: 900px) {
        .stColumns, .stColumn {
            flex-direction: column !important;
            width: 100% !important;
            min-width: 0 !important;
        }
        .section-header {
            font-size: 1.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Add a geometric/tech/AI-style header image
st.markdown('<h1 class="main-header"><img src="https://cdn.openart.ai/uploads/image_1687288576822_1024.jpg" width="60" style="vertical-align:middle; margin-bottom:10px;"> Enhanced Stroke Prediction ML Dashboard</h1>', unsafe_allow_html=True)

# Sidebar header with image
st.sidebar.markdown('<div style="display:flex;align-items:center;gap:0.5rem;"><img src="https://cdn.openart.ai/uploads/image_1687288576822_1024.jpg" width="28" style="border-radius:8px;"> <span style="font-size:1.2rem;font-weight:700;color:#6c63ff;">Dashboard Controls</span></div>', unsafe_allow_html=True)

# =============================
# 📊 Data Processing Functions
# =============================
@st.cache_data
def load_and_preprocess_data(uploaded_file=None):
    """Load and preprocess data with caching"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # Try to load default file
            df = pd.read_csv('stroke/healthcare-dataset-stroke-data.csv')
        
        # Handle missing values
        if 'bmi' in df.columns and df['bmi'].isnull().any():
            df['bmi'] = df['bmi'].fillna(
                df.groupby(['age', 'gender'])['bmi'].transform('median')
            )
            df['bmi'].fillna(df['bmi'].median(), inplace=True)
        
        # Handle 'Other' gender
        if 'gender' in df.columns:
            if (df['gender'] == 'Other').sum() < 5:
                df['gender'] = df['gender'].replace('Other', df['gender'].mode()[0])
        
        # Log transformations
        if 'avg_glucose_level' in df.columns:
            df['avg_glucose_level_log'] = np.log1p(df['avg_glucose_level'])
        if 'bmi' in df.columns:
            df['bmi_log'] = np.log1p(df['bmi'])
        
        # Binary encoding
        if 'ever_married' in df.columns:
            df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1})
        
        # One-hot encoding
        categorical_cols = ['gender', 'work_type', 'Residence_type', 'smoking_status']
        existing_cats = [col for col in categorical_cols if col in df.columns]
        df_encoded = pd.get_dummies(df, columns=existing_cats, drop_first=True)
        
        # Remove ID column if exists
        if 'id' in df_encoded.columns:
            df_encoded.drop(columns='id', inplace=True)
            
        return df, df_encoded
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def enhanced_feature_engineering(df):
    """Create enhanced features"""
    df_enhanced = df.copy()
    
    # BMI categorization
    def categorize_bmi(bmi):
        if pd.isna(bmi):
            return 'Unknown'
        elif bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    # Age categorization
    def categorize_age(age):
        if age < 18:
            return 'Child'
        elif age < 30:
            return 'Young_Adult'
        elif age < 50:
            return 'Middle_Aged'
        elif age < 65:
            return 'Pre_Senior'
        else:
            return 'Senior'
    
    # Glucose categorization
    def categorize_glucose(glucose):
        if glucose < 100:
            return 'Normal'
        elif glucose < 126:
            return 'Prediabetic'
        else:
            return 'Diabetic'
    
    # Apply categorizations
    if 'bmi' in df_enhanced.columns:
        df_enhanced['bmi_category'] = df_enhanced['bmi'].apply(categorize_bmi)
    if 'age' in df_enhanced.columns:
        df_enhanced['age_category'] = df_enhanced['age'].apply(categorize_age)
    if 'avg_glucose_level' in df_enhanced.columns:
        df_enhanced['glucose_category'] = df_enhanced['avg_glucose_level'].apply(categorize_glucose)
    
    # Risk score calculation
    def compute_risk_score(row):
        score = 0
        if 'age' in row and row['age'] >= 65:
            score += 2
        if 'hypertension' in row and row['hypertension'] == 1:
            score += 2
        if 'heart_disease' in row and row['heart_disease'] == 1:
            score += 3
        if 'bmi' in row and row['bmi'] >= 30:
            score += 1
        return score
    
    df_enhanced['risk_score'] = df_enhanced.apply(compute_risk_score, axis=1)
    
    return df_enhanced

@st.cache_data
def train_models(X_train, X_test, y_train, y_test, selected_models, use_smote=True, use_grid_search=False):
    """Train selected models"""
    
    # Apply SMOTE if selected
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
    
    # Model dictionary
    models = {
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Parameter grids for grid search
    param_grids = {
        'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        'Logistic Regression': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
        'Decision Tree': {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]},
        'Random Forest': {'n_estimators': [50, 100], 'max_depth': [3, 5, 7]},
        'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2]}
    }
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model_name in enumerate(selected_models):
        status_text.text(f'Training {model_name}...')
        
        model = models[model_name]
        start_time = time.time()
        
        # Grid search if enabled
        if use_grid_search and model_name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_name], cv=3, scoring='f1', n_jobs=-1
            )
            grid_search.fit(X_train_res, y_train_res)
            best_model = grid_search.best_estimator_
        else:
            best_model = model
            best_model.fit(X_train_res, y_train_res)
        
        end_time = time.time()
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train_res, y_train_res, cv=5, scoring='f1')
        
        # Metrics
        results.append({
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
            'CV_F1_Mean': cv_scores.mean(),
            'CV_F1_Std': cv_scores.std(),
            'Training_Time': end_time - start_time,
            'Model_Object': best_model,
            'Predictions': y_pred,
            'Probabilities': y_prob
        })
        
        progress_bar.progress((i + 1) / len(selected_models))
    
    status_text.text('Training completed!')
    return results

# =============================
# 📊 Visualization Functions
# =============================
def create_data_overview_plots(df):
    """Create data overview visualizations"""
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        stroke_count = df['stroke'].sum() if 'stroke' in df.columns else 0
        st.metric("Stroke Cases", stroke_count)
    with col4:
        stroke_rate = (stroke_count / len(df) * 100) if len(df) > 0 else 0
        st.metric("Stroke Rate (%)", f"{stroke_rate:.2f}")
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        if 'stroke' in df.columns:
            fig = px.pie(df, names='stroke', title='Stroke Distribution',
                        labels={'stroke': 'Stroke', 0: 'No Stroke', 1: 'Stroke'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'age' in df.columns:
            fig = px.histogram(df, x='age', color='stroke' if 'stroke' in df.columns else None,
                             title='Age Distribution', nbins=30)
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        fig = px.imshow(df[numeric_cols].corr(), 
                       title='Feature Correlation Heatmap',
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)

def create_model_comparison_plots(results):
    """Create model comparison visualizations"""
    
    results_df = pd.DataFrame([{k: v for k, v in r.items() 
                               if k not in ['Model_Object', 'Predictions', 'Probabilities']} 
                              for r in results])
    
    # Performance metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(results_df, x='Model', y='F1-Score', 
                    title='F1-Score Comparison',
                    color='F1-Score', color_continuous_scale='viridis')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(results_df, x='Model', y='AUC', 
                    title='AUC Score Comparison',
                    color='AUC', color_continuous_scale='plasma')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Precision vs Recall scatter
    fig = px.scatter(results_df, x='Recall', y='Precision', 
                    color='F1-Score', size='AUC',
                    hover_data=['Model'],
                    title='Precision vs Recall',
                    color_continuous_scale='viridis')
    
    for i, row in results_df.iterrows():
        fig.add_annotation(x=row['Recall'], y=row['Precision'],
                          text=row['Model'], showarrow=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Training time comparison
    fig = px.bar(results_df, x='Model', y='Training_Time',
                title='Training Time Comparison (seconds)',
                color='Training_Time', color_continuous_scale='oranges')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

def create_roc_curves(results, y_test):
    """Create ROC curves comparison"""
    
    fig = go.Figure()
    
    # Add ROC curve for each model
    for result in results:
        if result['Probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['Probabilities'])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f"{result['Model']} (AUC = {result['AUC']:.3f})",
                line=dict(width=2)
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800, height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_confusion_matrices(results, y_test):
    """Create confusion matrices for all models"""
    
    n_models = len(results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    for i in range(0, n_models, cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j < n_models:
                result = results[i + j]
                cm = confusion_matrix(y_test, result['Predictions'])
                
                fig = px.imshow(cm, text_auto=True, aspect="auto",
                               title=f"Confusion Matrix - {result['Model']}",
                               labels=dict(x="Predicted", y="Actual"))
                
                with col:
                    st.plotly_chart(fig, use_container_width=True)

# =============================
# 📱 Main Dashboard
# =============================
def main():
    """Main dashboard function"""
    # Load data FIRST
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file", type=['csv'],
        help="Upload your stroke dataset CSV file"
    )
    df_raw, df_processed = load_and_preprocess_data(uploaded_file)
    if df_raw is None:
        st.error("Please upload a valid CSV file or ensure 'healthcare-dataset-stroke-data.csv' exists")
        st.stop()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🔧 Feature Engineering", 
        "🤖 Model Training", 
        "📈 Results Analysis",
        "📋 Model Comparison"
    ])
    
    # =============================
    # Tab 1: Data Overview
    # =============================
    with tab1:
        st.subheader("📊 Dataset Overview")
        
        # --- Metrics Row ---
        st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Total Samples", len(df_raw))
        with metric_cols[1]:
            st.metric("Features", len(df_raw.columns) - 1)
        with metric_cols[2]:
            stroke_count = df_raw['stroke'].sum() if 'stroke' in df_raw.columns else 0
            st.metric("Stroke Cases", stroke_count)
        with metric_cols[3]:
            stroke_rate = (stroke_count / len(df_raw) * 100) if len(df_raw) > 0 else 0
            st.metric("Stroke Rate (%)", f"{stroke_rate:.2f}")
        
        # --- Data Overview Section ---
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        overview_cols = st.columns([2, 1])
        with overview_cols[0]:
            st.write("**Dataset Sample:**")
            st.dataframe(df_raw.head(10), use_container_width=True)
        with overview_cols[1]:
            st.write("**Dataset Info:**")
            info_df = pd.DataFrame({
                "Column": df_raw.columns,
                "Non-Null Count": df_raw.notnull().sum().values,
                "Dtype": df_raw.dtypes.astype(str).values
            })
            st.dataframe(info_df, use_container_width=True)
        
        # --- Data Quality Section ---
        st.markdown('<div class="section-header">Data Quality Analysis</div>', unsafe_allow_html=True)
        dq_cols = st.columns(2)
        with dq_cols[0]:
            missing_data = df_raw.isnull().sum()
            if missing_data.sum() > 0:
                fig = px.bar(x=missing_data.index, y=missing_data.values,
                           title="Missing Values by Column")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        
        with dq_cols[1]:
            duplicates = df_raw.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)
            if duplicates > 0:
                st.warning(f"⚠️ Found {duplicates} duplicate rows")
            else:
                st.success("✅ No duplicate rows found!")
        
        # --- Visualizations Section ---
        st.markdown('<div class="section-header">Data Visualizations</div>', unsafe_allow_html=True)
        viz_cols = st.columns(2)
        with viz_cols[0]:
            if 'stroke' in df_raw.columns:
                fig = px.pie(df_raw, names='stroke', title='Stroke Distribution',
                            labels={'stroke': 'Stroke', 0: 'No Stroke', 1: 'Stroke'})
                st.plotly_chart(fig, use_container_width=True)
        with viz_cols[1]:
            if 'age' in df_raw.columns:
                fig = px.histogram(df_raw, x='age', color='stroke' if 'stroke' in df_raw.columns else None,
                                 title='Age Distribution', nbins=30)
                st.plotly_chart(fig, use_container_width=True)
        # Correlation heatmap (full width)
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig = px.imshow(df_raw[numeric_cols].corr(), 
                           title='Feature Correlation Heatmap',
                           color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
    
    # =============================
    # Tab 2: Feature Engineering
    # =============================
    with tab2:
        st.subheader("🔧 Feature Engineering")
        
        # Feature engineering options
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Features:**")
            st.write(list(df_processed.columns))
        
        with col2:
            apply_advanced_fe = st.checkbox("Apply Advanced Feature Engineering", value=True)
        
        if apply_advanced_fe:
            df_enhanced = enhanced_feature_engineering(df_raw)
            
            st.write("**Enhanced Features:**")
            new_features = set(df_enhanced.columns) - set(df_raw.columns)
            st.write(f"Added {len(new_features)} new features:")
            st.write(list(new_features))
            
            # Show feature distributions
            if st.checkbox("Show Feature Distributions"):
                feature_to_plot = st.selectbox("Select feature to visualize:", 
                                             df_enhanced.select_dtypes(include=[np.number]).columns)
                
                fig = px.histogram(df_enhanced, x=feature_to_plot, 
                                 color='stroke' if 'stroke' in df_enhanced.columns else None,
                                 title=f'Distribution of {feature_to_plot}')
                st.plotly_chart(fig, use_container_width=True)
        else:
            df_enhanced = df_raw
    
    # =============================
    # Tab 3: Model Training
    # =============================
    with tab3:
        st.subheader("🤖 Model Training Configuration")
        
        # Model selection
        available_models = ['KNN', 'Logistic Regression', 'SVM', 'Naive Bayes', 
                           'Decision Tree', 'Random Forest', 'Gradient Boosting']
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_models = st.multiselect(
                "Select Models to Train:",
                available_models,
                default=['Logistic Regression', 'Random Forest', 'SVM']
            )
        
        with col2:
            use_smote = st.checkbox("Apply SMOTE (Handle Class Imbalance)", value=True)
            use_grid_search = st.checkbox("Use Grid Search (Hyperparameter Tuning)", value=False)
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.25, 0.05)
        
        if st.button("🚀 Train Models", type="primary"):
            if len(selected_models) == 0:
                st.error("Please select at least one model to train!")
            else:
                # Prepare data
                if 'stroke' not in df_processed.columns:
                    st.error("Target column 'stroke' not found in dataset!")
                    st.stop()
                
                X = df_processed.drop(columns='stroke')
                y = df_processed['stroke']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Display data split info
                st.success(f"✅ Data prepared successfully!")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Training Samples", len(X_train))
                with col2:
                    st.metric("Testing Samples", len(X_test))
                with col3:
                    st.metric("Features", X_train.shape[1])
                with col4:
                    st.metric("Stroke Rate", f"{y.mean()*100:.2f}%")
                
                # Train models
                with st.spinner("Training models... This may take a while."):
                    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test,
                                         selected_models, use_smote, use_grid_search)
                
                # Store results in session state
                st.session_state['training_results'] = results
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = X.columns.tolist()
                
                st.success("🎉 Training completed successfully!")
    
    # =============================
    # Tab 4: Results Analysis
    # =============================
    with tab4:
        st.subheader("📈 Results Analysis")
        
        if 'training_results' not in st.session_state:
            st.info("👆 Please train models first in the 'Model Training' tab.")
        else:
            results = st.session_state['training_results']
            y_test = st.session_state['y_test']
            
            # Performance summary
            st.subheader("🏆 Performance Summary")
            
            # Create results dataframe for display
            display_results = []
            for r in results:
                display_results.append({
                    'Model': r['Model'],
                    'Accuracy': f"{r['Accuracy']:.4f}",
                    'Precision': f"{r['Precision']:.4f}",
                    'Recall': f"{r['Recall']:.4f}",
                    'F1-Score': f"{r['F1-Score']:.4f}",
                    'AUC': f"{r['AUC']:.4f}",
                    'CV F1 (Mean ± Std)': f"{r['CV_F1_Mean']:.4f} ± {r['CV_F1_Std']:.4f}",
                    'Training Time (s)': f"{r['Training_Time']:.2f}"
                })
            
            results_df = pd.DataFrame(display_results)
            
            # Sort by F1-Score
            results_df['F1_numeric'] = [float(x) for x in results_df['F1-Score']]
            results_df = results_df.sort_values('F1_numeric', ascending=False)
            results_df = results_df.drop('F1_numeric', axis=1)
            
            st.dataframe(results_df, use_container_width=True)
            
            # Best model highlight
            best_model = max(results, key=lambda x: x['F1-Score'])
            st.markdown(f"""
            <div class="success-box">
                <h4>🏆 Best Performing Model: {best_model['Model']}</h4>
                <p><strong>F1-Score:</strong> {best_model['F1-Score']:.4f}</p>
                <p><strong>AUC:</strong> {best_model['AUC']:.4f}</p>
                <p><strong>Cross-validation F1:</strong> {best_model['CV_F1_Mean']:.4f} ± {best_model['CV_F1_Std']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualizations
            st.subheader("📊 Performance Visualizations")
            create_model_comparison_plots(results)
            
            # ROC Curves
            st.subheader("📈 ROC Curves")
            create_roc_curves(results, y_test)
            
            # Confusion Matrices
            st.subheader("🎯 Confusion Matrices")
            create_confusion_matrices(results, y_test)
    
    # =============================
    # Tab 5: Model Comparison
    # =============================
    with tab5:
        st.subheader("📋 Detailed Model Comparison")
        
        if 'training_results' not in st.session_state:
            st.info("👆 Please train models first in the 'Model Training' tab.")
        else:
            results = st.session_state['training_results']
            y_test = st.session_state['y_test']
            
            # Model selector for detailed analysis
            model_names = [r['Model'] for r in results]
            selected_model = st.selectbox("Select model for detailed analysis:", model_names)
            
            # Find selected model results
            model_result = next(r for r in results if r['Model'] == selected_model)
            
            # Detailed metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{model_result['Accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{model_result['Precision']:.4f}")
            with col3:
                st.metric("Recall", f"{model_result['Recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{model_result['F1-Score']:.4f}")
            
            # Classification report
            st.subheader(f"📊 Classification Report - {selected_model}")
            
            report = classification_report(y_test, model_result['Predictions'], 
                                         target_names=['No Stroke', 'Stroke'],
                                         output_dict=True)
            
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4))
            
            # Feature importance (if available)
            if hasattr(model_result['Model_Object'], 'feature_importances_'):
                st.subheader(f"🔍 Feature Importance - {selected_model}")
                
                feature_names = st.session_state['feature_names']
                importances = model_result['Model_Object'].feature_importances_
                
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance_df.head(15), 
                           x='Importance', y='Feature',
                           orientation='h',
                           title=f'Top 15 Feature Importances - {selected_model}')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Model coefficients (for Logistic Regression)
            elif hasattr(model_result['Model_Object'], 'coef_'):
                st.subheader(f"📈 Model Coefficients - {selected_model}")
                
                feature_names = st.session_state['feature_names']
                coefficients = model_result['Model_Object'].coef_[0]
                
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients,
                    'Abs_Coefficient': np.abs(coefficients)
                }).sort_values('Abs_Coefficient', ascending=False)
                
                fig = px.bar(coef_df.head(15), 
                           x='Coefficient', y='Feature',
                           orientation='h',
                           title=f'Top 15 Feature Coefficients - {selected_model}',
                           color='Coefficient',
                           color_continuous_scale='RdBu')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            st.subheader("💾 Download Results")
            
            # Prepare results for download
            results_for_download = []
            for r in results:
                results_for_download.append({
                    'Model': r['Model'],
                    'Accuracy': r['Accuracy'],
                    'Precision': r['Precision'],
                    'Recall': r['Recall'],
                    'F1-Score': r['F1-Score'],
                    'AUC': r['AUC'],
                    'CV_F1_Mean': r['CV_F1_Mean'],
                    'CV_F1_Std': r['CV_F1_Std'],
                    'Training_Time': r['Training_Time']
                })
            
            download_df = pd.DataFrame(results_for_download)
            
            # Convert to CSV
            csv = download_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"stroke_prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Model recommendations
            st.subheader("💡 Recommendations")
            
            best_model = max(results, key=lambda x: x['F1-Score'])
            
            recommendations = []
            
            if best_model['Model'] == 'Random Forest':
                recommendations.extend([
                    "✅ Random Forest provides good interpretability through feature importance",
                    "✅ Robust to overfitting and handles missing values well",
                    "✅ Consider ensemble methods for production deployment",
                    "⚠️ May require more computational resources for large datasets"
                ])
            elif best_model['Model'] == 'Logistic Regression':
                recommendations.extend([
                    "✅ Logistic Regression offers excellent interpretability",
                    "✅ Fast training and prediction times",
                    "✅ Good baseline model for medical predictions",
                    "⚠️ May struggle with complex non-linear relationships"
                ])
            elif best_model['Model'] == 'SVM':
                recommendations.extend([
                    "✅ SVM performs well on high-dimensional data",
                    "✅ Good generalization capabilities",
                    "⚠️ Less interpretable than linear models",
                    "⚠️ May require more computational resources"
                ])
            elif best_model['Model'] == 'Gradient Boosting':
                recommendations.extend([
                    "✅ Gradient Boosting often provides high accuracy",
                    "✅ Handles complex patterns well",
                    "⚠️ Prone to overfitting without proper tuning",
                    "⚠️ Longer training times"
                ])
            
            # General recommendations
            recommendations.extend([
                "",
                "**General Guidelines:**",
                "🔄 Implement proper cross-validation in production",
                "📊 Monitor model performance over time",
                "🏥 Consider clinical validation before deployment",
                "⚖️ Ensure compliance with healthcare regulations",
                "🔒 Implement proper data privacy measures"
            ])
            
            for rec in recommendations:
                if rec:
                    st.write(rec)
    
    # =============================
    # Sidebar - Additional Info
    # =============================
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
    This dashboard provides an interactive interface for stroke prediction analysis using multiple machine learning algorithms.
    
    **Features:**
    - Data exploration and visualization
    - Advanced feature engineering
    - Multiple ML model comparison
    - Comprehensive performance analysis
    - Downloadable results
    
    **Models Supported:**
    - K-Nearest Neighbors (KNN)
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Naive Bayes
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Technical Details")
    st.sidebar.info("""
    **Preprocessing:**
    - Missing value imputation
    - Log transformations
    - One-hot encoding
    - Feature scaling
    
    **Class Imbalance:**
    - SMOTE oversampling
    - Stratified sampling
    
    **Validation:**
    - Train/test split
    - Cross-validation
    - Multiple metrics evaluation
    """)
    
    # Add footer with developer credits
    st.markdown("""
    <div style='text-align: center; color: #aaa; margin-top: 2rem; font-size: 1.1rem;'>
        Developed by <b>Suhaimi, Anis, Syafiqah, Nizam</b>
    </div>
    """, unsafe_allow_html=True)

# =============================
# 🚀 Application Entry Point
# =============================
if __name__ == "__main__":
    main()