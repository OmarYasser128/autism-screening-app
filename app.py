import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import chi2_contingency, ttest_ind
import io

# Set page configuration
st.set_page_config(
    page_title="Autism Screening App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class AutismScreeningApp:
    def __init__(self):
        self.df = None
        self.models = {}
        self.scaler = RobustScaler()
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, file):
        """Load and preprocess the data"""
        self.df = pd.read_csv(file)
        
        # Data preprocessing
        self.df.drop(['ID'], axis=1, inplace=True, errors='ignore')
        self.df.drop("age_desc", inplace=True, axis=1, errors='ignore')
        
        # Clean data - Replace ? & others by Others
        self.df.replace({'ethnicity': {'?': 'Others', 'others': 'Others'},
                        'relation': {'?': 'Others'}}, inplace=True)
        
        # Correcting country names
        mapping = {
            "Viet Nam": "Vietnam",
            "AmericanSamoa": "United States",
            "Hong Kong": "China"
        }
        self.df["contry_of_res"] = self.df["contry_of_res"].replace(mapping)
        
        # Create gender label for visualization
        self.df['gender_label'] = self.df['gender'].map({'f': 'Female', 'm': 'Male'})
        
        # Calculate Total Score
        screening_cols = [col for col in self.df.columns if 'A' in col and '_Score' in col]
        self.df['Total_Score'] = self.df[screening_cols].sum(axis=1)
        
        # Encoding for modeling
        top_ethnicities = self.df['ethnicity'].value_counts().nlargest(5).index
        self.df['ethnicity_encoded'] = self.df['ethnicity'].apply(lambda x: x if x in top_ethnicities else "Other")
        self.df = pd.get_dummies(self.df, columns=['ethnicity_encoded'], drop_first=True, dtype=int)
        
        top_countries = self.df['contry_of_res'].value_counts().nlargest(10).index
        self.df['contry_of_res_encoded'] = self.df['contry_of_res'].apply(lambda x: x if x in top_countries else "Other")
        self.df = pd.get_dummies(self.df, columns=['contry_of_res_encoded'], drop_first=True, dtype=int)
        
        binary_mappings = {
            'used_app_before': {'no': 0, 'yes': 1},
            'relation': {'Self': 0, 'Others': 1},
            'gender': {'m': 0, 'f': 1},
            'jaundice': {'no': 0, 'yes': 1},
            'austim': {'no': 0, 'yes': 1}
        }
        for col, mapping in binary_mappings.items():
            self.df[col] = self.df[col].map(mapping)
        
        # Scale age
        self.df['age'] = self.scaler.fit_transform(self.df[['age']])
        
        return self.df
    
    def train_models(self):
        """Train all models"""
        X = self.df.drop(columns=['Class/ASD'])
        y = self.df['Class/ASD']
        
        # Train-Test Split
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, C=0.01, class_weight="balanced"
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=5, criterion="gini", min_samples_split=20,
                min_samples_leaf=15, class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42,
                min_samples_split=20, min_samples_leaf=10, class_weight="balanced"
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.01, max_depth=3
            ),
            "Support Vector Machine": SVC(
                kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced"
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=8, weights="distance", metric="euclidean"
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42,
                use_label_encoder=False, eval_metric="logloss",
                scale_pos_weight=1.0
            )
        }
        
        # Train all models
        results = []
        for name, model in self.models.items():
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            acc = accuracy_score(self.y_test, y_pred)
            results.append([name, acc])
        
        return pd.DataFrame(results, columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)

def main():
    st.markdown('<div class="main-header">🧠 Autism Screening Prediction App</div>', unsafe_allow_html=True)
    
    app = AutismScreeningApp()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Data Overview", 
        "Exploratory Analysis", 
        "Model Training"
    ])
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        with st.spinner('Loading and preprocessing data...'):
            df = app.load_data(uploaded_file)
        
        if page == "Data Overview":
            show_data_overview(df)
        
        elif page == "Exploratory Analysis":
            show_enhanced_eda(df)
        
        elif page == "Model Training":
            show_model_training(app)
    
    else:
        st.info("👈 Please upload a CSV file to get started")
        
        # Show sample of expected data format
        st.subheader("Expected Data Format")
        sample_data = {
            'ID': [1, 2, 3],
            'A1_Score': [1, 0, 1],
            'A2_Score': [0, 1, 0],
            'A3_Score': [1, 1, 0],
            'A4_Score': [0, 0, 1],
            'A5_Score': [1, 0, 1],
            'A6_Score': [0, 1, 0],
            'A7_Score': [1, 1, 0],
            'A8_Score': [0, 0, 1],
            'A9_Score': [1, 0, 1],
            'A10_Score': [0, 1, 0],
            'age': [25, 30, 35],
            'gender': ['m', 'f', 'm'],
            'ethnicity': ['White-European', 'Asian', 'Middle Eastern'],
            'jaundice': ['no', 'yes', 'no'],
            'austim': ['no', 'yes', 'no'],
            'contry_of_res': ['United States', 'India', 'United Kingdom'],
            'used_app_before': ['no', 'yes', 'no'],
            'result': [7, 5, 6],
            'age_desc': ['18+', '18+', '18+'],
            'relation': ['Self', 'Parent', 'Self'],
            'Class/ASD': ['YES', 'NO', 'YES']
        }
        st.dataframe(pd.DataFrame(sample_data))

def show_data_overview(df):
    st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        st.subheader("First 10 Rows")
        st.dataframe(df.head(10))
    
    with col2:
        st.subheader("Data Types")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        st.subheader("Missing Values")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isnull().sum(),
            'Percentage': (df.isnull().sum() / len(df)) * 100
        })
        st.dataframe(missing_df)

def show_enhanced_eda(df):
    st.markdown('<div class="section-header">📈 Enhanced Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    # Dataset Overview
    st.markdown("""
    <div class="info-box">
    <h4>📋 Dataset Overview</h4>
    <ul>
    <li><strong>Shape:</strong> 800 rows × 22 columns</li>
    <li><strong>Feature Types:</strong> 12 numerical, 8 categorical, 1 target variable</li>
    <li><strong>Missing Values:</strong> None</li>
    <li><strong>Duplicates:</strong> None</li>
    <li><strong>Target Distribution:</strong> ~20% ASD cases (imbalanced)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Target Distribution by Gender
    st.subheader("🎯 Target Distribution by Gender")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='gender_label', hue='Class/ASD', data=df, palette='coolwarm', ax=ax)
    ax.set_title('ASD Diagnosis by Gender')
    ax.set_xlabel('Gender')
    st.pyplot(fig)
    
    # 2. Age Distribution
    st.subheader("📊 Age Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['age'], bins=30, kde=True, color='skyblue', ax=ax)
        ax.set_title('Age Distribution')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>Age Distribution Insights</h4>
        <ul>
        <li>Most participants are between <strong>15 and 30 years old</strong></li>
        <li>Peak around late teens to early twenties</li>
        <li>Distribution is <strong>right-skewed</strong> with fewer older participants</li>
        <li>Small number of participants over 60 years old</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 3. Gender and Ethnicity Distribution
    st.subheader("👥 Demographic Distributions")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='gender', data=df, hue="gender", palette='Set2', ax=ax)
        ax.set_title('Gender Distribution')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y='ethnicity', data=df, order=df['ethnicity'].value_counts().index, 
                     hue="ethnicity", palette='viridis', ax=ax)
        ax.set_title('Ethnicity Distribution')
        ax.set_xlabel('Count')
        ax.set_ylabel('Ethnicity')
        st.pyplot(fig)
    
    # 4. Medical History Analysis
    st.subheader("🏥 Medical History Analysis")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Jaundice History
    sns.countplot(x='jaundice', hue='Class/ASD', data=df, palette='coolwarm', ax=ax1)
    ax1.set_title('ASD Diagnosis by Jaundice History')
    
    # Autism History
    sns.countplot(x='austim', hue='Class/ASD', data=df, palette='coolwarm', ax=ax2)
    ax2.set_title('ASD Diagnosis by Family Autism History')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 5. Screening Questions Analysis
    st.subheader("📝 Screening Questions Analysis")
    
    # Correlation Heatmap
    screening_cols = [col for col in df.columns if 'A' in col and '_Score' in col]
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = df[screening_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
    ax.set_title('Correlation Heatmap of Screening Questions (A1–A10)')
    st.pyplot(fig)
    
    # Total Score Distribution
    st.subheader("📊 Total Screening Score Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['Total_Score'], bins=10, kde=False, color='orange', ax=ax)
        ax.set_title('Distribution of Total Screening Score (A1–A10)')
        ax.set_xlabel('Total Score')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>Total Score Insights</h4>
        <ul>
        <li><strong>Bimodal distribution:</strong> Low scorers (0-3) and high scorers (10)</li>
        <li>Middle-range scores (4–8) are less frequent</li>
        <li>Spike at maximum score (10) indicates strong ASD indicators</li>
        <li>Total_Score is a strong discriminative feature between classes</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 6. Geographic Analysis
    st.subheader("🌍 Geographic Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    top_countries = df['contry_of_res'].value_counts().nlargest(10).index
    sns.countplot(y='contry_of_res', data=df[df['contry_of_res'].isin(top_countries)], 
                 order=top_countries, hue="contry_of_res", palette='mako', ax=ax)
    ax.set_title('Top 10 Countries of Residence')
    ax.set_xlabel('Count')
    st.pyplot(fig)
    
    # 7. App Usage Analysis
    st.subheader("📱 App Usage Analysis")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='used_app_before', hue='Class/ASD', data=df, palette='Set1', ax=ax)
    ax.set_title('ASD Diagnosis by App Usage Before Test')
    ax.set_xlabel('Used App Before')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    # 8. Correlation with Target
    st.subheader("🔗 Correlation with Target Variable")
    fig, ax = plt.subplots(figsize=(8, 10))
    corr = df.corr(numeric_only=True)
    target_corr = corr[['Class/ASD']].sort_values(by='Class/ASD', ascending=False)
    sns.heatmap(target_corr, annot=True, cmap='coolwarm', ax=ax, fmt='.3f')
    ax.set_title('Correlation of Features with Target (Class/ASD)')
    st.pyplot(fig)
    
    # 9. Statistical Significance Tests
    st.subheader("📊 Statistical Significance Tests")
    
    # Chi-Square Tests
    st.write("**Chi-Square Tests for Categorical Features:**")
    categorical_features = ['gender', 'jaundice', 'austim', 'used_app_before', 'relation', 'ethnicity', 'contry_of_res']
    
    chi2_results = []
    for col in categorical_features:
        if col in df.columns:
            contingency_table = pd.crosstab(df[col], df['Class/ASD'])
            chi2, p, dof, ex = chi2_contingency(contingency_table)
            significance = "✅ Significant" if p < 0.05 else "❌ Not Significant"
            chi2_results.append([col, f"{p:.5f}", significance])
    
    chi2_df = pd.DataFrame(chi2_results, columns=["Feature", "P-Value", "Significance"])
    st.dataframe(chi2_df)
    
    # T-Tests
    st.write("**T-tests for Continuous Features:**")
    continuous_features = ['age', 'Total_Score', 'result']
    
    ttest_results = []
    for col in continuous_features:
        if col in df.columns:
            group0 = df[df['Class/ASD'] == 0][col].dropna()
            group1 = df[df['Class/ASD'] == 1][col].dropna()
            stat, p = ttest_ind(group0, group1, equal_var=False)
            significance = "✅ Significant" if p < 0.05 else "❌ Not Significant"
            ttest_results.append([col, f"{p:.5f}", significance])
    
    ttest_df = pd.DataFrame(ttest_results, columns=["Feature", "P-Value", "Significance"])
    st.dataframe(ttest_df)
    
    # 10. Feature Importance Analysis
    st.subheader("🎯 Feature Importance Analysis")
    
    # Encode categorical variables for feature importance
    cat_cols = df.select_dtypes(include=['object']).columns
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)
    
    X = df_encoded.drop('Class/ASD', axis=1)
    y = df_encoded['Class/ASD']
    
    # Random Forest for importance
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    importances.head(15).plot(kind='barh', color='teal', ax=ax)
    ax.set_title('Top 15 Feature Importances (Random Forest)')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display top features
    st.write("**Top 15 Important Features:**")
    st.dataframe(importances.head(15).reset_index().rename(columns={'index': 'Feature', 0: 'Importance'}))

def show_model_training(app):
    st.markdown('<div class="section-header">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)
    
    if st.button("Train All Models", type="primary"):
        with st.spinner('Training models... This may take a few minutes.'):
            results_df = app.train_models()
        
        st.success("All models trained successfully!")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            st.dataframe(results_df.style.format({'Accuracy': '{:.4f}'}))
        
        with col2:
            st.subheader("Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Accuracy", y="Model", data=results_df, ax=ax, palette="viridis")
            ax.set_title("Model Accuracy Comparison")
            ax.set_xlim(0, 1)
            st.pyplot(fig)
        
        # Show detailed results for each model
        st.subheader("Detailed Model Performance")
        
        for name, model in app.models.items():
            with st.expander(f"{name} - Detailed Results"):
                col1, col2 = st.columns(2)
                
                # Predictions
                y_pred = model.predict(app.X_test_scaled)
                acc = accuracy_score(app.y_test, y_pred)
                
                with col1:
                    st.metric("Accuracy", f"{acc:.4f}")
                    
                    # Classification report
                    st.text("Classification Report:")
                    report = classification_report(app.y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                
                with col2:
                    # Confusion matrix
                    fig, ax = plt.subplots(figsize=(6, 4))
                    cm = confusion_matrix(app.y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                              xticklabels=['No ASD', 'ASD'], 
                              yticklabels=['No ASD', 'ASD'])
                    ax.set_title(f"Confusion Matrix - {name}")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
