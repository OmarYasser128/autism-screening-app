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
        
        # Clean data
        self.df['ethnicity'].replace({'?':'others','Others':'others'}, inplace=True)
        self.df['relation'].replace({'?':'others'}, inplace=True)
        
        self.df.relation = self.df.relation.replace({
            "?": "Others",
            "Relative": "Others",
            "Parent": "Others", 
            "Health care professional": "Others",
            "others": "Others"
        })
        
        mapping = {
            "Viet Nam": "Vietnam",
            "AmericanSamoa": "United States",
            "Hong Kong": "China"
        }
        self.df["contry_of_res"] = self.df["contry_of_res"].replace(mapping)
        
        # Encoding
        top_ethnicities = self.df['ethnicity'].value_counts().nlargest(5).index
        self.df['ethnicity'] = self.df['ethnicity'].apply(lambda x: x if x in top_ethnicities else "Other")
        self.df = pd.get_dummies(self.df, columns=['ethnicity'], drop_first=True, dtype=int)
        
        top_countries = self.df['contry_of_res'].value_counts().nlargest(10).index
        self.df['contry_of_res'] = self.df['contry_of_res'].apply(lambda x: x if x in top_countries else "Other")
        self.df = pd.get_dummies(self.df, columns=['contry_of_res'], drop_first=True, dtype=int)
        
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
        
        elif page == "Exploratory Analysis":
            st.markdown('<div class="section-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Target Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                target_counts = df['Class/ASD'].value_counts()
                colors = ['#ff9999', '#66b3ff']
                ax.pie(target_counts.values, labels=['No ASD', 'ASD'], autopct='%1.1f%%', 
                       colors=colors, startangle=90)
                ax.set_title("ASD vs No ASD Distribution")
                st.pyplot(fig)
            
            with col2:
                st.subheader("Age Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df['age'], kde=True, ax=ax)
                age_mean = df['age'].mean()
                age_median = df['age'].median()
                ax.axvline(age_mean, color="red", linestyle="--", label=f"Mean: {age_mean:.2f}")
                ax.axvline(age_median, color="green", linestyle="-", label=f"Median: {age_median:.2f}")
                ax.legend()
                ax.set_title("Distribution of Age")
                st.pyplot(fig)
        
        elif page == "Model Training":
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

if __name__ == "__main__":
    main()
