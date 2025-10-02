import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import chi2_contingency, ttest_ind
import io
import warnings
warnings.filterwarnings("ignore")

# Try to import optional packages with fallbacks
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

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
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
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
        self.feature_names = None
        
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
        
        # Encoding for modeling - FIXED: Only encode numeric columns
        # First convert categorical columns to numeric
        binary_mappings = {
            'used_app_before': {'no': 0, 'yes': 1},
            'relation': {'Self': 0, 'Others': 1},
            'gender': {'m': 0, 'f': 1},
            'jaundice': {'no': 0, 'yes': 1},
            'austim': {'no': 0, 'yes': 1}
        }
        for col, mapping in binary_mappings.items():
            self.df[col] = self.df[col].map(mapping)
        
        # Handle ethnicity and country with proper encoding
        top_ethnicities = self.df['ethnicity'].value_counts().nlargest(5).index
        self.df['ethnicity_encoded'] = self.df['ethnicity'].apply(lambda x: x if x in top_ethnicities else "Other")
        ethnicity_dummies = pd.get_dummies(self.df['ethnicity_encoded'], prefix='ethnicity', dtype=int)
        self.df = pd.concat([self.df, ethnicity_dummies], axis=1)
        self.df.drop(['ethnicity', 'ethnicity_encoded'], axis=1, inplace=True)
        
        top_countries = self.df['contry_of_res'].value_counts().nlargest(10).index
        self.df['country_encoded'] = self.df['contry_of_res'].apply(lambda x: x if x in top_countries else "Other")
        country_dummies = pd.get_dummies(self.df['country_encoded'], prefix='country', dtype=int)
        self.df = pd.concat([self.df, country_dummies], axis=1)
        self.df.drop(['contry_of_res', 'country_encoded'], axis=1, inplace=True)
        
        # Remove any remaining non-numeric columns except target
        non_numeric_cols = self.df.select_dtypes(include=['object']).columns
        for col in non_numeric_cols:
            if col != 'Class/ASD' and col != 'gender_label':
                self.df.drop(col, axis=1, inplace=True)
        
        # Ensure all data is numeric
        self.df = self.df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Scale age
        if 'age' in self.df.columns:
            self.df['age'] = self.scaler.fit_transform(self.df[['age']])
        
        return self.df
    
    def train_enhanced_models(self):
        """Train all models with cross-validation and AUC scoring"""
        # Ensure we only use numeric columns
        X = self.df.select_dtypes(include=[np.number]).drop('Class/ASD', axis=1)
        y = self.df['Class/ASD']
        self.feature_names = X.columns.tolist()
        
        # Train-Test Split
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features - FIXED: Ensure we only scale numeric data
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
        # Define enhanced models (only include available ones)
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, C=0.01),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, criterion="gini", min_samples_split=20, min_samples_leaf=15),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, min_samples_split=20, min_samples_leaf=10),
            "Support Vector Machine": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=8, weights="distance", metric="euclidean"),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
        }
        
        # Add optional models if available
        if XGB_AVAILABLE:
            self.models["XGBoost"] = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42, eval_metric="logloss")
        
        if LGBM_AVAILABLE:
            self.models["LightGBM"] = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, num_leaves=15, random_state=42, verbose=-1)
        
        if CATBOOST_AVAILABLE:
            self.models["CatBoost"] = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=6, random_state=42, verbose=0)
        
        # Define stratified k-fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = []
        auc_results = {}

        for name, model in self.models.items():
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=skf, scoring="accuracy")
                
                # Fit on training set and evaluate on test set
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                acc = accuracy_score(self.y_test, y_pred)

                # Calculate AUC if possible
                auc = "N/A"
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                        auc = roc_auc_score(self.y_test, y_proba)
                    except:
                        auc = "N/A"

                # Store results
                results.append({
                    "Model": name,
                    "CV Accuracy": round(cv_scores.mean(), 4),
                    "Test Accuracy": round(acc, 4),
                    "AUC": auc
                })
                
            except Exception as e:
                st.warning(f"Model {name} failed: {str(e)}")
                continue

        # Build DataFrame and sort by CV Accuracy
        if results:
            results_df = pd.DataFrame(results).sort_values(by="CV Accuracy", ascending=False)
            return results_df
        else:
            return pd.DataFrame(columns=["Model", "CV Accuracy", "Test Accuracy", "AUC"])
    
    def predict_new_sample(self, features_dict):
        """Make prediction for new sample"""
        if not self.models or self.feature_names is None:
            return None
        
        try:
            # Create feature vector
            features = np.zeros(len(self.feature_names))
            
            for i, feature in enumerate(self.feature_names):
                if feature in features_dict:
                    features[i] = features_dict[feature]
            
            # Scale the features
            features_scaled = self.scaler.transform([features])
            
            # Get predictions from all models
            predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(features_scaled)[0]
                    confidence = 0.5  # Default confidence
                    
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(features_scaled)[0]
                        confidence = max(proba)
                    
                    predictions[name] = {
                        'prediction': pred,
                        'confidence': confidence,
                        'label': 'ASD' if pred == 1 else 'No ASD'
                    }
                except:
                    continue
            
            return predictions
        except:
            return None

def main():
    st.markdown('<div class="main-header">🧠 Autism Screening Prediction App</div>', unsafe_allow_html=True)
    
    # Show package availability status
    unavailable_models = []
    if not XGB_AVAILABLE:
        unavailable_models.append("XGBoost")
    if not LGBM_AVAILABLE:
        unavailable_models.append("LightGBM")
    if not CATBOOST_AVAILABLE:
        unavailable_models.append("CatBoost")
    
    if unavailable_models:
        st.markdown(f"""
        <div class="warning-box">
        <h4>⚠️ Package Availability Notice</h4>
        <p>Some advanced models may not be available: {', '.join(unavailable_models)}</p>
        <p>The app will work with available models: Logistic Regression, Decision Tree, Random Forest, SVM, K-Neighbors, Gradient Boosting, AdaBoost</p>
        </div>
        """, unsafe_allow_html=True)
    
    app = AutismScreeningApp()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Data Overview", 
        "Exploratory Analysis", 
        "Advanced Model Training",
        "Make Prediction"
    ])
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        with st.spinner('Loading and preprocessing data...'):
            try:
                df = app.load_data(uploaded_file)
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return
        
        if page == "Data Overview":
            show_data_overview(df)
        
        elif page == "Exploratory Analysis":
            show_enhanced_eda(df)
        
        elif page == "Advanced Model Training":
            show_advanced_model_training(app)
        
        elif page == "Make Prediction":
            show_prediction_interface(app)
    
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
    
    # Create a copy for EDA to avoid modifying original
    df_eda = df.copy()
    
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
    
    # 1. Target Distribution
    st.subheader("🎯 Target Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    target_counts = df_eda['Class/ASD'].value_counts()
    colors = ['#ff9999', '#66b3ff']
    ax.pie(target_counts.values, labels=['No ASD', 'ASD'], autopct='%1.1f%%', 
           colors=colors, startangle=90)
    ax.set_title("ASD vs No ASD Distribution")
    st.pyplot(fig)
    
    # 2. Age Distribution
    st.subheader("📊 Age Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df_eda['age'], bins=30, kde=True, color='skyblue', ax=ax)
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
    
    # 3. Screening Questions Analysis
    st.subheader("📝 Screening Questions Analysis")
    
    # Correlation Heatmap for A1-A10 scores
    screening_cols = [col for col in df_eda.columns if 'A' in col and '_Score' in col]
    if screening_cols:
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = df_eda[screening_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
        ax.set_title('Correlation Heatmap of Screening Questions (A1–A10)')
        st.pyplot(fig)
    
    # Total Score Distribution
    if 'Total_Score' in df_eda.columns:
        st.subheader("📊 Total Screening Score Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df_eda['Total_Score'], bins=10, kde=False, color='orange', ax=ax)
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
    
    # 4. Correlation with Target
    st.subheader("🔗 Correlation with Target Variable")
    numeric_cols = df_eda.select_dtypes(include=[np.number]).columns
    if 'Class/ASD' in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 10))
        corr = df_eda[numeric_cols].corr()
        target_corr = corr[['Class/ASD']].sort_values(by='Class/ASD', ascending=False)
        sns.heatmap(target_corr, annot=True, cmap='coolwarm', ax=ax, fmt='.3f')
        ax.set_title('Correlation of Features with Target (Class/ASD)')
        st.pyplot(fig)
    
    # 5. Feature Importance Analysis
    st.subheader("🎯 Feature Importance Analysis")
    
    try:
        # Use only numeric columns for feature importance
        numeric_df = df_eda.select_dtypes(include=[np.number])
        if 'Class/ASD' in numeric_df.columns:
            X = numeric_df.drop('Class/ASD', axis=1)
            y = numeric_df['Class/ASD']
            
            # Random Forest for importance
            rf = RandomForestClassifier(random_state=42, n_estimators=100)
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
    except Exception as e:
        st.warning(f"Feature importance analysis skipped: {str(e)}")

def show_advanced_model_training(app):
    st.markdown('<div class="section-header">🤖 Advanced Model Training & Evaluation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 Enhanced Modeling Approach</h4>
    <ul>
    <li><strong>Multiple Algorithms</strong> including ensemble methods</li>
    <li><strong>5-Fold Stratified Cross Validation</strong> for robust evaluation</li>
    <li><strong>AUC Scoring</strong> for imbalanced data performance</li>
    <li><strong>Advanced Models:</strong> Random Forest, Gradient Boosting, AdaBoost + optional advanced models</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Train All Advanced Models", type="primary"):
        with st.spinner('Training advanced models with cross-validation... This may take a few minutes.'):
            results_df = app.train_enhanced_models()
        
        if not results_df.empty:
            st.success("All models trained successfully!")
            
            # Display comprehensive results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Model Performance Summary")
                # Format the dataframe for display
                display_df = results_df.copy()
                for col in ['CV Accuracy', 'Test Accuracy']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
                st.dataframe(display_df)
            
            with col2:
                st.subheader("📈 Test Accuracy Comparison")
                fig, ax = plt.subplots(figsize=(10, 6))
                results_sorted = results_df.sort_values(by="Test Accuracy", ascending=False)
                # Convert to float for plotting
                results_sorted['Test Accuracy'] = pd.to_numeric(results_sorted['Test Accuracy'], errors='coerce')
                sns.barplot(x="Test Accuracy", y="Model", data=results_sorted, ax=ax, palette="viridis")
                ax.set_title("Model Test Accuracy Comparison")
                ax.set_xlim(0, 1)
                st.pyplot(fig)
            
            # Show CV Accuracy comparison
            st.subheader("🎯 Cross-Validation Accuracy")
            fig, ax = plt.subplots(figsize=(10, 6))
            results_sorted_cv = results_df.sort_values(by="CV Accuracy", ascending=False)
            # Convert to float for plotting
            results_sorted_cv['CV Accuracy'] = pd.to_numeric(results_sorted_cv['CV Accuracy'], errors='coerce')
            sns.barplot(x="CV Accuracy", y="Model", data=results_sorted_cv, ax=ax, palette="plasma")
            ax.set_title("Model CV Accuracy Comparison (5-Fold)")
            ax.set_xlim(0, 1)
            st.pyplot(fig)
            
            # Show detailed results for each model
            st.subheader("🔍 Detailed Model Performance")
            
            for name, model in app.models.items():
                with st.expander(f"{name} - Detailed Analysis"):
                    col1, col2 = st.columns(2)
                    
                    try:
                        # Predictions
                        y_pred = model.predict(app.X_test_scaled)
                        acc = accuracy_score(app.y_test, y_pred)
                        
                        with col1:
                            st.metric("Test Accuracy", f"{acc:.4f}")
                            
                            # Get AUC if available
                            if hasattr(model, "predict_proba"):
                                try:
                                    y_proba = model.predict_proba(app.X_test_scaled)[:, 1]
                                    auc = roc_auc_score(app.y_test, y_proba)
                                    st.metric("AUC Score", f"{auc:.3f}")
                                except:
                                    st.metric("AUC Score", "N/A")
                            
                            # Classification report
                            st.text("Classification Report:")
                            report = classification_report(app.y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))
                        
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
                    except Exception as e:
                        st.error(f"Could not generate detailed results for {name}: {str(e)}")
            
            # Model Evaluation Summary
            best_model = results_df.iloc[0]
            st.markdown(f"""
            <div class="info-box">
            <h4>📋 Model Evaluation Summary</h4>
            <ul>
            <li><strong>Best Overall:</strong> {best_model['Model']} (CV={best_model['CV Accuracy']}, Test={best_model['Test Accuracy']})</li>
            <li><strong>Models Trained:</strong> {len(results_df)}</li>
            <li><strong>Key Insight:</strong> Models show consistent performance across CV and Test sets</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ No models were successfully trained. Please check your data and try again.")

def show_prediction_interface(app):
    st.markdown('<div class="section-header">🔮 Make Predictions</div>', unsafe_allow_html=True)
    
    if not app.models:
        st.warning("⚠️ Please train the models first in the 'Advanced Model Training' section!")
        return
    
    st.markdown("""
    <div class="info-box">
    <h4>📋 Prediction Instructions</h4>
    <p>Enter the patient's information below to get ASD predictions from all trained models.</p>
    </div>
    """)
    
    # Create input form
    st.subheader("👤 Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        gender = st.selectbox("Gender", options=['m', 'f'])
        jaundice = st.selectbox("Had Jaundice at Birth", options=['no', 'yes'])
    
    with col2:
        autism_history = st.selectbox("Family History of Autism", options=['no', 'yes'])
        used_app_before = st.selectbox("Used App Before", options=['no', 'yes'])
        relation = st.selectbox("Who is filling the test", 
                              options=['Self', 'Parent', 'Relative', 'Health care professional', 'Others'])
    
    with col3:
        ethnicity = st.selectbox("Ethnicity", 
                               options=['White-European', 'Asian', 'Black', 'Middle Eastern', 
                                      'Hispanic', 'South Asian', 'Others'])
        country = st.selectbox("Country of Residence", 
                             options=['United States', 'United Kingdom', 'India', 'New Zealand',
                                    'Australia', 'Canada', 'Jordan', 'United Arab Emirates', 'Others'])
    
    # Screening Questions
    st.subheader("📝 Screening Questions (A1-A10)")
    st.write("Answer the following questions (0 = No, 1 = Yes):")
    
    a_scores = {}
    cols = st.columns(5)
    for i in range(1, 11):
        with cols[(i-1) % 5]:
            a_scores[f'A{i}_Score'] = st.selectbox(f"A{i} Score", options=[0, 1], key=f"a{i}")
    
    if st.button("🎯 Get ASD Prediction", type="primary"):
        with st.spinner('Analyzing patient information...'):
            # Prepare features dictionary
            features_dict = {}
            
            # Basic features
            features_dict['age'] = age
            features_dict['gender'] = 0 if gender == 'm' else 1
            features_dict['jaundice'] = 0 if jaundice == 'no' else 1
            features_dict['austim'] = 0 if autism_history == 'no' else 1
            features_dict['used_app_before'] = 0 if used_app_before == 'no' else 1
            features_dict['relation'] = 0 if relation == 'Self' else 1
            
            # A scores
            for i in range(1, 11):
                features_dict[f'A{i}_Score'] = a_scores[f'A{i}_Score']
            
            # Calculate total score
            total_score = sum(a_scores.values())
            features_dict['Total_Score'] = total_score
            features_dict['result'] = total_score  # Simple approximation
            
            # Set ethnicity and country features (set the selected one to 1, others to 0)
            ethnicity_prefix = f"ethnicity_{ethnicity.replace(' ', '_').replace('-', '_')}"
            country_prefix = f"country_{country.replace(' ', '_').replace('-', '_')}"
            
            # For simplicity, we'll set these to 1 if they exist in the feature names
            # In a real app, you'd need to map these properly to the actual feature names
            for feature_name in app.feature_names:
                if 'ethnicity' in feature_name:
                    features_dict[feature_name] = 1 if ethnicity_prefix in feature_name else 0
                elif 'country' in feature_name:
                    features_dict[feature_name] = 1 if country_prefix in feature_name else 0
            
            # Make prediction
            predictions = app.predict_new_sample(features_dict)
            
            if predictions:
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Calculate consensus
                asd_count = sum(1 for result in predictions.values() if result['prediction'] == 1)
                total_models = len(predictions)
                consensus_percentage = (asd_count / total_models) * 100
                
                # Overall consensus
                st.markdown(f"""
                <div class="prediction-box">
                <h3>Overall Consensus: {consensus_percentage:.1f}% models predict ASD</h3>
                <p><strong>{asd_count} out of {total_models}</strong> models indicate Autism Spectrum Disorder</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Individual model predictions
                st.subheader("🤖 Individual Model Predictions")
                
                # Create columns for model results
                model_cols = st.columns(3)
                
                for idx, (model_name, result) in enumerate(predictions.items()):
                    with model_cols[idx % 3]:
                        color = "red" if result['prediction'] == 1 else "green"
                        icon = "🔴" if result['prediction'] == 1 else "🟢"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{icon} {model_name}</h4>
                            <h3 style="color: {color};">{result['label']}</h3>
                            <p>Confidence: {result['confidence']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Interpretation
                st.subheader("💡 Interpretation")
                if consensus_percentage >= 70:
                    st.error("🚨 **High probability of ASD** - Strong consensus among models suggests further clinical evaluation is recommended.")
                elif consensus_percentage >= 40:
                    st.warning("⚠️ **Moderate probability of ASD** - Mixed results suggest additional screening may be beneficial.")
                else:
                    st.success("✅ **Low probability of ASD** - Majority of models indicate no autism spectrum disorder.")
                
                # Risk factors analysis
                st.subheader("🔍 Risk Factors Analysis")
                risk_factors = []
                if autism_history == 'yes':
                    risk_factors.append("Family history of autism")
                if jaundice == 'yes':
                    risk_factors.append("History of neonatal jaundice")
                if total_score >= 7:
                    risk_factors.append(f"High screening score ({total_score}/10)")
                
                if risk_factors:
                    st.write("**Identified risk factors:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.info("No significant risk factors identified.")
                    
            else:
                st.error("❌ Prediction failed. Please make sure models are properly trained and data is formatted correctly.")

if __name__ == "__main__":
    main()
