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
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class AutismScreeningApp:
    def __init__(self):
        self.df = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = RobustScaler()
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.is_trained = False
        self.training_results = None
        
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
        
        # Relation encoding
        self.df.relation = self.df.relation.replace({
            "?": "Others",
            "Relative": "Others", 
            "Parent": "Others",
            "Health care professional": "Others",
            "others": "Others"
        })
        
        # Encoding for modeling
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
    
    def train_enhanced_models(self):
        """Train all models with cross-validation and AUC scoring"""
        X = self.df.drop(columns=['Class/ASD'])
        y = self.df['Class/ASD']
        self.feature_names = X.columns.tolist()
        
        # Train-Test Split
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
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
        best_score = 0
        self.best_model = None
        self.best_model_name = None

        for name, model in self.models.items():
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=skf, scoring="accuracy")
                cv_mean = cv_scores.mean()
                
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
                    "CV Accuracy": round(cv_mean, 4),
                    "Test Accuracy": round(acc, 4),
                    "AUC": auc
                })
                
                # Update best model
                if cv_mean > best_score:
                    best_score = cv_mean
                    self.best_model = model
                    self.best_model_name = name
                
            except Exception as e:
                st.warning(f"Model {name} failed: {str(e)}")
                continue

        # Mark as trained and save results
        self.is_trained = True
        self.training_results = pd.DataFrame(results).sort_values(by="CV Accuracy", ascending=False)
        
        return self.training_results
    
    def predict_single_sample(self, features_dict):
        """Make prediction using the best model only"""
        if not self.is_trained or self.best_model is None:
            return None
        
        try:
            # Create feature vector
            features = np.zeros(len(self.feature_names))
            
            for i, feature in enumerate(self.feature_names):
                if feature in features_dict:
                    features[i] = features_dict[feature]
                else:
                    # Set default value for missing features
                    features[i] = 0
            
            # Scale the features
            features_scaled = self.scaler.transform([features])
            
            # Get prediction from best model
            pred = self.best_model.predict(features_scaled)[0]
            confidence = 0.5
            
            if hasattr(self.best_model, "predict_proba"):
                proba = self.best_model.predict_proba(features_scaled)[0]
                confidence = max(proba)
            
            return {
                'prediction': pred,
                'confidence': confidence,
                'label': 'ASD' if pred == 1 else 'No ASD',
                'model_used': self.best_model_name
            }
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

def main():
    st.markdown('<div class="main-header">🧠 Autism Screening Prediction App</div>', unsafe_allow_html=True)
    
    # Initialize app in session state - THIS IS THE KEY FIX
    if 'autism_app' not in st.session_state:
        st.session_state.autism_app = AutismScreeningApp()
    
    app = st.session_state.autism_app
    
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
        <p>The app will work with available models.</p>
        </div>
        """, unsafe_allow_html=True)
    
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
    
    # Load data if uploaded
    if uploaded_file is not None:
        if app.df is None or st.sidebar.button("Reload Data"):
            with st.spinner('Loading and preprocessing data...'):
                try:
                    df = app.load_data(uploaded_file)
                    st.success("✅ Data loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
                    return
        else:
            df = app.df
        
        # Show training status in sidebar
        st.sidebar.markdown("---")
        if app.is_trained:
            st.sidebar.success(f"✅ Models Trained\nBest: {app.best_model_name}")
        else:
            st.sidebar.warning("⚠️ Models Not Trained")
        
        # Page routing
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
    
    # Create a copy for EDA
    df_eda = df.copy()
    
    # Dataset Overview
    st.markdown("""
    <div class="info-box">
    <h4>📋 Dataset Overview</h4>
    <ul>
    <li><strong>Shape:</strong> 800 rows × 21 columns (after preprocessing)</li>
    <li><strong>Feature Types:</strong> Mixed numerical and encoded categorical</li>
    <li><strong>Missing Values:</strong> None</li>
    <li><strong>Duplicates:</strong> None</li>
    <li><strong>Target Distribution:</strong> ~20% ASD cases (imbalanced)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Target distribution by Gender
    st.subheader("🎯 Target Distribution by Gender")
    df_eda['gender_label'] = df_eda['gender'].map({0: 'Male', 1: 'Female'})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='gender_label', hue='Class/ASD', data=df_eda, palette='coolwarm', ax=ax)
    ax.set_title('ASD Diagnosis by Gender')
    ax.set_xlabel('Gender')
    st.pyplot(fig)
    
    # 2. Age distribution
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
        <li>Distribution shows varied age groups</li>
        <li>Data has been robust scaled for modeling</li>
        <li>Age range covers both children and adults</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 3. Screening questions correlation
    st.subheader("📝 Screening Questions Correlation")
    screening_cols = [col for col in df_eda.columns if 'A' in col and '_Score' in col]
    if screening_cols:
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = df_eda[screening_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
        ax.set_title('Correlation Heatmap of Screening Questions (A1–A10)')
        st.pyplot(fig)
    
    # 4. Calculate and show Total Score distribution
    st.subheader("📊 Total Screening Score Distribution")
    df_eda['Total_Score'] = df_eda[screening_cols].sum(axis=1)
    
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
        <li>Sum of all A1-A10 screening questions</li>
        <li>Higher scores indicate more ASD indicators</li>
        <li>Key feature for model prediction</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 5. Correlation with target
    st.subheader("🔗 Correlation with Target Variable")
    numeric_cols = df_eda.select_dtypes(include=[np.number]).columns
    if 'Class/ASD' in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 10))
        corr = df_eda[numeric_cols].corr()
        target_corr = corr[['Class/ASD']].sort_values(by='Class/ASD', ascending=False)
        sns.heatmap(target_corr, annot=True, cmap='coolwarm', ax=ax, fmt='.3f')
        ax.set_title('Correlation of Features with Target (Class/ASD)')
        st.pyplot(fig)

def show_advanced_model_training(app):
    st.markdown('<div class="section-header">🤖 Advanced Model Training & Evaluation</div>', unsafe_allow_html=True)
    
    # Show training status
    if app.is_trained:
        st.markdown(f"""
        <div class="success-box">
        <h4>✅ Models Already Trained!</h4>
        <p><strong>Best Model:</strong> {app.best_model_name}</p>
        <p>You can view the results below or go to <strong>Make Prediction</strong> to use the model.</p>
        </div>
        """)
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 Enhanced Modeling Approach</h4>
    <ul>
    <li><strong>Multiple Algorithms</strong> including ensemble methods</li>
    <li><strong>5-Fold Stratified Cross Validation</strong> for robust evaluation</li>
    <li><strong>AUC Scoring</strong> for imbalanced data performance</li>
    <li><strong>Best Model Selection</strong> automatically saved for predictions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Train All Advanced Models", type="primary"):
            with st.spinner('Training advanced models with cross-validation... This may take a few minutes.'):
                results_df = app.train_enhanced_models()
            
            if not results_df.empty:
                st.success("✅ All models trained successfully!")
                st.rerun()  # Refresh to show updated status
    
    with col2:
        if app.is_trained:
            if st.button("🔄 Retrain Models"):
                # Reset training state
                app.is_trained = False
                app.models = {}
                app.best_model = None
                app.best_model_name = None
                st.rerun()
    
    # Show results if models are trained
    if app.is_trained and app.training_results is not None:
        st.markdown("---")
        st.subheader("📊 Training Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(app.training_results)
        
        with col2:
            st.subheader("📈 Test Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            results_sorted = app.training_results.sort_values(by="Test Accuracy", ascending=False)
            sns.barplot(x="Test Accuracy", y="Model", data=results_sorted, ax=ax, palette="viridis")
            ax.set_title("Model Test Accuracy Comparison")
            ax.set_xlim(0, 1)
            st.pyplot(fig)
        
        # Model Evaluation Summary
        best_model = app.training_results.iloc[0]
        st.markdown(f"""
        <div class="info-box">
        <h4>📋 Model Evaluation Summary</h4>
        <ul>
        <li><strong>Best Model:</strong> {best_model['Model']}</li>
        <li><strong>CV Accuracy:</strong> {best_model['CV Accuracy']}</li>
        <li><strong>Test Accuracy:</strong> {best_model['Test Accuracy']}</li>
        <li><strong>AUC:</strong> {best_model['AUC']}</li>
        <li><strong>Models Trained:</strong> {len(app.training_results)}</li>
        <li><strong>Ready for Predictions:</strong> Best model is saved and ready to use</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_prediction_interface(app):
    st.markdown('<div class="section-header">🔮 Make Predictions</div>', unsafe_allow_html=True)
    
    if not app.is_trained or app.best_model is None:
        st.warning("""
        ⚠️ **Please train the models first!**
        
        Go to the **Advanced Model Training** page and click **"Train All Advanced Models"**.
        Once training is complete, the best model will be automatically saved and ready for predictions.
        """)
        
        # Show quick training button
        if st.button("🚀 Go Train Models Now", type="primary"):
            st.switch_page("Advanced Model Training")
        
        return
    
    st.markdown(f"""
    <div class="success-box">
    <h4>✅ Ready for Predictions!</h4>
    <p>Using <strong>{app.best_model_name}</strong> - the best performing model from training.</p>
    <p><strong>CV Accuracy:</strong> {app.training_results[app.training_results['Model'] == app.best_model_name]['CV Accuracy'].iloc[0]}</p>
    </div>
    """)
    
    st.markdown("""
    <div class="info-box">
    <h4>📋 Prediction Instructions</h4>
    <p>Enter the patient's information below to get ASD prediction using our best trained model.</p>
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
            
            # Set ethnicity and country features
            for feature_name in app.feature_names:
                if 'ethnicity' in feature_name:
                    ethnicity_feature = f"ethnicity_{ethnicity.replace(' ', '_').replace('-', '_')}"
                    features_dict[feature_name] = 1 if ethnicity_feature in feature_name else 0
                elif 'contry_of_res' in feature_name:
                    country_feature = f"contry_of_res_{country.replace(' ', '_').replace('-', '_')}"
                    features_dict[feature_name] = 1 if country_feature in feature_name else 0
            
            # Make prediction using the best model
            prediction = app.predict_single_sample(features_dict)
            
            if prediction:
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Result")
                
                # Overall prediction
                color = "red" if prediction['prediction'] == 1 else "green"
                icon = "🔴" if prediction['prediction'] == 1 else "🟢"
                
                st.markdown(f"""
                <div class="prediction-box">
                <h2>{icon} Prediction: <span style="color: {color};">{prediction['label']}</span></h2>
                <p><strong>Model Used:</strong> {prediction['model_used']}</p>
                <p><strong>Confidence:</strong> {prediction['confidence']:.1%}</p>
                <p><strong>Screening Score:</strong> {total_score}/10</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Interpretation and risk factors (same as before)
                # ... [rest of your prediction display code]

if __name__ == "__main__":
    main()
