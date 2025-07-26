# Heart Disease Predictor ML

A machine learning project that predicts the likelihood of heart disease based on various medical parameters. The project includes data preprocessing, model training with multiple algorithms, and a user-friendly Streamlit web application for predictions.

## 🎯 Project Overview

This project uses machine learning to predict heart disease based on 11 key medical features. It implements and compares four different algorithms:
- **Logistic Regression** (85.86% accuracy)
- **Random Forest Classifier** (84.23% accuracy) 
- **Support Vector Machine** (84.22% accuracy)
- **Decision Tree Classifier** (80.97% accuracy)

## 📊 Dataset

The project uses the Heart Disease dataset containing 918 records with the following features:

### Input Features:
- **Age**: Age of the patient (years)
- **Sex**: Gender (M/F)
- **ChestPainType**: Type of chest pain (TA, ATA, NAP, ASY)
- **RestingBP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Serum cholesterol (mg/dl)
- **FastingBS**: Fasting blood sugar (>120 mg/dl or ≤120 mg/dl)
- **RestingECG**: Resting electrocardiogram results (Normal, ST, LVH)
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (Y/N)
- **Oldpeak**: ST depression induced by exercise
- **ST_Slope**: Slope of peak exercise ST segment (Up, Flat, Down)

### Target Variable:
- **HeartDisease**: 1 = heart disease, 0 = no heart disease

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning library
- **Plotly** - Interactive visualizations
- **Pickle** - Model serialization

## 📁 Project Structure

```
heart-disease-predictor-ml/
├── app.py                 # Streamlit web application
├── notebook.ipynb         # Jupyter notebook with ML workflow
├── requirements.txt       # Python dependencies
├── dataset/
│   └── heart.csv         # Heart disease dataset
├── DTC_model.pkl         # Trained Decision Tree model
├── LR_model.pkl          # Trained Logistic Regression model
├── RFC_model.pkl         # Trained Random Forest model
├── SVM_model.pkl         # Trained Support Vector Machine model
└── README.md             # Project documentation
```

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd heart-disease-predictor-ml
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

## 💻 Usage

### Web Application Features

The Streamlit app provides three main tabs:

#### 1. **Predict Tab**
- Enter individual patient data through an intuitive form
- Get predictions from all four trained models
- Instant results showing heart disease likelihood

#### 2. **Bulk Predict Tab**
- Upload CSV files for batch predictions
- Uses the Logistic Regression model (best performing)
- Download results as CSV file
- **CSV Requirements:**
  - No missing values (NaN)
  - Exact column names: `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`
  - Categorical values should be encoded as numbers

#### 3. **Model Information Tab**
- Interactive bar chart comparing model accuracies
- Performance metrics visualization

### Input Data Format

For categorical variables, use the following encoding:

- **Sex**: Male = 0, Female = 1
- **ChestPainType**: ATA = 0, NAP = 1, ASY = 2, TA = 3
- **RestingECG**: Normal = 0, ST = 1, LVH = 2
- **ExerciseAngina**: No = 0, Yes = 1
- **ST_Slope**: Up = 0, Flat = 1, Down = 2

## 🔬 Machine Learning Pipeline

### Data Preprocessing
1. **Missing Value Imputation**: Used KNN Imputer for cholesterol and resting BP zero values
2. **Categorical Encoding**: Converted categorical variables to numerical format
3. **Feature Scaling**: Applied StandardScaler for Logistic Regression
4. **Train-Test Split**: 80-20 split with stratification

### Model Training & Evaluation
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Performance Metrics**: Accuracy, F1-score for model comparison

### Model Performance
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 85.86% |
| Random Forest | 84.23% |
| Support Vector Machine | 84.22% |
| Decision Tree | 80.97% |

## 📈 Key Features

- **Multiple ML Algorithms**: Compare predictions across four different models
- **Data Validation**: Robust input validation and error handling
- **Batch Processing**: Handle multiple predictions efficiently
- **Interactive Visualization**: Model performance comparison charts
- **User-Friendly Interface**: Intuitive Streamlit web application
- **Export Functionality**: Download prediction results

## 🔍 Model Details

### Logistic Regression (Best Performer)
- **Solver**: Optimized through hyperparameter tuning
- **Preprocessing**: StandardScaler normalization
- **Accuracy**: 85.86%

### Random Forest Classifier
- **Hyperparameters**: Tuned via GridSearchCV
- **Features**: Handles feature importance automatically
- **Accuracy**: 84.23%

### Support Vector Machine
- **Kernel**: Optimized kernel selection (linear, poly, rbf, sigmoid)
- **Evaluation**: F1-score weighted average
- **Accuracy**: 84.22%

### Decision Tree Classifier
- **Parameters**: Balanced class weights
- **Pruning**: Optimized depth and leaf parameters
- **Accuracy**: 80.97%

## 📋 Requirements

See `requirements.txt` for complete dependencies:
- numpy
- pandas
- scikit-learn
- streamlit
- plotly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue in the repository.

---

**Built with ❤️ using Python and Streamlit**
