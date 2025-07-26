import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model information'])

predictions = []
def predict_heart_diseases(data):
    for modelname in model_names:
        model = pickle.load(open(modelname, 'rb'))
        prediction = model.predict(data)
        predictions.append(prediction)
    return predictions

def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    bs64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{bs64}" download="predictions.csv">Download Predictions </a>'
    return href


with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=140)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-induced angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])

    # convert categorical inputs to numerical
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # create dataframe with user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    algo_names = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    model_names = ['DTC_model.pkl', 'LR_model.pkl', 'RFC_model.pkl', 'SVM_model.pkl']

    # create a submit button to make predictions
    if st.button("Submit"):
        st.subheader('Results...')
        st.markdown('----------------')
        result = predict_heart_diseases(input_data)
        for i in range(len(result)):
            st.subheader(algo_names[i])
            if result[i][0] == 0:
                st.write('No heart disease detected')
            else:
                st.write("Heart disease detected.")
            st.markdown('--------------------------')
 
with tab2:
    st.title("Upload CSV file")
    st.subheader('Instructions to note before uploading a file')
    st.info("""
            1. No NaN values allowed
            2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG',
                'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_slope'). \n
            3. Check spellings of the feature names
    """)

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open('LR_model.pkl', 'rb'))

        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_slope']

        if set(expected_columns).issubset(input_data.columns):

            input_data['Prediction LR'] = ''

            for i in range(len(input_data)):
                arr = input_data.iloc[i, :-1].values
                input_data['Prediction LR'][i] = model.predict([arr])[0]
            input_data.to_csv('PredictedHeartLR.csv')

            #display the predictions
            st.subheader("View The Predictions")
            st.write(input_data)

            # button to download
            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning('make sure uploaded CSV has expected columns')
    else:
        st.warning("Upload CSV to get predictions")

with tab3:
    import plotly.express as px
    data = {
        'Decision Trees': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 84.23,
        'Support Vector Machine': 84.22
    }
    models = list(data.keys())
    accuracies = list(data.values())
    df = pd.DataFrame(list(zip(models, accuracies)), columns=["Models", "Accuracies"])
    fig = px.bar(df, y='Accuracies', x="Models")
    st.plotly_chart(fig)