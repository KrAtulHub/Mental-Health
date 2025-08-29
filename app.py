

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_PATH = 'random_forest_tuned_model.joblib'
model_pipeline = None
if os.path.exists(MODEL_PATH):
    model_pipeline = joblib.load(MODEL_PATH)
else:
    model_pipeline = None

# Define possible values for categorical features (should match training data)
gender_options = ['Male', 'Female', 'Non-binary', 'Prefer not to say']
country_options = ['USA', 'Canada', 'UK', 'Germany', 'Australia', 'India']
state_options = ['CA', 'NY', 'TX', 'WA', 'ON', 'QC', 'LDN', 'BLN', 'SYD', 'MH', 'nan']
self_employed_options = ['Yes', 'No']
yes_no_options = ['Yes', 'No']
work_interfere_options = ['Often', 'Sometimes', 'Never', 'Rarely']
no_employees_options = ['1-5', '6-25', '26-100', '101-500', '501-1000', 'More than 1000']
benefits_options = ['Yes', 'No', "Don't know"]
care_options_options = ['Yes', 'No', 'Not sure']
wellness_program_options = ['Yes', 'No', "Don't know"]
seek_help_options = ['Yes', 'No', "Don't know"]
anonymity_options = ['Yes', 'No', "Don't know"]
leave_options = ['Somewhat easy', 'Very easy', 'Don\'t know', 'Somewhat difficult', 'Very difficult']
mh_consequence_options = ['Yes', 'No', 'Maybe']
ph_consequence_options = ['Yes', 'No', 'Maybe']
coworkers_supervisor_options = ['Yes', 'No', 'Some of them']
interview_options = ['Yes', 'No', 'Maybe']
mental_vs_physical_options = ['Yes', 'No', "Don't know"]
observed_consequence_options = ['Yes', 'No']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    risk_level = None
    confidence = None
    error = None
    if request.method == 'POST':
        try:
            user_data = pd.DataFrame({
                'Age': [int(request.form['Age'])],
                'Gender': [request.form['Gender']],
                'Country': [request.form['Country']],
                'State': [request.form['State'] if request.form['State'] != 'nan' else np.nan],
                'Self_employed': [request.form['Self_employed']],
                'Family_history': [request.form['Family_history']],
                'Work_interfere': [request.form['Work_interfere']],
                'No_employees': [request.form['No_employees']],
                'Remote_work': [request.form['Remote_work']],
                'Tech_company': [request.form['Tech_company']],
                'Benefits': [request.form['Benefits']],
                'Care_options': [request.form['Care_options']],
                'Wellness_program': [request.form['Wellness_program']],
                'Seek_help': [request.form['Seek_help']],
                'Anonymity': [request.form['Anonymity']],
                'Leave': [request.form['Leave']],
                'Mental_health_consequence': [request.form['Mental_health_consequence']],
                'Phys_health_consequence': [request.form['Phys_health_consequence']],
                'Coworkers': [request.form['Coworkers']],
                'Supervisor': [request.form['Supervisor']],
                'Mental_health_interview': [request.form['Mental_health_interview']],
                'Phys_health_interview': [request.form['Phys_health_interview']],
                'Mental_vs_physical': [request.form['Mental_vs_physical']],
                'Observed_consequence': [request.form['Observed_consequence']],
            })
            if model_pipeline:
                prediction_proba = model_pipeline.predict_proba(user_data)[:, 1][0]
                confidence = f"{prediction_proba:.2%}"
                if prediction_proba >= 0.7:
                    risk_level = "High"
                elif prediction_proba >= 0.4:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
                prediction = f"Predicted Mental Health Risk: {risk_level} (Confidence: {confidence})"
            else:
                error = "Model file not found. Please ensure 'random_forest_tuned_model.joblib' is in the same directory."
        except Exception as e:
            error = f"Error during prediction: {e}"
    return render_template('index.html',
        gender_options=gender_options,
        country_options=country_options,
        state_options=state_options,
        self_employed_options=self_employed_options,
        yes_no_options=yes_no_options,
        work_interfere_options=work_interfere_options,
        no_employees_options=no_employees_options,
        benefits_options=benefits_options,
        care_options_options=care_options_options,
        wellness_program_options=wellness_program_options,
        seek_help_options=seek_help_options,
        anonymity_options=anonymity_options,
        leave_options=leave_options,
        mh_consequence_options=mh_consequence_options,
        ph_consequence_options=ph_consequence_options,
        coworkers_supervisor_options=coworkers_supervisor_options,
        interview_options=interview_options,
        mental_vs_physical_options=mental_vs_physical_options,
        observed_consequence_options=observed_consequence_options,
        prediction=prediction,
        risk_level=risk_level,
        confidence=confidence,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)


