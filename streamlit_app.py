import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and label encoder
try:
    model_pipeline = joblib.load('random_forest_tuned_model.joblib')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'random_forest_tuned_model.joblib' not found. Please ensure it's in the same directory.")
    st.stop()

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

st.set_page_config(page_title="Mental Health Risk Predictor in Tech", layout="wide")
st.title("🧠 Mental Health Risk Predictor in Tech Industry")
st.markdown("This application predicts the likelihood of an individual seeking **Mental Health Treatment** based on survey responses.")
st.markdown("---")
st.header("Please fill out the survey below:")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Demographics & Work Environment")
    age = st.slider("Age", 18, 70, 30)
    gender = st.selectbox("Gender", gender_options)
    country = st.selectbox("Country", country_options)
    state = st.selectbox("State (if applicable)", state_options)
    self_employed = st.radio("Are you self-employed?", yes_no_options)
    no_employees = st.selectbox("How many employees does your company have?", no_employees_options)
    remote_work = st.radio("Do you work remotely?", yes_no_options)
    tech_company = st.radio("Is your employer primarily a tech company/organization?", yes_no_options)
with col2:
    st.subheader("Mental Health History & Support")
    family_history = st.radio("Do you have a family history of mental illness?", yes_no_options)
    work_interfere = st.selectbox("If you have a mental health condition, do you feel that it interferes with your work?", work_interfere_options)
    st.markdown("---")
    st.subheader("Employer Provided Resources")
    benefits = st.selectbox("Does your employer provide mental health benefits?", benefits_options)
    care_options = st.selectbox("Do you know the options for mental health care your employer provides?", care_options_options)
    wellness_program = st.selectbox("Has your employer ever discussed mental health as part of an employee wellness program?", wellness_program_options)
    seek_help = st.selectbox("Does your employer provide resources to learn more about mental health issues and how to seek help?", seek_help_options)
    anonymity = st.selectbox("Is your anonymity protected if you choose to seek mental health care through your employer?", anonymity_options)
    leave = st.selectbox("How easy is it to take medical leave for a mental health condition?", leave_options)
    st.markdown("---")
    st.subheader("Perceived Stigma & Support")
    mh_consequence = st.selectbox("Do you think discussing a mental health issue with your employer would have negative consequences?", mh_consequence_options)
    ph_consequence = st.selectbox("Do you think discussing a physical health issue with your employer would have negative consequences?", ph_consequence_options)
    coworkers = st.selectbox("Would you be willing to discuss a mental health issue with your coworkers?", coworkers_supervisor_options)
    supervisor = st.selectbox("Would you be willing to discuss a mental health issue with your direct supervisor(s)?", coworkers_supervisor_options)
    mental_health_interview = st.selectbox("Would you bring up a mental health issue with a potential employer in an interview?", interview_options)
    phys_health_interview = st.selectbox("Would you bring up a physical health issue with a potential employer in an interview?", interview_options)
    mental_vs_physical = st.selectbox("Do you feel that your employer takes mental health as seriously as physical health?", mental_vs_physical_options)
    observed_consequence = st.radio("Have you heard or observed negative consequences for coworkers with mental health conditions in your workplace?", observed_consequence_options)

if st.button("Predict Mental Health Risk"):
    user_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Country': [country],
        'State': [state if state != 'nan' else np.nan],
        'Self_employed': [self_employed],
        'Family_history': [family_history],
        'Work_interfere': [work_interfere],
        'No_employees': [no_employees],
        'Remote_work': [remote_work],
        'Tech_company': [tech_company],
        'Benefits': [benefits],
        'Care_options': [care_options],
        'Wellness_program': [wellness_program],
        'Seek_help': [seek_help],
        'Anonymity': [anonymity],
        'Leave': [leave],
        'Mental_health_consequence': [mh_consequence],
        'Phys_health_consequence': [ph_consequence],
        'Coworkers': [coworkers],
        'Supervisor': [supervisor],
        'Mental_health_interview': [mental_health_interview],
        'Phys_health_interview': [phys_health_interview],
        'Mental_vs_physical': [mental_vs_physical],
        'Observed_consequence': [observed_consequence],
    })
    try:
        prediction_proba = model_pipeline.predict_proba(user_data)[:, 1][0]
        if prediction_proba >= 0.7:
            risk_level = "High"
            st.error(f"## Predicted Mental Health Risk: {risk_level}")
            st.write(f"Confidence Score: {prediction_proba:.2%}")
            st.warning("Based on your inputs, there is a **high likelihood** that you might benefit from mental health treatment. Please consider seeking support.")
        elif prediction_proba >= 0.4:
            risk_level = "Medium"
            st.warning(f"## Predicted Mental Health Risk: {risk_level}")
            st.write(f"Confidence Score: {prediction_proba:.2%}")
            st.info("Your responses suggest a **medium likelihood** of benefiting from mental health treatment. It might be helpful to explore available resources.")
        else:
            risk_level = "Low"
            st.success(f"## Predicted Mental Health Risk: {risk_level}")
            st.write(f"Confidence Score: {prediction_proba:.2%}")
            st.success("Your responses indicate a **low likelihood** of needing mental health treatment. Continue to prioritize your well-being!")
        st.markdown("---")
        st.subheader("Supportive Feedback & Resources:")
        st.write("""
        Mental health is just as important as physical health. If you or someone you know is struggling, help is available.
        - **National Suicide Prevention Lifeline:** Call or text 988 (US)
        - **Crisis Text Line:** Text HOME to 741741
        - **The Trevor Project:** 1-866-488-7386 (for LGBTQ youth)
        - **Find a Therapist:** Psychology Today, SAMHSA National Helpline (1-800-662-HELP)
        - **Workplace Resources:** Check with your HR department for Employee Assistance Programs (EAPs) or mental health benefits.
        """)
        st.markdown("---")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input fields are filled correctly.")
st.markdown("---")
st.caption("Disclaimer: This tool is for informational purposes only and does not constitute medical advice. Always consult with a qualified healthcare professional for any health concerns.")
