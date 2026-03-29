import streamlit as st

st.set_page_config(
    page_title="PM2.5 HCMC",
    page_icon="🌫️",
    layout="wide",
)

st.title("Predict PM2.5 – CMT8 Station HCMC")
st.markdown("Choose option from sidebar to view current air quality or forecast PM2.5 levels.")

# streamlit run app.py