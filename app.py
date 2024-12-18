import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import os
from streamlit.web import cli as stcli
import warnings

warnings.filterwarnings('ignore')

st.title("Classifying the complaints raised by customers to categories using Artificial Intelligence")

classifier = pipeline('text-classification', model= 'bank_customer_ticket_category_classifier')#,device='cuda')

text = st.text_area("Please describe your issue.")

if st.button("Predict"):
    result = classifier(text)
    st.write(result)