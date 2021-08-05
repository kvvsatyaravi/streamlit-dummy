import streamlit as st
import streamlit.components.v1 as components
from backend import chat

st.title("Dummy bot")

inputvalue,but = st.beta_columns(2)
inputvalue.text_input("write your input here")
submit = but.button("Submit")

if submit:
	st.text("User: "+ str(inputvalue))
	st.text("Dummy: "+ botvalue)