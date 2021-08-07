from urllib import request
import streamlit as st
import streamlit.components.v1 as components
import random
import json
import torch
from backend.nltk_utils import bag_of_words, tokenize, stem
from backend.Model import NeuralNet
from backend.train import trained
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('backend/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "backend/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

print("Let's test! (type 'quit' to exit)")

def chat():
    while True:
        # sentence = "do you use credit cards?"
        sentence = name
        if sentence == "quit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    botvalue = f"{random.choice(intent['responses'])}"
        else:
            botvalue = f"I do not understand..."

        return botvalue

st.title("Dummy bot")
form = st.form(key='my-form')
name = form.text_input('Enter input')
submit = form.form_submit_button('Submit')

if name == '/train':
    os.system('python backend/train.py')
    st.write(trained())
    
    st.write("trained data successful")
elif submit:
	st.write("User: "+ name)
	st.write("Dummy: "+ chat())
else:
    st.write("there is some error in your code")