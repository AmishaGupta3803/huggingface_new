from langchain.llms import HuggingFaceHub
import streamlit as st
from streamlit_chat import message
import os
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

prompt = PromptTemplate(
    input_variables=["temp"],
    template='{temp}, give clear answer and explain in steps'
)

st.title('Chatbot')
llm = HuggingFaceHub(
    huggingfacehub_api_token=os.environ.get('HUGGINGFACEHUB_API_TOKEN'),
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={'temperature': 0.2, 'max_tokens': -1}
)

if 'messages' not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Enter prompt: ")
st.session_state.messages.append(user_input)
if user_input:
    message(user_input, is_user=True)
    response = llm(user_input)
    st.session_state.messages.append(response)
    message(response, is_user=False)

