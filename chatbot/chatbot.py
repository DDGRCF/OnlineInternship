import os
import time

import streamlit as st
from loguru import logger

from util import Config, ModelFactory, parse_files

CONFIG = Config()

def parameters_set(st: st):
    if 'chat_dialogue' not in st.session_state:
        st.session_state['chat_dialogue'] = []
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = CONFIG.temperature
    if 'top_p' not in st.session_state:
        st.session_state['top_p'] = CONFIG.top_p
    if 'top_k' not in st.session_state:
        st.session_state['top_k'] = CONFIG.top_k
    if 'max_seq_len' not in st.session_state:
        st.session_state['max_seq_len'] = CONFIG.max_seq_len 
    if 'system_prompt' not in st.session_state:
        st.session_state['system_prompt'] =  CONFIG.system_prompt
    if 'string_dialogue' not in st.session_state:
        st.session_state['string_dialogue'] = ''
    if 'llm' not in st.session_state:
        st.session_state['llm'] = "LLaMA2"

def ui_set(st: st):
    # main config
    custom_css = """
        <style>
            .stTextArea textarea {font-size: 13px;}
            div[data-baseweb="select"] > div {font-size: 13px !important;}
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    # hidden footer and menu
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # left sider 
    st.sidebar.title("Hello🤗, I'am CodeAssistant")

    # model config
    model_expander_css = """
        <style>
        .streamlit-expanderHeader p { font-weight: bold; }
        </style>
    """

    left_sidebar = st.sidebar

    model_title = left_sidebar.container()

    model_expander = left_sidebar.expander("👉Change config? Expand here!👈")
    model_expander.markdown(model_expander_css, unsafe_allow_html=True)

    selected_option = model_expander.selectbox('Choose a LLaMA2 model:', ['LLaMA2-70B', 'LLaMA2-13B', 'LLaMA2-7B'], key='model')
    st.session_state['temperature'] = model_expander.slider('Temperature:', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    st.session_state['top_p'] = model_expander.slider('Top P:', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    st.session_state['max_seq_len'] = model_expander.slider('Max Sequence Length:', min_value=64, max_value=4096, value=2048, step=8)
    st.session_state['top_k'] = model_expander.slider('Top K:', min_value=1, max_value=100, value=40, step=1)
    model_title.markdown(f"<h2>Model: {selected_option}</h2>", unsafe_allow_html=True)

    # add code context TODO:
    left_sidebar.markdown(f"<h2>Add Context</h2>", unsafe_allow_html=True)
    upload_files = left_sidebar.file_uploader("Please choose files or dirs to add context", accept_multiple_files=True, label_visibility="collapsed")
    parse_files(upload_files)


def load_check(st: st):
    pass


def chat(st):
    # recycle to render chat ui
    model = ModelFactory.get(st.session_state.llm)
    for message in st.session_state.chat_dialogue:
        with st.chat_message("user"):
            st.markdown(message[0])
        if len(message[1]):
            with st.chat_message("assistant"):
                st.markdown(message[1])
    
    # chat message
    if prompt := st.chat_input("Please enter your code"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # string dialogue for debugging
            string_dialogue = "System: " + st.session_state.system_prompt + "\n\n"
            for message_tuple in st.session_state.chat_dialogue:
                string_dialogue += "User: " + message_tuple[0] + "\n\n"
                string_dialogue += "Assistant: " + message_tuple[1] + "\n\n"
            st.session_state.string_dialogue = string_dialogue # for debug

            # to generate iterator
            generator = model.do(prompt, 
                                   st.session_state.chat_dialogue, 
                                   st.session_state.system_prompt, 
                                   st.session_state.max_seq_len,
                                   st.session_state.temperature,
                                   st.session_state.top_p,
                                   st.session_state.top_k)

            # chat response
            chat_response = ""
            for item in generator:
                chat_response = item
                message_placeholder.markdown(chat_response + "▌")
            message_placeholder.markdown(chat_response)

        # history
        st.session_state.chat_dialogue.append((prompt, chat_response))


def run_app_ui():
    parameters_set(st)
    ui_set(st)
    load_check(st)
    chat(st)

# streamlit will to recycle to call main func
def main():
    run_app_ui()

if __name__ == "__main__":
    main()


