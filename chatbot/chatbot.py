import os
import time

import streamlit as st
from util import Config, parse_files, model_run

CONFIG = Config()

def ui_set(st: st):
    # main config
    custom_css = """
        <style>
            .stTextArea textarea {font-size: 13px;}
            div[data-baseweb="select"] > div {font-size: 13px !important;}
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    # 将streamlit自带的footer和menu隐藏
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # 设置侧栏的标题
    st.sidebar.title("Hello, I'am CodeAssistant")

    # 模型配置
    model_expander_css = """
        <style>
        .streamlit-expanderHeader p { font-weight: bold; }
        </style>
    """

    left_sidebar = st.sidebar

    model_title = left_sidebar.container()

    model_expander = left_sidebar.expander("Change model config? Expand here!")
    model_expander.markdown(model_expander_css, unsafe_allow_html=True)

    selected_option = model_expander.selectbox('Choose a LLaMA2 model:', ['LLaMA2-70B', 'LLaMA2-13B', 'LLaMA2-7B'], key='model')
    st.session_state['temperature'] = model_expander.slider('Temperature:', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    st.session_state['top_p'] = model_expander.slider('Top P:', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    st.session_state['max_seq_len'] = model_expander.slider('Max Sequence Length:', min_value=64, max_value=4096, value=2048, step=8)
    model_title.markdown(f"<h2>Model: {selected_option}</h2>", unsafe_allow_html=True)

    # 添加上下文
    left_sidebar.markdown(f"<h2>Add Context</h2>", unsafe_allow_html=True)
    upload_files = left_sidebar.file_uploader("Please choose files or dirs to add context", accept_multiple_files=True, label_visibility="collapsed")
    parse_files(upload_files)

def load_check(st: st):
    # TODO:
    pass

def parameters_set(st: st):
    if 'chat_dialogue' not in st.session_state:
        st.session_state['chat_dialogue'] = []
    if 'llm' not in st.session_state:
        #st.session_state['llm'] = REPLICATE_MODEL_ENDPOINT13B
        st.session_state['llm'] = "TODO"
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = CONFIG.temperature
    if 'top_p' not in st.session_state:
        st.session_state['top_p'] = CONFIG.top_p
    if 'max_seq_len' not in st.session_state:
        st.session_state['max_seq_len'] = CONFIG.max_seq_len 
    if 'pre_prompt' not in st.session_state:
        st.session_state['pre_prompt'] =  CONFIG.pre_prompt
    if 'string_dialogue' not in st.session_state:
        st.session_state['string_dialogue'] = ''

def chat(st):
    # 设置初始变量
    for message in st.session_state.chat_dialogue:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 产生对话框，然后输入问题 prompt = 输入内容
    if prompt := st.chat_input("Please enter your code"):
        st.session_state.chat_dialogue.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # 这里是全部信息
            string_dialogue = st.session_state['pre_prompt']
            for dict_message in st.session_state.chat_dialogue:
                if dict_message["role"] == "user":
                    string_dialogue = string_dialogue + "User: " + dict_message["content"] + "\n\n"
                else:
                    string_dialogue = string_dialogue + "Assistant: " + dict_message["content"] + "\n\n"
            print(string_dialogue)
            
            # 要把全部上下问送入模型
            output = model_run(st.session_state['llm'], string_dialogue + "Assistant: ",  st.session_state['max_seq_len'], st.session_state['temperature'], st.session_state['top_p'], "token")
            
            # 回复信息
            chat_response = ""
            for item in output:
                chat_response += item
                message_placeholder.markdown(chat_response + "▌")
            message_placeholder.markdown(chat_response)
        # Add assistant response to chat history
        st.session_state.chat_dialogue.append({"role": "assistant", "content": chat_response})


def run_app_ui():
    parameters_set(st)
    ui_set(st)
    load_check(st)
    chat(st)

# streamlit 会一遍遍地调用main函数，也就说下面将是一个循环
def main():
    run_app_ui()

if __name__ == "__main__":
    main()


