import os
import time

import streamlit as st
from loguru import logger

from util import Config, ModelFactory, ModelType, parse_files

CONFIG = Config()

# logger
if len(CONFIG.log_dir) == 0:
    CONFIG.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "logs")
    os.makedirs(CONFIG.log_dir, exist_ok = True)
    current_time = time.localtime(time.time())
    current_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)
    log_file = os.path.join(CONFIG.log_dir, f"{current_time_str}_chatbot.log")
    logger.add(log_file, rotation="50MB")


def run_app_ui():
    # parameters
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
    if 'model' not in st.session_state:
        st.session_state['model'] = CONFIG.model
    if 'model_path' not in st.session_state:
        st.session_state['model_path'] = CONFIG.model_path

    # main css
    custom_css = """
        <style>
            .stTextArea textarea {font-size: 10px;}
            div[data-baseweb="select"] > div {font-size: 10px !important;}
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
    left_sidebar = st.sidebar
    left_sidebar.title("Hello🤗, I'am CodeAssistant")

    model_options = left_sidebar.container()
    model_selected_title = model_options.container()

    model_expander = model_options.expander("👉Change config? Expand here!👈")

    model_expander_selected_option = model_expander.container()

    # TODO: add new model
    model_proto_selected, model_backend_selected, model_path_input = model_expander.columns([1, 1, 1])
    model_proto_text = model_proto_selected.selectbox("model proto", ModelType.get_all_model_name(), label_visibility = "collapsed", key="model_proto")
    model_backend_text = model_backend_selected.selectbox("model backend", ModelType.get_all_model_name(), label_visibility = "collapsed", key="model_backend")
    model_path_text = model_path_input.text_input("new model path: ", "", label_visibility = "collapsed") # TODO:

    st.session_state['temperature'] = model_expander.slider('Temperature:', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    st.session_state['top_p'] = model_expander.slider('Top P:', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    st.session_state['max_seq_len'] = model_expander.slider('Max Sequence Length:', min_value=64, max_value=4096, value=2048, step=8)
    st.session_state['top_k'] = model_expander.slider('Top K:', min_value=1, max_value=100, value=40, step=1)

    # model load
    model = ModelFactory.get(st.session_state.model, model_path = st.session_state.model_path)
    st.session_state.model_path = model.model_path

    model_proto_list = ModelFactory.keys()
    model_name_list = []
    for model_proto in model_proto_list:
        model_name_list.extend(model_proto.get_name_list())
    model_index = 0
    if model.model_name in model_name_list:
        model_index = model_name_list.index(model.model_name)

    model_selected_option = model_expander_selected_option.selectbox('Choose a model or add new model:', model_name_list, index=model_index)
    st.session_state.model_path = model.get_model_path()[model_selected_option]

    model_selected_title.markdown(f"<h2>Model: {model.model_name}</h2>", unsafe_allow_html=True)

    # add code context TODO:
    left_sidebar.markdown(f"<h2>Add Context</h2>", unsafe_allow_html=True)
    upload_files = left_sidebar.file_uploader("Please choose files or dirs to add context", accept_multiple_files=True, label_visibility="collapsed")
    parse_files(upload_files)

    
    # chat message
    for message in st.session_state.chat_dialogue:
        with st.chat_message("user"):
            st.markdown(message[0])
        if len(message[1]):
            with st.chat_message("assistant"):
                st.markdown(message[1])


    if prompt := st.chat_input("Please enter your code"):
        st.session_state.prompt = None
        with st.chat_message("user"):
            st.markdown(prompt)
    
        validity = False
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # string dialogue for debugging
            string_dialogue = "System: " + st.session_state.system_prompt + "\n\n"
            for message_tuple in st.session_state.chat_dialogue:
                string_dialogue += "User: " + message_tuple[0] + "\n\n"
                string_dialogue += "Assistant: " + message_tuple[1] + "\n\n"
            st.session_state.string_dialogue = string_dialogue # for debug

            # limit the input
            validity, st.session_state.chat_dialogue = model.reduce_input_token(prompt, 
                                                                      st.session_state.chat_dialogue, 
                                                                      st.session_state.system_prompt,
                                                                      st.session_state.max_seq_len)
            if not validity: 
                st.error("The input is too long!", icon="🚨")
            else:
                # stream generator
                generator = model.do(prompt, 
                                     st.session_state.chat_dialogue, 
                                     st.session_state.system_prompt, 
                                     st.session_state.max_seq_len,
                                     st.session_state.temperature,
                                     st.session_state.top_p,
                                     st.session_state.top_k)

                # chat response
                with st.spinner(""):
                    chat_response = next(generator)

                for item in generator:
                    message_placeholder.markdown(chat_response + "▌")
                    chat_response = item
                message_placeholder.markdown(chat_response)

        if validity:
            st.session_state.chat_dialogue.append((prompt, chat_response))

# streamlit will to recycle to call main func
def main():
    run_app_ui()

if __name__ == "__main__":
    main()


