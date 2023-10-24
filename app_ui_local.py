import os
import re
from datetime import datetime
import streamlit as st
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from dotenv import load_dotenv
from htmlTemplates import bot_template, user_template
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from streamlit_js_eval import streamlit_js_eval
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers").setLevel(logging.DEBUG)

def refresh_chat_history(container):
    count = 1
    expander = None
    with container:
        for message in st.session_state.messages:
            role = message["role"]
            if role == "user":
                expander = st.expander(message["content"])
                count = count + 1
            if "content" in message:
                write_message(role, message["content"], expander)
            if st.session_state.display_source is True and "source_ref" in message:
                write_message(role, message["source_ref"], expander)

def write_message(role, message, container):
    with container:
        if role == "user":
            st.write(user_template, unsafe_allow_html=True)
            st.markdown(message,unsafe_allow_html=False)
        else:
            output = "<pre>"+message+"</pre>"
            st.write(bot_template, unsafe_allow_html=True)
            st.markdown(output,unsafe_allow_html=True)

def write_simple_message(message, container):
    with container:
        st.markdown(message)

@st.cache_data
def save_chat_history(prompt, response, sources):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.messages.append({"role": "assistant", "source_ref": sources})
    return st.session_state.messages

def main():
    # tab1, tab2, col1, col2, sidebar = init()
    tab1, tab2, sidebar = init()
    with tab1:
        prompt = st.text_input("Ask your Digital Assistant a question")
        if prompt:
            write_message("user", prompt, tab1)
            response, sources = handle_user_request(prompt)
            write_message("assistant", response, tab1)
            if st.session_state.display_source is True:
                write_message("assistant", sources, tab1)
            save_chat_history(prompt, response, sources)

    with sidebar:
        on = st.toggle("Display Source Reference")
        if on:
            st.session_state.display_source = True
            refresh_chat_history(tab2)
        else:
            st.session_state.display_source = False
            refresh_chat_history(tab2)
        
        on_mq_retriever = st.toggle("Use MultiQuery Retriever")
        if on_mq_retriever:
            st.session_state.use_mq_retriever = True
        else:
            st.session_state.use_mq_retriever = False


        if st.button("clear chat history"):
            st.cache_data.clear()
            streamlit_js_eval(js_expressions="parent.window.location.reload()")


def init():
    load_dotenv()
    get_doc_retriever()
    get_doc_retriever_multi_q()
    get_llm()
    st.title("Banking Compliance Bot")
    tab1, tab2 = st.tabs(["Current Chat","Chat History"])
    tab1.write("Our Digital Assistant will help with your questions here")
    tab2.write("Your chat history with our Digital Assistant can be found here")
    # col1, col2 = st.columns(2)
    # Initialize session data
    if "display_source" not in st.session_state:
        st.session_state.display_source = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "use_mq_retriever" not in st.session_state:
        st.session_state.use_mq_retriever = False
    # return tab1, tab2, col1, col2, st.sidebar
    return tab1, tab2, st.sidebar
      

@st.cache_resource
def load_vec_db(file_path):
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vec_db = FAISS.load_local(file_path,embeddings)
    print("Done Loading the vec database")
    return vec_db

@st.cache_resource
def load_mem_store(file_path):
    fs = LocalFileStore(file_path)
    mem_store = create_kv_docstore(fs)
    return mem_store

@st.cache_resource
def get_doc_retriever():
    vec_db = load_vec_db(os.environ["vec_db_index"])
    mem_store = load_mem_store(os.environ["mem_store_path"])
    max_doc_retrieved=int(os.environ["max_num_docs_local"])
    # llm = VertexAI(model_name="text-bison", max_output_tokens=1000, temperature=0.1, top_k=40, top_p=0.8)
    llm = get_llm()
    compressor = LLMChainExtractor.from_llm(llm)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
    retriever = ParentDocumentRetriever(
    vectorstore=vec_db, 
    docstore=mem_store, 
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k":max_doc_retrieved}
    )
    return retriever
    # compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    # return compression_retriever

@st.cache_resource
def get_doc_retriever_multi_q():
    vec_db = load_vec_db(os.environ["vec_db_index"])
    mem_store = load_mem_store(os.environ["mem_store_path"])
    max_doc_retrieved=int(os.environ["max_num_docs_local"])
    llm = get_llm()
    compressor = LLMChainExtractor.from_llm(llm)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
    retriever = ParentDocumentRetriever(
    vectorstore=vec_db, 
    docstore=mem_store, 
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k":max_doc_retrieved}
    )
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever_from_llm)
    return compression_retriever


@st.cache_resource
def get_llm():
    n_gpu_layers = 32  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 4096  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
    model_path="/Users/jackeyng/Projects/llama/cpp/llama.cpp/models/7B/ggml-model-q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=4096,
    temperature=0.2,
    max_tokens=1024,
    top_p=0.8,
    top_k=40,
    # callback_manager=callback_manager,
    # verbose=True
    )
    # llm = VertexAI(model_name="text-bison", max_output_tokens=1000, temperature=0.1, top_k=40, top_p=0.8)
    return llm

def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def get_prompt_template(context, question):
    template = """You are given the regulatory rules which are between "[START RULE]" and "[END RULE]". You are also given a question which is between "[START QUESTION]" and "[END QUESTION]". 
    Please answer the question like a compliance officer in a bank. Elaborate your answer to give examples and supporting rationales. 
    Don't make up answers for questioins which you do not have sufficient information to answer. Just say: "Sorry, I do not have sufficient information to answer the question"
    Always give your answer in English.
    
    [START RULE] 
    {context} 
    [END RULE]

    [START QUESTION] 
    {question} 
    [END QUESTION]

   """
    prompt = PromptTemplate(template=template, input_variables=["question","context"])
    return prompt.format(context=context,question=question)

def format_response(response):
    response = remove_html_tags(response)
    return response

def format_response_for_sources(docs):
    response = "\n\nThe following sources are considered: \n "
    sources = "<ul>"
    for d in docs:
        sources = sources + "<li>"+d.metadata["source"]+"</li>"
        sources = sources + "<pre>" + d.page_content+"</pre>"
    sources = sources+"</ul>"
    response = response + sources
    return response

@st.cache_data
def handle_user_request(user_question):
    retriever = None
    if st.session_state.use_mq_retriever:
        retriever = get_doc_retriever_multi_q()
    else:
        retriever = get_doc_retriever()
    print_time("Retrieving relevant documents")
    docs=retriever.get_relevant_documents(user_question)
    print_time("Retrieved relevant documents")
    context = remove_html_tags(f"\n{'-' * 100}\n".join([f"Rule {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))
    prompt = get_prompt_template(context,user_question)
    llm = get_llm()
    print_time("Start quering LLM")
    response = llm(prompt)
    print_time("Got response from LLM")
    return format_response(response), format_response_for_sources(docs)

def print_time(message):
    print(datetime.now().strftime("%H:%M:%S")+" "+message)


if __name__ == '__main__':
    main()

