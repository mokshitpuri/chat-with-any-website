import os
from dotenv import load_dotenv
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

st.set_page_config(page_title="Chat with Any Website", page_icon="🌐", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: bold;
            color: #3b82f6;
        }
        .subtext {
            text-align: center;
            color: gray;
            margin-bottom: 25px;
        }
        .chat-bubble {
            padding: 12px 16px;
            border-radius: 12px;
            margin-bottom: 10px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .user {
            background-color: #2b5c8a;
            align-self: flex-end;
            color: white;
            margin-left: auto;
        }
        .bot {
            background-color: #444654;
            align-self: flex-start;
            color: white;
            margin-right: auto;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow-y: auto;
            max-height: 400px;
            padding: 15px;
            border: 1px solid #444;
            border-radius: 10px;
            background-color: #262730;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Chat with Any Website</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Enter a website URL, scrape its content, and start chatting!</div>', unsafe_allow_html=True)

# sidebar
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)

llm = ChatOpenAI(model_name=model_choice, temperature=temperature)

# Session State 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_url" not in st.session_state:
    st.session_state.current_url = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

#Helper Function to Create RAG Chain
def create_rag_chain(vectorstore, url):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. DO NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    qa_system_prompt = f"""
    You are a helpful assistant strictly answering based on the scraped website content.
    The website URL you are referring to is: {url}
    Context from the website:
    {{context}}
    If you cannot find the answer within this context, say:
    "The website does not contain information about that topic."
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
#URL Input
url = st.text_input("Enter a Website URL (with https://)")

# Scrape Button
if st.button("Scrape Website"):
    if not url:
        st.warning("⚠️ Please enter a website URL first.")
    
    #Block YouTube URLs
    elif "youtube.com" in url or "youtu.be" in url:
        st.error("❌ This app cannot scrape YouTube video transcripts. Please use a text-based website.")
    
    #New Caching Logic
    else:
        try:
            url_hash = str(hash(url))
            persist_dir = f"chroma_cache_{url_hash}"

            if os.path.exists(persist_dir):
                with st.spinner("⏳ Loading from cache..."):
                    embeddings = OpenAIEmbeddings()
                    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
                    
                    st.session_state.vectorstore = vectorstore
                    st.session_state.current_url = url
                    st.session_state.messages = []  # reset chat
                    st.session_state.rag_chain = create_rag_chain(vectorstore, url)
                    
                    st.success(f"✅ Loaded '{url}' from cache!")
            
            # If not cached, scrape it for the first time
            else:
                with st.spinner("🔍 Scraping website..."):
                    response = requests.get(url, timeout=10)
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    main_content = soup.find("main") or soup.find("body")
                    
                    if main_content:
                        text = main_content.get_text(separator=" ", strip=True)
                    else:
                        text = soup.get_text(separator=" ", strip=True) # Fallback

                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_text(text)
                    
                    if not chunks:
                        st.error("❌ Error: Could not extract any text. The website might be heavily JavaScript-based.")
                    else:
                        embeddings = OpenAIEmbeddings()
                        # Create the new, unique cache directory
                        vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=persist_dir)

                        st.session_state.vectorstore = vectorstore
                        st.session_state.current_url = url
                        st.session_state.messages = []  # reset chat
                        st.session_state.rag_chain = create_rag_chain(vectorstore, url)
                        
                        st.success(f"✅ Scraped and cached '{url}' successfully!")

        except Exception as e:
            st.error(f"❌ Error: {e}")

if st.session_state.messages:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role_class = "user" if msg["role"] == "user" else "bot"
        st.markdown(f'<div class="chat-bubble {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
elif st.session_state.vectorstore:
    st.info("💬 Start chatting below once you're ready!")
else:
    st.markdown("<p style='text-align:center;color:gray;'>🔒 Please scrape a website first to unlock the chat.</p>", unsafe_allow_html=True)

# Disable Chat Until Scraping
disabled = st.session_state.rag_chain is None
placeholder = "Ask something about the website content..." if not disabled else "Please scrape a website first!"

user_input = st.chat_input(placeholder=placeholder, disabled=disabled)

if user_input and not disabled:
    
    #1. Format Chat History
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    #2. Add User's New Message to UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    #3. Invoke the RAG Chain (from session state)
    response = st.session_state.rag_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    bot_response = response["answer"]
    
    # 4. Add Bot's Response to UI
    st.session_state.messages.append({"role": "bot", "content": bot_response})
    
    #5. Rerun to Display New Messages
    st.rerun()