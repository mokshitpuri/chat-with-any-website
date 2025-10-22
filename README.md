# Chat with Any Website — LangChain App

A powerful LangChain-based web application that lets you chat with any website just by entering its URL.
It scrapes the site, creates embeddings, and allows you to ask context-aware questions about the site’s content — all powered by LangChain, OpenAI, and Chroma.

## 🚀 Features

🔍 URL-based scraping — Enter any website URL to fetch content dynamically

🧩 Smart chunking & embeddings — Uses RecursiveCharacterTextSplitter and OpenAIEmbeddings for efficient context storage

💬 Conversational AI — Chat naturally about the website using an LLM (OpenAI)

🧠 RAG pipeline — Retrieval-Augmented Generation ensures factual, site-based answers

⚡ Chroma VectorStore — Fast semantic search for relevant responses

🧰 Streamlit UI — Simple, responsive interface to interact with your model

## 🏗️ Tech Stack
Layer	Tools
Frontend/UI	Streamlit
Backend	Python
Framework	LangChain
Embeddings	OpenAIEmbeddings
Vector DB	Chroma
Scraping	BeautifulSoup, Requests
Environment Management	dotenv

## 🧪 Setup Instructions

Clone the repository

git clone https://github.com/yourusername/chat-with-any-website.git
cd chat-with-any-website


Create a virtual environment

python -m venv venv
source venv/bin/activate      # for macOS/Linux
venv\Scripts\activate         # for Windows


Install dependencies

pip install -r requirements.txt


Add your environment variables

Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_api_key


Run the app

streamlit run app.py


Enter any website URL (with https://) and start chatting!

## 🧠 How It Works

1. The app scrapes all text content from the given URL using BeautifulSoup.
2. The text is split into smaller chunks using LangChain’s RecursiveCharacterTextSplitter.
3. Each chunk is converted into embeddings via OpenAIEmbeddings.
4. The embeddings are stored in ChromaDB for fast retrieval.
5. When you ask a question, LangChain’s RetrievalQA pipeline fetches relevant chunks and generates an LLM-based response — grounded in the website’s content.