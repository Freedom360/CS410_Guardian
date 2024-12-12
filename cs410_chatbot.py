# -*- coding: utf-8 -*-
"""cs410_chatbot.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SGHgBIvHK-OVSYzRvqb71lI5hfKTjp8X
"""

import os
import langchain
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = 'sk-mtYM6tMoovmKzZ_JjsczfBWJzZMJMRVltSXPcdISJCT3BlbkFJ3M5KrkowzA0ulAlgkgp9yt4y3_iWQBwmYIvoph1fUA'
llm = ChatOpenAI(model_kwargs={'temperature': 0})

from langchain_core.prompts import PromptTemplate

template = """ You are an assistant for UIUC's CS 410: Text Information Systems class.
You are an expert on lecture content and concepts related to text mining.
ONLY use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.


{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

import bs4
# from langchain import hub
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load, chunk and index the contents of the blog.
loader = TextLoader('/Users/michaelparekh/Downloads/week1_lecture_1.txt')


docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = custom_rag_prompt

def format_docs(docs):
      formatted_content = []
      for doc in docs:
        formatted_content.append(doc.page_content)
        print(formatted_content)
      result = "\n\n".join(formatted_content)
      return result


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("what are the three things covered in this lecture"))

print(rag_chain.invoke("give an analogy to undersand nlp"))

rag_chain.invoke('what is nlp? analogy')

print(rag_chain.invoke("give an analogy to undersand nlp"))

import os
import subprocess
import streamlit as st

# Streamlit app
st.title("💬 Coursera Transcript Downloader")
st.write(
    "This is a chatbot that uses OpenAI's GPT model + RAG to generate responses. "
    "It allows you to learn more about topics taught in CS 410 lectures. "
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    stream = rag_chain.invoke(prompt)

    with st.chat_message("assistant"):
        response = st.write(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
