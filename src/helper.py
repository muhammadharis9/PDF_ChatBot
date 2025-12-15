import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    loader = PyPDFLoader(file_path=file_path)
    load = loader.load()
    return load

def splitting(load):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap= 100)

    splitted_text = text_splitter.split_documents(load)
    return splitted_text

def embeds(splitted_text):
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embed = HuggingFaceEmbeddings(model=embedding_model)
    PERSIST_DIR = "./chroma_db"
    final_embed = Chroma.from_documents(
        documents=splitted_text,
        embedding=embed,
        collection_name="pdfs",
        persist_directory=PERSIST_DIR)
    
    return final_embed
    

#If you want to load your own model then use the following function
 
def custom_embeds(load):
    embedding_model1 = input("Please provide the model for embedding which you want to load: ")
    embed = HuggingFaceEmbeddings(model=embedding_model)
    
    final_embed = Chroma.from_documents(
        documents=splitted_text,
        embedding=embed,
        collection_name="pdfs",
        persist_directory=PERSIST_DIR
        )

    return final_embed


