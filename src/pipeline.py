from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os 
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

def retrieval(final_embed , question):
    llm = ChatOpenAI(
        model="alibaba/tongyi-deepresearch-30b-a3b:free",
        openai_api_base="https://openrouter.ai/api/v1", 
        openai_api_key=OPENROUTER_API_KEY,              
        temperature=0.2,                                
        max_retries=3)

    ret = final_embed.as_retriever(search_kwargs = {"k" : 5})

    template = """
    You are an expert AI document analyst. Your task is to provide answers based STRICTLY on the context provided below from a document.

    Guidelines:
    1. If the answer is NOT found in the context, clearly state: "The requested information is not available in the document."
    2. Keep your answer clear, structured, and easy to read.
    3. Use bullet points or lists if listing multiple items.
    4. Do NOT make up facts or information that is not supported by the Context section.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """

    my_prompt = PromptTemplate(
        template=template,
        input_variables= ["context" , "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = ret,
        return_source_documents=True,
        chain_type_kwargs = {"prompt" : my_prompt})

    response = qa_chain.invoke({"query": question})
    
    return response['result']
