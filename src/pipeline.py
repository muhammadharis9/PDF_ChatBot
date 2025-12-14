from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

def retrieval(final_embed , question):
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1, 
    max_tokens=None,
    timeout=None,
    max_retries=2)

    ret = final_embed.as_retriever(search_kwargs = {"k" : 2})

    template = """
    You are an expert in AI. Your task is to answer questions based ONLY on the following context provided from a PDF document.

    Guidelines:
    2. Keep your answer clear, structured, and easy to read.
    3. Use bullet points if listing multiple items.
    4. Do not make up information.
    5. You can also search information from internet and give him suggestion

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
    