from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

def retrieval():
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1, 
    max_tokens=None,
    timeout=None,
    google_api_key = "AIzaSyB1Gm74LHNB9kT2tnI4xyay1Pc1iqB_lk4",
    max_retries=2)

    ret = final_embed.as_retriever(search_kwargs = {"k" : 2})

    template = """
    You are an expert and professional in reading PDFs. Your task is to answer questions based ONLY on the following context provided from a PDF document.

    Guidelines:
    1. Suggest him about job portals as well.
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

    print("--- PDF Chatbot Ready! (Type 'exit' or 'quit' to stop) ---")


    response = qa_chain.invoke({"query": question})
    print("-" * 50)
    print("Answer:")
    print(response['result'])
    

