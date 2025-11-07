# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA
# from langchain_core.messages import HumanMessage
# from vector_store import load_faiss

# def get_llm():
#     return ChatGroq(
#         groq_api_key="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",
#         model="llama-3.1-8b-instant",
#         temperature=0.0,
#         max_tokens=512
#     )

# def load_qa(index_path="faiss_index"):
#     index = load_faiss(index_path)
#     retriever = index.as_retriever(search_kwargs={"k": 4})
#     llm = get_llm()
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True
#     )

# def ask_question(qa, question, role_prompt, q):
#     answer_pack = qa({"query": question})
#     raw_answer, docs = answer_pack["result"], answer_pack["source_documents"]

#     llm = get_llm()
#     reframed = llm.invoke([HumanMessage(content=f"{role_prompt}\n\nQ: {q}\n\nA: {raw_answer}")])
#     return reframed.content if hasattr(reframed, "content") else reframed, docs

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from vector_store import load_faiss


def get_llm():
    return ChatGroq(
        groq_api_key="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",
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=512
    )


def load_qa(index_path="faiss_index"):
    # Load FAISS
    index = load_faiss(index_path)
    retriever = index.as_retriever(search_kwargs={"k": 4})

    # Define response prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful policy assistant.
Answer using ONLY the provided documents.

Documents:
{context}

Question:
{question}

If answer is not found in the documents, reply with:
"I don’t know based on the provided documents."
""")

    llm = get_llm()

    # Build Runnable retrieval chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain


def ask_question(qa_chain, question, role_prompt, q):
    # Run retrieval → returns ChatMessage
    raw = qa_chain.invoke(question)
    answer_text = raw.content if hasattr(raw, "content") else str(raw)

    # Reframe tone with role prompt
    llm = get_llm()
    reframed = llm.invoke(
        f"{role_prompt}\n\nQ: {q}\n\nA: {answer_text}"
    )

    return reframed.content if hasattr(reframed, "content") else reframed, []
