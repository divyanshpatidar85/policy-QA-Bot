# import streamlit as st
# import os
# from vector_store import build_faiss, load_faiss
# from qa_engine import get_llm
# from langchain.chains import RetrievalQA

# UPLOAD_DIR = "uploads"
# INDEX_DIR = "indexes"

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(INDEX_DIR, exist_ok=True)

# st.set_page_config(page_title="Policy Chatbot", layout="centered")
# st.title("📑 Policy Chatbot")

# # ---- File Upload ----
# uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
# if uploaded:
#     file_path = os.path.join(UPLOAD_DIR, uploaded.name)
#     with open(file_path, "wb") as f:
#         f.write(uploaded.getbuffer())
#     st.success(f"Saved {uploaded.name}")

#     # Build FAISS index for this file
#     index_path = os.path.join(INDEX_DIR, uploaded.name.replace(".pdf", ""))
#     st.text(file_path)
#     build_faiss(file_path, index_path=index_path)
#     st.session_state["active_index"] = index_path
#     st.success("Index built ✅ — Start chatting below!")

# # ---- Chat Window ----
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# for msg in st.session_state["messages"]:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])

# if prompt := st.chat_input("Ask me anything about the policy..."):
#     # Show user message
#     st.session_state["messages"].append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.write(prompt)

#     # Load retriever
#     if "active_index" in st.session_state:
#         index = load_faiss(st.session_state["active_index"])
#         retriever = index.as_retriever(search_kwargs={"k": 4})
#         llm = get_llm()
#         qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

#         answer = qa.run(prompt)

#         # Bot answer
#         st.session_state["messages"].append({"role": "assistant", "content": answer})
#         with st.chat_message("assistant"):
#             st.write(answer)
#     else:
#         with st.chat_message("assistant"):
#             st.write("⚠️ Please upload a PDF first.")



import streamlit as st
import os
from vector_store import build_faiss, load_faiss
from qa_engine import get_llm, load_qa  # ✅ use updated functions

UPLOAD_DIR = "uploads"
INDEX_DIR = "indexes"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

st.set_page_config(page_title="Policy Chatbot", layout="centered")
st.title("📑 Policy Chatbot")

# ---- File Upload ----
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded:
    file_path = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved {uploaded.name}")

    # Build FAISS index for this file
    index_path = os.path.join(INDEX_DIR, uploaded.name.replace(".pdf", ""))
    st.text(file_path)
    build_faiss(file_path, index_path=index_path)
    st.session_state["active_index"] = index_path
    st.session_state["qa_chain"] = load_qa(index_path)  # ✅ create retrieval chain
    st.success("Index built ✅ — Start chatting below!")

# ---- Chat Window ----
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask me anything about the policy..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if "qa_chain" in st.session_state:
        qa_chain = st.session_state["qa_chain"]

        # ✅ Run using LangChain 1.x pipeline
        raw = qa_chain.invoke(prompt)
        answer = raw.content if hasattr(raw, "content") else str(raw)

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

    else:
        with st.chat_message("assistant"):
            st.write("⚠️ Please upload a PDF first.")
