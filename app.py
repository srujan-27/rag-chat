import os
import tempfile
import uuid
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- Page config ---
st.set_page_config(page_title="Chat with your PDFs", page_icon="")
st.title(" Chat with your PDFs")

# --- Sidebar: file uploader ---
with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Drop PDFs here", type="pdf", accept_multiple_files=True
    )
    process_btn = st.button("Process documents", type="primary")

# --- Initialize session state ---
import uuid
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Process uploaded PDFs ---
if process_btn and uploaded_files:
    with st.spinner("Reading, chunking, and embedding your PDFs..."):
        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for uploaded_file in uploaded_files:
            # Save to temp file so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Add filename to metadata so citations show which doc
            for page in pages:
                page.metadata["source"] = uploaded_file.name

            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            os.unlink(tmp_path)  # clean up temp file

        # Build vector store
        db = Chroma.from_documents(
            all_chunks,
            OpenAIEmbeddings(),
            collection_name=f"session_{st.session_state.session_id}"
        )
        retriever = db.as_retriever(search_kwargs={"k": 4})

        # Build prompt
        template = """Answer the question based ONLY on the following context.
After your answer, list the specific sources you used in this format:
- [Source: filename, Page X]

If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model="gpt-4o-mini")

        def format_docs(docs):
            # Store retrieved docs in session state for citation display
            st.session_state.last_sources = docs
            return "\n\n".join(
                f"[Source: {d.metadata['source']}, Page {d.metadata['page'] + 1}]: {d.page_content}"
                for d in docs
            )

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        st.session_state.chain = chain
        st.session_state.messages = []

    st.success(f"Processed {len(uploaded_files)} file(s) into {len(all_chunks)} chunks!")

# --- Chat interface ---
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about your documents"):
    if st.session_state.chain is None:
        st.error("Please upload and process a PDF first.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response = st.session_state.chain.invoke(question)
                answer = response.content

            st.markdown(answer)

            # Show source chunks in an expander
            if hasattr(st.session_state, "last_sources"):
                with st.expander(" View source passages"):
                    for i, doc in enumerate(st.session_state.last_sources):
                        st.caption(
                            f"**Source {i+1}** — {doc.metadata['source']}, "
                            f"Page {doc.metadata['page'] + 1}"
                        )
                        st.text(doc.page_content[:500])
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})