\# 📄 Chat with your PDFs



A RAG (Retrieval-Augmented Generation) app that lets you upload PDF documents and ask questions about them, with answers grounded in the actual content and source citations.



\*\*Live demo:\*\* \[srujanragchat.streamlit.app](https://srujanragchat.streamlit.app)



\## Architecture



```

1\. PDF Upload → PyPDF (extract text per page)

2\. Text Splitting → RecursiveCharacterTextSplitter (1000 char chunks, 200 overlap)

3\. Embedding → OpenAI text-embedding-ada-002 (text → 1536-dim vectors)

4\. Storage → ChromaDB (vector store)

5\. Query → Same embedding model → Cosine similarity search → Top 4 chunks

6\. Answer → Prompt template (context + question) → GPT-4o-mini → Cited answer

```



\## Tech Stack



\- \*\*LLM \& Embeddings:\*\* OpenAI GPT-4o-mini + text-embedding-ada-002

\- \*\*Vector Store:\*\* ChromaDB (in-memory)

\- \*\*Framework:\*\* LangChain (chains, retrievers, prompt templates)

\- \*\*UI:\*\* Streamlit

\- \*\*Deployment:\*\* Streamlit Community Cloud



\## Features



\- Upload multiple PDFs simultaneously

\- Semantic search across all uploaded documents

\- Answers cite specific source files and page numbers

\- Expandable source passages for verification

\- Chat history maintained during session



\## Run Locally



```bash

git clone https://github.com/srujan-27/rag-chat.git

cd rag-chat

python -m venv venv

source venv/bin/activate  # Windows: .\\venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

```



Create a `.env` file:

```

OPENAI\_API\_KEY=sk-your-key-here

```



Run:

```bash

streamlit run app.py

```



\## Known Limitations



\- \*\*Ephemeral storage:\*\* ChromaDB runs in-memory on Streamlit Cloud — uploaded documents are lost when the app sleeps or restarts. Each session starts fresh.

\- \*\*No persistent chat:\*\* Conversation history resets on page refresh.

\- \*\*PDF only:\*\* Does not support Word docs, images, or other file types.

\- \*\*Token limits:\*\* Very large PDFs (500+ pages) may hit OpenAI token limits during embedding.



\## Built With



Built in \~4 hours as a learn-by-building project to understand RAG pipelines, vector embeddings, and LangChain.

