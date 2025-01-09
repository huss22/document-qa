import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Databricks API Configuration ---
# Use st.secrets to store sensitive information
DATABRICKS_HOST = st.secrets["DATABRICKS_HOST"]
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]

# --- Endpoint Names ---
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-70b-instruct"  # Replace with your LLM endpoint name
EMBEDDING_ENDPOINT_NAME = "databricks-gte-large-en"  # Replace with your embedding endpoint name

# --- Helper Functions ---
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""
    return text

def chunk_text(text, chunk_size=768, chunk_overlap=50):
    """Splits the text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_text(text)

def generate_embedding(text):
    """Generates an embedding for a given text using the Databricks serving endpoint."""
    url = f"{DATABRICKS_HOST}/serving-endpoints/{EMBEDDING_ENDPOINT_NAME}/invocations"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {"inputs": [text]}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()["predictions"][0]
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating embedding for text: '{text[:50]}...'")  # Print part of the text
        st.error(f"Request error: {e}")
        if 'response' in locals():
          st.error(f"Response status code: {response.status_code}")
          st.error(f"Response text: {response.text}")
        else:
          st.error("Response object not available")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"Error parsing embedding response: {e}")
        return None

def generate_embeddings(text_chunks):
    """Generates embeddings for a list of text chunks."""
    embeddings = []
    for chunk in text_chunks:
        embedding = generate_embedding(chunk)
        if embedding is not None:  # Only add if embedding was generated successfully
            embeddings.append(embedding)
    return np.array(embeddings)

def add_embeddings_to_index(index, embeddings):
    """Adds embeddings to a FAISS index."""
    index.add(np.array(embeddings).astype('float32'))

def search_index(index, query_embedding, k=5):
    """Searches the FAISS index for the top k nearest neighbors."""
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
    return indices[0]

def query_llm(context, question):
    """Queries the LLM serving endpoint with a context and a question."""
    url = f"{DATABRICKS_HOST}/serving-endpoints/{LLM_ENDPOINT_NAME}/invocations"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant chatbot that answers questions based on a provided context. "
                       "If the question cannot be answered using the information in the context, say 'I don't know.'",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}",
        },
    ]
    data = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        st.error(f"Error querying LLM: {response.text}")
        return None
    return response.json()["choices"][0]["message"]["content"]

# --- Streamlit App ---
st.title("📄 Chat with your PDFs using Databricks Endpoints")
st.write(
    "Upload one or more PDF documents and ask questions about their content. "
    f"Powered by {LLM_ENDPOINT_NAME} and {EMBEDDING_ENDPOINT_NAME}."
)

uploaded_files = st.file_uploader(
    "Upload your PDF documents", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    all_chunks = []
    for uploaded_file in uploaded_files:
        # Process uploaded files directly using BytesIO
        pdf_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(pdf_text)
        all_chunks.extend([(uploaded_file.name, chunk) for chunk in chunks])

    # Using st.cache_data to cache embeddings and index
    @st.cache_data(show_spinner=True)
    def process_and_index(all_chunks):
        all_text_chunks = [chunk for _, chunk in all_chunks]

        # Handle empty all_text_chunks
        if not all_text_chunks:
            st.warning("No text chunks found in the uploaded documents.")
            return None, None  # Or raise an exception, or return a default index

        embeddings = generate_embeddings(all_text_chunks)

        # Handle cases where no embeddings were generated
        if embeddings is None or embeddings.size == 0:
            st.warning("Could not generate embeddings for any text chunks.")
            return None, None

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Using Inner Product
        add_embeddings_to_index(index, embeddings)
        return embeddings, index

    with st.spinner('Processing documents and creating index...'):
        embeddings, index = process_and_index(all_chunks)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question about the documents:"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        query_embedding = generate_embedding(question)
        if query_embedding is not None:
            relevant_indices = search_index(index, query_embedding, k=5)
            relevant_chunks = [all_chunks[i][1] for i in relevant_indices]
            context = "\n\n".join(relevant_chunks)

            # Query the LLM serving endpoint
            with st.spinner("Generating response..."):
                response = query_llm(context, question)

            if response:
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.write("Please upload some PDF documents to get started.")