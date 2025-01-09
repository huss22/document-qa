import streamlit as st
import fitz  # PyMuPDF
import requests
import json
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import pytesseract

# --- Databricks API Configuration ---
# Use st.secrets to store sensitive information
DATABRICKS_HOST = st.secrets["DATABRICKS_HOST"]
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]

# --- Endpoint Names ---
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-70b-instruct"  # Replace with your LLM endpoint name

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file, using OCR if necessary."""
    text = ""
    try:
        # Open the PDF from the BytesIO object
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            print(f"Number of pages: {doc.page_count}")
            for page in doc:
                # Try to get text directly
                page_text = page.get_text()
                if page_text.strip():  # If text is found, use it
                    print(f"Page {page.number + 1} (text):\n{page_text[:200]}...")
                    text += page_text
                else:  # Otherwise, try OCR
                    print(f"Page {page.number + 1} (image): Performing OCR...")
                    # Get image of the page
                    image_list = page.get_images(full=True)
                    if image_list:
                        # Get the xref of the image
                        xref = image_list[0][0]
                        # Extract the image
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        # Convert to PIL image
                        image = Image.open(io.BytesIO(image_bytes))
                        # Perform OCR
                        page_text = pytesseract.image_to_string(image)
                        print(f"Page {page.number + 1} (OCR):\n{page_text[:200]}...")
                        text += page_text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        print(f"Error processing PDF: {e}")
        return ""
    print(f"Extracted text length: {len(text)}")
    return text

def chunk_text(text, chunk_size=3000, chunk_overlap=100):  # Tweaked chunk size
    """Splits the text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(text)
    print(f"Number of chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk[:200]}...")
    return chunks

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
st.title("📄 Chat with your PDFs using Llama 3")
st.write(
    "Upload one or more PDF documents and ask questions about their content. "
    f"Powered by {LLM_ENDPOINT_NAME}."
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
        all_chunks.extend(chunks)  # Store chunks directly

    print(f"all_chunks: {all_chunks}")

    # No need for caching embeddings and index

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

        # Concatenate all chunks for simplicity (consider more advanced strategies for long documents)
        context = "\n\n".join(all_chunks)

        # Query the Llama 3 serving endpoint
        with st.spinner("Generating response..."):
            response = query_llm(context, question)

        if response:
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.write("Please upload some PDF documents to get started.")