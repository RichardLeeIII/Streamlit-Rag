from PyPDF2 import PdfReader
import openai
import streamlit as st
import numpy as np
import faiss

openai.api_key = st.secrets["OPEN_API_KEY"]

# Function to extract text from PDF
@st.cache_data
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to chunk text
@st.cache_data
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to get embeddings from OpenAI
def get_openai_embedding(text, engine="text-embedding-3-small"):
    response = openai.Embedding.create(input=[text], model=engine)
    return response['data'][0]['embedding']

# Function to get vector store using FAISS
def get_vectorstore(text_chunks):
    # Retrieve embeddings for each text chunk
    embeddings = [get_openai_embedding(chunk) for chunk in text_chunks]

    # Convert list of embeddings to numpy array and ensure data type is float32
    embeddings_array = np.array(embeddings).astype('float32')

    # Ensure there is at least one embedding to get dimension from
    if len(embeddings) == 0:
        raise ValueError("No embeddings were returned. Check your input.")

    dimension = len(embeddings_array[0])  # Dimension of embeddings

    # Create a FAISS index and add the embeddings to it
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    return index, text_chunks

# Function to query embeddings
def query_embeddings(query, index, text_chunks):
    query_embedding = get_openai_embedding(query, engine="text-embedding-3-small")
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k=3)  # Retrieve top 3 relevant chunks
    return [text_chunks[i] for i in indices[0]]

# Function to generate response based on relevant chunks
def generate_response(relevant_chunks, query):
    context = " ".join(relevant_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"].strip()

st.title("Chat with PDFs :books:")

with st.chat_message(name='assistant', avatar="./assets/profile_picture.png"):
    st.write("This chatbot is retrieval augmented with the PDF info. Please restrict your question related to content of PDF. There are limits in OpenAI token usage, so if an error message pops up, it means the monthly token usage is met.")

with st.chat_message(name='assistant', avatar="./assets/profile_picture.png"):
    st.write("Try download the Sample PDF from the Below button if you don't have any at the moment, this is American Declaration of Independence. For example, you can ask why the US wanted the independence.")

with open('./assets/dein.pdf', 'rb') as file:
    # Create a download button for the PDF
    st.download_button(
        label="Download Sample PDF",
        data=file,
        file_name='Declaration of Independence.pdf',
        mime='application/pdf'
    )

# File uploader for PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if pdf_file:
    # Extract text from uploaded PDF
    pdf_text = get_pdf_text(pdf_file)

    # Chunk text and get vector store
    chunks = chunk_text(pdf_text)
    index, text_chunks = get_vectorstore(chunks)

    # Accept user input
    if prompt := st.chat_input("Ask me anything based on the PDF information:"):
        # Query embeddings and get relevant chunks
        relevant_chunks = query_embeddings(prompt, index, text_chunks)
        
        # Generate response
        response = generate_response(relevant_chunks, prompt)
        
        # Display user message and assistant response
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
