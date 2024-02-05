import logging
from openai import OpenAI

client = OpenAI(api_key=st.secrets.openai_key)
from annoy import AnnoyIndex
import streamlit as st
from langchain.document_loaders import UnstructuredAPIFileLoader

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI API key

# Initialize text storage
text_storage = {}

# Error handling function
def handle_error(error_message):
    logging.error(f"An error occurred: {error_message}")
    st.error(f"An error occurred: {error_message}")
    st.warning("Please try again.")

# Function to extract text from PDF
def extract_text_from_pdf(file, pages_per_chunk=8):
    try:
        loader = UnstructuredAPIFileLoader(
            file_path=file,
            api_key=st.secrets["unstructured_api"]["key"],
            strategy="fast",
            mode="elements"
        )
        docs = loader.load()
        combined_text = " ".join([doc.text for doc in docs])
        return [combined_text]
    except Exception as e:
        handle_error(f"Error opening PDF file: {e}")
        return []

# Function to set up Annoy index
def setup_annoy(dimension=1536):
    index = AnnoyIndex(dimension, 'angular')
    return index

# Function to create OpenAI embedding
def create_openai_embedding(text, max_words=700):
    if not text:
        handle_error("Input text is empty or None.")
        return None

    try:
        words = text.split()
        chunks = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
        embeddings = []
        for chunk in chunks:
            response = client.embeddings.create(model="text-embedding-ada-002",
            input=chunk)
            embeddings.append(response.data[0].embedding)
        embedding = [sum(x) / len(x) for x in zip(*embeddings)]
        return embedding
    except Exception as e:
        handle_error(f"Error creating OpenAI embedding: {e}")
        return None

# Function to upsert to Annoy index
def upsert_to_annoy(index, i, embedding, text):
    index.add_item(i, embedding)
    text_storage[i] = text

# Function to query Annoy index
def query_annoy(index, question_embedding, top_k=3):
    try:
        query_results = index.get_nns_by_vector(question_embedding, top_k, include_distances=True)
        return query_results
    except Exception as e:
        handle_error(f"Error querying Annoy: {e}")
        return [], []

# Function to process query results
def process_query_results(query_results):
    results, scores = query_results
    context_data = []
    for i, result in enumerate(results):
        context_data.append(f"{result}: {scores[i]} - {text_storage[result]}")
    return context_data

# Function to generate an answer
def generate_answer(context_data, user_question):
    prompt = f"Context: {', '.join(context_data)}\nQuestion: {user_question}\nAnswer:"
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.0)
        return response.choices[0].message.content.strip()
    except Exception as e:
        handle_error(f"Error generating answer: {e}")
        return None

# Updated main() function with chat history and user input logic
def main():
    logging.info("Starting the Streamlit application")
    st.title("PDF Question Answering")
    st.markdown("**Use AI to find answers to your questions from PDF documents.**")
    
    # Initialize 'annoy_index' and 'index_built' if they're not already in session state
    if 'annoy_index' not in st.session_state:
        st.session_state.annoy_index = setup_annoy()
        st.session_state.index_built = False  # Initialize index_built here

    # Initialize chat history if not in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Controls")
        with st.form("file_upload_form", clear_on_submit=True):
            file = st.file_uploader("Upload your PDF Document", type=['pdf'])
            submit_button = st.form_submit_button("Submit")
    
    if file is not None:
        logging.info("File uploaded. Extracting text from PDF.")
        with st.spinner('Extracting text from PDF...'):
            st.session_state.text_chunks = extract_text_from_pdf(file)
        
        # Creating embeddings and upserting to Annoy
        for i, text in enumerate(st.session_state.text_chunks):
            if text:
                logging.info(f"Creating OpenAI embedding for chunk {i}")
                embedding = create_openai_embedding(text)
                if embedding is not None:
                    logging.info(f"Upserting to Annoy index for chunk {i}")
                    upsert_to_annoy(st.session_state.annoy_index, i, embedding, text)
        
        if not st.session_state.index_built:  # Check if the index has already been built
            logging.info("Building the Annoy index.")
            st.session_state.annoy_index.build(20)
            st.session_state.index_built = True  # Mark the index as built
    
        st.success("Setup complete!")
    
    st.subheader("Chat History")
    for chat_message in st.session_state.chat_history:
        if "role" in chat_message:
            with st.chat_message(chat_message["role"]):
                st.markdown(chat_message["content"])
    
    with st.form("chat_form"):
        user_input = st.text_input("Type your question here...")
        submit_button = st.form_submit_button("Submit")
    
        if submit_button:
            logging.info("User submitted a question.")
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            logging.info("Creating OpenAI embedding for the question.")
            question_embedding = create_openai_embedding(user_input)
            if question_embedding is not None:
                logging.info("Querying Annoy index.")
                query_results = query_annoy(st.session_state.annoy_index, question_embedding)
                context_data = process_query_results(query_results)

                logging.info("Generating an answer using OpenAI.")
                answer = generate_answer(context_data, user_input)
                
                # Append the full answer to the chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
