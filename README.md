# **README for Python Project: PDF Question Answering with Streamlit and OpenAI**

## **Project Overview**

The "PDF Question Answering" project provides a Streamlit-based interface where users can upload PDF documents and ask questions related to their content. Utilizing OpenAI's language model and the Annoy library for approximate nearest neighbor search, the application generates answers based on the text extracted from the PDFs.

## **Key Features**

1. **PDF File Upload**: Users can upload a PDF file for analysis.
2. **Question-Answering Interface**: After processing the PDF, users can ask questions and receive AI-generated answers based on the document's content.
3. **Text Extraction and Embedding**: The application extracts text from the PDF and creates embeddings using OpenAI's models.
4. **Annoy Index for Efficient Search**: Utilizes Annoy to efficiently search through text embeddings to find relevant content for answering user queries.
5. **Interactive Chat History**: The application displays a chat history that includes both user queries and AI-generated responses.

## **Requirements**

- **`openai`**
- **`annoy`**
- **`streamlit`**
- **`langchain.document_loaders`**
- **`logging`**

## **Installation**

Install the required packages using:

```bash
pip install openai annoy streamlit langchain.document_loaders logging
```

## **Usage**

1. Run the application using Streamlit: **`streamlit run [filename].py`**.
2. Upload a PDF document through the user interface.
3. Type a question in the provided input field and submit it.
4. View the AI-generated answers based on the PDF content.

## **Functional Overview**

- **`handle_error`**: Captures and displays errors to the user interface.
- **`extract_text_from_pdf`**: Extracts text from the uploaded PDF using the UnstructuredAPIFileLoader.
- **`setup_annoy`**: Initializes the Annoy index for text embeddings.
- **`create_openai_embedding`**: Generates embeddings for text using OpenAI's model.
- **`upsert_to_annoy`**: Inserts embeddings into the Annoy index and stores the associated text.
- **`query_annoy`**: Searches the Annoy index using question embeddings to find relevant text.
- **`process_query_results`**: Processes the results from the Annoy index query.
- **`generate_answer`**: Generates an answer using OpenAI based on the context obtained from the query results.
- **`main`**: Orchestrates the application's workflow, including file upload, chat history, and processing user questions.

## **Error Handling**

The script includes robust error handling to ensure user-friendly notifications in case of any operational issues.

## **User Interface**

The application offers a simple and intuitive Streamlit interface for uploading PDFs, asking questions, and viewing responses.

## **License**

Specify your project's license here.

## **Contributors**

List the contributors to this project if any.

## **Acknowledgements**

Acknowledge the resources, libraries, or individuals that contributed to the project.
