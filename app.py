import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from flask_cors import CORS
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Workaround to prevent OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables from .env file (for API_KEY and other sensitive details)
load_dotenv()

# Set the environment variable for Google Cloud authentication (use your Service Account JSON file path)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account-key.json"  # Update with your actual file path

# Configuration for Google API Key (Gemini API Key)
API_KEY = os.getenv("MY_API_KEY")  # Ensure your API key is saved in .env
if not API_KEY:
    raise ValueError("API Key is not set. Please add your API Key to the .env file.")
genai.configure(api_key=API_KEY)

# Initialize embedding models
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
hf_embedding_model = HuggingFaceEmbeddings()

# Path to the LTN PDF
LTN_PDF_PATH = "cycle-infrastructure-design-ltn-1-20.pdf"  # Update with the actual path to your LTN PDF

# Utility Functions
def load_pdf_text(pdf_path):
    """
    Extract text content from a PDF file.
    """
    pdf_reader = PdfReader(pdf_path)
    return "".join([page.extract_text() for page in pdf_reader.pages])

def create_chunks(text):
    """
    Splits text into manageable chunks using a RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return [{"text": chunk} for chunk in text_splitter.split_text(text)]

def create_vector_store(text):
    """
    Create a FAISS vector store for the LTN document.
    """
    # Split text into chunks
    chunks = create_chunks(text)
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": "ltn"}] * len(chunks)

    # Create the FAISS vector store
    return FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)

def generate_response(prompt):
    """
    Generate a response using the Gemini Pro model.
    """
    model = genai.GenerativeModel("gemini-1.5-pro-002")
    response = model.generate_content(prompt)
    return response.text.strip()

# Load LTN PDF content and create the vector store
ltn_text = load_pdf_text(LTN_PDF_PATH)
vectorstore = create_vector_store(ltn_text)

@app.route("/query", methods=["POST"])
def query_ltn():
    """
    API endpoint to query the LTN document and retrieve responses with threshold logic.
    """
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Retrieve relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    relevant_docs = retriever.get_relevant_documents(query)

    if not relevant_docs:
        return jsonify({"response": "No relevant information found."}), 200

    # Calculate similarity scores
    query_embedding = hf_embedding_model.embed_query(query)
    max_similarity_score = 0

    for doc in relevant_docs:
        doc_embedding = hf_embedding_model.embed_query(doc.page_content)
        similarity_score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        max_similarity_score = max(max_similarity_score, similarity_score)

    # Set relevance threshold
    relevance_threshold = 0.3
    if max_similarity_score < relevance_threshold:
        return jsonify({"response": "Sorry, please ask something more related to the LTN document."}), 200

    # Combine the relevant documents' text
    combined_text = "\n".join([doc.page_content for doc in relevant_docs])

    # Formulate the prompt
    prompt = f"Based on the following information:\n\n{combined_text}\n\nAnswer the query: {query}"
    response_text = generate_response(prompt)

    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)
