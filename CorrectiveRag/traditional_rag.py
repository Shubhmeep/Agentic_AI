import os
import pathlib
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# loading data
from langchain_community.document_loaders import PyPDFLoader

# token-based chunking 
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# build embeddings + FAISS index.
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()

# Load environment variables
HF_API_KEY = os.getenv("HF_API_TOKEN")
GEMINI_API_KEY = os.getenv("API_ONE")
EMBEDDING_MODEL_NAME = os.getenv("HF_EMBEDDING_MODEL")

# Defining the parameters for the RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
QUESTION = "What is a transformer in deep learning?"

if not HF_API_KEY:
    raise ValueError("HF_API_KEY is not set in the environment variables.")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

if not EMBEDDING_MODEL_NAME:
    raise ValueError("HF_EMBEDDING_MODEL_NAME is not set in the environment variables.")

# Define Data directory
print(20*"-", "Loading PDF Data", 20*"-")
DATA_DIR = pathlib.Path("CorrectiveRag/data_pdf")
print(f"Data directory set to: {DATA_DIR}")

# load the pdf data in docs list
docs=[]
for file in DATA_DIR.glob("*.pdf"):
    print(f"Loading {file}...")
    print()
    loader = PyPDFLoader(str(file))
    loaded_docs = loader.load()
    print(f"Loaded {len(loaded_docs)} pages from {file}.")
    docs.extend(loaded_docs)

# Clean non-string/invalid content before tokenization
clean_docs = []
for d in docs:
    text = d.page_content
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    if text.strip():
        d.page_content = text
        clean_docs.append(d)
docs = clean_docs

print(f"Total documents loaded: {len(docs)}")
print()
print(20*"-", "Split into chunks - Token based chunking", 20*"-")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, token=HF_API_KEY)
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=400,      # tokens
    chunk_overlap=50,    # tokens
)

# Split the documents into chunks based on token count rather than character count
documents = text_splitter.split_documents(docs)
print(f"Total document chunks created: {len(documents)}")

# Check the maximum input tokens for the embedding model
max_input_tokens = tokenizer.model_max_length
print(f"Embedding model max input tokens: {max_input_tokens}")

max_chunk_tokens = 0
for d in documents:
    token_len = len(tokenizer.encode(d.page_content, add_special_tokens=False))
    if token_len > max_chunk_tokens:
        max_chunk_tokens = token_len

if max_chunk_tokens <= max_input_tokens:
    print("safe : max_chunk_tokens <= max_input_tokens")
else:
    print(f"unsafe: max chunk tokens {max_chunk_tokens} > {max_input_tokens}")

print()
print(20*"-", "Build Embeddings + Index", 20*"-")
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY,
    model_name=EMBEDDING_MODEL_NAME,
)
vector_store = FAISS.from_documents(documents, embeddings)
print(f"FAISS index ready: {vector_store.index.ntotal} vectors")


