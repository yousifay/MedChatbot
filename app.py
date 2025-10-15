# -----------------------------------------------------
# app.py: RAG Chatbot for Hugging Face Spaces (Revised)
# -----------------------------------------------------

# Install required packages (done in environment setup)
# !pip install faiss-cpu transformers torch gradio accelerate bitsandbytes tiktoken -q
import faiss
import json
import torch
import numpy as np
import os # Import os for file checking
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig, # Import for 4-bit quantization
    # Removed LlamaTokenizer import as we are moving to AutoTokenizer for the new model
)
import gradio as gr

# --- Constants ---
FAISS_INDEX_FILE = "faiss_index.bin"
DOCUMENTS_FILE = "chunked_documents.json"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# New LLM model: StableLM 2 1.6B Chat (efficient, instruction-tuned)
LLM_MODEL_ID = "stabilityai/stablelm-2-1_6b-chat"

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1️⃣ Load FAISS Index + Documents
# -------------------------------
index = None
chunked_documents = []

if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(DOCUMENTS_FILE):
    print("FATAL ERROR: RAG data files not found. Ensure 'faiss_index.bin' and 'chunked_documents.json' are in the same directory.")
    # Exit or use dummy data if files are critical. Here we continue with a null index.
else:
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(DOCUMENTS_FILE, "r") as f:
            # We assume documents are a list of strings
            chunked_documents = json.load(f)
        print(f"Loaded {len(chunked_documents)} documents and FAISS index.")
    except Exception as e:
        print(f"Error loading RAG components: {e}")

# -------------------------------
# 2️⃣ Load Embedding Model
# -------------------------------
embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID)
embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_ID).to(DEVICE)

def get_embedding(text):
    if not index: return np.array([[0.0]])
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = embed_model(**inputs).last_hidden_state
        # Use mean pooling to get sentence embedding
        embedding = outputs.mean(dim=1).cpu().numpy()
    return embedding.astype("float32")

# -------------------------------
# 3️⃣ Retrieval Function
# -------------------------------
def retrieve_context(query, k=7):
    if not index or not chunked_documents:
        return "ERROR: RAG data not available. Cannot retrieve context."

    query_emb = get_embedding(query)
    # Ensure query_emb is 2D for faiss.search
    if query_emb.ndim == 1:
        query_emb = np.expand_dims(query_emb, axis=0)

    # If the index dimension doesn't match the query embedding dimension, retrieval will fail.
    if query_emb.shape[1] != index.d:
        print(f"Embedding dimension mismatch: Query dim {query_emb.shape[1]}, Index dim {index.d}")
        return "ERROR: Embedding dimension mismatch during retrieval."

    distances, indices = index.search(query_emb, k)
    
    # Filter out invalid indices (e.g., -1 if k > N) and ensure they are within bounds
    valid_indices = [i for i in indices[0] if i != -1 and i < len(chunked_documents)]
    
    retrieved_chunks = [chunked_documents[i] for i in valid_indices]
    
    if not retrieved_chunks:
        return "No relevant context found."

    return "\n\n".join(retrieved_chunks)

# -------------------------------
# 4️⃣ Load LLM for Answer Generation
# -------------------------------

# Configuration for 4-bit quantization (significantly reduces VRAM/RAM usage)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Use NF4 quantization
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 # Better for modern GPUs
)

# Load the model with quantization if running on CUDA (GPU)
if DEVICE.type == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=nf4_config,
        device_map="auto", # device_map="auto" works best with bnb
    )
    # Ensure the pipeline uses the high-precision dtype for computation
    pipeline_dtype = torch.bfloat16
else:
    # On CPU, load without quantization and use float32
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID)
    pipeline_dtype = torch.float32

# Load the tokenizer, using AutoTokenizer for maximum compatibility
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

# Text-generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto" if DEVICE.type == "cuda" else None,
    torch_dtype=pipeline_dtype,
)

# -------------------------------
# 5️⃣ Full RAG Response Function
# -------------------------------
# ChatInterface passes (query, history) so we must accept both
def answer_question(query, history):
    # 1. Retrieve context based on the current query
    context = retrieve_context(query, k=7)
    
    if context.startswith("ERROR") or context == "No relevant context found.":
        # If retrieval fails completely, let the LLM handle the response
        final_answer = f"I'm sorry, I couldn't find any relevant information in the medical textbooks to answer: \"{query}\". Please try a different question or keyword."
        if context.startswith("ERROR"):
            print(context) # Print system error for debugging
        return final_answer
        
    # 2. Format the Conversation History (optional but improves coherence)
    # The history comes as a list of lists: [[user_msg, bot_msg], [user_msg, bot_msg], ...]
    formatted_history = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])

    # 3. Construct the RAG prompt with clear system instructions and separators
    # StableLM-2 uses a conversational format, so we use the standard prompt but AutoTokenizer
    # might apply a chat template if available, which can be beneficial.
    prompt = f"""
### System Instruction
You are a highly detailed and helpful medical assistant and tutor for a student.
Your primary task is to answer the student's question *extensively* and accurately using *only* the provided context.
You must not use external knowledge. If the provided context does not contain the answer, state clearly and politely: "I don't have enough information in the provided textbooks to answer this question."
### Context from Medical Textbooks
{context}
### Conversation History
{formatted_history}
### Current Question
{query}
### Answer
"""
    
    # 4. Generate the response
    response = generator(
        prompt, 
        max_new_tokens=400, 
        temperature=0.4, 
        top_p=0.9, 
        do_sample=True,
        # Crucial to ensure the output only contains the generated answer text
        return_full_text=False 
    )
    
    # 5. Extract the answer
    if response and response[0] and response[0]["generated_text"]:
        answer = response[0]["generated_text"].strip()
    else:
        answer = "I apologize, the text generation failed. Please try again."

    return answer

# -------------------------------
# 6️⃣ Gradio Chat Interface
# -------------------------------
gr.ChatInterface(
    fn=answer_question,
    title="💬 Medbot — Personal Medical Tutor",
    description="Ask me anything from your medical textbooks. I provide detailed explanations suitable for a medical student.",
    # Add an example query for users
    examples=[["What is the function of the renal corpuscle?"], ["Describe the mechanism of action for insulin."]],
    theme=gr.themes.Soft(), # A visually appealing theme
).launch()
