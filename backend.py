# backend.py

import nltk
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

nltk.download("punkt")
from nltk.tokenize import sent_tokenize
print("\u2705 Punkt successfully reloaded.")

# Preload embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Preload QA model from local cache (no repeated downloads)
MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
qa_model = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def scrape_website(url, depth=1, visited=None):
    if visited is None:
        visited = set()
    if url in visited or depth == 0:
        return []

    visited.add(url)
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()  # Raises error for bad responses (404, 500, etc.)
        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text(separator=' ', strip=True)
        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
        result = [(url, text)]
        for link in links[:5]:
            result += scrape_website(link, depth - 1, visited)
        return result
    except requests.exceptions.RequestException as e:
        raise ValueError(f"🚫 Failed to load the website: {e}")


def smart_chunk_text(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    return chunks


def load_site(url, depth=2):
    docs = scrape_website(url, depth)
    if not docs:
        raise ValueError("⚠️ No content found at the given URL.")

    chunks = []
    for _, text in docs:
        chunks += smart_chunk_text(text)

    if not chunks:
        raise ValueError("⚠️ Website loaded but no usable text content was found.")

    return chunks


def build_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity

def ask_question(question, index, chunks, k=5, question_type="Open-ended", show_chunks=True):
    question_embedding = embedding_model.encode([question])
    D, I = index.search(np.array(question_embedding).astype("float32"), k)

    selected_chunks = [chunks[i] for i in I[0]]
    chunk_embeddings = embedding_model.encode(selected_chunks)
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]

    threshold = 0.2
    filtered_chunks = [
        chunk for chunk, sim in zip(selected_chunks, similarities)
        if sim > threshold and len(chunk) > 50
    ]

    if not filtered_chunks:
        return (
            "❌ I couldn't find relevant information on the website to answer your question.",
            []
        )

    context = "\n\n".join(filtered_chunks)[:1500]

    # Prompt instructions
    if question_type == "Close-ended (Yes/No)":
        instruction = "Answer with a clear Yes or No, followed by a short explanation."
    elif question_type == "Fact-based":
        instruction = "Answer with a precise fact or statistic from the context."
    elif question_type == "Definition":
        instruction = "Provide a clear definition based on the context."
    else:
        instruction = "Answer clearly in one or two sentences."

    prompt = f"""
You are a helpful assistant answering questions based only on the website content below.
If the answer is not in the content, clearly state that you cannot answer.

Context:
{context}

Question:
{question}

Answer: {instruction}
"""

    result = qa_model(prompt.strip(), max_length=500, do_sample=False)[0]['generated_text']
    return result.strip(), filtered_chunks
