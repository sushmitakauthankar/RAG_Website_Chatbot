
import nltk
import os
import re
import requests
import numpy as np
import faiss
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
from collections import deque
import fitz

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def scrape_website(url, depth=1, max_pages=10):
    visited = set()
    queue = deque([(url, 0)])
    result = []

    while queue and len(visited) < max_pages:
        current_url, current_depth = queue.popleft()

        if current_url in visited or current_depth > depth:
            continue

        try:
            res = requests.get(current_url, timeout=5)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            text = soup.get_text(separator=' ', strip=True)
            result.append((current_url, text))
            visited.add(current_url)

            if current_depth < depth:
                for a in soup.find_all("a", href=True):
                    link = urljoin(current_url, a["href"])
                    if link.startswith("http") and link not in visited:
                        queue.append((link, current_depth + 1))

        except requests.exceptions.RequestException:
            continue  # skip broken links

    if not result:
        raise ValueError("❌ Failed to extract any content from the site.")
    return result


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


def load_content(source, depth=2):
    if source.lower().endswith(".pdf"):
        with fitz.open(source) as doc:
            text = "\n".join([page.get_text() for page in doc])
        chunks = smart_chunk_text(text)
        if not chunks:
            raise ValueError("⚠ PDF loaded but no usable text content was found.")
        return chunks
    else:
        docs = scrape_website(source, depth=depth)
        if not docs:
            raise ValueError("⚠ Website loaded but no usable text content was found.")
        text_blocks = [text for _, text in docs]
        all_text = "\n".join(text_blocks)
        return smart_chunk_text(all_text)


def build_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks


def ask_question_streaming(question, index, chunks, k=5, question_type="Open-ended"):
    if index is None or not chunks:
        yield "⚠ Please load a website first."
        return

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
        yield "❌ I couldn't find relevant information on the website to answer your question."
        return

    context = "\n\n".join(filtered_chunks)[:1500]

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

Instruction:
{instruction}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt.strip(),
                "stream": True
            },
            stream=True
        )

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                token = data.get("response", "")
                if token:
                    yield token
    except Exception as e:
        yield f"❌ Failed to connect to the model{e}"