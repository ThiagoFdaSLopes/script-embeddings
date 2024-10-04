from sentence_transformers import SentenceTransformer
import pdfplumber
import PyPDF2
import json
import os

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text

def generate_embeddings_for_pages(pdf_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []

    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        if text:
            embeddings.append(model.encode([text]))

    return embeddings

def save_embeddings_as_json(embeddings, filename):
    # Criar diretório se não existir
    save_path = os.path.join("src", "embeddings")
    os.makedirs(save_path, exist_ok=True)

    # Definir o caminho completo do arquivo
    filepath = os.path.join(save_path, filename)

    # Converter cada NumPy array dentro da lista para uma lista Python
    embeddings_list = [embedding.tolist() for embedding in embeddings]

    # Salvar os embeddings no formato JSON
    with open(filepath, 'w') as f:
        json.dump(embeddings_list, f)

pdf_path = "/home/ubuntu/script-embeddings/assets/AltaLaudoCompleto20240719.pdf"
page_embeddings = generate_embeddings_for_pages(pdf_path)

save_embeddings_as_json(page_embeddings, 'embeddings')