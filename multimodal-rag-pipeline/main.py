import os
import requests
import base64
import pymupdf
import tabula
from tqdm import tqdm

from src.config import Config
from src.embedding import generate_multimodal_embeddings
from src.vectorstore import FaissVectorStore

config = Config()
paths = config.get_data_paths()
dim = config.get_embedding_dim()
top_k = config.get_retriever_config().get("top_k", 5)
vs_paths = config.get_vectorstore_config()

url = "https://arxiv.org/pdf/1706.03762.pdf"
filename = "attention_paper.pdf"
filepath = os.path.join(paths["input_dir"], filename)

# Download PDF
os.makedirs(paths["input_dir"], exist_ok=True)
if not os.path.exists(filepath):
    response = requests.get(url)
    with open(filepath, 'wb') as f:
        f.write(response.content)

# Load PDF
doc = pymupdf.open(filepath)
items = []

for page_num, page in enumerate(doc):
    text = page.get_text()
    if text.strip():
        items.append({"page": page_num, "type": "text", "text": text})

    images = page.get_images()
    for idx, image in enumerate(images):
        xref = image[0]
        pix = pymupdf.Pixmap(doc, xref)
        image_name = f"image_{page_num}_{idx}.png"
        pix.save(image_name)
        with open(image_name, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")
        items.append({"page": page_num, "type": "image", "image": encoded_image})

# Generate embeddings
all_embeddings = []
with tqdm(total=len(items), desc="Embedding items") as pbar:
    for item in items:
        if item['type'] == 'text':
            emb = generate_multimodal_embeddings(prompt=item['text'])
        elif item['type'] == 'image':
            emb = generate_multimodal_embeddings(image=item['image'])
        else:
            emb = None
        item["embedding"] = emb
        all_embeddings.append(emb)
        pbar.update(1)

# Save index and metadata
store = FaissVectorStore(
    index_path=vs_paths["index_path"],
    metadata_path=vs_paths["metadata_path"]
)
store.build(all_embeddings, items)
store.save()

# Test a query
query = "Which optimizer was used for training?"
query_embedding = generate_multimodal_embeddings(prompt=query)
store.load()
results = store.search(query_embedding, top_k=top_k)

print("\n🔍 Top results for query:")
for r in results:
    print(f"Page {r['page']} ({r['type']}):")
    if 'text' in r:
        print(r['text'][:300])
    print("---")
