"""
Author: GHNAMI Helmi
Date: 2025-09-26
Position: Data-Science
"""

import os
import json
import base64
import requests
import logging
import warnings
import tabula
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def download_pdf(url: str, save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully: {filepath}")
    else:
        raise RuntimeError(f"Failed to download file from {url} (status code {response.status_code})")
    return filepath

def create_directories(base_dir: str):
    directories = ["images", "text", "tables", "page_images"]
    for dir_name in directories:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)

def process_tables(filepath: str, doc, page_num: int, base_dir: str, items: list):
    try:
        tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True)
        if not tables:
            return
        for table_idx, table in enumerate(tables):
            table_text = "\n".join([" | ".join(map(str, row)) for row in table.values])
            table_file_name = f"{base_dir}/tables/{os.path.basename(filepath)}_table_{page_num}_{table_idx}.txt"
            with open(table_file_name, 'w') as f:
                f.write(table_text)
            items.append({"page": page_num, "type": "table", "text": table_text, "path": table_file_name})
    except Exception as e:
        logger.error(f"Error extracting tables from page {page_num}: {str(e)}")

def process_text_chunks(text: str, text_splitter, page_num: int, base_dir: str, items: list):
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        text_file_name = f"{base_dir}/text/{os.path.basename(text_splitter.filepath)}_text_{page_num}_{i}.txt"
        with open(text_file_name, 'w') as f:
            f.write(chunk)
        items.append({"page": page_num, "type": "text", "text": chunk, "path": text_file_name})

def process_images(doc, page, page_num: int, base_dir: str, items: list):
    images = page.get_images()
    for idx, image in enumerate(images):
        xref = image[0]
        pix = pymupdf.Pixmap(doc, xref)
        image_name = f"{base_dir}/images/{os.path.basename(doc.name)}_image_{page_num}_{idx}_{xref}.png"
        pix.save(image_name)
        with open(image_name, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf8')
        items.append({"page": page_num, "type": "image", "path": image_name, "image": encoded_image})

def process_page_images(page, page_num: int, base_dir: str, items: list):
    pix = page.get_pixmap()
    page_path = os.path.join(base_dir, f"page_images/page_{page_num:03d}.png")
    pix.save(page_path)
    with open(page_path, 'rb') as f:
        page_image = base64.b64encode(f.read()).decode('utf8')
    items.append({"page": page_num, "type": "page", "path": page_path, "image": page_image})
