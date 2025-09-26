"""
Author: GHANMI Helmi
Date: 2025-09-26
Position: Data-Science
"""

import os, json, base64, requests, logging, warnings, tabula, pymupdf
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
        raise RuntimeError(f"Failed to download {url} (status {response.status_code})")
    return filepath

def create_directories(base_dir: str):
    for d in ["images","text","tables","page_images"]:
        os.makedirs(os.path.join(base_dir,d), exist_ok=True)

def process_tables(filepath, doc, page_num, base_dir, items):
    try:
        tables = tabula.read_pdf(filepath, pages=page_num+1, multiple_tables=True)
        if not tables: return
        for t_idx, table in enumerate(tables):
            table_text = "\n".join([" | ".join(map(str,row)) for row in table.values])
            out = f"{base_dir}/tables/{os.path.basename(filepath)}_table_{page_num}_{t_idx}.txt"
            with open(out,'w') as f: f.write(table_text)
            items.append({"page":page_num,"type":"table","text":table_text,"path":out})
    except Exception as e:
        logger.error(f"Error extracting tables on page {page_num}: {e}")

def process_text_chunks(filepath, text, splitter, page_num, base_dir, items):
    chunks = splitter.split_text(text)
    for i,chunk in enumerate(chunks):
        out = f"{base_dir}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
        with open(out,'w') as f: f.write(chunk)
        items.append({"page":page_num,"type":"text","text":chunk,"path":out})

def process_images(doc, page, page_num, base_dir, items):
    for idx,img in enumerate(page.get_images()):
        xref = img[0]; pix = pymupdf.Pixmap(doc,xref)
        name = f"{base_dir}/images/{os.path.basename(doc.name)}_img_{page_num}_{idx}_{xref}.png"
        pix.save(name)
        with open(name,'rb') as f: encoded = base64.b64encode(f.read()).decode("utf8")
        items.append({"page":page_num,"type":"image","path":name,"image":encoded})

def process_page_images(page, page_num, base_dir, items):
    pix = page.get_pixmap()
    path = os.path.join(base_dir,f"page_images/page_{page_num:03d}.png")
    pix.save(path)
    with open(path,'rb') as f: encoded = base64.b64encode(f.read()).decode("utf8")
    items.append({"page":page_num,"type":"page","path":path,"image":encoded})
