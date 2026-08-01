import io, os, re, json, unicodedata
from typing import List, Dict, Optional

import fitz
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from tqdm.auto import tqdm
from google.colab import files





print('PyMuPDF version:', fitz.__version__)

# ── Cargar PDF ──────────────────────────────────────────────
uploaded = files.upload()
PDF_PATH = next(iter(uploaded.keys()))
print(f'PDF cargado: {PDF_PATH}')

# ── Funciones de extracción ──────────────────────────────────
MIN_CHARS_OK = 15

def extract_with_pymupdf(pdf_path, page_num):
    try:
        doc = fitz.open(pdf_path)
        text = doc[page_num].get_text('text')
        doc.close()
        return text.strip()
    except Exception as e:
        print(f'  [PyMuPDF] página {page_num}: {e}')
        return None

def extract_with_pdfplumber(pdf_path, page_num):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = pdf.pages[page_num].extract_text() or ''
        return text.strip()
    except Exception as e:
        print(f'  [pdfplumber] página {page_num}: {e}')
        return None

def extract_with_tesseract(pdf_path, page_num, lang='spa+eng'):
    try:
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=300)
        if not images:
            return None
        return pytesseract.image_to_string(images[0], lang=lang).strip()
    except Exception as e:
        print(f'  [Tesseract] página {page_num}: {e}')
        return None

_easyocr_reader = None

def extract_with_easyocr(pdf_path, page_num, langs=['es', 'en']):
    global _easyocr_reader
    try:
        import easyocr, numpy as np
        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(langs, gpu=True)
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=300)
        if not images:
            return None
        result = _easyocr_reader.readtext(np.array(images[0]), detail=0, paragraph=True)
        return '\n'.join(result).strip()
    except Exception as e:
        print(f'  [EasyOCR] página {page_num}: {e}')
        return None

EXTRACTORS = [
    ('pymupdf',    extract_with_pymupdf),
    ('pdfplumber', extract_with_pdfplumber),
    ('tesseract',  extract_with_tesseract),
    ('easyocr',    extract_with_easyocr),
]

# ── Ciclo de extracción ──────────────────────────────────────
def get_num_pages(pdf_path):
    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    return n

def extract_all_pages(pdf_path):
    num_pages = get_num_pages(pdf_path)
    results = {}
    pending = set(range(num_pages))
    for lib_name, extractor_fn in EXTRACTORS:
        if not pending:
            break
        print(f'\n=== Librería: {lib_name} | Páginas pendientes: {len(pending)} ===')
        for page_num in tqdm(sorted(pending), desc=lib_name):
            text = extractor_fn(pdf_path, page_num)
            if text and len(text) >= MIN_CHARS_OK:
                results[page_num] = {'text': text, 'source': lib_name}
                pending.discard(page_num)
    if pending:
        print(f'\nAdvertencia: sin texto en {len(pending)} página(s): {sorted(pending)}')
        for p in pending:
            results[p] = {'text': '', 'source': 'none'}
    else:
        print('\nExito: todas las páginas tienen texto.')
    return dict(sorted(results.items()))

pages_result = extract_all_pages(PDF_PATH)
for p, info in pages_result.items():
    print(f'  Pág {p+1}: {len(info["text"])} chars | fuente: {info["source"]}')

# ── Parseo a entradas ────────────────────────────────────────
def parse_page_text(raw_text):
    entries = []
    if not raw_text:
        return entries
    pattern = re.compile(
        r'cat(?:egor[íi]a)?\s*[:\-]\s*(?P<category>.+?)\s*[\|\n]\s*'
        r'nombre\s*[:\-]\s*(?P<name>.+?)\s*[\|\n]\s*'
        r'(?:descripci[óo]n)\s*[:\-]\s*(?P<description>.+)',
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(raw_text))
    if matches:
        for m in matches:
            entries.append({
                'category':    m.group('category').strip(),
                'name':        m.group('name').strip(),
                'description': m.group('description').strip(),
            })
        return entries
    for block in re.split(r'\n\s*\n', raw_text.strip()):
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) >= 3:
            entries.append({'category': lines[0], 'name': lines[1],
                            'description': ' '.join(lines[2:])})
        elif len(lines) == 2:
            entries.append({'category': 'Sin categoría', 'name': lines[0],
                            'description': lines[1]})
    return entries

raw_entries = []
for page_num, info in pages_result.items():
    raw_entries.extend(parse_page_text(info['text']))
print(f'Entradas detectadas: {len(raw_entries)}')

# ── Limpieza ─────────────────────────────────────────────────
def clean_ws(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    return text.strip()

def title(text):
    return ' '.join(w.capitalize() for w in clean_ws(text).split())

def fix_desc(text):
    text = clean_ws(text)
    if not text:
        return text
    text = re.sub(r'\s+([,.;:])', r'\1', text)
    text = text[0].upper() + text[1:]
    if text[-1] not in '.!?':
        text += '.'
    return text

cleaned_entries, seen = [], set()
for e in raw_entries:
    cat  = title(e.get('category', ''))
    name = title(e.get('name', ''))
    desc = fix_desc(e.get('description', ''))
    if name and desc and name not in seen:
        seen.add(name)
        cleaned_entries.append({'category': cat, 'name': name, 'description': desc})
print(f'Entradas finales: {len(cleaned_entries)}')

# ── Generar, validar y descargar JSON ────────────────────────
OUTPUT_JSON = 'dataset_extraido.json'
REQUIRED = {'category', 'name', 'description'}

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(cleaned_entries, f, ensure_ascii=False, indent=2)

with open(OUTPUT_JSON, encoding='utf-8') as f:
    data = json.load(f)

errors = []
for i, item in enumerate(data):
    missing = REQUIRED - item.keys()
    if missing:
        errors.append(f'Elemento {i}: faltan claves {missing}')
    for k in REQUIRED & item.keys():
        if not isinstance(item[k], str) or not item[k].strip():
            errors.append(f'Elemento {i}: "{k}" vacío o no es texto')

if errors:
    print(f'{len(errors)} error(es):')
    for err in errors[:20]:
        print(f'  - {err}')
else:
    print(f'JSON válido: {len(data)} entradas.')
    files.download(OUTPUT_JSON)