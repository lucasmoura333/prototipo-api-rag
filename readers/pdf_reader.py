from pathlib import Path

from llama_index.core import Document


def load_pdfs(pdf_files: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for pdf_path in pdf_files:
        try:
            docs.extend(_load_with_pymupdf(pdf_path))
            print(f"[pdf_reader] {pdf_path.name} — OK (PyMuPDF)")
        except Exception as e:
            print(f"[pdf_reader] PyMuPDF failed for {pdf_path.name}: {e} — trying OCR")
            try:
                docs.extend(_load_with_ocr(pdf_path))
                print(f"[pdf_reader] {pdf_path.name} — OK (OCR)")
            except Exception as e2:
                print(f"[pdf_reader] OCR also failed for {pdf_path.name}: {e2}")
    return docs


def _load_with_pymupdf(pdf_path: Path) -> list[Document]:
    from llama_index.readers.file import PyMuPDFReader

    reader = PyMuPDFReader()
    return reader.load(file_path=str(pdf_path))


def _load_with_ocr(pdf_path: Path) -> list[Document]:
    import io

    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image

    doc_fitz = fitz.open(str(pdf_path))
    docs: list[Document] = []
    for page_num, page in enumerate(doc_fitz):
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="por+eng")
        if text.strip():
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": pdf_path.name,
                        "page": page_num + 1,
                        "source": "ocr",
                    },
                )
            )
    doc_fitz.close()
    return docs
