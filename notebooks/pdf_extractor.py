import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text


def save_extracted_text(pdf_path, out_path):
    text = extract_text_from_pdf(pdf_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    pdfs = ["./data/raw/EVChargingStation.pdf"]
    for pdf in pdfs:
        fname = os.path.splitext(os.path.basename(pdf))[0] + ".txt"
        save_extracted_text(pdf, f"./data/processed/{fname}")














