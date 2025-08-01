from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from dotenv import load_dotenv
from langchain_docling.loader import ExportType
load_dotenv()

def extract_docling(file__path):
    """Extracts documents using Docling."""

    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    EXPORT_TYPE = ExportType.DOC_CHUNKS
    topic = "EV Charging Stations"  # Example topic, can be customized
    # Initialize the DoclingLoader with the specified parameters
    loader = DoclingLoader(
        file_path=file__path,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
        prompt_template=f"extract the text from the document which is relevant to the {topic}",
    )
    # Load the documents
    docs = loader.load()
    # Return the loaded documents
    return docs


def save_scraped_text(text, out_path):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    # Run the extraction function
    #FILE_PATH_ = "./data/raw/EVChargingStation.pdf" # Docling Technical Report
    file__path="./data/raw/EVChargingStation.pdf"
    extracted_docs = extract_docling(file__path="./data/raw/EVChargingStation.pdf")
    print(f"Extracted {len(extracted_docs)} documents.")
    print(f"Documents saved in temporary directory: {extracted_docs[0]}")
    for d in extracted_docs[:len(extracted_docs)]:
        print(f"- {d.page_content=}")
        print("...")
        fname = file__path.split("/")[-1].split(".")[0] + ".txt"
        save_scraped_text(d.page_content, f"./data/processed/1{fname}")    
        print(f"Saved extracted text to {fname}")
    print("Extraction complete.")
