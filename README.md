# MLOps-pipeline

A domain-specific MLOps pipeline for electric vehicle (EV) charging station analytics, question-answer generation, and document extraction.

## Features

- **Document Extraction:** Extracts relevant content from technical reports and PDFs using Docling and LangChain.
- **Web Scraping:** Automated scraping and reporting using agentic RAG (Retrieval-Augmented Generation) with CrewAI.
- **Synthetic QA Generation:** Generates questions and responses for EV charging station topics using LLMs.
- **Data Processing:** Handles raw, processed, and QA datasets for analytics and model training.
- **Configurable Pipeline:** Modular configuration via YAML and Python files for easy customization.

## Project Structure

- `src/`: Main source code for extraction, QA generation, and web scraping.
- `data/`: Raw, processed, and QA datasets.
- `config/`: Configuration files for pipeline and environment.
- `notebooks/`: Jupyter notebooks for data cleaning and extraction.
- `models/`: Model artifacts and registry.
- `tests/`: Test scripts and pipeline validation.

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   
2. **Install dependencies:**
```pip install -r requirements.txt```
3. **Set up environment variables:**
Copy .env.example to .env and fill in your API keys and credentials.
## Usage
* Document Extraction: Run src/doc_langchain_extractor.py to extract text from PDFs.

* Question-Answer Generation: Run src/question_answer_generation.py to generate synthetic QA pairs.

* Web Scraping: Use src/web_scraping/main.py and src/web_scraping/crew.py for agentic scraping and reporting.

## Configuration
Edit config/config.yaml and config/config.py for pipeline settings.
Environment variables are managed via .env.
## License
MIT License