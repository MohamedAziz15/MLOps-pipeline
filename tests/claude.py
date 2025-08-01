# ML Pipeline for Domain-Specific Language Model Fine-tuning
# Complete End-to-End Implementation

"""
Project Structure:
ml-pipeline/
├── config/
│   ├── config.yaml
│   └── .env.example
├── src/
│   ├── __init__.py
│   ├── config/
│   ├── data_collection/
│   ├── data_processing/
│   ├── dataset_generation/
│   ├── fine_tuning/
│   ├── evaluation/
│   ├── deployment/
│   └── orchestration/
├── tests/
├── docker/
├── scripts/
└── requirements.txt
"""

# ============================================================================
# requirements.txt
# ============================================================================

"""
# Core ML/NLP
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
peft>=0.4.0
accelerate>=0.20.0
bitsandbytes>=0.41.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
PyMuPDF>=1.22.0
pdfplumber>=0.9.0
beautifulsoup4>=4.12.0
scrapy>=2.9.0
requests>=2.31.0

# ML Ops
mlflow>=2.5.0
wandb>=0.15.0
prometheus-client>=0.17.0

# API & Deployment
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
redis>=4.6.0

# Orchestration
apache-airflow>=2.6.0
prefect>=2.10.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0
loguru>=0.7.0
tqdm>=4.65.0
psutil>=5.9.0

# Evaluation
rouge-score>=0.1.2
sacrebleu>=2.3.0
nltk>=3.8.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
alembic>=1.11.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
"""

# ============================================================================
# config/config.yaml
# ============================================================================

config_yaml = """
# Configuration for ML Pipeline
project:
  name: "domain-llm-pipeline"
  version: "1.0.0"
  description: "End-to-end pipeline for domain-specific LLM fine-tuning"

# Domain Configuration
domain:
  topic: "electric vehicle charging stations"
  use_case: "QA"
  description: "Question-answering system for EV charging station information"

# Data Sources
data_sources:
  web_scraping:
    enabled: true
    urls:
      - "https://www.energy.gov/eere/electricvehicles"
      - "https://afdc.energy.gov/fuels/electricity_locations.html"
    max_pages: 100
    delay: 1
  pdf_sources:
    enabled: true
    directories:
      - "./data/pdfs/"
    max_files: 50

# Data Processing
data_processing:
  chunk_size: 512
  overlap: 50
  min_text_length: 100
  max_text_length: 2048
  deduplication_threshold: 0.85
  quality_filters:
    min_word_count: 10
    max_word_count: 500

# Model Configuration
model:
  base_model: "microsoft/DialoGPT-small"  # Using smaller model for demo
  cache_dir: "./models/"
  device: "auto"
  
# Fine-tuning Configuration
fine_tuning:
  method: "lora"
  lora_config:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: ["q_proj", "v_proj"]
  training:
    batch_size: 4
    learning_rate: 5e-4
    num_epochs: 3
    warmup_steps: 100
    logging_steps: 10
    save_steps: 500
    eval_steps: 250
    max_grad_norm: 1.0

# Dataset Generation
dataset_generation:
  llm_api:
    provider: "openai"  # or "anthropic"
    model: "gpt-3.5-turbo"
    max_tokens: 150
    temperature: 0.7
  qa_generation:
    questions_per_chunk: 3
    max_questions: 1000

# Evaluation
evaluation:
  metrics: ["rouge", "bleu", "exact_match", "f1"]
  benchmark_size: 200
  test_split: 0.2

# Deployment
deployment:
  api:
    host: "0.0.0.0"
    port: 8000
    workers: 2
  model_serving:
    max_length: 512
    temperature: 0.7
    top_p: 0.9

# Storage
storage:
  database_url: "postgresql://user:pass@localhost/mlpipeline"
  s3_bucket: "ml-pipeline-data"
  model_registry: "./models/registry/"

# Monitoring
monitoring:
  mlflow:
    tracking_uri: "http://localhost:5000"
    experiment_name: "domain-llm-finetuning"
  prometheus:
    port: 9090
  logging:
    level: "INFO"
    format: "<green>{time}</green> | <level>{level}</level> | {message}"

# Orchestration
orchestration:
  scheduler: "airflow"  # or "prefect"
  schedule_interval: "0 2 * * *"  # Daily at 2 AM
"""

# ============================================================================
# .env.example
# ============================================================================

env_example = """
# API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
WANDB_API_KEY=your_wandb_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mlpipeline

# Storage
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=ml-pipeline-data

# Monitoring
MLFLOW_TRACKING_URI=http://localhost:5000
PROMETHEUS_GATEWAY=localhost:9091

# Security
JWT_SECRET_KEY=your_jwt_secret_key
API_KEY=your_api_key
"""

# ============================================================================
# src/__init__.py
# ============================================================================

init_content = """
\"\"\"
ML Pipeline for Domain-Specific Language Model Fine-tuning
\"\"\"

__version__ = "1.0.0"
__author__ = "ML Pipeline Team"
"""

# ============================================================================
# src/config/settings.py
# ============================================================================

settings_py = """
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
from loguru import logger

class Settings(BaseSettings):
    \"\"\"Application settings loaded from config and environment variables.\"\"\"
    
    # Project settings
    project_name: str = "domain-llm-pipeline"
    version: str = "1.0.0"
    debug: bool = False
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    wandb_api_key: Optional[str] = Field(None, env="WANDB_API_KEY")
    
    # Database
    database_url: str = Field("sqlite:///./pipeline.db", env="DATABASE_URL")
    
    # Storage
    aws_access_key_id: Optional[str] = Field(None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    s3_bucket: str = Field("ml-pipeline-data", env="S3_BUCKET")
    
    # Monitoring
    mlflow_tracking_uri: str = Field("http://localhost:5000", env="MLFLOW_TRACKING_URI")
    
    # Security
    jwt_secret_key: str = Field("default-secret-key", env="JWT_SECRET_KEY")
    api_key: str = Field("default-api-key", env="API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    \"\"\"Load configuration from YAML file.\"\"\"
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {}
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

# Global settings instance
settings = Settings()
config = load_config()
"""

# ============================================================================
# src/utils/logging.py
# ============================================================================

logging_py = """
import sys
from pathlib import Path
from loguru import logger
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    \"\"\"Setup logging configuration.\"\"\"
    
    # Remove default handler
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Console handler
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True
    )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    logger.info(f"Logging setup complete. Level: {level}")

def get_logger(name: str):
    \"\"\"Get a logger instance.\"\"\"
    return logger.bind(name=name)
"""

# ============================================================================
# src/utils/database.py
# ============================================================================

database_py = """
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from typing import Generator
import os

from ..config.settings import settings

# Database setup
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class DataSource(Base):
    \"\"\"Track data sources and collection metadata.\"\"\"
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(50), nullable=False)  # 'web', 'pdf'
    source_url = Column(String(500))
    file_path = Column(String(500))
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class ProcessedData(Base):
    \"\"\"Store processed text chunks.\"\"\"
    __tablename__ = "processed_data"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text_content = Column(Text, nullable=False)
    metadata = Column(Text)  # JSON string
    embedding_vector = Column(Text)  # JSON string of embedding
    quality_score = Column(Float)
    created_at = Column(DateTime, server_default=func.now())

class TrainingDataset(Base):
    \"\"\"Store generated QA pairs for training.\"\"\"
    __tablename__ = "training_dataset"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    context = Column(Text)
    source_id = Column(Integer)
    split = Column(String(10), default="train")  # train, val, test
    metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, server_default=func.now())

class ModelExperiment(Base):
    \"\"\"Track model training experiments.\"\"\"
    __tablename__ = "model_experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_name = Column(String(100), nullable=False)
    model_name = Column(String(100), nullable=False)
    config = Column(Text)  # JSON string
    status = Column(String(20), default="running")  # running, completed, failed
    metrics = Column(Text)  # JSON string
    model_path = Column(String(500))
    created_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime)

class ModelDeployment(Base):
    \"\"\"Track model deployments.\"\"\"
    __tablename__ = "model_deployments"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, nullable=False)
    deployment_name = Column(String(100), nullable=False)
    model_path = Column(String(500), nullable=False)
    endpoint_url = Column(String(200))
    status = Column(String(20), default="deploying")  # deploying, active, inactive, failed
    performance_metrics = Column(Text)  # JSON string
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

def create_tables():
    \"\"\"Create all database tables.\"\"\"
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    \"\"\"Database session dependency.\"\"\"
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    \"\"\"Initialize database.\"\"\"
    create_tables()
"""

# ============================================================================
# src/data_collection/web_scraper.py
# ============================================================================

web_scraper_py = """
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Set
import time
import json
from pathlib import Path

from ..utils.logging import get_logger
from ..utils.database import get_db, DataSource

logger = get_logger(__name__)

class WebScraper:
    \"\"\"Asynchronous web scraper for collecting domain-specific data.\"\"\"
    
    def __init__(self, config: Dict):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.visited_urls: Set[str] = set()
        self.scraped_data: List[Dict] = []
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; DomainBot/1.0)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scrape_url(self, url: str) -> Optional[Dict]:
        \"\"\"Scrape a single URL and extract relevant content.\"\"\"
        if url in self.visited_urls:
            return None
            
        self.visited_urls.add(url)
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: {response.status}")
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract main content
                content = self._extract_content(soup)
                
                if not content or len(content.strip()) < 100:
                    logger.info(f"Insufficient content from {url}")
                    return None
                
                # Extract metadata
                metadata = self._extract_metadata(soup, url)
                
                scraped_data = {
                    'url': url,
                    'title': metadata.get('title', ''),
                    'content': content,
                    'metadata': metadata,
                    'scraped_at': time.time()
                }
                
                logger.info(f"Successfully scraped {url}: {len(content)} characters")
                return scraped_data
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        \"\"\"Extract main text content from HTML.\"\"\"
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main', 'article', '.content', '#content', 
            '.main-content', '.post-content'
        ]
        
        content_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content_text = " ".join([elem.get_text() for elem in elements])
                break
        
        # Fallback to body content
        if not content_text:
            body = soup.find('body')
            if body:
                content_text = body.get_text()
        
        # Clean up text
        lines = (line.strip() for line in content_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        content_text = ' '.join(chunk for chunk in chunks if chunk)
        
        return content_text
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        \"\"\"Extract metadata from HTML.\"\"\"
        metadata = {'url': url}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            name = tag.get('name') or tag.get('property')
            content = tag.get('content')
            if name and content:
                metadata[name] = content
        
        # Links for potential crawling
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            if self._is_valid_url(full_url):
                links.append(full_url)
        
        metadata['links'] = links[:50]  # Limit to prevent memory issues
        
        return metadata
    
    def _is_valid_url(self, url: str) -> bool:
        \"\"\"Check if URL is valid for scraping.\"\"\"
        try:
            parsed = urlparse(url)
            
            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Skip non-web protocols
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Skip file extensions we don't want
            skip_extensions = ['.pdf', '.doc', '.docx', '.jpg', '.png', '.gif', '.mp4']
            if any(url.lower().endswith(ext) for ext in skip_extensions):
                return False
            
            return True
        except:
            return False
    
    async def scrape_domain(self, urls: List[str], max_pages: int = 100) -> List[Dict]:
        \"\"\"Scrape multiple URLs with crawling.\"\"\"
        logger.info(f"Starting domain scraping for {len(urls)} seed URLs")
        
        to_scrape = set(urls)
        scraped_count = 0
        
        while to_scrape and scraped_count < max_pages:
            # Take next URL to scrape
            current_url = to_scrape.pop()
            
            # Add delay to be respectful
            if scraped_count > 0:
                await asyncio.sleep(self.config.get('delay', 1))
            
            # Scrape the URL
            result = await self.scrape_url(current_url)
            
            if result:
                self.scraped_data.append(result)
                scraped_count += 1
                
                # Add linked URLs for crawling (limited)
                if scraped_count < max_pages * 0.8:  # Only crawl for first 80% of quota
                    new_urls = result['metadata'].get('links', [])[:5]  # Max 5 new URLs per page
                    for new_url in new_urls:
                        if new_url not in self.visited_urls and len(to_scrape) < 20:
                            to_scrape.add(new_url)
            
            if scraped_count % 10 == 0:
                logger.info(f"Scraped {scraped_count} pages so far...")
        
        logger.info(f"Scraping completed. Total pages: {len(self.scraped_data)}")
        return self.scraped_data
    
    def save_to_database(self):
        \"\"\"Save scraped data to database.\"\"\"
        db = next(get_db())
        
        try:
            for data in self.scraped_data:
                source = DataSource(
                    source_type="web",
                    source_url=data['url'],
                    status="completed",
                    metadata=json.dumps(data['metadata'])
                )
                db.add(source)
            
            db.commit()
            logger.info(f"Saved {len(self.scraped_data)} web sources to database")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            db.rollback()
        finally:
            db.close()

async def run_web_scraping(config: Dict) -> List[Dict]:
    \"\"\"Main function to run web scraping.\"\"\"
    scraping_config = config.get('data_sources', {}).get('web_scraping', {})
    
    if not scraping_config.get('enabled', False):
        logger.info("Web scraping disabled in config")
        return []
    
    urls = scraping_config.get('urls', [])
    max_pages = scraping_config.get('max_pages', 50)
    
    async with WebScraper(scraping_config) as scraper:
        scraped_data = await scraper.scrape_domain(urls, max_pages)
        scraper.save_to_database()
        
        return scraped_data
"""

# ============================================================================
# src/data_collection/pdf_extractor.py
# ============================================================================

pdf_extractor_py = ""
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import hashlib

from ..utils.logging import get_logger
from ..utils.database import get_db, DataSource

logger = get_logger(__name__)

class PDFExtractor:
    #\"\"Extract text and metadata from PDF files with layout preservation.\"\"\"
    
    def __init__(self, config: Dict):
        self.config = config
        self.extracted_data: List[Dict] = []
    
    def extract_from_file(self, file_path: Path) -> Optional[Dict]:
        #"\"\"Extract content from a single PDF file.\"\"\"
        try:
            logger.info(f"Processing PDF: {file_path}")
            
            # Try PyMuPDF first (faster)
            content = self._extract_with_pymupdf(file_path)
            
            # Fallback to pdfplumber for better layout (slower)
            if not content or len(content.get('text', '')) < 100:
                logger.info(f"Trying pdfplumber for {file_path}")
                content = self._extract_with_pdfplumber(file_path)
            
            if not content:
                logger.warning(f"No content extracted from {file_path}")
                return None
            
            # Add file metadata
            content.update({
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_hash': self._calculate_file_hash(file_path)
            })
            
            logger.info(f"Successfully extracted {len(content['text'])} characters from {file_path.name}")
            return content
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _extract_with_pymupdf(self, file_path: Path) -> Optional[Dict]:
        #\"\"\"Extract using PyMuPDF (faster, basic layout).\"\"\"
        try:
            doc = fitz.open(file_path)
            
            pages_content = []
            metadata = {
                'total_pages': len(doc),
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', '')
            }
            
            full_text = ""
            
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                
                # Extract text blocks for better structure
                blocks = page.get_text("dict")
                structured_text = self._process_text_blocks(blocks)
                
                page_content = {
                    'page_number': page_num + 1,
                    'text': text,
                    'structured_text': structured_text,
                    'bbox': page.rect  # Page bounding box
                }
                
                pages_content.append(page_content)
                full_text += f"\\n\\n--- Page {page_num + 1} ---\\n\\n" + text
            
            doc.close()
            
            return {
                'text': full_text.strip(),
                'pages': pages_content,
                'metadata': metadata,
                'extraction_method': 'pymupdf'
            }
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return None
    
    def _extract_with_pdfplumber(self, file_path: Path) -> Optional[Dict]:
        #\"\"\"Extract using pdfplumber (slower, better layout).\"\"\"
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata = pdf.metadata or {}
                
                pages_content = []
                full_text = ""
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text with layout
                    text = page.extract_text() or ""
                    
                    # Extract tables if any
                    tables = page.extract_tables()
                    
                    # Extract text with coordinates for better structure
                    words = page.extract_words()
                    
                    page_content = {
                        'page_number': page_num + 1,
                        'text': text,
                        'tables': tables,
                        'words_count': len(words),
                        'bbox': (page.bbox if hasattr(page, 'bbox') else None)
                    }
                    
                    pages_content.append(page_content)
                    full_text += f"\\n\\n--- Page {page_num + 1} ---\\n\\n" + text
                    
                    # Add table content to text
                    for table in tables:
                        table_text = self._table_to_text(table)
                        full_text += f"\\n\\nTable {len(tables)}:\\n{table_text}\\n"
                
                return {
                    'text': full_text.strip(),
                    'pages': pages_content,
                    'metadata': metadata,
                    'extraction_method': 'pdfplumber'
                }
                
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return None
    
    def _process_text_blocks(self, blocks_dict: Dict) -> List[Dict]:
        #\"\"\"Process PyMuPDF text blocks for structure.\"\"\"
        structured_blocks = []
        
        for block in blocks_dict.get("blocks", []):
            if "lines" in block:  # Text block
                block_text = ""
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    block_text += line_text + " "
                
                structured_blocks.append({
                    'type': 'text',
                    'text': block_text.strip(),
                    'bbox': block["bbox"],
                    'font_info': line["spans"][0] if line.get("spans") else None
                })
        
        return structured_blocks
    
    def _table_to_text(self, table: List[List]) -> str:
        #\"\"\"Convert table data to text format.\"\"\"
        if not table:
            return ""
        
        text_rows = []
        for row in table:
            # Clean and join row cells
            clean_row = [str(cell).strip() if cell else "" for cell in row]
            text_rows.append(" | ".join(clean_row))
        
        return "\\n".join(text_rows)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        #\"\"\"Calculate MD5 hash of file for deduplication.\"\"\"
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def extract_from_directory(self, directory: Path) -> List[Dict]:
        #\"\"\"Extract content from all PDFs in a directory.\"\"\"
        pdf_files = list(directory.glob("*.pdf"))
        max_files = self.config.get('max_files', 50)
        
        logger.info(f"Found {len(pdf_files)} PDF files, processing up to {max_files}")
        
        for pdf_file in pdf_files[:max_files]:
            content = self.extract_from_file(pdf_file)
            if content:
                self.extracted_data.append(content)
        
        return self.extracted_data
    
    def save_to_database(self):
        #\"\"\"Save extracted PDF data to database.\"\"\"
        db = next(get_db())
        
        try:
            for data in self.extracted_data:
                source = DataSource(
                    source_type="pdf",
                    file_path=data['file_path'],
                    status="completed",
                    metadata=json.dumps({
                        'file_size': data['file_size'],
                        'file_hash': data['file_hash'],
                        'pages': len(data['pages']),
                        'extraction_method': data['extraction_method'],
                        'pdf_metadata': data['metadata']
                    })
                )
                db.add(source)
            
            db.commit()
            logger.info(f"Saved {len(self.extracted_data)} PDF sources to database")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            db.rollback()
        finally:
            db.close()

# ============================================================================
# src/data_processing/text_processor.py
# ============================================================================

text_processor_py = ""
import re
import hashlib
from typing import List, Dict, Tuple, Set
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import json

from ..utils.logging import get_logger
from ..utils.database import get_db, ProcessedData

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextProcessor:
    #\"\"\"Process and clean raw text data for training.\"\"\"
    
    def __init__(self, config: Dict):
        self.config = config
        self.chunk_size = config.get('chunk_size', 512)
        self.overlap = config.get('overlap', 50)
        self.min_length = config.get('min_text_length', 100)
        self.max_length = config.get('max_text_length', 2048)
        self.dedup_threshold = config.get('deduplication_threshold', 0.85)
        
        self.stop_words = set(stopwords.words('english'))
        self.processed_chunks: List[Dict] = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def clean_text(self, text: str) -> str:
        #\"\"\"Clean and normalize text content.\"\"\"
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Normalize spacing around punctuation
        text = re.sub(r'\s*([.!?;:,])\s*', r'\1 ', text)
        
        # Remove lines with mostly special characters
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            if len(re.sub(r'[^\w\s]', '', line).strip()) > 5:
                clean_lines.append(line.strip())
        
        text = '\n'.join(clean_lines)
        
        return text.strip()
    
    def chunk_text(self, text: str, source_id: int = 0) -> List[Dict]:
        #\"\"\"Split text into overlapping chunks.\"\"\"
        if len(text) < self.min_length:
            return []
        
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                if len(current_chunk.strip()) >= self.min_length:
                    chunk_data = {
                        'text': current_chunk.strip(),
                        'source_id': source_id,
                        'chunk_index': len(chunks),
                        'sentence_count': len(current_sentences),
                        'word_count': len(word_tokenize(current_chunk))
                    }
                    chunks.append(chunk_data)
                
                # Start new chunk with overlap
                if self.overlap > 0 and current_sentences:
                    overlap_sentences = current_sentences[-self.overlap:]
                    current_chunk = " ".join(overlap_sentences) + " "
                    current_sentences = overlap_sentences[:]
                else:
                    current_chunk = ""
                    current_sentences = []
            
            current_chunk += sentence + " "
            current_sentences.append(sentence)
        
        # Add final chunk
        if len(current_chunk.strip()) >= self.min_length:
            chunk_data = {
                'text': current_chunk.strip(),
                'source_id': source_id,
                'chunk_index': len(chunks),
                'sentence_count': len(current_sentences),
                'word_count': len(word_tokenize(current_chunk))
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def calculate_quality_score(self, text: str) -> float:
        #\"\"\"Calculate quality score for text chunk.\"\"\"
        if not text:
            return 0.0
        
        score = 1.0
        words = word_tokenize(text.lower())
        
        # Word count factor
        word_count = len(words)
        filters = self.config.get('quality_filters', {})
        min_words = filters.get('min_word_count', 10)
        max_words = filters.get('max_word_count', 500)
        
        if word_count < min_words:
            score *= 0.5
        elif word_count > max_words:
            score *= 0.7
        
        # Stop word ratio (too many stop words = lower quality)
        stop_word_count = sum(1 for word in words if word in self.stop_words)
        stop_word_ratio = stop_word_count / len(words) if words else 0
        
        if stop_word_ratio > 0.7:
            score *= 0.6
        elif stop_word_ratio < 0.3:
            score *= 0.8
        
        # Character diversity
        unique_chars = set(text.lower())
        char_diversity = len(unique_chars) / len(text) if text else 0
        
        if char_diversity < 0.05:  # Too repetitive
            score *= 0.4
        
        # Sentence structure (prefer balanced sentences)
        sentences = sent_tokenize(text)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 5 or avg_sentence_length > 30:
                score *= 0.8
        
        # Penalize excessive repetition
        word_freq = Counter(words)
        most_common_freq = word_freq.most_common(1)[0][1] if word_freq else 1
        if most_common_freq > len(words) * 0.2:  # Single word > 20% of text
            score *= 0.3
        
        return min(score, 1.0)
    
    def remove_duplicates(self, chunks: List[Dict]) -> List[Dict]:
        #\"\"\"Remove duplicate chunks using TF-IDF similarity.\"\"\"
        if len(chunks) < 2:
            return chunks
        
        logger.info(f"Removing duplicates from {len(chunks)} chunks...")
        
        # Create TF-IDF vectors
        texts = [chunk['text'] for chunk in chunks]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(tfidf_matrix)
            
            # Mark duplicates
            to_remove = set()
            for i in range(len(chunks)):
                if i in to_remove:
                    continue
                    
                for j in range(i + 1, len(chunks)):
                    if j in to_remove:
                        continue
                    
                    if similarities[i][j] >= self.dedup_threshold:
                        # Keep the chunk with higher quality score
                        if chunks[i].get('quality_score', 0) >= chunks[j].get('quality_score', 0):
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break
            
            # Remove duplicates
            unique_chunks = [chunk for i, chunk in enumerate(chunks) if i not in to_remove]
            
            logger.info(f"Removed {len(chunks) - len(unique_chunks)} duplicate chunks")
            return unique_chunks
            
        except Exception as e:
            logger.error(f"Error in deduplication: {e}")
            return chunks
    
    def process_data_source(self, source_data: Dict) -> List[Dict]:
        #\"\"\"Process a single data source (web or PDF).\"\"\"
        text = source_data.get('text', '') or source_data.get('content', '')
        source_id = source_data.get('source_id', 0)
        
        if not text:
            logger.warning(f"No text found in source {source_id}")
            return []
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        if len(cleaned_text) < self.min_length:
            logger.info(f"Text too short after cleaning: {len(cleaned_text)} chars")
            return []
        
        # Chunk text
        chunks = self.chunk_text(cleaned_text, source_id)
        
        # Calculate quality scores
        for chunk in chunks:
            chunk['quality_score'] = self.calculate_quality_score(chunk['text'])
            chunk['text_hash'] = hashlib.md5(chunk['text'].encode()).hexdigest()
            chunk['metadata'] = {
                'source_type': source_data.get('source_type', 'unknown'),
                'original_length': len(text),
                'cleaned_length': len(cleaned_text)
            }
        
        # Filter by quality
        quality_threshold = 0.3
        quality_chunks = [c for c in chunks if c['quality_score'] >= quality_threshold]
        
        logger.info(f"Processed source {source_id}: {len(chunks)} chunks, {len(quality_chunks)} after quality filter")
        
        return quality_chunks
    
    def process_all_sources(self, sources_data: List[Dict]) -> List[Dict]:
        #\"\"\"Process all data sources.\"\"\"
        logger.info(f"Processing {len(sources_data)} data sources")
        
        all_chunks = []
        
        for i, source_data in enumerate(sources_data):
            source_data['source_id'] = i
            chunks = self.process_data_source(source_data)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks before deduplication: {len(all_chunks)}")
        
        # Remove duplicates
        unique_chunks = self.remove_duplicates(all_chunks)
        
        # Store processed chunks
        self.processed_chunks = unique_chunks
        
        logger.info(f"Final processed chunks: {len(unique_chunks)}")
        return unique_chunks
    
    def save_to_database(self):
        #\"\"\"Save processed chunks to database.\"\"\"
        db = next(get_db())
        
        try:
            for chunk in self.processed_chunks:
                processed_data = ProcessedData(
                    source_id=chunk['source_id'],
                    chunk_index=chunk['chunk_index'],
                    text_content=chunk['text'],
                    metadata=json.dumps(chunk['metadata']),
                    quality_score=chunk['quality_score']
                )
                db.add(processed_data)
            
            db.commit()
            logger.info(f"Saved {len(self.processed_chunks)} processed chunks to database")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            db.rollback()
        finally:
            db.close()

def run_text_processing(sources_data: List[Dict], config: Dict) -> List[Dict]:
    #\"\"\"Main function to run text processing.\"\"\"
    processing_config = config.get('data_processing', {})
    
    processor = TextProcessor(processing_config)
    processed_chunks = processor.process_all_sources(sources_data)
    
    if processed_chunks:
        processor.save_to_database()
    
    return processed_chunks