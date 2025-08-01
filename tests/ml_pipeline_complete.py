    def create_domain_benchmark(self, qa_pairs: List[Dict], benchmark_size: int = 200) -> List[Dict]:
        \"\"\"Create domain-specific benchmark dataset.\"\"\"
        logger.info(f"Creating domain benchmark with {benchmark_size} samples")
        
        # Filter test pairs
        test_pairs = [pair for pair in qa_pairs if pair.get('split') == 'test']
        
        if len(test_pairs) >= benchmark_size:
            benchmark = random.sample(test_pairs, benchmark_size)
        else:
            # Use all test pairs and sample from train if needed
            benchmark = test_pairs[:]
            remaining_needed = benchmark_size - len(test_pairs)
            
            train_pairs = [pair for pair in qa_pairs if pair.get('split') == 'train']
            if train_pairs and remaining_needed > 0:
                additional = random.sample(train_pairs, min(remaining_needed, len(train_pairs)))
                benchmark.extend(additional)
        
        logger.info(f"Domain benchmark created with {len(benchmark)} samples")
        return benchmark
    
    def evaluate_model(self, model, tokenizer, benchmark: List[Dict], model_name: str = "model") -> Dict:
        \"\"\"Evaluate a model on the benchmark dataset.\"\"\"
        logger.info(f"Evaluating {model_name} on {len(benchmark)} samples")
        
        questions = [item['question'] for item in benchmark]
        reference_answers = [item['answer'] for item in benchmark]
        
        # Generate predictions
        predictions = []
        for question in questions:
            answer = self.generate_answer(model, tokenizer, question)
            predictions.append(answer)
        
        # Calculate metrics
        results = {'model_name': model_name}
        
        if 'rouge' in self.metrics:
            rouge_scores = self.calculate_rouge_scores(predictions, reference_answers)
            results.update(rouge_scores)
        
        if 'bleu' in self.metrics:
            bleu_score = self.calculate_bleu_score(predictions, reference_answers)
            results['bleu'] = bleu_score
        
        if 'exact_match' in self.metrics:
            em_score = self.calculate_exact_match(predictions, reference_answers)
            results['exact_match'] = em_score
        
        if 'f1' in self.metrics:
            f1_score = self.calculate_f1_score(predictions, reference_answers)
            results['f1'] = f1_score
        
        # Measure inference speed
        speed_metrics = self.measure_inference_speed(model, tokenizer, questions)
        results.update(speed_metrics)
        
        logger.info(f"{model_name} evaluation completed")
        return results
    
    def run_evaluation(self, finetuned_model_path: str, base_model_name: str, qa_pairs: List[Dict]) -> Dict:
        \"\"\"Run full evaluation comparing base and fine-tuned models.\"\"\"
        logger.info("Starting model evaluation")
        
        # Create benchmark
        benchmark = self.create_domain_benchmark(qa_pairs, self.eval_config.get('benchmark_size', 200))
        
        # Load models
        self.load_base_model(base_model_name)
        self.load_finetuned_model(finetuned_model_path, base_model_name)
        
        # Evaluate base model
        base_results = self.evaluate_model(
            self.base_model, 
            self.base_tokenizer, 
            benchmark, 
            "base_model"
        )
        
        # Evaluate fine-tuned model
        finetuned_results = self.evaluate_model(
            self.finetuned_model, 
            self.finetuned_tokenizer, 
            benchmark, 
            "finetuned_model"
        )
        
        # Calculate improvements
        improvements = {}
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bleu', 'exact_match', 'f1']:
            if metric in base_results and metric in finetuned_results:
                base_score = base_results[metric]
                ft_score = finetuned_results[metric]
                improvement = ((ft_score - base_score) / base_score * 100) if base_score > 0 else 0
                improvements[f"{metric}_improvement_%"] = improvement
        
        # Compile results
        evaluation_results = {
            'benchmark_size': len(benchmark),
            'base_model_results': base_results,
            'finetuned_model_results': finetuned_results,
            'improvements': improvements,
            'evaluation_timestamp': time.time()
        }
        
        # Log summary
        logger.info("Evaluation Summary:")
        logger.info(f"Base Model ROUGE-L: {base_results.get('rougeL', 0):.4f}")
        logger.info(f"Fine-tuned Model ROUGE-L: {finetuned_results.get('rougeL', 0):.4f}")
        logger.info(f"ROUGE-L Improvement: {improvements.get('rougeL_improvement_%', 0):.2f}%")
        
        return evaluation_results
    
    def save_evaluation_results(self, results: Dict, output_path: str = "evaluation_results.json"):
        \"\"\"Save evaluation results to file.\"\"\"
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")

def run_evaluation(finetuned_model_path: str, base_model_name: str, qa_pairs: List[Dict], config: Dict) -> Dict:
    \"\"\"Main function to run model evaluation.\"\"\"
    evaluator = ModelEvaluator(config)
    results = evaluator.run_evaluation(finetuned_model_path, base_model_name, qa_pairs)
    
    # Save results
    evaluator.save_evaluation_results(results)
    
    return results

# ============================================================================
# src/deployment/api_server.py
# ============================================================================

api_server_py = """
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
import json
from typing import Dict, List, Optional
import asyncio
from pathlib import Path
import logging

from ..utils.logging import get_logger
from ..utils.database import get_db, ModelDeployment
from ..config.settings import settings

logger = get_logger(__name__)

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    max_length: Optional[int] = 150
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class ChatResponse(BaseModel):
    answer: str
    response_time: float
    model_info: Dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime: float

# Security
security = HTTPBearer()

class ModelServer:
    \"\"\"Model serving server with API endpoints.\"\"\"
    
    def __init__(self, model_path: str, base_model_name: str):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.model_info = {}
        
        # Server stats
        self.start_time = time.time()
        self.request_count = 0
        
        self.load_model()
    
    def load_model(self):
        \"\"\"Load the# ============================================================================
# src/dataset_generation/qa_generator.py
# ============================================================================

qa_generator_py = """
import openai
import anthropic
import json
import time
import random
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from ..utils.logging import get_logger
from ..utils.database import get_db, TrainingDataset
from ..config.settings import settings

logger = get_logger(__name__)

class QAGenerator:
    \"\"\"Generate question-answer pairs from processed text using LLM APIs.\"\"\"
    
    def __init__(self, config: Dict):
        self.config = config
        self.api_config = config.get('llm_api', {})
        self.qa_config = config.get('qa_generation', {})
        
        # Initialize API clients
        self.openai_client = None
        self.anthropic_client = None
        
        self._setup_api_clients()
        
        self.generated_pairs: List[Dict] = []
        self.question_templates = self._load_question_templates()
    
    def _setup_api_clients(self):
        \"\"\"Setup API clients based on configuration.\"\"\"
        provider = self.api_config.get('provider', 'openai')
        
        if provider == 'openai' and settings.openai_api_key:
            openai.api_key = settings.openai_api_key
            self.openai_client = openai
            logger.info("OpenAI client initialized")
        
        elif provider == 'anthropic' and settings.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            logger.info("Anthropic client initialized")
        
        else:
            logger.error(f"No API key found for provider: {provider}")
    
    def _load_question_templates(self) -> List[Dict]:
        \"\"\"Load question generation templates.\"\"\"
        return [
            {
                'type': 'factual',
                'template': 'Based on the following text, generate a factual question and its answer:\\n\\nText: {context}\\n\\nQuestion: ',
                'system_prompt': 'You are an expert at creating educational questions. Generate clear, specific questions that can be answered directly from the given text.'
            },
            {
                'type': 'conceptual',
                'template': 'Based on the following text, generate a conceptual question that tests understanding and its answer:\\n\\nText: {context}\\n\\nQuestion: ',
                'system_prompt': 'You are an expert at creating questions that test conceptual understanding. Generate questions that require comprehension of the main ideas.'
            },
            {
                'type': 'analytical',
                'template': 'Based on the following text, generate an analytical question and its answer:\\n\\nText: {context}\\n\\nQuestion: ',
                'system_prompt': 'You are an expert at creating analytical questions. Generate questions that require analysis or inference from the given information.'
            },
            {
                'type': 'practical',
                'template': 'Based on the following text, generate a practical application question and its answer:\\n\\nText: {context}\\n\\nQuestion: ',
                'system_prompt': 'You are an expert at creating practical questions. Generate questions about real-world applications of the information.'
            }
        ]
    
    async def generate_qa_openai(self, context: str, template: Dict) -> Optional[Dict]:
        \"\"\"Generate QA pair using OpenAI API.\"\"\"
        try:
            # Create the prompt
            prompt = template['template'].format(context=context[:1500])  # Limit context length
            
            # Make API call
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model=self.api_config.get('model', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "system", "content": template['system_prompt']},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.api_config.get('max_tokens', 150),
                temperature=self.api_config.get('temperature', 0.7),
                top_p=0.9
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # Parse question and answer
            qa_pair = self._parse_qa_response(generated_text, context, template['type'])
            return qa_pair
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
    
    async def generate_qa_anthropic(self, context: str, template: Dict) -> Optional[Dict]:
        \"\"\"Generate QA pair using Anthropic API.\"\"\"
        try:
            prompt = f"{template['system_prompt']}\\n\\n{template['template'].format(context=context[:1500])}"
            
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-sonnet-20240229",
                max_tokens=self.api_config.get('max_tokens', 150),
                temperature=self.api_config.get('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            
            generated_text = response.content[0].text.strip()
            
            # Parse question and answer
            qa_pair = self._parse_qa_response(generated_text, context, template['type'])
            return qa_pair
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None
    
    def _parse_qa_response(self, response: str, context: str, question_type: str) -> Optional[Dict]:
        \"\"\"Parse the LLM response to extract question and answer.\"\"\"
        try:
            # Try to find question and answer patterns
            lines = response.strip().split('\\n')
            
            question = ""
            answer = ""
            
            # Look for common patterns
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Question patterns
                if any(pattern in line.lower() for pattern in ['question:', 'q:', 'ask:']):
                    question = line.split(':', 1)[1].strip() if ':' in line else line
                    
                # Answer patterns
                elif any(pattern in line.lower() for pattern in ['answer:', 'a:', 'response:']):
                    answer = line.split(':', 1)[1].strip() if ':' in line else line
                    
                # If no explicit markers, try to infer
                elif '?' in line and not question:
                    question = line
                elif question and not answer and line:
                    answer = line
            
            # Fallback: split by common patterns
            if not question or not answer:
                if '\\nAnswer:' in response:
                    parts = response.split('\\nAnswer:', 1)
                    question = parts[0].replace('Question:', '').strip()
                    answer = parts[1].strip()
                elif '?' in response:
                    parts = response.split('?', 1)
                    question = parts[0].strip() + '?'
                    answer = parts[1].strip() if len(parts) > 1 else ""
            
            # Validate QA pair
            if not question or not answer or len(question) < 10 or len(answer) < 5:
                logger.warning(f"Invalid QA pair generated: Q='{question[:50]}...', A='{answer[:50]}...'")
                return None
            
            return {
                'question': question,
                'answer': answer,
                'context': context,
                'question_type': question_type,
                'metadata': {
                    'generated_by': self.api_config.get('provider', 'unknown'),
                    'model': self.api_config.get('model', 'unknown'),
                    'generation_timestamp': time.time()
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing QA response: {e}")
            return None
    
    async def generate_questions_for_chunk(self, chunk: Dict) -> List[Dict]:
        \"\"\"Generate multiple questions for a single text chunk.\"\"\"
        context = chunk.get('text', '')
        if len(context) < 50:  # Skip very short contexts
            return []
        
        questions_per_chunk = self.qa_config.get('questions_per_chunk', 2)
        generated_pairs = []
        
        # Select random templates for variety
        selected_templates = random.sample(
            self.question_templates, 
            min(questions_per_chunk, len(self.question_templates))
        )
        
        tasks = []
        for template in selected_templates:
            if self.api_config.get('provider') == 'openai' and self.openai_client:
                task = self.generate_qa_openai(context, template)
            elif self.api_config.get('provider') == 'anthropic' and self.anthropic_client:
                task = self.generate_qa_anthropic(context, template)
            else:
                continue
                
            tasks.append(task)
        
        # Execute generation tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and result:
                    # Add chunk metadata
                    result['source_chunk_id'] = chunk.get('chunk_index', 0)
                    result['source_id'] = chunk.get('source_id', 0)
                    generated_pairs.append(result)
        
        return generated_pairs
    
    async def generate_dataset(self, processed_chunks: List[Dict]) -> List[Dict]:
        \"\"\"Generate QA dataset from processed text chunks.\"\"\"
        logger.info(f"Generating QA pairs from {len(processed_chunks)} chunks")
        
        max_questions = self.qa_config.get('max_questions', 1000)
        
        # Sort chunks by quality score (highest first)
        sorted_chunks = sorted(
            processed_chunks, 
            key=lambda x: x.get('quality_score', 0), 
            reverse=True
        )
        
        all_qa_pairs = []
        processed_count = 0
        
        # Process chunks in batches to avoid API rate limits
        batch_size = 10
        
        for i in range(0, len(sorted_chunks), batch_size):
            if len(all_qa_pairs) >= max_questions:
                break
                
            batch = sorted_chunks[i:i + batch_size]
            batch_tasks = []
            
            for chunk in batch:
                task = self.generate_questions_for_chunk(chunk)
                batch_tasks.append(task)
            
            # Process batch
            if batch_tasks:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, list):
                        all_qa_pairs.extend(result)
                        
                        # Stop if we have enough questions
                        if len(all_qa_pairs) >= max_questions:
                            break
            
            processed_count += len(batch)
            
            # Add delay to respect API rate limits
            if processed_count % 50 == 0:
                logger.info(f"Generated {len(all_qa_pairs)} QA pairs from {processed_count} chunks")
                await asyncio.sleep(1)  # Rate limiting delay
        
        # Limit to max questions
        final_pairs = all_qa_pairs[:max_questions]
        
        # Store generated pairs
        self.generated_pairs = final_pairs
        
        logger.info(f"QA generation completed: {len(final_pairs)} pairs generated")
        return final_pairs
    
    def create_train_test_split(self, qa_pairs: List[Dict], test_ratio: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
        \"\"\"Split QA pairs into train and test sets.\"\"\"
        random.shuffle(qa_pairs)
        
        split_index = int(len(qa_pairs) * (1 - test_ratio))
        train_pairs = qa_pairs[:split_index]
        test_pairs = qa_pairs[split_index:]
        
        # Mark splits
        for pair in train_pairs:
            pair['split'] = 'train'
        
        for pair in test_pairs:
            pair['split'] = 'test'
        
        logger.info(f"Dataset split: {len(train_pairs)} train, {len(test_pairs)} test")
        return train_pairs, test_pairs
    
    def save_to_database(self):
        \"\"\"Save generated QA pairs to database.\"\"\"
        db = next(get_db())
        
        try:
            for pair in self.generated_pairs:
                dataset_entry = TrainingDataset(
                    question=pair['question'],
                    answer=pair['answer'],
                    context=pair['context'],
                    source_id=pair.get('source_id', 0),
                    split=pair.get('split', 'train'),
                    metadata=json.dumps(pair.get('metadata', {}))
                )
                db.add(dataset_entry)
            
            db.commit()
            logger.info(f"Saved {len(self.generated_pairs)} QA pairs to database")
            
        except Exception as e:
            logger.error(f"Error saving QA pairs to database: {e}")
            db.rollback()
        finally:
            db.close()
    
    def export_to_json(self, filepath: str):
        \"\"\"Export generated QA pairs to JSON file.\"\"\"
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.generated_pairs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"QA pairs exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting QA pairs: {e}")

async def run_qa_generation(processed_chunks: List[Dict], config: Dict) -> List[Dict]:
    \"\"\"Main function to run QA generation.\"\"\"
    dataset_config = config.get('dataset_generation', {})
    
    generator = QAGenerator(dataset_config)
    qa_pairs = await generator.generate_dataset(processed_chunks)
    
    if qa_pairs:
        # Create train/test split
        train_pairs, test_pairs = generator.create_train_test_split(qa_pairs)
        all_pairs = train_pairs + test_pairs
        
        # Update generator with split data
        generator.generated_pairs = all_pairs
        
        # Save to database
        generator.save_to_database()
        
        # Export to JSON
        generator.export_to_json("data/qa_dataset.json")
        
        return all_pairs
    
    return []

# ============================================================================
# src/fine_tuning/trainer.py
# ============================================================================

trainer_py = """
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import mlflow
import wandb
from datetime import datetime

from ..utils.logging import get_logger
from ..utils.database import get_db, ModelExperiment
from ..config.settings import settings

logger = get_logger(__name__)

class QADataset(Dataset):
    \"\"\"Dataset class for QA pairs.\"\"\"
    
    def __init__(self, qa_pairs: List[Dict], tokenizer, max_length: int = 512):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        pair = self.qa_pairs[idx]
        
        # Format as conversation
        prompt = f"Question: {pair['question']}\\nAnswer: {pair['answer']}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding.input_ids.flatten(),
            'attention_mask': encoding.attention_mask.flatten(),
            'labels': encoding.input_ids.flatten()
        }

class FineTuner:
    \"\"\"Fine-tune language models using LoRA/QLoRA.\"\"\"
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config.get('model', {})
        self.training_config = config.get('fine_tuning', {})
        
        self.base_model_name = self.model_config.get('base_model', 'microsoft/DialoGPT-small')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tracking
        self.experiment_name = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._setup_tracking()
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def _setup_tracking(self):
        \"\"\"Setup MLflow and WandB tracking.\"\"\"
        try:
            # MLflow setup
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.get('monitoring', {}).get('mlflow', {}).get('experiment_name', 'default'))
            
            # WandB setup (if key available)
            if settings.wandb_api_key:
                wandb.init(
                    project="domain-llm-finetuning",
                    name=self.experiment_name,
                    config=self.config
                )
            
            logger.info("Experiment tracking setup complete")
            
        except Exception as e:
            logger.warning(f"Could not setup experiment tracking: {e}")
    
    def load_model_and_tokenizer(self):
        \"\"\"Load base model and tokenizer.\"\"\"
        logger.info(f"Loading model: {self.base_model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                cache_dir=self.model_config.get('cache_dir', './models/'),
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                cache_dir=self.model_config.get('cache_dir', './models/'),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto' if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def setup_lora(self):
        \"\"\"Setup LoRA configuration and apply to model.\"\"\"
        lora_config = self.training_config.get('lora_config', {})
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            target_modules=lora_config.get('target_modules', ['q_proj', 'v_proj']),
            bias="none",
            inference_mode=False
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return self.peft_model
    
    def prepare_datasets(self, qa_pairs: List[Dict]) -> Tuple[Dataset, Dataset]:
        \"\"\"Prepare train and validation datasets.\"\"\"
        # Split data
        train_pairs = [pair for pair in qa_pairs if pair.get('split') == 'train']
        val_pairs = [pair for pair in qa_pairs if pair.get('split') == 'test']
        
        # If no split specified, create one
        if not val_pairs:
            split_idx = int(0.8 * len(qa_pairs))
            train_pairs = qa_pairs[:split_idx]
            val_pairs = qa_pairs[split_idx:]
        
        # Create datasets
        train_dataset = QADataset(train_pairs, self.tokenizer)
        val_dataset = QADataset(val_pairs, self.tokenizer)
        
        logger.info(f"Dataset prepared: {len(train_dataset)} train, {len(val_dataset)} validation")
        
        return train_dataset, val_dataset
    
    def get_training_arguments(self, output_dir: str) -> TrainingArguments:
        \"\"\"Get training arguments configuration.\"\"\"
        training_config = self.training_config.get('training', {})
        
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=training_config.get('num_epochs', 3),
            per_device_train_batch_size=training_config.get('batch_size', 4),
            per_device_eval_batch_size=training_config.get('batch_size', 4),
            warmup_steps=training_config.get('warmup_steps', 100),
            learning_rate=training_config.get('learning_rate', 5e-4),
            fp16=torch.cuda.is_available(),
            logging_steps=training_config.get('logging_steps', 10),
            save_steps=training_config.get('save_steps', 500),
            eval_steps=training_config.get('eval_steps', 250),
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["mlflow", "wandb"] if settings.wandb_api_key else ["mlflow"],
            logging_dir=f"{output_dir}/logs",
            run_name=self.experiment_name,
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            dataloader_num_workers=2,
            remove_unused_columns=False
        )
    
    def train(self, qa_pairs: List[Dict]) -> str:
        \"\"\"Train the model with QA pairs.\"\"\"
        logger.info("Starting model fine-tuning")
        
        # Setup model
        self.load_model_and_tokenizer()
        peft_model = self.setup_lora()
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(qa_pairs)
        
        # Setup output directory
        output_dir = f"./models/finetuned/{self.experiment_name}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = self.get_training_arguments(output_dir)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if torch.cuda.is_available() else None
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Start training
        try:
            with mlflow.start_run(run_name=self.experiment_name):
                # Log parameters
                mlflow.log_params({
                    'base_model': self.base_model_name,
                    'num_train_samples': len(train_dataset),
                    'num_val_samples': len(val_dataset),
                    **self.training_config
                })
                
                # Train
                train_result = trainer.train()
                
                # Log metrics
                mlflow.log_metrics({
                    'final_train_loss': train_result.training_loss,
                    'train_runtime': train_result.metrics['train_runtime'],
                    'train_samples_per_second': train_result.metrics['train_samples_per_second']
                })
                
                # Save model
                trainer.save_model()
                self.tokenizer.save_pretrained(output_dir)
                
                # Save LoRA adapters separately
                peft_model.save_pretrained(f"{output_dir}/lora_adapters")
                
                logger.info(f"Training completed. Model saved to {output_dir}")
                
                # Save to database
                self._save_experiment_to_db(output_dir, train_result.metrics)
                
                return output_dir
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _save_experiment_to_db(self, model_path: str, metrics: Dict):
        \"\"\"Save experiment details to database.\"\"\"
        db = next(get_db())
        
        try:
            experiment = ModelExperiment(
                experiment_name=self.experiment_name,
                model_name=self.base_model_name,
                config=json.dumps(self.config),
                status="completed",
                metrics=json.dumps(metrics),
                model_path=model_path
            )
            db.add(experiment)
            db.commit()
            
            logger.info(f"Experiment {self.experiment_name} saved to database")
            
        except Exception as e:
            logger.error(f"Error saving experiment to database: {e}")
            db.rollback()
        finally:
            db.close()

def run_fine_tuning(qa_pairs: List[Dict], config: Dict) -> str:
    \"\"\"Main function to run fine-tuning.\"\"\"
    if not qa_pairs:
        logger.error("No QA pairs provided for training")
        return ""
    
    trainer = FineTuner(config)
    model_path = trainer.train(qa_pairs)
    
    return model_path

# ============================================================================
# src/evaluation/evaluator.py
# ============================================================================

evaluator_py = """
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
import re
from collections import Counter
import random

from ..utils.logging import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    \"\"\"Evaluate fine-tuned models against baselines.\"\"\"
    
    def __init__(self, config: Dict):
        self.config = config
        self.eval_config = config.get('evaluation', {})
        self.metrics = self.eval_config.get('metrics', ['rouge', 'bleu', 'exact_match', 'f1'])
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Models
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        self.finetuned_tokenizer = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_base_model(self, model_name: str):
        \"\"\"Load base model for comparison.\"\"\"
        try:
            logger.info(f"Loading base model: {model_name}")
            
            self.base_tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto' if torch.cuda.is_available() else None
            )
            
            logger.info("Base model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            raise
    
    def load_finetuned_model(self, model_path: str, base_model_name: str):
        \"\"\"Load fine-tuned model.\"\"\"
        try:
            logger.info(f"Loading fine-tuned model from: {model_path}")
            
            # Load tokenizer
            self.finetuned_tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.finetuned_tokenizer.pad_token is None:
                self.finetuned_tokenizer.pad_token = self.finetuned_tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto' if torch.cuda.is_available() else None
            )
            
            # Load LoRA adapters
            lora_path = f"{model_path}/lora_adapters"
            if Path(lora_path).exists():
                self.finetuned_model = PeftModel.from_pretrained(base_model, lora_path)
                logger.info("LoRA adapters loaded successfully")
            else:
                # Fallback to loading full model
                self.finetuned_model = AutoModelForCausalLM.from_pretrained(model_path)
                logger.info("Full fine-tuned model loaded")
                
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            raise
    
    def generate_answer(self, model, tokenizer, question: str, max_length: int = 150) -> str:
        \"\"\"Generate answer using model.\"\"\"
        try:
            prompt = f"Question: {question}\\nAnswer:"
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer (remove prompt)
            if "Answer:" in full_response:
                answer = full_response.split("Answer:")[-1].strip()
            else:
                answer = full_response[len(prompt):].strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return ""
    
    def calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict:
        \"\"\"Calculate ROUGE scores.\"\"\"
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            rouge_scores = self.rouge_scorer.score(ref, pred)
            
            scores['rouge1'].append(rouge_scores['rouge1'].fmeasure)
            scores['rouge2'].append(rouge_scores['rouge2'].fmeasure)
            scores['rougeL'].append(rouge_scores['rougeL'].fmeasure)
        
        # Calculate averages
        avg_scores = {
            'rouge1': np.mean(scores['rouge1']),
            'rouge2': np.mean(scores['rouge2']),
            'rougeL': np.mean(scores['rougeL'])
        }
        
        return avg_scores
    
    def calculate_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        \"\"\"Calculate BLEU score.\"\"\"
        try:
            # Format references for sacrebleu
            formatted_refs = [[ref] for ref in references]
            
            bleu_score = corpus_bleu(predictions, formatted_refs)
            return bleu_score.score / 100.0  # Convert to 0-1 scale
            
        except Exception as e:
            logger.error(f"Error calculating BLEU: {e}")
            return 0.0
    
    def calculate_exact_match(self, predictions: List[str], references: List[str]) -> float:
        \"\"\"Calculate exact match accuracy.\"\"\"
        exact_matches = 0
        
        for pred, ref in zip(predictions, references):
            # Normalize text
            pred_norm = re.sub(r'\s+', ' ', pred.lower().strip())
            ref_norm = re.sub(r'\s+', ' ', ref.lower().strip())
            
            if pred_norm == ref_norm:
                exact_matches += 1
        
        return exact_matches / len(predictions) if predictions else 0.0
    
    def calculate_f1_score(self, predictions: List[str], references: List[str]) -> float:
        \"\"\"Calculate F1 score based on token overlap.\"\"\"
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if not ref_tokens:
                f1_scores.append(0.0)
                continue
            
            common_tokens = pred_tokens & ref_tokens
            
            if not common_tokens:
                f1_scores.append(0.0)
                continue
            
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(ref_tokens)
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    def measure_inference_speed(self, model, tokenizer, questions: List[str], num_samples: int = 50) -> Dict:
        \"\"\"Measure inference latency and throughput.\"\"\"
        if len(questions) > num_samples:
            test_questions = random.sample(questions, num_samples)
        else:
            test_questions = questions
        
        latencies = []
        
        for question in test_questions:
            start_time = time.time()
            _ = self.generate_answer(model, tokenizer, question)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        avg_latency = np.mean(latencies)
        throughput = 1.0 / avg_latency if avg_latency > 0 else 0
        
        return {
            'avg_latency_seconds': avg_latency,
            'throughput_qps': throughput,
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'std_latency': np.std(latencies)
        }
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

pdf_extractor_py = """
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
    \"\"\"Extract text and metadata from PDF files with layout preservation.\"\"\"
    
    def __init__(self, config: Dict):
        self.config = config
        self.extracted_data: List[Dict] = []
    
    def extract_from_file(self, file_path: Path) -> Optional[Dict]:
        \"\"\"Extract content from a single PDF file.\"\"\"
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
        \"\"\"Extract using PyMuPDF (faster, basic layout).\"\"\"
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
        \"\"\"Extract using pdfplumber (slower, better layout).\"\"\"
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
        \"\"\"Process PyMuPDF text blocks for structure.\"\"\"
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
        \"\"\"Convert table data to text format.\"\"\"
        if not table:
            return ""
        
        text_rows = []
        for row in table:
            # Clean and join row cells
            clean_row = [str(cell).strip() if cell else "" for cell in row]
            text_rows.append(" | ".join(clean_row))
        
        return "\\n".join(text_rows)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        \"\"\"Calculate MD5 hash of file for deduplication.\"\"\"
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def extract_from_directory(self, directory: Path) -> List[Dict]:
        \"\"\"Extract content from all PDFs in a directory.\"\"\"
        pdf_files = list(directory.glob("*.pdf"))
        max_files = self.config.get('max_files', 50)
        
        logger.info(f"Found {len(pdf_files)} PDF files, processing up to {max_files}")
        
        for pdf_file in pdf_files[:max_files]:
            content = self.extract_from_file(pdf_file)
            if content:
                self.extracted_data.append(content)
        
        return self.extracted_data
    
    def save_to_database(self):
        \"\"\"Save extracted PDF data to database.\"\"\"
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

text_processor_py = """
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
    \"\"\"Process and clean raw text data for training.\"\"\"
    
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
        \"\"\"Clean and normalize text content.\"\"\"
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
        \"\"\"Split text into overlapping chunks.\"\"\"
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
        \"\"\"Calculate quality score for text chunk.\"\"\"
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
        \"\"\"Remove duplicate chunks using TF-IDF similarity.\"\"\"
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
        \"\"\"Process a single data source (web or PDF).\"\"\"
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
        \"\"\"Process all data sources.\"\"\"
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
        \"\"\"Save processed chunks to database.\"\"\"
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
    \"\"\"Main function to run text processing.\"\"\"
    processing_config = config.get('data_processing', {})
    
    processor = TextProcessor(processing_config)
    processed_chunks = processor.process_all_sources(sources_data)
    
    if processed_chunks:
        processor.save_to_database()
    
    return processed_chunks
"""