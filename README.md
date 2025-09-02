# ClaRE - Classification of Harmful and Benign Prompts

This project compiles and processes datasets containing harmful and benign prompts for AI safety research.

## Project Overview

ClaRE (Classification of Harmful and Benign Prompts) is a comprehensive system for compiling, processing, and analyzing datasets containing harmful and benign prompts. The system compiles multiple datasets from Hugging Face, automatically separating prompts into harmful and benign categories for AI safety research.

## Features

- **Multi-Dataset Compilation**: Downloads and processes 6+ major datasets from Hugging Face
- **Intelligent Classification**: Separates prompts into harmful and benign categories using response analysis
- **Data Validation**: Comprehensive quality checks and deduplication
- **Multiple Output Formats**: JSONL, CSV, and train/validation/test splits
- **Interactive Analysis**: Jupyter notebooks for data exploration
- **Modular Design**: Easy to extend with additional datasets
- **Open Source Model Ready**: Formatted for testing with various AI models

## Comprehensive Dataset Statistics

The compiled dataset contains:
- **Total Prompts**: 618,966
- **Harmful Prompts**: 236,358 (38.2%)
- **Benign Prompts**: 382,608 (61.8%)
- **Average Prompt Length**: 763 characters
- **Sources**: 6 major datasets from Hugging Face

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Comprehensive Data Compilation

```bash
# Compile ALL datasets (recommended)
python3 scripts/run_comprehensive_compilation.py

# Or compile only Anthropic HH dataset
python3 scripts/run_compilation.py
```

### Analyze Dataset

```bash
python3 scripts/analyze_dataset.py
```

### Interactive Analysis

```bash
jupyter notebook notebooks/dataset_analysis.ipynb
```

## Project Structure

```
ClaRE/
├── src/                           # Source code
│   ├── data_compiler.py          # Anthropic HH compilation logic
│   ├── multi_dataset_compiler.py # Multi-dataset compilation logic
│   ├── config.py                 # Configuration settings
│   └── utils.py                  # Utility functions
├── data/                         # Processed datasets
│   ├── raw/                     # Raw downloaded data from all sources
│   │   ├── anthropic_hh/        # Anthropic HH dataset
│   │   ├── allenai_harmful/     # AllenAI toxicity prompts
│   │   ├── onepaneai_harmful/   # OnepaneAI adversarial prompts
│   │   ├── nask_harmful/        # NASK-PIB harmful prompts
│   │   ├── guychuk_harmful/     # Guychuk classification dataset
│   │   ├── openassistant_benign/# OpenAssistant conversations
│   │   ├── alpaca_benign/       # Stanford Alpaca dataset
│   │   └── dolly_benign/        # Databricks Dolly dataset
│   ├── harmful/                 # All harmful prompts
│   │   ├── all_harmful_prompts.jsonl
│   │   └── all_harmful_prompts.csv
│   ├── benign/                  # All benign prompts
│   │   ├── all_benign_prompts.jsonl
│   │   └── all_benign_prompts.csv
│   └── processed/               # Combined datasets and splits
│       ├── all_combined_dataset.csv
│       ├── all_train.jsonl      # Training split (80%)
│       ├── all_val.jsonl        # Validation split (10%)
│       └── all_test.jsonl       # Test split (10%)
├── notebooks/                   # Analysis notebooks
├── scripts/                     # Utility scripts
└── requirements.txt             # Dependencies
```

## Compiled Datasets

### Harmful Prompt Datasets

| Dataset | Source | Size | Description | Variables |
|---------|--------|------|-------------|-----------|
| **OnepaneAI Adversarial** | `onepaneai/harmful-prompts` | 200 | Adversarial prompts designed to elicit harmful behavior | `prompt`, `is_harmful`, `category` |
| **NASK-PIB Harmful** | `NASK-PIB/harmful_prompts_sample` | 428 | Manually reviewed harmful prompts with categories | `prompt`, `category`, `subcategory` |
| **Guychuk Classification** | `guychuk/benign-malicious-prompt-classification` | 235,730 | Large-scale harmful prompt classification | `prompt`, `label` (0=benign, 1=harmful) |
| **Anthropic HH** | `Anthropic/hh-rlhf` | 8,786 | Human preference data with harmful responses | `prompt`, `response`, `label` |

### Benign Prompt Datasets

| Dataset | Source | Size | Description | Variables |
|---------|--------|------|-------------|-----------|
| **OpenAssistant** | `OpenAssistant/oasst1` | 86,856 | Human-generated assistant conversations | `prompt`, `text`, `role` |
| **Stanford Alpaca** | `tatsu-lab/alpaca` | 52,002 | Instruction-following prompts for fine-tuning | `prompt`, `instruction` |
| **Databricks Dolly** | `databricks/databricks-dolly-15k` | 15,011 | Human-generated instruction-response pairs | `prompt`, `instruction`, `category` |
| **Guychuk Classification** | `guychuk/benign-malicious-prompt-classification` | 228,739 | Large-scale benign prompt classification | `prompt`, `label` (0=benign, 1=harmful) |
| **Anthropic HH** | `Anthropic/hh-rlhf` | 58,161 | Human preference data with helpful responses | `prompt`, `response`, `label` |

## Classification Methodology

The system classifies prompts as harmful or benign based on:

### Harmful Indicators
- Responses containing refusal patterns ("I can't help with that", "I cannot assist")
- Content related to illegal, dangerous, or inappropriate activities
- Responses that indicate the AI is refusing to provide harmful information
- Explicit harmful content categories (violence, hate speech, etc.)

### Benign Indicators
- Helpful and constructive responses
- Educational content
- Responses that provide useful information without harm
- Normal conversational and instructional content

## Output Files

The system generates multiple output formats:

- **JSONL Files**: `all_harmful_prompts.jsonl`, `all_benign_prompts.jsonl`
- **CSV Files**: For easy analysis in spreadsheet applications
- **Train/Val/Test Splits**: Ready for machine learning experiments
- **Statistics**: Comprehensive dataset metrics and quality scores

## Data Variables & Structure

### Standard Variables (All Datasets)
- **`prompt`**: The actual prompt text (primary variable for testing)
- **`label`**: Classification label (`'harmful'` or `'benign'`)
- **`source`**: Dataset source identifier
- **`id`**: Unique identifier for each prompt
- **`split`**: Original dataset split (train/test/val)

### Additional Variables (Dataset-Specific)
- **`category`**: Content category (violence, hate speech, etc.)
- **`subcategory`**: More specific content classification
- **`toxicity_score`**: Toxicity rating (0-1) where available
- **`response`**: AI response text (for Anthropic HH dataset)
- **`instruction`**: Instruction text (for instruction-following datasets)
- **`role`**: Conversation role (for conversational datasets)

## Testing with Open Source Models

### Quick Data Access

```python
import pandas as pd
import json

# Load all harmful prompts
harmful_df = pd.read_csv('data/harmful/all_harmful_prompts.csv')
print(f"Loaded {len(harmful_df)} harmful prompts")

# Load all benign prompts
benign_df = pd.read_csv('data/benign/all_benign_prompts.csv')
print(f"Loaded {len(benign_df)} benign prompts")

# Load combined dataset
combined_df = pd.read_csv('data/processed/all_combined_dataset.csv')
print(f"Total dataset: {len(combined_df)} prompts")
```

### Testing with Hugging Face Models

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# Load dataset
df = pd.read_csv('data/processed/all_combined_dataset.csv')

# Test with a text classification model
classifier = pipeline("text-classification", 
                     model="microsoft/DialoGPT-medium")

# Test with a causal language model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Sample testing function
def test_prompts(model, tokenizer, prompts, max_samples=100):
    results = []
    for i, prompt in enumerate(prompts[:max_samples]):
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            'prompt': prompt,
            'response': response,
            'label': df.iloc[i]['label']
        })
    return results

# Test harmful prompts
harmful_prompts = df[df['label'] == 'harmful']['prompt'].tolist()
harmful_results = test_prompts(model, tokenizer, harmful_prompts)

# Test benign prompts
benign_prompts = df[df['label'] == 'benign']['prompt'].tolist()
benign_results = test_prompts(model, tokenizer, benign_prompts)
```

### Testing with OpenAI-Compatible Models

```python
import openai
import pandas as pd

# Load dataset
df = pd.read_csv('data/processed/all_combined_dataset.csv')

# Test with local models (Ollama, LM Studio, etc.)
def test_with_openai_api(prompts, model_name="gpt-3.5-turbo", max_samples=100):
    results = []
    for prompt in prompts[:max_samples]:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            results.append({
                'prompt': prompt,
                'response': response.choices[0].message.content,
                'model': model_name
            })
        except Exception as e:
            results.append({
                'prompt': prompt,
                'response': f"Error: {str(e)}",
                'model': model_name
            })
    return results

# Test sample prompts
sample_prompts = df.sample(100)['prompt'].tolist()
test_results = test_with_openai_api(sample_prompts)
```

### Testing with Local Models (Ollama)

```python
import requests
import pandas as pd

def test_with_ollama(prompts, model_name="llama2", max_samples=100):
    results = []
    for prompt in prompts[:max_samples]:
        try:
            response = requests.post('http://localhost:11434/api/generate',
                                   json={
                                       'model': model_name,
                                       'prompt': prompt,
                                       'stream': False
                                   })
            if response.status_code == 200:
                result = response.json()
                results.append({
                    'prompt': prompt,
                    'response': result['response'],
                    'model': model_name
                })
            else:
                results.append({
                    'prompt': prompt,
                    'response': f"Error: {response.status_code}",
                    'model': model_name
                })
        except Exception as e:
            results.append({
                'prompt': prompt,
                'response': f"Error: {str(e)}",
                'model': model_name
            })
    return results

# Load and test
df = pd.read_csv('data/processed/all_combined_dataset.csv')
harmful_prompts = df[df['label'] == 'harmful']['prompt'].tolist()
results = test_with_ollama(harmful_prompts, model_name="llama2")
```

### Batch Testing Script

```python
import pandas as pd
import json
from tqdm import tqdm

def batch_test_model(dataset_path, model_function, output_path, batch_size=100):
    """
    Test a model on the entire dataset in batches
    """
    df = pd.read_csv(dataset_path)
    results = []
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        batch_prompts = batch['prompt'].tolist()
        batch_results = model_function(batch_prompts)
        
        # Add metadata
        for j, result in enumerate(batch_results):
            result['original_label'] = batch.iloc[j]['label']
            result['source'] = batch.iloc[j]['source']
            result['batch_id'] = i // batch_size
        
        results.extend(batch_results)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# Example usage
results = batch_test_model(
    'data/processed/all_combined_dataset.csv',
    test_with_ollama,
    'model_test_results.json'
)
```

## Usage Examples

### Comprehensive Compilation

```python
from src.multi_dataset_compiler import MultiDatasetCompiler

# Compile all datasets
compiler = MultiDatasetCompiler()
harmful_prompts, benign_prompts = compiler.compile_all_datasets(exclude_anthropic=True)
compiler.save_compiled_datasets(harmful_prompts, benign_prompts)
compiler.generate_comprehensive_statistics(harmful_prompts, benign_prompts)
```

### Basic Compilation (Anthropic Only)

```python
from src.data_compiler import DatasetCompiler

compiler = DatasetCompiler()
raw_path = compiler.download_anthropic_hh_dataset()
harmful_prompts, benign_prompts = compiler.process_anthropic_hh_dataset(raw_path)
compiler.save_datasets(harmful_prompts, benign_prompts)
```

### Load Processed Data

```python
import pandas as pd

# Load all harmful prompts
harmful_df = pd.read_csv('data/harmful/all_harmful_prompts.csv')
print(f"Loaded {len(harmful_df)} harmful prompts")

# Load all benign prompts  
benign_df = pd.read_csv('data/benign/all_benign_prompts.csv')
print(f"Loaded {len(benign_df)} benign prompts")

# Load combined dataset
combined_df = pd.read_csv('data/processed/all_combined_dataset.csv')
print(f"Total dataset: {len(combined_df)} prompts")

# Load train/val/test splits
train_df = pd.read_csv('data/processed/all_train.csv')
val_df = pd.read_csv('data/processed/all_val.csv')
test_df = pd.read_csv('data/processed/all_test.csv')
```

### Filter by Dataset Source

```python
# Filter by specific dataset source
anthropic_harmful = harmful_df[harmful_df['source'] == 'anthropic_hh']
guychuk_harmful = harmful_df[harmful_df['source'] == 'guychuk_harmful']
nask_harmful = harmful_df[harmful_df['source'] == 'nask_harmful']

print(f"Anthropic harmful: {len(anthropic_harmful)}")
print(f"Guychuk harmful: {len(guychuk_harmful)}")
print(f"NASK harmful: {len(nask_harmful)}")
```

### Filter by Category

```python
# Filter harmful prompts by category
violence_prompts = harmful_df[harmful_df['category'] == 'violence']
hate_speech_prompts = harmful_df[harmful_df['category'] == 'hate_speech']

print(f"Violence prompts: {len(violence_prompts)}")
print(f"Hate speech prompts: {len(hate_speech_prompts)}")
```

## Research Applications

This comprehensive dataset can be used for:

### AI Safety Research
- **Harmful Content Detection**: Train classifiers to identify harmful prompts
- **Model Alignment Studies**: Analyze how different models respond to harmful vs benign prompts
- **Safety Benchmarking**: Evaluate model safety across diverse prompt types
- **Refusal Pattern Analysis**: Study how models refuse harmful requests

### Content Moderation
- **Automated Moderation Systems**: Train content filters using the large-scale dataset
- **Bias Detection**: Analyze harmful content patterns across different categories
- **Prompt Engineering**: Understand effective vs harmful prompt construction

### Machine Learning
- **Binary Classification**: Harmful vs benign prompt classification
- **Multi-class Classification**: Category-based harmful content classification
- **Text Generation Safety**: Train safer text generation models
- **Adversarial Training**: Improve model robustness against harmful prompts

## Dataset Summary by Source

| Source | Total | Harmful | Benign | Type | Key Features |
|--------|-------|---------|--------|------|--------------|
| **Guychuk** | 464,469 | 235,730 | 228,739 | Both | Largest dataset, binary classification |
| **OpenAssistant** | 86,856 | 0 | 86,856 | Benign | Conversational, multi-language |
| **Anthropic HH** | 66,947 | 8,786 | 58,161 | Both | Human preferences, refusal patterns |
| **Alpaca** | 52,002 | 0 | 52,002 | Benign | Instruction-following, educational |
| **Dolly** | 15,011 | 0 | 15,011 | Benign | Human-generated, categorized |
| **NASK-PIB** | 428 | 428 | 0 | Harmful | Manually reviewed, categorized |
| **OnepaneAI** | 200 | 200 | 0 | Harmful | Adversarial, designed to elicit harm |
| **TOTAL** | **618,966** | **236,358** | **382,608** | **Both** | **Comprehensive coverage** |

## Key Features

- **Scale**: 618,966 total prompts for robust analysis
- **Diversity**: 6 different dataset sources with varied content types
- **Balance**: 38.2% harmful, 61.8% benign for realistic testing
- **Categories**: Detailed categorization of harmful content types
- **Formats**: Multiple output formats (JSONL, CSV, splits)
- **Quality**: Validated, deduplicated, and ready for ML

## Important Notes

- **Content Warning**: The dataset contains potentially offensive, harmful, or upsetting content
- **Research Purpose**: Intended for AI safety research to make systems safer
- **Ethical Use**: Handle the data responsibly and in accordance with ethical guidelines
- **No Harmful Training**: Do not use this data to train models that could generate harmful content
- **Authentication**: Some datasets (like HEx-PHI) require Hugging Face authentication

## 📄 License

This project is for research purposes. Please refer to the original dataset licenses for usage terms:
- Anthropic HH: Research use only
- Other datasets: Check individual Hugging Face dataset pages for specific licenses
