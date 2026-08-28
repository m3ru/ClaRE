"""
Configuration settings for anon-repo data compilation.
"""

# Dataset URLs and sources
DATASET_SOURCES = {
    'anthropic_hh': {
        'name': 'Anthropic Helpful and Harmless',
        'huggingface_id': 'Anthropic/hh-rlhf',
        'description': 'Human preference data on helpfulness and harmlessness',
        'url': 'https://huggingface.co/datasets/Anthropic/hh-rlhf',
        'type': 'both'  # contains both harmful and benign
    },
    # Harmful Prompt Datasets
    'allenai_harmful': {
        'name': 'AllenAI Harmful Conversations',
        'huggingface_id': 'allenai/real-toxicity-prompts',
        'description': '~160k+ conversations with refusals in natural settings',
        'url': 'https://huggingface.co/datasets/allenai/real-toxicity-prompts',
        'type': 'harmful'
    },
    'onepaneai_harmful': {
        'name': 'OnepaneAI Adversarial Prompts',
        'huggingface_id': 'onepaneai/harmful-prompts',
        'description': '~120k+ adversarial prompts designed to elicit harmful behavior',
        'url': 'https://huggingface.co/datasets/onepaneai/harmful-prompts',
        'type': 'harmful'
    },
    'nask_harmful': {
        'name': 'NASK-PIB Harmful Prompts',
        'huggingface_id': 'NASK-PIB/harmful_prompts_sample',
        'description': '428 manually reviewed harmful prompts with categories',
        'url': 'https://huggingface.co/datasets/NASK-PIB/harmful_prompts_sample',
        'type': 'harmful'
    },
    'hex_phi_harmful': {
        'name': 'HEx-PHI Harmful Instructions',
        'huggingface_id': 'LLM-Tuning-Safety/HEx-PHI',
        'description': '330 harmful instructions across 11 prohibited categories',
        'url': 'https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI',
        'type': 'harmful'
    },
    'guychuk_harmful': {
        'name': 'Guychuk Harmful Prompts',
        'huggingface_id': 'guychuk/benign-malicious-prompt-classification',
        'description': '464k prompts labeled as benign or malicious',
        'url': 'https://huggingface.co/datasets/guychuk/benign-malicious-prompt-classification',
        'type': 'both'  # contains both harmful and benign
    },
    # Benign Prompt Datasets
    'openassistant_benign': {
        'name': 'OpenAssistant Conversations',
        'huggingface_id': 'OpenAssistant/oasst1',
        'description': '88k human-generated assistant conversations across 35 languages',
        'url': 'https://huggingface.co/datasets/OpenAssistant/oasst1',
        'type': 'benign'
    },
    'alpaca_benign': {
        'name': 'Stanford Alpaca',
        'huggingface_id': 'tatsu-lab/alpaca',
        'description': '52k instruction-following prompts for fine-tuning',
        'url': 'https://huggingface.co/datasets/tatsu-lab/alpaca',
        'type': 'benign'
    },
    'dolly_benign': {
        'name': 'Databricks Dolly 15k',
        'huggingface_id': 'databricks/databricks-dolly-15k',
        'description': '15k human-generated instruction-response pairs',
        'url': 'https://huggingface.co/datasets/databricks/databricks-dolly-15k',
        'type': 'benign'
    }
}

# Classification criteria
HARMFUL_INDICATORS = [
    "I can't help with that",
    "I'm not able to",
    "I cannot and will not",
    "I can't provide",
    "I'm not programmed to",
    "I cannot assist",
    "I won't help",
    "I cannot help",
    "I'm unable to",
    "I cannot comply",
    "I cannot and will not provide",
    "I cannot provide information",
    "I cannot give you",
    "I cannot create",
    "I cannot write",
    "I cannot generate",
    "I cannot share",
    "I cannot recommend",
    "I cannot suggest",
    "I cannot tell you how to"
]

BENIGN_INDICATORS = [
    "here's how",
    "here are some",
    "I can help",
    "I'd be happy to",
    "let me explain",
    "here's what",
    "I can provide",
    "I can assist",
    "I can explain",
    "I can show you",
    "I can guide you",
    "I can walk you through",
    "I can help you understand",
    "I can help you with",
    "I can help you learn",
    "I can help you find",
    "I can help you create",
    "I can help you write",
    "I can help you develop"
]

HARMFUL_PATTERNS = [
    "illegal", "harmful", "dangerous", "violent", "explicit",
    "inappropriate", "offensive", "hate speech", "discrimination",
    "self-harm", "suicide", "violence", "weapon", "drug",
    "fraud", "scam", "phishing", "malware", "hack"
]

# Data processing settings
MIN_RESPONSE_LENGTH = 50
MAX_RESPONSE_LENGTH = 10000

# Output settings
OUTPUT_FORMATS = ['jsonl', 'csv', 'json']
DEFAULT_OUTPUT_FORMAT = 'jsonl'

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
