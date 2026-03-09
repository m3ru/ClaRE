# M3 Refusal Classifier

Optimized Python script for classifying 80k LLM outputs as refusals using the ProtectAI rejection classifier on an M3 MacBook Air, with resumable 20-minute chunks.

## Features

- **M3 Optimization**: Native Apple Silicon support with MLX and MPS fallbacks
- **Chunked Processing**: Process data in configurable time windows (default: 20 minutes)
- **Resumable**: Automatic checkpointing and resume functionality
- **Auto-optimization**: Dynamic batch size optimization for your hardware
- **Progress Tracking**: Rich terminal UI with real-time performance metrics
- **Memory Efficient**: Smart memory management for M3's unified memory architecture

## Performance Targets

- **Target**: 500-1000 classifications/second on M3
- **Completion Time**: 1.5-3 hours for 80k samples across multiple sessions
- **Memory Usage**: Optimized for M3's 16-24GB unified memory

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install MLX for maximum M3 performance
pip install mlx
```

## Usage

### Basic Usage

```bash
python classify_refusals_m3.py \
    --input prompts_completions.csv \
    --output classified_results.csv \
    --chunk_time_minutes 20 \
    --batch_size auto
```

### Advanced Usage

```bash
# Resume from checkpoint
python classify_refusals_m3.py \
    --input prompts_completions.csv \
    --output classified_results.csv \
    --resume_from_checkpoint \
    --continuous

# Custom batch size and disable MLX
python classify_refusals_m3.py \
    --input prompts_completions.csv \
    --output classified_results.csv \
    --batch_size 128 \
    --disable_mlx \
    --chunk_time_minutes 30
```

## Input Format

Your CSV file must contain these columns:
- `prompt`: The input prompt text
- `generation`: The LLM's generated response

```csv
prompt,generation
"What is the capital of France?","The capital of France is Paris."
"How do I hack a computer?","I cannot provide instructions for illegal activities."
```

## Output Format

The output CSV includes:
- `prompt`: Original prompt
- `generation`: Original generation
- `is_refusal`: Boolean indicating if response is a refusal
- `confidence_score`: Confidence score (0-1) for the refusal classification

```csv
prompt,generation,is_refusal,confidence_score
"What is the capital of France?","The capital of France is Paris.",false,0.05
"How do I hack a computer?","I cannot provide instructions for illegal activities.",true,0.95
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input CSV file path | Required |
| `--output` | Output CSV file path | Required |
| `--chunk_time_minutes` | Processing time per chunk | 20 |
| `--batch_size` | Batch size or 'auto' | auto |
| `--resume_from_checkpoint` | Resume from existing checkpoint | false |
| `--continuous` | Run without pausing between chunks | false |
| `--model_name` | HuggingFace model to use | protectai/distilroberta-base-rejection-v1 |
| `--disable_mlx` | Disable MLX optimization | false |

## Performance Optimization

### Device Selection Priority
1. **MLX** (Apple Silicon native) - Best performance on M3
2. **MPS** (Metal Performance Shaders) - Good fallback
3. **CPU** - Last resort

### Automatic Batch Size Optimization
The script benchmarks different batch sizes (16, 32, 64, 128, 256) and selects the optimal one for your hardware.

### Memory Management
- Automatic memory cleanup between chunks
- Gradient checkpointing for memory efficiency
- Smart cache management for M3's unified memory

## Checkpoint System

The script automatically saves checkpoints:
- **Interval**: Every 1000 classifications
- **Location**: `{output_file}.checkpoint.json`
- **Resume**: Automatically detects and resumes from last checkpoint
- **Cleanup**: Checkpoint files are automatically removed on completion

## Example Session

```bash
$ python classify_refusals_m3.py --input data.csv --output results.csv

🔧 Initializing classifier on device: mlx
✅ MLX available - using Apple Silicon optimization
📥 Loading tokenizer and model...
✅ Model loaded successfully!
🔍 Finding optimal batch size...
Batch size 64: 847 items/sec
✅ Optimal batch size: 64 (847 items/sec)

📊 Starting classification with 80,000 remaining rows
Batch size: 64
Chunk duration: 20 minutes
Device: mlx

--- Chunk 1 ---
Chunk 1 • 923/sec • 4.2GB RAM ████████████████████ 100% 10,000/10,000 • 0:10:52 • 0:00:00

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric             ┃ Value                    ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Processed in chunk │ 10,000                   │
│ Total processed    │ 10,000                   │
│ Remaining          │ 70,000                   │
│ Chunk duration     │ 20.0m                    │
│ Average throughput │ 923.4 items/sec         │
│ Memory usage       │ 4.2GB                    │
└────────────────────┴──────────────────────────┘

Chunk complete. 70,000 rows remaining. Continue with next chunk? (y/n/auto): y
```

## Demo Script

Run the example script to test with sample data:

```bash
python refusal_classifier_example.py
```

This creates sample data and shows you how to run the classifier.

## Troubleshooting

### MLX Installation Issues
```bash
# If MLX fails to install
pip install --upgrade pip
pip install mlx

# Or disable MLX
python classify_refusals_m3.py --disable_mlx ...
```

### Memory Issues
- Reduce batch size: `--batch_size 32`
- Ensure other memory-intensive apps are closed
- Use smaller chunk times: `--chunk_time_minutes 10`

### Performance Issues
- Enable MLX: `pip install mlx`
- Close other applications
- Check Activity Monitor for background processes

## Technical Details

### Model
- **Base Model**: `protectai/distilroberta-base-rejection-v1`
- **Type**: DistilRoBERTa for sequence classification
- **Classes**: [normal, refusal]
- **Input Length**: 512 tokens (configurable)

### M3 Optimizations
- **MLX**: Apple's ML framework for native Silicon performance
- **MPS**: Metal Performance Shaders for GPU acceleration
- **Unified Memory**: Efficient use of M3's shared memory architecture
- **Batch Optimization**: Dynamic sizing based on hardware capabilities

### File Structure
```
├── classify_refusals_m3.py      # Main classifier script
├── refusal_classifier_example.py # Demo/example script  
├── requirements.txt              # Dependencies
└── README_refusal_classifier.md  # This documentation
```

## License

This script uses the ProtectAI rejection classifier model. Please refer to their licensing terms for commercial usage.