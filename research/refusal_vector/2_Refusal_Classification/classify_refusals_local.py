#!/usr/bin/env python3
"""
Optimized refusal classifier for Apple Silicon laptops
Classifies 80k LLM outputs using ProtectAI rejection classifier with resumable 20-minute chunks
"""

import argparse
import csv
import gc
import json
import psutil
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Union
import pandas as pd
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()

class LocalRefusalClassifier:
    """Optimized refusal classifier for Apple Silicon laptops"""

    def __init__(
        self,
        model_name: str = "ProtectAI/distilroberta-base-rejection-v1",
        batch_size: Union[int, str] = "auto",
        use_mlx: bool = True,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_mlx = use_mlx
        self.max_length = max_length
        self.device = self._get_optimal_device()

        console.print(f"[blue]Initializing classifier on device: {self.device}[/blue]")
        
        # Load model and tokenizer directly for efficiency
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            low_cpu_mem_usage=True,
        )
        
        if self.device == "mps":
            self.model = self.model.to("mps")
        
        self.model.eval()
        
        # Check labels - ProtectAI model: {0: 'NORMAL', 1: 'REJECTION'}
        id2label = self.model.config.id2label
        console.print(f"[green]Model labels: {id2label}[/green]")

        self.optimal_batch_size = 64
        if self.batch_size == "auto":
            self._find_optimal_batch_size()
    
    def _get_optimal_device(self) -> str:
        """Determine optimal device for Apple Silicon laptops"""
        if self.use_mlx:
            try:
                import mlx.core as mx
                console.print("[green]MLX available - using Apple Silicon optimization[/green]")
                return "mlx"
            except ImportError:
                console.print("[yellow]MLX not available, falling back to MPS[/yellow]")
        
        if torch.backends.mps.is_available():
            return "mps"
        else:
            console.print("[yellow]MPS not available, using CPU[/yellow]")
            return "cpu"
    
    
    
    
    
    def _find_optimal_batch_size(self) -> int:
        """Find optimal batch size through benchmarking"""
        console.print("[blue]Finding optimal batch size...[/blue]")
        
        test_texts = ["This is a test response for optimization"] * 1000
        batch_sizes = [16, 32, 64, 128, 256]
        best_batch_size = 64
        best_throughput = 0
        
        for batch_size in batch_sizes:
            try:
                start_time = time.time()
                
                # Test this batch size
                for i in range(0, min(200, len(test_texts)), batch_size):
                    batch = test_texts[i:i + batch_size]
                    _ = self._classify_batch(batch)
                
                elapsed = time.time() - start_time
                throughput = 200 / elapsed if elapsed > 0 else 0
                
                console.print(f"Batch size {batch_size}: {throughput:.1f} items/sec")
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
                    
                # Clear memory
                if self.device == "mps":
                    torch.mps.empty_cache()
                gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    console.print(f"[red]Batch size {batch_size} too large (OOM)[/red]")
                    break
                else:
                    raise e
        
        self.optimal_batch_size = best_batch_size
        console.print(f"[green]Optimal batch size: {best_batch_size} ({best_throughput:.1f} items/sec)[/green]")
        return best_batch_size
    
    def _classify_batch(self, texts: List[str]) -> List[Dict[str, Union[bool, float]]]:
        """Classify a batch of response texts for refusal detection using direct model inference"""
        if not texts:
            return []

        # Tokenize the batch
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device if using MPS
        if self.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Move back to CPU for processing
            if self.device == "mps":
                probabilities = probabilities.cpu()
            
            # Get predictions - index 1 is REJECTION, index 0 is NORMAL
            predicted_classes = torch.argmax(probabilities, dim=-1)
            confidence_scores = torch.max(probabilities, dim=-1)[0]
        
        results = []
        for i in range(len(texts)):
            is_refusal = bool(predicted_classes[i].item() == 1)  # 1 = REJECTION
            confidence = float(confidence_scores[i].item())
            
            results.append({
                "is_refusal": is_refusal,
                "confidence_score": confidence
            })

        return results

    
    def classify_texts(self, texts: List[str], progress_callback=None) -> List[Dict[str, Union[bool, float]]]:
        """Classify a list of texts with progress tracking"""
        results = []
        batch_size = self.optimal_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._classify_batch(batch)
            results.extend(batch_results)
            
            if progress_callback:
                progress_callback(len(batch))
            
            # Memory management
            if i % (batch_size * 10) == 0:  # Every 10 batches
                if self.device == "mps":
                    torch.mps.empty_cache()
                gc.collect()
        
        return results


class CheckpointManager:
    """Manages checkpoints and resumable processing"""
    
    def __init__(self, output_file: str, checkpoint_interval: int = 1000):
        self.output_file = Path(output_file)
        self.checkpoint_file = self.output_file.with_suffix('.checkpoint.json')
        self.checkpoint_interval = checkpoint_interval
        self.processed_count = 0
    
    def load_checkpoint(self) -> Dict:
        """Load existing checkpoint data"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"processed_rows": 0, "last_timestamp": None}
    
    def save_checkpoint(self, processed_rows: int, additional_data: Dict = None):
        """Save checkpoint data"""
        checkpoint_data = {
            "processed_rows": processed_rows,
            "last_timestamp": datetime.now().isoformat(),
            "output_file": str(self.output_file),
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def should_save_checkpoint(self, current_count: int) -> bool:
        """Check if it's time to save a checkpoint"""
        return current_count % self.checkpoint_interval == 0
    
    def get_resume_point(self, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Get dataframe slice to resume from"""
        checkpoint = self.load_checkpoint()
        start_row = checkpoint.get("processed_rows", 0)
        
        if start_row > 0:
            console.print(f"[yellow]Resuming from row {start_row}[/yellow]")
            return input_df.iloc[start_row:].copy(), start_row
        
        return input_df.copy(), 0


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_gb": memory_info.rss / (1024**3),  # Resident Set Size
        "vms_gb": memory_info.vms / (1024**3),  # Virtual Memory Size
        "percent": process.memory_percent(),
    }


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def main():
    parser = argparse.ArgumentParser(
        description="Classify LLM outputs for refusal using ProtectAI model (Apple Silicon optimized)"
    )
    parser.add_argument("--input", required=True, help="Input CSV file with 'prompt' and 'response' columns")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--chunk_time_minutes", type=int, default=20, help="Processing time per chunk (minutes)")
    parser.add_argument("--batch_size", default="auto", help="Batch size (or 'auto' for optimization)")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument("--continuous", action="store_true", help="Run continuously without pausing between chunks")
    parser.add_argument("--model_name", default="protectai/distilroberta-base-rejection-v1", help="Model to use")
    parser.add_argument("--disable_mlx", action="store_true", help="Disable MLX optimization")
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file {input_path} not found[/red]")
        return 1
    
    # Load input data
    console.print(f"[blue]Loading data from {input_path}...[/blue]")
    try:
        df = pd.read_csv(input_path)
        if 'prompt' not in df.columns or 'response' not in df.columns:
            console.print("[red]Error: Input CSV must have 'prompt' and 'response' columns[/red]")
            return 1
    except Exception as e:
        console.print(f"[red]Error loading CSV: {e}[/red]")
        return 1
    
    console.print(f"[green]Loaded {len(df)} rows[/green]")
    
    # Initialize classifier
    try:
        batch_size = args.batch_size if args.batch_size != "auto" else "auto"
        if batch_size != "auto":
            batch_size = int(batch_size)
        
        classifier = LocalRefusalClassifier(
            model_name=args.model_name,
            batch_size=batch_size,
            use_mlx=not args.disable_mlx,
        )
    except Exception as e:
        console.print(f"[red]Error initializing classifier: {e}[/red]")
        return 1
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(args.output)
    
    # Resume from checkpoint if requested
    if args.resume_from_checkpoint:
        df_to_process, start_row = checkpoint_manager.get_resume_point(df)
    else:
        df_to_process = df.copy()
        start_row = 0
    
    if len(df_to_process) == 0:
        console.print("[green]All rows already processed![/green]")
        return 0
    
    # Prepare output file
    output_path = Path(args.output)
    output_exists = output_path.exists()
    
    if not output_exists or start_row == 0:
        # Create new output file with headers
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['prompt', 'response', 'is_refusal', 'confidence_score'])
    
    # Processing setup
    chunk_duration = timedelta(minutes=args.chunk_time_minutes)
    total_processed = start_row
    chunk_number = 1
    
    console.print(f"\n[bold blue]Starting classification with {len(df_to_process)} remaining rows[/bold blue]")
    console.print(f"Batch size: {classifier.optimal_batch_size}")
    console.print(f"Chunk duration: {args.chunk_time_minutes} minutes")
    console.print(f"Device: {classifier.device}")
    
    # Main processing loop
    while len(df_to_process) > 0:
        console.print(f"\n[bold green]--- Chunk {chunk_number} ---[/bold green]")
        
        chunk_start_time = time.time()
        chunk_end_time = chunk_start_time + chunk_duration.total_seconds()
        
        processed_in_chunk = 0
        chunk_results = []
        
        # Progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task(
                f"Chunk {chunk_number}",
                total=len(df_to_process)
            )
            
            # Process rows in this chunk
            row_index = 0
            while row_index < len(df_to_process) and time.time() < chunk_end_time:
                # Determine batch size for remaining time
                remaining_time = chunk_end_time - time.time()
                if remaining_time < 10:  # Less than 10 seconds left
                    break
                
                # Get next batch
                batch_end = min(
                    row_index + classifier.optimal_batch_size,
                    len(df_to_process)
                )
                batch_df = df_to_process.iloc[row_index:batch_end]
                
                # Prepare texts for classification - ONLY the response text
                texts = [
                    row['response']
                    for _, row in batch_df.iterrows()
                ]
                
                # Classify batch
                batch_start = time.time()
                try:
                    results = classifier.classify_texts(texts)
                    
                    # Store results
                    for i, (_, row) in enumerate(batch_df.iterrows()):
                        result = results[i]
                        chunk_results.append({
                            'prompt': row['prompt'],
                            'response': row['response'],
                            'is_refusal': result['is_refusal'],
                            'confidence_score': result['confidence_score']
                        })
                    
                    # Update counters
                    processed_in_chunk += len(batch_df)
                    total_processed += len(batch_df)
                    row_index = batch_end
                    
                    # Update progress
                    progress.update(task, advance=len(batch_df))
                    
                    # Calculate and display stats
                    batch_time = time.time() - batch_start
                    if batch_time > 0:
                        throughput = len(batch_df) / batch_time
                        memory = get_memory_usage()
                        
                        progress.update(
                            task,
                            description=f"Chunk {chunk_number} • {throughput:.0f}/sec • {memory['rss_gb']:.1f}GB RAM"
                        )
                    
                    # Save checkpoint periodically
                    if checkpoint_manager.should_save_checkpoint(total_processed):
                        checkpoint_manager.save_checkpoint(
                            total_processed,
                            {"chunk_number": chunk_number, "throughput": throughput}
                        )
                
                except Exception as e:
                    console.print(f"[red]Error processing batch: {e}[/red]")
                    break
        
        # Save chunk results
        if chunk_results:
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['prompt', 'response', 'is_refusal', 'confidence_score'])
                writer.writerows(chunk_results)
        
        # Update dataframe for next chunk
        df_to_process = df_to_process.iloc[row_index:].copy()
        
        # Chunk summary
        chunk_duration_actual = time.time() - chunk_start_time
        avg_throughput = processed_in_chunk / chunk_duration_actual if chunk_duration_actual > 0 else 0
        
        table = Table(title=f"Chunk {chunk_number} Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Processed in chunk", f"{processed_in_chunk:,}")
        table.add_row("Total processed", f"{total_processed:,}")
        table.add_row("Remaining", f"{len(df_to_process):,}")
        table.add_row("Chunk duration", format_duration(chunk_duration_actual))
        table.add_row("Average throughput", f"{avg_throughput:.1f} items/sec")
        table.add_row("Memory usage", f"{get_memory_usage()['rss_gb']:.1f}GB")
        
        console.print(table)
        
        # Save final checkpoint for this chunk
        checkpoint_manager.save_checkpoint(
            total_processed,
            {
                "chunk_number": chunk_number,
                "chunk_throughput": avg_throughput,
                "remaining_rows": len(df_to_process)
            }
        )
        
        chunk_number += 1
        
        # Check if we should continue
        if len(df_to_process) == 0:
            break
        
        if not args.continuous:
            user_input = console.input(
                f"\n[yellow]Chunk complete. {len(df_to_process):,} rows remaining. "
                "Continue with next chunk? (y/n/auto): [/yellow]"
            )
            if user_input.lower() in ['n', 'no']:
                break
            elif user_input.lower() in ['auto']:
                args.continuous = True
        
        # Memory cleanup between chunks
        gc.collect()
        if classifier.device == "mps":
            torch.mps.empty_cache()
    
    # Final summary
    console.print(f"\n[bold green]🎉 Classification Complete![/bold green]")
    console.print(f"Total processed: {total_processed:,} rows")
    console.print(f"Output saved to: {output_path}")
    
    # Clean up checkpoint file
    if checkpoint_manager.checkpoint_file.exists():
        checkpoint_manager.checkpoint_file.unlink()
        console.print("[dim]Checkpoint file cleaned up[/dim]")
    
    return 0


if __name__ == "__main__":
    exit(main())