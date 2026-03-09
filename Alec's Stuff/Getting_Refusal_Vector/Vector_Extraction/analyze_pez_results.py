#!/usr/bin/env python3
"""
Analyze and aggregate PEZ optimization results across multiple runs.

This script processes results from multiple PEZ runs, generates comparative
visualizations, and exports discovered prompts for further analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import csv

import numpy as np


def load_run_results(run_dir: Path) -> Dict[str, Any]:
    """Load results from a single PEZ run directory"""
    results_file = run_dir / "results.json"
    config_file = run_dir / "config.json"

    if not results_file.exists():
        return None

    with open(results_file, 'r') as f:
        results = json.load(f)

    config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

    return {
        'run_name': run_dir.name,
        'run_dir': str(run_dir),
        'config': config,
        'results': results
    }


def aggregate_results(results_dir: Path, pattern: str = "*") -> List[Dict[str, Any]]:
    """Aggregate results from all runs matching pattern"""
    all_runs = []

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return []

    # Find all run directories
    run_dirs = [d for d in results_dir.glob(pattern) if d.is_dir()]

    if not run_dirs:
        print(f"No run directories found in {results_dir}")
        return []

    print(f"Found {len(run_dirs)} run directories")

    for run_dir in run_dirs:
        run_data = load_run_results(run_dir)
        if run_data is not None:
            all_runs.append(run_data)

    print(f"Successfully loaded {len(all_runs)} runs")
    return all_runs


def export_prompts_csv(all_runs: List[Dict[str, Any]], output_file: Path):
    """Export all discovered prompts to CSV"""
    print(f"\nExporting prompts to {output_file}")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'run_name',
            'objective',
            'target_activation',
            'init_mode',
            'layers',
            'best_prompt',
            'best_activation',
            'best_loss',
            'best_step',
            'num_steps',
            'learning_rate',
            'prompt_length'
        ])
        writer.writeheader()

        for run in all_runs:
            config = run['config']
            results = run['results']

            writer.writerow({
                'run_name': run['run_name'],
                'objective': config.get('objective', 'unknown'),
                'target_activation': config.get('target_activation', 'N/A'),
                'init_mode': config.get('init_mode', 'unknown'),
                'layers': config.get('layers', 'unknown'),
                'best_prompt': results.get('best_prompt', ''),
                'best_activation': results.get('best_activation', 0.0),
                'best_loss': results.get('best_loss', 0.0),
                'best_step': results.get('best_step', 0),
                'num_steps': config.get('num_steps', 0),
                'learning_rate': config.get('learning_rate', 0.0),
                'prompt_length': config.get('prompt_length', 0)
            })

    print(f"Exported {len(all_runs)} prompts")


def print_summary_statistics(all_runs: List[Dict[str, Any]]):
    """Print summary statistics across all runs"""
    print("\n" + "="*80)
    print("PEZ Results Summary")
    print("="*80)

    # Group by objective
    by_objective = {}
    for run in all_runs:
        obj = run['config'].get('objective', 'unknown')
        if obj not in by_objective:
            by_objective[obj] = []
        by_objective[obj].append(run)

    print(f"\nTotal runs: {len(all_runs)}")
    print(f"Objectives: {', '.join(by_objective.keys())}")

    # Statistics per objective
    for objective, runs in by_objective.items():
        print(f"\n--- Objective: {objective.upper()} ({len(runs)} runs) ---")

        activations = [r['results']['best_activation'] for r in runs]
        losses = [r['results']['best_loss'] for r in runs]

        print(f"Activation range: [{min(activations):.4f}, {max(activations):.4f}]")
        print(f"Activation mean:  {np.mean(activations):.4f} ± {np.std(activations):.4f}")
        print(f"Loss range:       [{min(losses):.4f}, {max(losses):.4f}]")
        print(f"Loss mean:        {np.mean(losses):.4f} ± {np.std(losses):.4f}")

        # Show top 3 prompts by activation
        sorted_runs = sorted(runs, key=lambda r: r['results']['best_activation'], reverse=True)

        print(f"\nTop 3 prompts (highest activation):")
        for i, run in enumerate(sorted_runs[:3], 1):
            act = run['results']['best_activation']
            prompt = run['results']['best_prompt']
            print(f"  {i}. [{act:+.4f}] {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        print(f"\nTop 3 prompts (lowest activation):")
        sorted_runs_low = sorted(runs, key=lambda r: r['results']['best_activation'])
        for i, run in enumerate(sorted_runs_low[:3], 1):
            act = run['results']['best_activation']
            prompt = run['results']['best_prompt']
            print(f"  {i}. [{act:+.4f}] {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    print("\n" + "="*80)


def plot_results(all_runs: List[Dict[str, Any]], output_dir: Path):
    """Create comparative plots across runs"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("\nMatplotlib not available, skipping plots")
        return

    print(f"\nGenerating plots...")

    # Group by objective
    by_objective = {}
    for run in all_runs:
        obj = run['config'].get('objective', 'unknown')
        if obj not in by_objective:
            by_objective[obj] = []
        by_objective[obj].append(run)

    # 1. Activation distribution across objectives
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PEZ Results Analysis', fontsize=16, fontweight='bold')

    # Subplot 1: Activation distribution by objective
    ax = axes[0, 0]
    data_by_obj = []
    labels = []
    for obj, runs in sorted(by_objective.items()):
        activations = [r['results']['best_activation'] for r in runs]
        data_by_obj.append(activations)
        labels.append(f"{obj}\n(n={len(runs)})")

    if data_by_obj:
        bp = ax.boxplot(data_by_obj, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero')
        ax.set_ylabel('Refusal Activation')
        ax.set_title('Activation Distribution by Objective')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Subplot 2: Convergence curves (sample runs)
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(by_objective)))

    for (obj, runs), color in zip(sorted(by_objective.items()), colors):
        # Plot up to 3 representative runs per objective
        for run in runs[:3]:
            activations = run['results']['activations']
            if activations:
                steps = np.arange(len(activations))
                ax.plot(steps, activations, color=color, alpha=0.3, linewidth=0.5)

        # Plot mean trajectory
        max_len = max(len(r['results']['activations']) for r in runs)
        mean_traj = np.zeros(max_len)
        counts = np.zeros(max_len)

        for run in runs:
            acts = run['results']['activations']
            mean_traj[:len(acts)] += acts
            counts[:len(acts)] += 1

        mean_traj = mean_traj / np.maximum(counts, 1)
        ax.plot(mean_traj, color=color, linewidth=2, label=obj)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Refusal Activation')
    ax.set_title('Optimization Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 3: Loss vs Activation scatter
    ax = axes[1, 0]
    for (obj, runs), color in zip(sorted(by_objective.items()), colors):
        activations = [r['results']['best_activation'] for r in runs]
        losses = [r['results']['best_loss'] for r in runs]
        ax.scatter(activations, losses, color=color, label=obj, alpha=0.6, s=50)

    ax.set_xlabel('Best Activation')
    ax.set_ylabel('Best Loss')
    ax.set_title('Loss vs Activation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 4: Success rate (reaching target)
    ax = axes[1, 1]
    success_rates = []
    obj_labels = []

    for obj, runs in sorted(by_objective.items()):
        if obj == "target":
            # Calculate how many runs got close to target
            successes = []
            for run in runs:
                target = run['config'].get('target_activation', 0.0)
                actual = run['results']['best_activation']
                error = abs(actual - target)
                successes.append(error < 0.1)  # Within 0.1 of target

            success_rate = 100 * np.mean(successes)
            success_rates.append(success_rate)
            obj_labels.append(f"{obj}\n(±0.1)")
        elif obj == "maximize":
            # Activation > 0.5
            successes = [r['results']['best_activation'] > 0.5 for r in runs]
            success_rate = 100 * np.mean(successes)
            success_rates.append(success_rate)
            obj_labels.append(f"{obj}\n(>0.5)")
        elif obj == "minimize":
            # Activation < -0.5
            successes = [r['results']['best_activation'] < -0.5 for r in runs]
            success_rate = 100 * np.mean(successes)
            success_rates.append(success_rate)
            obj_labels.append(f"{obj}\n(<-0.5)")
        elif obj == "boundary":
            # Activation near 0
            successes = [abs(r['results']['best_activation']) < 0.15 for r in runs]
            success_rate = 100 * np.mean(successes)
            success_rates.append(success_rate)
            obj_labels.append(f"{obj}\n(|x|<0.15)")

    if success_rates:
        bars = ax.bar(obj_labels, success_rates, color=colors[:len(obj_labels)])
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Optimization Success Rate')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')

    plt.tight_layout()

    plot_path = output_dir / "pez_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved analysis plot to {plot_path}")

    # 2. Per-objective detailed plots
    for obj, runs in by_objective.items():
        if len(runs) < 2:
            continue

        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig)

        fig.suptitle(f'Objective: {obj.upper()} ({len(runs)} runs)', fontsize=14, fontweight='bold')

        # All trajectories
        ax1 = fig.add_subplot(gs[0, :])
        for i, run in enumerate(runs):
            acts = run['results']['activations']
            ax1.plot(acts, alpha=0.5, linewidth=1, label=f"Run {i+1}")

        if obj == "target" and runs:
            target = runs[0]['config'].get('target_activation', 0.0)
            ax1.axhline(y=target, color='r', linestyle='--', linewidth=2, label='Target')

        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Refusal Activation')
        ax1.set_title('All Optimization Trajectories')
        ax1.grid(True, alpha=0.3)
        if len(runs) <= 10:
            ax1.legend(loc='best', ncol=2)

        # Final activation histogram
        ax2 = fig.add_subplot(gs[1, 0])
        final_acts = [r['results']['best_activation'] for r in runs]
        ax2.hist(final_acts, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(final_acts), color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(final_acts):.3f}')
        if obj == "target" and runs:
            target = runs[0]['config'].get('target_activation', 0.0)
            ax2.axvline(x=target, color='g', linestyle='--', linewidth=2, label=f'Target: {target:.3f}')
        ax2.set_xlabel('Best Activation')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Final Activations')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Loss curves
        ax3 = fig.add_subplot(gs[1, 1])
        for i, run in enumerate(runs):
            losses = run['results']['losses']
            ax3.plot(losses, alpha=0.5, linewidth=1)

        ax3.set_xlabel('Step')
        ax3.set_ylabel('Loss')
        ax3.set_title('Loss Curves')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = output_dir / f"pez_analysis_{obj}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved {obj} analysis plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PEZ optimization results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--results_dir", required=True,
                       help="Directory containing PEZ run results")
    parser.add_argument("--pattern", default="*",
                       help="Pattern to match run directories")
    parser.add_argument("--output_dir", default=None,
                       help="Output directory for analysis (default: same as results_dir)")
    parser.add_argument("--skip_plots", action="store_true",
                       help="Skip generating plots")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    all_runs = aggregate_results(results_dir, args.pattern)

    if not all_runs:
        print("No results found to analyze")
        return 1

    # Print summary statistics
    print_summary_statistics(all_runs)

    # Export to CSV
    csv_path = output_dir / "pez_prompts.csv"
    export_prompts_csv(all_runs, csv_path)

    # Generate plots
    if not args.skip_plots:
        plot_results(all_runs, output_dir)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
