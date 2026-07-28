"""
Robust training script with checkpointing and error recovery
"""

import json
import pickle
import time
from pathlib import Path
from typing import Optional

from astronomy_recognizer.ml_classifier import train_and_evaluate


def load_checkpoint(checkpoint_path: Path) -> Optional[dict]:
    """Load training checkpoint if it exists"""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return None


def save_checkpoint(checkpoint_path: Path, data: dict):
    """Save training checkpoint"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)


def robust_train_and_evaluate(
    dataset_images_root: Path | str,
    model_output_path: Path | str,
    checkpoint_path: Optional[Path] = None
) -> dict:
    """
    Robust training with checkpointing and error recovery
    """
    dataset_images_root = Path(dataset_images_root)
    model_output_path = Path(model_output_path)

    if checkpoint_path is None:
        checkpoint_path = model_output_path.parent / "training_checkpoint.json"

    # Check if we have a completed model
    if model_output_path.exists():
        print("✓ Model already exists! Loading existing results...")
        metrics_path = model_output_path.parent / "space_ml_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
        else:
            print("Model exists but no metrics found. Retraining...")

    # Check for checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        print(f"✓ Found checkpoint from {checkpoint.get('timestamp', 'unknown time')}")
        print(f"  Last stage: {checkpoint.get('stage', 'unknown')}")

        # If we have a saved model from checkpoint, use it
        checkpoint_model_path = checkpoint_path.parent / "checkpoint_model.pkl"
        if checkpoint_model_path.exists() and checkpoint.get('stage') == 'completed':
            print("✓ Loading completed model from checkpoint...")
            # Copy checkpoint model to final location
            import shutil
            shutil.copy2(checkpoint_model_path, model_output_path)
            return checkpoint['metrics']

    print("Starting fresh training with checkpointing...")

    try:
        # Stage 1: Training
        save_checkpoint(checkpoint_path, {
            'stage': 'training',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': str(dataset_images_root),
            'model_output': str(model_output_path)
        })

        print("🚀 Starting training...")
        start_time = time.time()

        metrics = train_and_evaluate(dataset_images_root, model_output_path)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f} seconds")
        # Stage 2: Save results
        save_checkpoint(checkpoint_path, {
            'stage': 'completed',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'training_time_seconds': training_time,
            'metrics': metrics
        })

        # Save checkpoint model copy
        checkpoint_model_path = checkpoint_path.parent / "checkpoint_model.pkl"
        import shutil
        shutil.copy2(model_output_path, checkpoint_model_path)

        print("✅ Training completed successfully!")
        return metrics

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

        # Save error checkpoint
        save_checkpoint(checkpoint_path, {
            'stage': 'failed',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e),
            'error_type': type(e).__name__
        })

        raise  # Re-raise the exception


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Robust training with checkpointing")
    parser.add_argument("--dataset", default="SpaceDataset/images", help="Dataset path")
    parser.add_argument("--model-output", default="artifacts/space_category_model.pkl", help="Model output path")
    parser.add_argument("--checkpoint", default="artifacts/training_checkpoint.json", help="Checkpoint file")
    parser.add_argument("--force-restart", action="store_true", help="Force restart even if checkpoint exists")

    args = parser.parse_args()

    if args.force_restart:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("🗑️  Removed old checkpoint")

    try:
        metrics = robust_train_and_evaluate(
            args.dataset,
            args.model_output,
            Path(args.checkpoint) if not args.force_restart else None
        )

        print("\n" + "="*60)
        print("TRAINING RESULTS:")
        print("="*60)
        print(f"Accuracy: {metrics['model']['accuracy']}%")
        print(f"Macro F1: {metrics['model']['macro_f1']}%")
        print(f"Weighted F1: {metrics['model']['weighted_f1']}%")
        print(f"Top-3 Accuracy: {metrics['model']['top3_accuracy']}%")
        print("="*60)

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        print("Check the checkpoint file for recovery information")
        exit(1)


if __name__ == "__main__":
    main()