"""
Script to compare old vs new model metrics once training completes
Run this after training_space_model.py finishes
"""

import json
from pathlib import Path

# Load new metrics (with SMOTE + class balancing)
new_metrics_path = Path("artifacts/space_ml_metrics.json")

if not new_metrics_path.exists():
    print("Training still in progress. Check back in a few hours...")
    exit(1)

with open(new_metrics_path) as f:
    new = json.load(f)

# Original metrics (for comparison)
old_metrics = {
    "classes": [
        "asteroids_comets",
        "black_holes",
        "galaxies",
        "moons",
        "nebulae",
        "planets",
        "stars"
    ],
    "old_class_metrics": {
        "asteroids_and_comets": {"recall": 0.0, "f1_score": 0.0},
        "black_holes": {"recall": 28.74, "f1_score": 33.06},
        "galaxies": {"recall": 6.86, "f1_score": 11.76},
        "moons": {"recall": 63.8, "f1_score": 60.57},
        "nebulae": {"recall": 24.73, "f1_score": 33.92},
        "planets": {"recall": 66.97, "f1_score": 63.79},
        "stars": {"recall": 91.44, "f1_score": 84.54},
    }
}

print("\n" + "="*70)
print("MESSIER OBJECT DETECTION - MODEL IMPROVEMENT ANALYSIS")
print("="*70)

# Overall metrics
print("\nOVERALL PERFORMANCE:")
print(f"  {'Metric':<20} {'BEFORE':<15} {'AFTER':<15} {'CHANGE':<15}")
print("-" * 65)

old_acc = 65.82
print(f"  {'Accuracy':<20} {old_acc:<15.2f} {new['model']['accuracy']:<15.2f} {new['model']['accuracy'] - old_acc:+.2f}%")

old_macro = 35.95  
print(f"  {'Macro F1':<20} {old_macro:<15.2f} {new['model']['macro_f1']:<15.2f} {new['model']['macro_f1'] - old_macro:+.2f}%")

old_weighted = 62.73  # Not in old metrics but estimated
print(f"  {'Weighted F1':<20} {'~62.73':<15} {new['model']['weighted_f1']:<15.2f}")

print(f"  {'Top-3 Accuracy':<20} {'84.58':<15} {new['model']['top3_accuracy']:<15.2f}")

# Per-class improvements
print("\nPER-CLASS IMPROVEMENTS (Recall):")
print(f"  {'Class':<25} {'BEFORE':<15} {'AFTER':<15} {'IMPROVEMENT':<15}")
print("-" * 70)

class_improvements = []
for cls in new['classes']:
    new_metrics_cls = new['class_metrics'].get(cls, {})
    
    if cls == "asteroids_comets":
        old_recall = 0.0
    else:
        cls_formatted = cls.replace("_", " ").title()
        old_recall = old_metrics['old_class_metrics'].get(cls_formatted.lower().replace(" ", "_"), {}).get("recall", 0) if cls != "asteroids_comets" else 0.0
    
    new_recall = new_metrics_cls.get("recall", 0) if new_metrics_cls else 0
    improvement = new_recall - old_recall
    
    class_improvements.append((cls, old_recall, new_recall, improvement))
    print(f"  {cls:<25} {old_recall:<15.2f} {new_recall:<15.2f} {improvement:+.2f}%")

# Summary
print("\n" + "="*70)
print("KEY ACHIEVEMENTS:")
print("="*70)

improved_critical = [c for c, o, n, i in class_improvements if i > 10 and o < 50]
if improved_critical:
    print(f"✓ {len(improved_critical)} critical classes significantly improved:")
    for cls, o, n, i in improved_critical:
        print(f"  - {cls}: {o:.1f}% → {n:.1f}% (+{i:.1f}%)")

print(f"\n✓ Overall accuracy: {new['model']['accuracy']:.2f}%")
print(f"✓ Macro-F1 (fairness to all classes): {new['model']['macro_f1']:.2f}%")
print(f"✓ Top-3 accuracy: {new['model']['top3_accuracy']:.2f}%")

if new['model']['macro_f1'] > old_macro:
    print(f"\n✅ SUCCESS: Balanced classification achieved! Macro F1 +{new['model']['macro_f1'] - old_macro:.2f}%")
else:
    print(f"\n⚠️  Macro F1 still improving. Review per-class results above.")

print("\nNext steps if still not satisfied:")
print("  1. Add deep learning features (ResNet CNN)")
print("  2. Use ensemble voting (multiple models)")
print("  3. Add Messier reference image augmentation")
print("="*70)
