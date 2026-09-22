from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from fmg_fca_vision_lite import ACTIONS, KenyonLayer, SensoryHash


def target_action(technical_score, semantic_defect, evidence_confidence, has_weak_regions, has_boxes):
    semantic_risk = semantic_defect * evidence_confidence
    if semantic_risk >= 0.58:
        return "local_repair" if has_boxes else "global_repair"
    if semantic_risk >= 0.34:
        return "local_repair" if (has_boxes or has_weak_regions) else "global_repair"
    if technical_score < 0.38:
        return "local_repair" if has_weak_regions else "global_repair"
    if technical_score < 0.56 and has_weak_regions:
        return "local_repair"
    return "accept"


def prior_scores(technical_score, semantic_defect, evidence_confidence, has_weak_regions, has_boxes):
    technical_defect = 1.0 - technical_score
    return (
        0.34 + 0.72 * technical_score - 0.82 * semantic_defect,
        0.10 + 0.76 * technical_defect
        + (0.24 if has_weak_regions else -0.18)
        + 0.18 * semantic_defect
        + (0.12 if has_boxes else 0.0),
        -0.05 + 1.05 * semantic_defect * evidence_confidence
        + (0.10 if not has_boxes else -0.04),
    )


def build_patterns(seed=2401):
    sensory = SensoryHash(128)
    kc = KenyonLayer(inputs=128, kcs=256, fan_in=6, winners=16)
    rng = random.Random(seed)
    technical_values = [0.24 + i * 0.055 for i in range(14)]
    semantic_values = [0.0, 0.12, 0.25, 0.40, 0.58, 0.75, 0.92]
    confidence_values = [0.0, 0.25, 0.50, 0.75, 0.95]
    intents = [
        ("landscape", (0, 0, 0)),
        ("portrait face", (1, 0, 0)),
        ("hands fingers", (0, 1, 0)),
        ('sign "HELLO" text', (0, 0, 1)),
        ('person holding sign "HELLO"', (1, 1, 1)),
    ]
    patterns = []
    for tech in technical_values:
        for sem in semantic_values:
            for conf in confidence_values:
                for weak in (False, True):
                    for boxes in (False, True):
                        prompt, intent = intents[rng.randrange(len(intents))]
                        features = {
                            "technical_defect": 1.0 - tech,
                            "semantic_defect": sem,
                            "evidence_confidence": conf,
                            "has_weak_regions": 1.0 if weak else 0.0,
                            "intent_face": float(intent[0]),
                            "intent_hand": float(intent[1]),
                            "intent_text": float(intent[2]),
                        }
                        obs = (
                            f"prompt={prompt} technical={tech:.3f} "
                            f"semantic_defect={sem:.3f} confidence={conf:.3f}"
                        )
                        pattern = kc.activate(sensory.encode(obs, features), {})
                        priors = prior_scores(tech, sem, conf, weak, boxes)
                        target = ACTIONS.index(target_action(tech, sem, conf, weak, boxes))
                        patterns.append((pattern.active, pattern.values, priors, target))
    rng.shuffle(patterns)
    split = int(len(patterns) * 0.82)
    return patterns[:split], patterns[split:]


def accuracy(patterns, weights):
    correct = 0
    confusion = [[0, 0, 0] for _ in range(3)]
    for active, values, priors, target in patterns:
        scores = []
        for a in range(3):
            learned = sum(weights[a][k] * v for k, v in zip(active, values))
            scores.append(priors[a] + 0.18 * learned)
        chosen = max(range(3), key=lambda a: (scores[a], -a))
        confusion[target][chosen] += 1
        correct += int(chosen == target)
    return correct / len(patterns), confusion


def train(trials, seed):
    train_patterns, holdout = build_patterns(seed)
    weights = [[0.0] * 256 for _ in range(3)]
    baseline = 0.0
    lr = 0.055
    rng = random.Random(seed ^ 0xFCA2401)
    before_holdout, before_confusion = accuracy(holdout, weights)
    started = time.perf_counter()
    correct = 0
    rewards = 0.0

    for _ in range(trials):
        active, values, priors, target = train_patterns[rng.randrange(len(train_patterns))]
        learned_scores = [0.0, 0.0, 0.0]
        scores = [0.0, 0.0, 0.0]
        for a in range(3):
            w = weights[a]
            learned = sum(w[k] * v for k, v in zip(active, values))
            learned_scores[a] = learned
            scores[a] = priors[a] + 0.18 * learned
        chosen = max(range(3), key=lambda a: (scores[a], -a))
        reward = 1.0 if chosen == target else -1.0
        correct += int(chosen == target)
        rewards += reward
        rpe = reward - learned_scores[chosen] - baseline
        w = weights[chosen]
        for k, v in zip(active, values):
            w[k] = max(-1.5, min(1.5, w[k] + lr * rpe * v))
        baseline = 0.9995 * baseline + 0.0005 * reward

    after_holdout, after_confusion = accuracy(holdout, weights)
    return {
        "weights": weights,
        "baseline": baseline,
        "report": {
            "trials": trials,
            "seed": seed,
            "distinct_train_patterns": len(train_patterns),
            "distinct_holdout_patterns": len(holdout),
            "online_training_accuracy": correct / trials,
            "reward_mean": rewards / trials,
            "holdout_accuracy_before": before_holdout,
            "holdout_accuracy_after": after_holdout,
            "holdout_confusion_before": before_confusion,
            "holdout_confusion_after": after_confusion,
            "runtime_seconds": time.perf_counter() - started,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=2401)
    args = ap.parse_args()
    result = train(args.trials, args.seed)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
