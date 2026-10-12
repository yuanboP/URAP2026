"""Small CPU-only controls shared by full-data training and its readback."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def round_spans(presentations, microbatch=2, accumulation=2):
    if presentations < 1 or microbatch < 1 or accumulation < 1:
        raise ValueError('positive dataset and batch sizes are required')
    width = microbatch * accumulation
    return [(start, min(start + width, presentations)) for start in range(0, presentations, width)]


def validation_rounds(rounds, quarters=4):
    if rounds < 1 or quarters < 1:
        raise ValueError('positive round and interval counts are required')
    return sorted({math.ceil(rounds * i / quarters) for i in range(1, quarters + 1)})


def source_score(by_domain):
    required = {'DJI_task3', 'ARD100_A_val'}
    if set(by_domain) != required:
        raise ValueError('selection requires exactly the two frozen source validation domains')
    values = [float(by_domain[name]['map50_95']) for name in sorted(required)]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError('source validation scores must be finite probabilities')
    return sum(values) / len(values)


def validate_resume_identity(saved, current):
    required = ('manifest_sha256', 'plans_sha256', 'runtime_sha256', 'epochs')
    for key in required:
        if key not in saved or key not in current or saved[key] != current[key]:
            raise ValueError(f'resume identity differs: {key}')


def scheduled_lr(step, total_steps, peak_lr=.0002, warmup_steps=1000):
    if total_steps < 1 or not 0 <= step < total_steps:
        raise ValueError('step must be inside the frozen budget')
    warmup = min(warmup_steps, total_steps)
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / (total_steps - warmup - 1) if total_steps - warmup > 1 else 1.
    return peak_lr * (.1 + .45 * (1 + math.cos(math.pi * progress)))
