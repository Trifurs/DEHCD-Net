"""Small, atomic progress snapshots; no GPU operations or training decisions."""
from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from pathlib import Path


def write_status(path, value):
    """Atomic but not fsync'd: status is disposable; checkpoints are not."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def read_status(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def duration(seconds):
    if seconds is None:
        return "calibrating"
    seconds = max(int(seconds), 0)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f"{days}d " if days else "") + f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class RunProgress:
    def __init__(self, path=None, stage="train", epochs=1, completed=0):
        self.path = Path(os.environ.get("DEHCD_PROGRESS_FILE", str(path))) if path or os.environ.get("DEHCD_PROGRESS_FILE") else None
        self.started = time.monotonic()
        previous = read_status(self.path) if self.path else {}
        self.durations = previous.get("epoch_durations", [])[-20:] if stage == "train" else []
        self.previous_seconds = previous.get("elapsed_seconds", 0) if completed else 0
        self.state = {"stage": stage, "status": "running", "epoch": completed,
                      "epochs": epochs, "completed_epoch": completed, "phase": "initializing",
                      "phase_done": 0, "phase_total": 0, "eta_seconds": None}
        self.last_write = 0.0
        self.epoch_started = self.phase_started = self.started
        self.train_batches = self.val_batches = 0
        self.write(force=True)

    def training_health(self, epoch, best_name, best_metric, best_epoch, tracker, monitor, patience):
        """Observe checkpoint selection and protections without changing their state."""
        def protection(count, limit, warmup):
            enabled = limit > 0
            status = ('disabled' if not enabled else 'warmup' if epoch < warmup else
                      'triggered' if count >= limit else 'warning' if count else 'healthy')
            return {'enabled': enabled, 'count': int(count), 'patience': int(limit),
                    'warmup_epochs': int(warmup), 'status': status}
        self.state.update(
            best_metric_name=best_name, best_metric=float(best_metric) if best_epoch > 0 else None,
            best_epoch=int(best_epoch), epochs_since_best=max(0, epoch - best_epoch) if best_epoch > 0 else None,
            early_stopping={'enabled': patience > 0, 'stale_validation_checks': int(tracker.stale),
                            'patience': int(patience), 'remaining_checks': max(patience - tracker.stale, 0) if patience > 0 else None},
            protection={'collapse': protection(monitor.count, monitor.patience, monitor.warmup),
                        'foreground_stall': protection(monitor.stall_count, monitor.stall_patience, monitor.stall_warmup)})
        self.write(force=True)

    def start_epoch(self, epoch, train_batches, val_batches):
        self.epoch_started = time.monotonic()
        self.train_batches, self.val_batches = train_batches, val_batches
        self.state.update(epoch=epoch, phase="initializing", phase_done=0, phase_total=0)

    def phase(self, name, total=0):
        self.phase_started = time.monotonic()
        self.state.update(phase=name, phase_done=0, phase_total=total, phase_eta_seconds=None)
        self.write(force=True)

    def batch(self, completed, **metrics):
        self.state.update(phase_done=completed, **metrics)
        self.write()

    def epoch_done(self, epoch, **metrics):
        self.durations.append(time.monotonic() - self.epoch_started)
        self.durations = self.durations[-20:]
        self.state.update(completed_epoch=epoch, **metrics)
        self.write(force=True)

    def finish(self, status="complete"):
        self.state.update(status=status)
        self.write(force=True)

    def write(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_write < 2.0:
            return
        self.last_write = now
        state = self.state
        done, total = state["phase_done"], state["phase_total"]
        phase_seconds = now - self.phase_started
        state["phase_eta_seconds"] = phase_seconds * (total - done) / done if done and total else None
        state["elapsed_seconds"] = self.previous_seconds + now - self.started
        state["updated_at"] = time.time()
        state["epoch_durations"] = self.durations
        if state["stage"] == "test":
            state["fraction"] = done / total if total else 0.0
            state["eta_seconds"] = state["phase_eta_seconds"]
        else:
            weight = max(self.train_batches + 0.4 * self.val_batches, 1)
            fraction = done / weight if state["phase"] == "train" else (self.train_batches + 0.4 * done) / weight
            if state["phase"] == "checkpoint": fraction = 1.0
            if state["completed_epoch"] == state["epoch"]: fraction = 0.0
            fraction = max(0.0, min(1.0, fraction))
            state["fraction"] = min(1.0, (state["completed_epoch"] + fraction) / max(state["epochs"], 1))
            rate = statistics.median(self.durations) if self.durations else (
                (now - self.epoch_started) / fraction if fraction > 0 else None)
            remaining = max(state["epochs"] - state["completed_epoch"] - fraction, 0)
            state["eta_seconds"] = rate * remaining if rate is not None else None
            state["estimated_epoch_seconds"] = rate
        if state["status"] == "complete":
            state.update(fraction=1.0, eta_seconds=0.0, phase_eta_seconds=0.0)
        if self.path: write_status(self.path, state)


def training_health_text(state):
    """One display format shared by terminal bars, plain logs and progress.txt."""
    best = state.get('best_metric')
    name = state.get('best_metric_name', 'foreground_miou')
    best_text = 'pending' if best is None else f"{float(best):.6f}"
    since = state.get('epochs_since_best')
    early = state.get('early_stopping', {})
    early_text = (f"{early.get('stale_validation_checks', 0)}/{early['patience']} checks"
                  if early.get('enabled') else 'disabled (fixed budget)')
    parts = [f"Best val {name}={best_text} @epoch {state.get('best_epoch', 0)}",
             f"since best={since if since is not None else '-'} epochs", f"early stop={early_text}"]
    for name, label in (('collapse', 'collapse'), ('foreground_stall', 'foreground stall')):
        guard = state.get('protection', {}).get(name, {})
        if guard:
            counter = f" {guard.get('count', 0)}/{guard.get('patience', 0)} checks" if guard.get('enabled') else ''
            parts.append(f"{label}={guard['status']}{counter}")
    return ' | '.join(parts)


_current = None


def set_reporter(reporter):
    global _current
    _current = reporter


def get_reporter():
    global _current
    if _current is None: _current = RunProgress()
    return _current
