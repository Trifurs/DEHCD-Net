"""Campaign dashboard and explicitly provisional wall-clock ETA."""
from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

from utils.campaign_layout import run_path, result_path, summary_path
from utils.progress import read_status, write_status, duration, training_health_text


class CampaignProgress:
    def __init__(self, plan, output):
        self.plan, self.output = plan, Path(output)
        self.started = time.monotonic()
        self.last_plain = self.last_write = 0.0
        self.durations, self.completed = {}, set()
        self.health_snapshots = {}
        manifest = read_status(self.output / 'reuse_manifest.json')
        self.reuse_tasks = {row['id']: row for row in manifest.get('tasks', [])}
        self.jobs = {job["id"]: job for job in plan["jobs"]}
        for job in plan["jobs"]:
            summary = read_status(run_path(output, job, plan) / "training_summary.json")
            run = run_path(output, job, plan)
            task = self.reuse_tasks.get(job['id'])
            if summary and summary.get('seconds'):
                self.durations[job['id']] = summary['seconds']
            if task is not None:
                if task['state'] == 'reuse_complete':
                    self.completed.add(job['id'])
            elif (summary and result_path(output, job, plan).exists()
                    and not (run / "failure.json").exists() and not (run / "test/failure.json").exists()):
                self.completed.add(job["id"])
        self.bars = []
        if sys.stderr.isatty():
            from tqdm import tqdm
            self.bars = [tqdm(total=len(self.jobs), desc="Overall", position=0, dynamic_ncols=True,
                              bar_format="{desc}: {percentage:5.1f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]"),
                         tqdm(total=1, desc="Epoch", position=1, dynamic_ncols=True,
                              bar_format="{desc}: {percentage:5.1f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]"),
                         tqdm(total=1, desc="Batch", position=2, dynamic_ncols=True,
                              bar_format="{desc}: {percentage:5.1f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]"),
                         tqdm(total=1, desc="Validation", position=3, dynamic_ncols=True,
                              bar_format="{desc}"),
                         tqdm(total=1, desc="Protection", position=4, dynamic_ncols=True,
                              bar_format="{desc}")]
        self.write_index()

    def work(self, job):
        cfg = job["config"]
        splits = self.plan["data"][job["data_key"]]["splits"]
        pixels = int(cfg["dataset"].get("patch_size", 256)) ** 2
        return max(1, int(cfg["training"]["epochs"]) *
                   (splits["train"]["samples"] + 0.4 * splits["val"]["samples"]) * pixels)

    def model_key(self, job):
        model = job["config"].get("model", {})
        return model.get("compare_model") or (model.get("name"), model.get("variant", model.get("size", model.get("backbone"))))

    def remaining(self, current, state):
        observations = [(self.jobs[key], seconds / self.work(self.jobs[key])) for key, seconds in self.durations.items()]
        if current is not None and state.get("stage") == "train" and state.get("estimated_epoch_seconds"):
            full_time = state["estimated_epoch_seconds"] * current["config"]["training"]["epochs"]
            observations.append((current, full_time / self.work(current)))
        if not observations:
            return None
        total = 0.0
        for job in self.plan["jobs"]:
            if job["id"] in self.completed: continue
            exact = [rate for known, rate in observations if known["experiment"] == job["experiment"]]
            family = [rate for known, rate in observations if self.model_key(known) == self.model_key(job)]
            dataset = [rate for known, rate in observations if known["dataset"] == job["dataset"]]
            rate = statistics.median(exact or family or dataset or [rate for _, rate in observations])
            task = self.reuse_tasks.get(job['id'], {})
            initial_fraction = min(1.0, task.get('evidence', {}).get('completed_epoch', 0)
                                   / max(int(job['config']['training']['epochs']), 1))
            if task.get('state') == 'reevaluate_only':
                initial_fraction = 1.0
            if current and job["id"] == current["id"]:
                if state.get("stage") == "test":
                    current_eta = state.get("eta_seconds")
                    if current_eta is not None:
                        total += current_eta
                        continue
                    fraction = 1.0
                else:
                    fraction = max(initial_fraction, state.get("fraction", 0.0))
            else:
                fraction = initial_fraction
            total += max(0.0, 1.0 - fraction) * self.work(job) * rate
            splits = self.plan["data"][job["data_key"]]["splits"]
            total += 0.4 * splits["test"]["samples"] * int(job["config"]["dataset"].get("patch_size", 256)) ** 2 * rate
        return total

    def refresh(self, job=None, state=None, status="running", force=False):
        state = dict(state or {})
        if job is not None:
            fields = ('best_metric_name', 'best_metric', 'best_epoch', 'epochs_since_best', 'early_stopping', 'protection')
            if 'best_metric_name' in state:
                self.health_snapshots[job['id']] = {key: state[key] for key in fields if key in state}
            elif state.get('stage') == 'test':
                if job['id'] not in self.health_snapshots:
                    summary = read_status(run_path(self.output, job, self.plan) / 'training_summary.json')
                    self.health_snapshots[job['id']] = {key: summary[key] for key in fields if key in summary}
                    if summary.get('best_epoch'):
                        self.health_snapshots[job['id']]['epochs_since_best'] = max(0, summary.get('completed_epoch', 0) - summary['best_epoch'])
                state.update(self.health_snapshots[job['id']])
        now = time.monotonic()
        if not force and now - self.last_write < 1.0: return
        self.last_write = now
        eta = self.remaining(job, state)
        payload = {"status": status, "completed_jobs": len(self.completed), "total_jobs": len(self.jobs),
                   "percent": 100 * len(self.completed) / len(self.jobs), "session_elapsed_seconds": now - self.started,
                   "estimated_remaining_seconds": eta, "eta_note": "Provisional; unseen models use measured family/dataset rates. Audited completed epochs are excluded; evaluation-only tasks include test time only.",
                   "measured_experiments": len({self.jobs[key]["experiment"] for key in self.durations}),
                   "current_job": job["id"] if job else None, "current": state, "updated_at": time.time()}
        if status == "complete": payload["estimated_remaining_seconds"] = 0.0
        write_status(self.output / "progress.json", payload)
        phase = state.get("phase", "preparing")
        lines = [f"Overall {len(self.completed)}/{len(self.jobs)} | ETA ~{duration(payload['estimated_remaining_seconds'])} (estimate)",
                 f"Current: {job['id'] if job else status}",
                 f"Epoch {state.get('epoch', 0)}/{state.get('epochs', 0)} | {phase} {state.get('phase_done', 0)}/{state.get('phase_total', 0)} | run ETA ~{duration(state.get('eta_seconds'))}",
                 training_health_text(state),
                 f"Output: {self.output}"]
        (self.output / "progress.txt").write_text("\n".join(lines) + "\n")
        if self.bars:
            overall, epoch, batch, health, protection = self.bars
            overall.n = len(self.completed)
            overall.set_postfix_str(f"ETA ~{duration(eta)} (estimate)", refresh=False)
            epoch.total = max(state.get("epochs", 1), 1)
            epoch.n = state.get("completed_epoch", 0)
            label = job["id"] if job else "Epoch"
            epoch.set_description_str(label if len(label) <= 46 else label[:43] + "...", refresh=False)
            epoch.set_postfix_str(f"run ETA ~{duration(state.get('eta_seconds'))}", refresh=False)
            batch.total = max(state.get("phase_total", 1), 1)
            batch.n = state.get("phase_done", 0)
            batch.set_description_str(phase.capitalize(), refresh=False)
            loss_text = f"loss={state['loss']:.4f} " if state.get("loss") is not None else ""
            batch.set_postfix_str(f"{loss_text}ETA ~{duration(state.get('phase_eta_seconds'))}", refresh=False)
            best = state.get('best_metric')
            metric = state.get('best_metric_name', 'foreground_miou')
            best_value = 'pending' if best is None else f'{best:.6f}'
            since = state.get('epochs_since_best')
            early = state.get('early_stopping', {})
            stop = (f"{early.get('stale_validation_checks', 0)}/{early['patience']} checks" if early.get('enabled') else 'off')
            health.set_description_str(f"Best val {metric}={best_value} @{state.get('best_epoch', 0)} | since={since if since is not None else '-'} ep | early stop={stop}", refresh=False)
            guards = training_health_text(state).split(' | ')[3:]
            protection.set_description_str('Protection: ' + (' | '.join(guards) or 'pending'), refresh=False)
            for bar in self.bars: bar.refresh()
        elif force or now - self.last_plain >= 30:
            print(" | ".join(lines[:4]), flush=True)
            self.last_plain = now

    def finish_job(self, job):
        self.completed.add(job["id"])
        summary = read_status(run_path(self.output, job, self.plan) / "training_summary.json")
        self.durations[job["id"]] = summary["seconds"]
        self.write_index()
        self.refresh(force=True)

    def write_index(self):
        path = summary_path(self.output, self.plan, "jobs.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            fields = ["dataset", "suite", "experiment", "seed", "status", "run_directory", "test_result"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for job in self.plan["jobs"]:
                run = run_path(self.output, job, self.plan)
                state = "complete" if job["id"] in self.completed else ("failed" if (run / "failure.json").exists() or (run / "test/failure.json").exists() else
                         ("trained" if (run / "training_summary.json").exists() else "pending"))
                writer.writerow({"dataset": job["dataset"], "suite": job["config"]["experiment"]["suite"],
                    "experiment": job["experiment"], "seed": job["seed"], "status": state,
                    "run_directory": str(run.relative_to(self.output)),
                    "test_result": str(result_path(self.output, job, self.plan).relative_to(self.output))})

    def close(self):
        for bar in reversed(self.bars): bar.close()
