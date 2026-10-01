"""Single source of truth for current and historical campaign paths."""
from pathlib import Path


def run_path(output, job, plan):
    output = Path(output)
    if plan.get("layout_version", 1) == 1:
        return output / "train" / job["id"]
    # Catalog ids are identifiers, never paths.
    parts = [job["dataset"], job["config"]["experiment"]["suite"], job["experiment"], f"seed_{job['seed']}"]
    if any(not part or Path(part).name != part or part in {".", ".."} for part in parts):
        raise ValueError("Unsafe experiment identifier in output path")
    return output / "runs" / Path(*parts)


def result_path(output, job, plan):
    if plan.get("layout_version", 1) == 1:
        return Path(output) / "test" / f"{job['id']}.json"
    return run_path(output, job, plan) / "test/result.json"


def summary_path(output, plan, name):
    return Path(output) / ("summary" if plan.get("layout_version", 1) >= 2 else "") / name
