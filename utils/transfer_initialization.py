"""Weight initialization is distinct from continuation of a training run."""
from pathlib import Path

from utils.checkpoint import load_model_state
from utils.protocol import file_digest, verify_checkpoint_config


def initialize_weights(model, config):
    import json
    import torch

    training = config["training"]
    path = Path(training["init_checkpoint"]).expanduser().resolve(strict=True)
    sha = file_digest(path)
    if sha != training.get("init_checkpoint_sha256"):
        raise ValueError("Initialization checkpoint SHA256 differs from the sealed configuration")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    verify_checkpoint_config(checkpoint, config)
    source_protocol = path.parent.parent / "protocol.json"
    if not source_protocol.exists():
        raise ValueError("Source weights need a training protocol to check held-out event contamination")
    protocol = json.loads(source_protocol.read_text())
    seen_events = sorted({record["event"] for value in protocol["datasets"].values()
                          for record in value["records"] if record.get("event")})
    target = config.get("experiment", {}).get("held_out_event")
    if target in seen_events:
        raise ValueError(f"Initialization checkpoint already saw held-out event {target}")
    manifest = config.get("dataset", {}).get("manifest")
    if manifest:
        from utils.split_manifest import read_manifest
        pretrained_groups = {row.get("group") for row in protocol["datasets"]["train"]["records"]}
        protected_groups = {row["group"] for row in read_manifest(manifest) if row["split"] in {"val", "test"}}
        if None in pretrained_groups or pretrained_groups & protected_groups:
            raise ValueError("Source checkpoint training scenes overlap current validation/test scenes or lack group metadata")
    load_model_state(model, checkpoint["model"])
    return {"checkpoint": str(path), "sha256": sha, "epoch": checkpoint.get("epoch"),
            "source_events": seen_events, "optimizer_state_imported": False}
