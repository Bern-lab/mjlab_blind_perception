"""Watch stair latent report outputs while long embeddings are running."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tyro

FEATURE_ALIASES = {
  "actor": "latent",
  "actor_latent": "latent",
  "latent": "latent",
  "h_t": "h_t",
  "hidden": "hidden",
  "memory": "z_memory",
  "z_memory": "z_memory",
  "state": "z_state_memory",
  "z_state": "z_state_memory",
  "z_state_memory": "z_state_memory",
  "shape": "z_shape_memory",
  "z_shape": "z_shape_memory",
  "z_shape_memory": "z_shape_memory",
  "semantic": "semantic",
  "shadow_semantic": "shadow_semantic",
}

OUTPUT_ALIASES = {
  "latent": "actor_latent",
  "h_t": "h_t",
  "hidden": "hidden",
  "z_memory": "z_memory",
  "z_state_memory": "z_state",
  "z_shape_memory": "z_shape",
  "semantic": "semantic",
  "shadow_semantic": "semantic",
}

CORE_FILES = (
  "latent_state_switch_episode.png",
  "semantic_dynamics_episode.png",
  "z_memory_heatmap_episode.png",
  "gate_state_raster.png",
  "semantic_correlation_matrix.png",
  "semantic_invariant_report.json",
  "head_prediction_summary.png",
  "latent_state_switch_episode.csv",
  "hidden_cluster_pca.png",
  "actor_latent_cluster_pca.png",
  "z_memory_cluster_pca.png",
  "state_memory_cluster_pca.png",
  "shape_memory_cluster_pca.png",
)


@dataclass(frozen=True)
class WatchLatentReportProgressConfig:
  input_file: str | None = None
  output_dir: str | None = None
  interactive_features: str = "auto"
  embedding_method: str = "tsne"
  embedding_dim: int = 3
  interval_s: float = 5.0
  once: bool = False
  show_processes: bool = True


def _resolve_output_dir(cfg: WatchLatentReportProgressConfig) -> Path:
  if cfg.output_dir is not None:
    return Path(cfg.output_dir)
  if cfg.input_file is not None:
    return Path(cfg.input_file).parent
  raise ValueError("Use --input-file or --output-dir.")


def _load_keys(input_file: str | None) -> set[str]:
  if input_file is None or not Path(input_file).exists():
    return set()
  with np.load(input_file, allow_pickle=False) as data:
    return set(data.files)


def _semantic_key(keys: set[str]) -> str | None:
  if "semantic" in keys:
    return "semantic"
  if "shadow_semantic" in keys:
    return "shadow_semantic"
  return None


def _interactive_keys(
  cfg: WatchLatentReportProgressConfig,
  keys: set[str],
) -> tuple[list[str], list[str]]:
  missing = []
  if cfg.interactive_features.strip().lower() != "auto":
    resolved = []
    for raw in cfg.interactive_features.split(","):
      requested = raw.strip()
      if not requested:
        continue
      key = FEATURE_ALIASES.get(requested, requested)
      if key == "semantic":
        key = _semantic_key(keys) or key
      if key == "h_t" and key not in keys and "hidden" in keys:
        key = "hidden"
      if keys and key not in keys:
        missing.append(requested)
        continue
      resolved.append(key)
    return list(dict.fromkeys(resolved)), missing

  preferred = [
    "h_t",
    "hidden",
    "z_memory",
    "z_state_memory",
    "z_shape_memory",
    _semantic_key(keys) or "",
    "latent",
  ]
  return [
    key for key in dict.fromkeys(preferred) if key and (not keys or key in keys)
  ], []


def _expected_files(
  output_dir: Path,
  features: list[str],
  cfg: WatchLatentReportProgressConfig,
) -> list[Path]:
  files = [output_dir / name for name in CORE_FILES]
  suffix = f"{cfg.embedding_method}{cfg.embedding_dim}d"
  for feature in features:
    alias = OUTPUT_ALIASES.get(feature, feature)
    files.append(output_dir / "interactive" / f"{alias}_{suffix}.html")
    for plane in ("xy", "xz", "yz"):
      files.append(output_dir / "static" / f"{alias}_{suffix}_{plane}.png")
  files.append(output_dir / "representation_probe_summary.csv")
  files.append(output_dir / "representation_probe_summary.png")
  files.append(output_dir / "latent_report_summary.json")
  return files


def _format_size(path: Path) -> str:
  if not path.exists():
    return "-"
  size = path.stat().st_size
  for unit in ("B", "KB", "MB", "GB"):
    if size < 1024.0:
      return f"{size:.1f}{unit}"
    size /= 1024.0
  return f"{size:.1f}TB"


def _format_mtime(path: Path) -> str:
  if not path.exists():
    return "-"
  return datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")


def _process_lines() -> list[str]:
  result = subprocess.run(
    ["ps", "-eo", "pid,etime,state,%cpu,%mem,cmd"],
    check=False,
    capture_output=True,
    text=True,
  )
  lines = []
  for line in result.stdout.splitlines():
    if "plot_stair_latent_report.py" in line and "watch_latent_report" not in line:
      lines.append(line.strip())
  return lines


def _render_once(cfg: WatchLatentReportProgressConfig) -> bool:
  output_dir = _resolve_output_dir(cfg)
  keys = _load_keys(cfg.input_file)
  features, missing = _interactive_keys(cfg, keys)
  files = _expected_files(output_dir, features, cfg)
  done = [path for path in files if path.exists()]
  percent = 100.0 * len(done) / max(len(files), 1)

  print(f"Latent report progress | {datetime.now().strftime('%H:%M:%S')}")
  print(f"Output: {output_dir}")
  print(f"Done: {len(done)}/{len(files)} ({percent:.1f}%)")
  if missing:
    print(f"Skipped missing requested arrays: {', '.join(missing)}")
  if cfg.show_processes:
    processes = _process_lines()
    print("Processes:")
    if processes:
      for line in processes:
        print(f"  {line}")
    else:
      print("  none")
  print()
  for path in files:
    rel = path.relative_to(output_dir)
    status = "done" if path.exists() else "wait"
    print(f"[{status:4}] {rel}  {_format_size(path):>8}  {_format_mtime(path)}")
  return len(done) == len(files)


def main() -> None:
  cfg = tyro.cli(WatchLatentReportProgressConfig)
  while True:
    print("\033[2J\033[H", end="")
    complete = _render_once(cfg)
    if cfg.once or complete:
      break
    time.sleep(max(cfg.interval_s, 0.5))


if __name__ == "__main__":
  main()
