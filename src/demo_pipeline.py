"""Run the existing CCTV stages and report progress to the desktop interface."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.run_vit_pipeline import stage_commands


ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "vit": ("Vehicle tracking", "Motion candidates", "ViT detection", "Fire, smoke and severity"),
    "fusion": ("Fusion detection", "Fire, smoke and severity"),
}


@dataclass(frozen=True)
class DemoSettings:
    backend: str = "vit"
    calibration: Path = ROOT / "outputs/eval/vit/vit_evaluation.json"
    fusion_report: Path = ROOT / "outputs/eval/fusion/fusion_evaluation.json"
    detector: Path = ROOT / "outputs/models/yolo11n.pt"
    hazards: Path = ROOT / "outputs/models/firewatch-yolo11s.onnx"


class RunCancelled(RuntimeError):
    """The operator stopped this run."""


class DemoPipeline:
    def __init__(self, settings: DemoSettings):
        self.settings = settings
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()

    def run(self, video: Path, update: Callable[[int, str, str], None]) -> Path:
        model = self.settings
        stages = STAGES[model.backend]
        requirements = [("Source video", video), ("Firewatch weights", model.hazards)]
        if model.backend == "vit":
            requirements.extend([("ViT calibration", model.calibration), ("YOLO weights", model.detector)])
        else:
            requirements.append(("Fusion report", model.fusion_report))
        for label, path in requirements:
            if not path.is_file():
                raise FileNotFoundError(f"{label} missing at {path}")

        runs = ROOT / "outputs" / "desktop_runs"
        runs.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="run-", dir=runs))
        phase3 = output / "detection"
        report = output / "hazard_incidents.json"
        preview = output / "final_preview.mp4"
        incident_args = []
        if model.backend == "vit":
            commands = [
                (stage, command[2:])
                for stage, command in zip((0, 1, 2, 2), stage_commands(video, model.calibration, phase3, str(model.detector)))
            ]
            clip_result = phase3 / "vit" / f"{video.stem}_result.json"
            incident_args = ["--incidents", phase3 / "confirmed_incidents.json"]
        else:
            commands = [(0, ["src.fusion_workflow", "--video", video, "--report", model.fusion_report,
                             "--output-dir", phase3])]
            clip_result = phase3 / f"{video.stem}_result.json"
        commands.append((len(stages) - 1, ["src.run_phase4", "--video", video, "--clip-result", clip_result,
                                          *incident_args, "--weights", model.hazards, "--output-dir", output]))
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        runtime = ROOT / "outputs" / "runtime"
        if runtime.is_dir():
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(runtime), env.get("PYTHONPATH"))))

        log_path = output / "pipeline.log"
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            for index, (stage, arguments) in enumerate(commands):
                if self._cancel.is_set():
                    raise RunCancelled("Analysis stopped.")
                update(stage, "running", stages[stage])
                command = [sys.executable, "-u", "-B", "-m", *(str(value) for value in arguments)]
                log.write(subprocess.list2cmdline(command) + "\n")
                with self._lock:
                    if self._cancel.is_set():
                        raise RunCancelled("Analysis stopped.")
                    process = subprocess.Popen(
                        command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    self._process = process
                try:
                    with process.stdout as lines:
                        for line in lines:
                            log.write(line)
                            if line.strip():
                                update(stage, "detail", line.strip())
                    exit_code = process.wait()
                finally:
                    with self._lock:
                        self._process = None
                if self._cancel.is_set():
                    raise RunCancelled("Analysis stopped.")
                if exit_code:
                    raise RuntimeError(f"{stages[stage]} failed. See {log_path}")
                if index + 1 == len(commands) or commands[index + 1][0] != stage:
                    update(stage, "complete", stages[stage])
        if not all(path.is_file() for path in (report, preview, output / "review_summary.json")):
            raise RuntimeError(f"Analysis did not produce a preview and incident report. See {log_path}")
        return output
