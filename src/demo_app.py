"""Local desktop interface for CCTV accident analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from src.clip_buffer import compute_resized_dimensions
from src.demo_pipeline import STAGES, DemoPipeline, DemoSettings, RunCancelled


BG, PANEL, VIDEO = "#0D1420", "#172233", "#080E17"
TEXT, MUTED, ACCENT = "#EDF4FC", "#A4B4CC", "#5DE2CA"
LINE, HOVER = "#27384D", "#25364C"
FONT = "Segoe UI"

BACKEND_NAMES = {"vit": "Pretrained ViT", "fusion": "ViT + R3D-18 + YOLO11x"}
HAZARD_COLUMNS = (("time", "Scan time", 77), ("score", "Score", 72), ("scope", "Scope", 91),
                  ("fire", "Fire", 100), ("smoke", "Smoke", 100), ("risk", "Visual severity", 128))
CANDIDATE_COLUMNS = (("time", "Candidate time", 115), ("score", "ViT score", 95),
                     ("motion", "Motion score", 100), ("status", "Review status", 155))


def clock_label(seconds: float) -> str:
    whole = int(max(0, seconds))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def load_candidate_rows(folder: Path) -> list[dict]:
    audit = folder / "detection" / "confirmation_results.json"
    if not audit.is_file():
        return []
    proposals_path = audit.with_name("candidates.json")
    proposals = json.loads(proposals_path.read_text(encoding="utf-8")) if proposals_path.is_file() else []
    proposals_by_id = {row["candidate_id"]: row for row in proposals}
    scored = json.loads(audit.read_text(encoding="utf-8"))
    return sorted(
        ({**proposals_by_id.get(row["candidate_id"], {}), **row} for row in scored),
        key=lambda row: float(row["timestamp_sec"]),
    )


class VideoPlayer:
    def __init__(self, parent):
        self.capture = None
        self.frame = None
        self.candidate = None
        self.photo = None
        self.timer = None
        self.index = self.count = 0
        self.fps = 1.0
        self.track_position = tk.DoubleVar(value=0)
        self.area = tk.Canvas(parent, bg=VIDEO, highlightthickness=0, height=260)
        self.area.pack(fill="both", expand=True)
        self.area.bind("<Configure>", lambda _event: self.redraw())
        controls = ttk.Frame(parent, style="Panel.TFrame", padding=(12, 9))
        controls.pack(fill="x")
        self.play_button = ttk.Button(controls, text="Play", width=8, command=self.toggle, state="disabled")
        self.play_button.pack(side="left")
        self.slider = ttk.Scale(controls, from_=0, to=1, variable=self.track_position, command=self.seek, state="disabled")
        self.slider.pack(side="left", fill="x", expand=True, padx=12)
        self.clock = ttk.Label(controls, text="00:00 / 00:00", style="Muted.TLabel")
        self.clock.pack(side="right")

    def pause(self):
        if self.timer is not None:
            self.area.after_cancel(self.timer)
            self.timer = None
        self.play_button.configure(text="Play")

    def clear(self):
        self.pause()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.frame = None
        self.candidate = None
        self.track_position.set(0)
        self.slider.configure(state="disabled")
        self.play_button.configure(state="disabled")
        self.clock.configure(text="00:00 / 00:00")
        self.redraw()

    def load(self, video: Path):
        capture = cv2.VideoCapture(str(video))
        fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = capture.read()
        if not ok or not math.isfinite(fps) or fps <= 0 or count < 1:
            capture.release()
            raise ValueError(f"Could not play {video.name}.")
        self.clear()
        self.capture, self.fps, self.count = capture, fps, count
        self.slider.configure(to=max(1, count - 1), state="normal")
        self.play_button.configure(state="normal")
        self.set_frame(0, frame)

    def set_frame(self, index, frame):
        self.index, self.frame = index, frame
        self.track_position.set(index)
        self.clock.configure(text=f"{clock_label(index / self.fps)} / {clock_label(self.count / self.fps)}")
        self.redraw()

    def redraw(self):
        self.area.delete("preview")
        width, height = self.area.winfo_width(), self.area.winfo_height()
        if self.frame is None:
            self.area.create_text(width // 2, height // 2, text="Choose a video or open a saved analysis",
                                  fill=MUTED, font=(FONT, 13), tags="preview")
            return
        original_h, original_w = self.frame.shape[:2]
        ratio = min(width / original_w, height / original_h)
        size = max(1, int(original_w * ratio)), max(1, int(original_h * ratio))
        resized = cv2.resize(self.frame, size, interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.area.create_image(width // 2, height // 2, image=self.photo, tags="preview")
        if self.candidate is not None:
            self.draw_candidate(width // 2 - size[0] / 2, height // 2 - size[1] / 2, size,
                                original_w, original_h)

    def draw_candidate(self, left, top, size, original_w, original_h):
        timestamp, box = self.candidate
        if abs(self.index / self.fps - timestamp) > 0.40:
            return
        tracking_w, tracking_h = compute_resized_dimensions(original_w, original_h, 960)
        x1, y1, x2, y2 = (min(1.0, max(0.0, value / extent)) for value, extent in
                          zip(box, (tracking_w, tracking_h, tracking_w, tracking_h)))
        if x2 <= x1 or y2 <= y1:
            return
        a, b = left + x1 * size[0], top + y1 * size[1]
        color = "#6ECFFF"
        self.area.create_rectangle(a, b, left + x2 * size[0], top + y2 * size[1],
                                   outline=color, width=2, dash=(5, 3), tags=("preview", "review-candidate"))
        self.area.create_text(a + 4, max(top + 13, b - 9), text="CANDIDATE", fill=color,
                              anchor="w", font=(FONT, 9, "bold"), tags=("preview", "review-candidate"))

    def review_candidate(self, row):
        self.candidate = None
        if row is not None and not row.get("confirmed"):
            box = row.get("candidate_region")
            if isinstance(box, list) and len(box) == 4 and all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in box
            ):
                self.candidate = float(row["timestamp_sec"]), box
        self.redraw()

    def show_frame(self, target):
        if self.capture is None or target == self.index:
            return True
        gap = target - self.index
        if 1 <= gap <= 6:
            for _ in range(gap - 1):
                self.capture.grab()
        else:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self.capture.read()
        if ok:
            self.set_frame(target, frame)
        return ok

    def seek(self, position):
        if self.capture is not None:
            self.pause()
            self.show_frame(min(self.count - 1, max(0, round(float(position)))))

    def go_to(self, seconds):
        self.seek(seconds * self.fps)

    def toggle(self):
        if self.timer is not None:
            self.pause()
        elif self.capture is not None:
            if self.index >= self.count - 1:
                self.show_frame(0)
            self.play_button.configure(text="Pause")
            self.start_time, self.start_frame = time.monotonic(), self.index
            self.tick()

    def tick(self):
        self.timer = None
        target = min(self.count - 1, self.start_frame + int((time.monotonic() - self.start_time) * self.fps))
        if self.show_frame(target) and target < self.count - 1:
            self.timer = self.area.after(25, self.tick)
        else:
            self.pause()


class DemoApp:
    def __init__(self, root: tk.Tk, settings: DemoSettings):
        self.root, self.settings = root, settings
        self.messages = queue.Queue()
        self.runner = None
        self.current_report = None
        self.incidents = []
        self.candidates = []
        self.show_candidates = False
        self.selected = tk.StringVar()
        self.status = tk.StringVar(value="Ready to review footage")
        self.stage_states = []

        root.title("CCTV Incident Review")
        root.geometry(f"{min(1260, root.winfo_screenwidth() - 64)}x{min(780, root.winfo_screenheight() - 104)}")
        root.minsize(940, 610)
        root.configure(bg=BG)
        self.style()

        header = tk.Frame(root, bg=BG, padx=25, pady=16)
        header.pack(fill="x")
        self.label(header, "CCTV Incident Review", BG, size=24, weight="bold").pack(side="left")

        body = tk.Frame(root, bg=BG, padx=24, pady=0)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        sidebar = tk.Frame(body, bg=PANEL, width=260, padx=16, pady=16)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        self.sidebar(sidebar)

        review = tk.Frame(body, bg=PANEL)
        review.grid(row=0, column=1, sticky="nsew")
        review.columnconfigure(0, weight=1)
        review.rowconfigure(1, weight=1)
        title = tk.Frame(review, bg=PANEL, padx=13, pady=9)
        title.grid(row=0, column=0, sticky="ew")
        self.preview_title = self.label(title, "Video preview", PANEL, size=13, weight="bold")
        self.preview_title.pack(side="left")
        self.pill = self.label(title, "READY", PANEL, color=ACCENT, size=9, weight="bold")
        self.pill.pack(side="right")

        player_frame = tk.Frame(review, bg=VIDEO)
        player_frame.grid(row=1, column=0, sticky="nsew")
        self.player = VideoPlayer(player_frame)
        table_panel = tk.Frame(review, bg=PANEL, padx=12, pady=8)
        table_panel.grid(row=2, column=0, sticky="ew")
        self.result_table(table_panel)

        footer = tk.Frame(root, bg=BG, padx=24, pady=8)
        footer.pack(fill="x")
        self.label(footer, "", BG, color=MUTED, textvariable=self.status, size=9).pack(side="left")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(80, self.poll)

    def style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=(FONT, 9))
        style.configure("TButton", background=LINE, foreground=TEXT, font=(FONT, 10), padding=(9, 6), borderwidth=0)
        style.map("TButton", background=[("active", HOVER)], foreground=[("disabled", MUTED)])
        style.configure("Primary.TButton", background=ACCENT, foreground=BG, font=(FONT, 10, "bold"))
        style.map("Primary.TButton", background=[("disabled", LINE), ("active", "#8FF0E0")],
                  foreground=[("disabled", MUTED), ("!disabled", BG)])
        style.configure("TEntry", fieldbackground=VIDEO, foreground=TEXT, insertcolor=TEXT, padding=6)
        style.configure("Horizontal.TScale", background=ACCENT, troughcolor=LINE)
        style.configure("Treeview", background=VIDEO, foreground=TEXT, fieldbackground=VIDEO,
                        borderwidth=0, font=(FONT, 10), rowheight=27)
        style.configure("Treeview.Heading", background=LINE, foreground=TEXT, font=(FONT, 9, "bold"), padding=(4, 4))
        style.map("Treeview", background=[("selected", "#235469")], foreground=[("selected", "#FFFFFF")])

    @staticmethod
    def label(parent, text, bg, size=10, color=TEXT, weight="normal", **kwargs):
        return tk.Label(parent, text=text, bg=bg, fg=color, font=(FONT, size, weight), anchor="w", **kwargs)

    def sidebar(self, parent):
        self.label(parent, "Analyze a video", PANEL, size=15, weight="bold").pack(fill="x")
        self.label(parent, BACKEND_NAMES[self.settings.backend], PANEL, color=MUTED, size=9).pack(fill="x", pady=(2, 9))
        ttk.Entry(parent, textvariable=self.selected, width=28).pack(fill="x", pady=(0, 8))
        self.browse_button = ttk.Button(parent, text="Choose video", command=self.browse)
        self.browse_button.pack(fill="x", pady=(0, 7))
        self.analyze_button = ttk.Button(parent, text="Analyze video", style="Primary.TButton", command=self.start)
        self.analyze_button.pack(fill="x")
        self.stop_button = ttk.Button(parent, text="Stop analysis", state="disabled", command=self.stop)
        self.stop_button.pack(fill="x", pady=(7, 0))

        self.saved_button = ttk.Button(parent, text="Open saved analysis", command=self.open_saved)
        self.saved_button.pack(fill="x", pady=(16, 0))

        self.label(parent, "Pipeline", PANEL, size=12, weight="bold").pack(fill="x", pady=(12, 7))
        for stage in STAGES[self.settings.backend]:
            row = tk.Frame(parent, bg=PANEL)
            row.pack(fill="x", pady=3)
            self.label(row, stage, PANEL, size=9).pack(side="left")
            state = self.label(row, "Ready", PANEL, color=MUTED, size=8)
            state.pack(side="right", padx=(5, 0))
            self.stage_states.append(state)

    def result_table(self, parent):
        header = tk.Frame(parent, bg=PANEL)
        header.pack(fill="x", pady=(0, 6))
        self.count_label = self.label(header, "Incident review", PANEL, size=12, weight="bold")
        self.count_label.pack(side="left")
        self.report_button = ttk.Button(header, text="Open results folder", command=self.open_folder, state="disabled")
        self.report_button.pack(side="right")
        self.candidates_button = ttk.Button(header, text="All candidates", command=self.toggle_candidates,
                                            state="disabled")
        self.candidates_button.pack(side="right", padx=(0, 8))
        self.verdict_label = self.label(parent, "", PANEL, color=MUTED, size=9)
        self.verdict_label.pack(fill="x", pady=(0, 5))
        self.reason_label = self.label(parent, "", PANEL, color=MUTED, size=9, wraplength=580, justify="left")
        self.reason_label.pack(side="bottom", fill="x", pady=(5, 0))
        self.table = ttk.Treeview(parent, show="headings", height=3, selectmode="browse")
        self.configure_table(HAZARD_COLUMNS)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.table.pack(fill="x", expand=True)
        self.table.bind("<<TreeviewSelect>>", self.select_incident)

    def configure_table(self, columns):
        self.table.configure(columns=tuple(key for key, _, _ in columns))
        for key, name, width in columns:
            self.table.heading(key, text=name, anchor="w")
            self.table.column(key, width=width, anchor="w")

    def select_incident(self, _event=None):
        selected = self.table.selection()
        if selected:
            if self.show_candidates:
                row = self.candidates[int(selected[0])]
                self.player.go_to(float(row["timestamp_sec"]))
                self.player.review_candidate(row)
                reason = "Localized alert confirmed" if row["confirmed"] else row.get("rejection_reason")
                start, end = row.get("clip_start_sec"), row.get("clip_end_sec")
                window = f"  •  Scored window {start:.2f}–{end:.2f} s" if start is not None and end is not None else ""
                self.reason_label.configure(text=f"{reason or 'Flagged for review'}{window}")
                return
            incident = self.incidents[int(selected[0])]
            timestamp = float(incident["timestamp_sec"])
            reason = incident.get("visual_risk_reason", "")
            scan = incident["hazard_scan"]
            if scan["scope"] == "full_frame":
                reason = f"{reason} Scene-wide scan."
                strongest = scan.get("strongest_detection", {})
                cue = strongest.get("fire") or strongest.get("smoke")
                if cue:
                    timestamp = cue["timestamp_sec"]
                    reason = f"{reason} Strongest hazard cue at {timestamp:.2f} s."
            self.player.go_to(timestamp)
            self.reason_label.configure(text=reason)

    def browse(self):
        path = filedialog.askopenfilename(title="Choose a CCTV video", filetypes=[
            ("Video files", "*.mp4 *.webm *.mov *.avi *.mkv *.m4v"), ("All files", "*.*"),
        ])
        if path:
            self.selected.set(path)

    def set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.browse_button.configure(state=state)
        self.analyze_button.configure(state=state)
        self.stop_button.configure(state="normal" if busy else "disabled")
        self.saved_button.configure(state=state)

    def clear_review(self):
        self.player.clear()
        self.table.delete(*self.table.get_children())
        self.incidents = []
        self.candidates = []
        self.show_candidates = False
        self.current_report = None
        self.configure_table(HAZARD_COLUMNS)
        self.count_label.configure(text="Incident review")
        self.verdict_label.configure(text="")
        self.reason_label.configure(text="")
        self.report_button.configure(state="disabled")
        self.candidates_button.configure(text="All candidates", state="disabled")

    def load_review(self, folder):
        report = folder / "hazard_incidents.json"
        summary = json.loads((folder / "review_summary.json").read_text(encoding="utf-8"))
        incidents = json.loads(report.read_text(encoding="utf-8"))
        candidates = load_candidate_rows(folder)
        if not isinstance(incidents, list) or any(
            not isinstance(row, dict) or "timestamp_sec" not in row for row in incidents
        ):
            raise ValueError("Invalid incident report.")
        self.clear_review()
        self.player.load(folder / "final_preview.mp4")
        self.incidents = sorted(incidents, key=lambda row: float(row["timestamp_sec"]))
        self.candidates = candidates
        self.current_report = report
        self.preview_title.configure(text=Path(summary["video"]).name)
        self.pill.configure(text="ANALYSIS COMPLETE")
        verdict = "Accident signal" if summary["predicted_accident"] else "Below cutoff"
        self.verdict_label.configure(text=(
            f"{BACKEND_NAMES[summary['backend']]}   ·   Clip verdict: {verdict}   ·   "
            f"Score {summary['score']:.2f} / cutoff {summary['threshold']:.2f}"
        ))
        self.report_button.configure(state="normal")
        has_hazard_cues = any(any(row["hazard_scan"].get("hits", {}).values()) for row in self.incidents)
        self.show_candidates = bool(self.candidates) and not summary["localized_incidents"] and not has_hazard_cues
        self.populate_results()

    def toggle_candidates(self):
        self.show_candidates = not self.show_candidates
        self.populate_results()

    def populate_results(self):
        self.table.delete(*self.table.get_children())
        self.player.review_candidate(None)
        self.reason_label.configure(text="")
        self.candidates_button.configure(
            text="Hazard scans" if self.show_candidates else f"All candidates ({len(self.candidates)})",
            state="normal" if self.candidates else "disabled",
        )
        if self.show_candidates:
            self.configure_table(CANDIDATE_COLUMNS)
            self.count_label.configure(text=f"All candidates  ·  {len(self.candidates)}")
            for index, row in enumerate(self.candidates):
                motion = row.get("motion_score")
                self.table.insert("", "end", iid=str(index), values=(
                    f"{row['timestamp_sec']:.2f} s", f"{row['p_accident']:.4f}",
                    "—" if motion is None else f"{motion}/9",
                    "Confirmed" if row["confirmed"] else "Unconfirmed",
                ))
            self.status.set("Select a row to inspect its proposed region and score.")
            return

        self.configure_table(HAZARD_COLUMNS)
        for index, row in enumerate(self.incidents):
            def hazard(label):
                value = row.get(f"{label}_detected")
                if value is True:
                    return "Detected"
                if value is False:
                    return "Intermittent" if row["hazard_scan"].get("hits", {}).get(label) else "Below cutoff"
                return "Unknown"
            score = row.get("accident_confidence", row.get("clip_score"))
            time_prefix = "~" if row.get("timestamp_is_approximate") else ""
            scope = "Scene-wide" if row["hazard_scan"]["scope"] == "full_frame" else "Region"
            self.table.insert("", "end", iid=str(index), values=(
                f"{time_prefix}{float(row['timestamp_sec']):.2f} s", "—" if score is None else f"{score:.2f}",
                scope,
                hazard("fire"), hazard("smoke"), row.get("visual_risk") or "Unknown",
            ))
        self.count_label.configure(text=f"Hazard scans  ·  {len(self.incidents)}")
        self.status.set("Select a row to review its scan. ~ marks an approximate time."
                        if self.incidents else "Analysis complete. No incident window to scan.")

    def open_saved(self):
        folder = filedialog.askdirectory(title="Choose a saved analysis folder")
        if not folder:
            return
        try:
            self.load_review(Path(folder))
            for state in self.stage_states:
                state.configure(text="–", fg=MUTED)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("Could not open saved analysis", str(exc), parent=self.root)

    def open_folder(self):
        if self.current_report is not None:
            folder = self.current_report.parent
            if hasattr(os, "startfile"):
                os.startfile(str(folder))

    def start(self):
        typed = self.selected.get().strip().strip('"')
        if not typed or not Path(typed).is_file():
            messagebox.showinfo("Choose a video", "Select an existing CCTV video to analyze.", parent=self.root)
            return
        video = Path(typed).resolve()
        self.clear_review()
        self.preview_title.configure(text=video.name)
        self.pill.configure(text="ANALYZING")
        for state in self.stage_states:
            state.configure(text="Queued", fg=MUTED)
        self.set_busy(True)
        self.status.set("Starting analysis…")
        self.runner = DemoPipeline(self.settings)
        threading.Thread(target=self.work, args=(self.runner, video), daemon=True).start()

    def work(self, runner, video):
        try:
            output = runner.run(video, lambda stage, status, detail: self.messages.put((status, stage, detail)))
            self.messages.put(("done", video.name, output))
        except (RunCancelled, OSError, ValueError, RuntimeError) as exc:
            self.messages.put(("stopped" if isinstance(exc, RunCancelled) else "error", None, str(exc)))

    def poll(self):
        while not self.messages.empty():
            kind, subject, detail = self.messages.get_nowait()
            if kind in ("running", "complete"):
                state = self.stage_states[subject]
                state.configure(text="Running" if kind == "running" else "Done", fg=ACCENT)
            elif kind == "detail":
                if not detail.startswith(("Annotated video", "Trajectory CSV")):
                    self.status.set(detail if len(detail) < 118 else detail[:115] + "…")
            else:
                self.runner = None
                self.set_busy(False)
                if kind == "done":
                    try:
                        self.load_review(detail)
                    except (OSError, KeyError, TypeError, ValueError) as exc:
                        self.pill.configure(text="ERROR")
                        self.status.set(str(exc))
                        messagebox.showerror("Could not open results", str(exc), parent=self.root)
                else:
                    self.pill.configure(text="STOPPED" if kind == "stopped" else "ERROR")
                    self.status.set("Analysis stopped." if kind == "stopped" else "Analysis failed. See details.")
                    for state in self.stage_states:
                        if state.cget("text") == "Running":
                            state.configure(text="Stopped" if kind == "stopped" else "Error", fg=MUTED)
                    if kind == "error":
                        messagebox.showerror("Analysis failed", detail, parent=self.root)
        self.root.after(80, self.poll)

    def stop(self):
        if self.runner is not None:
            self.stop_button.configure(state="disabled")
            self.status.set("Stopping analysis…")
            self.runner.cancel()

    def close(self):
        self.stop()
        self.player.clear()
        self.root.destroy()


def main(argv=None):
    defaults = DemoSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(STAGES), default=defaults.backend)
    parser.add_argument("--calibration", type=Path, default=defaults.calibration)
    parser.add_argument("--fusion-report", type=Path, default=defaults.fusion_report)
    parser.add_argument("--detector", type=Path, default=defaults.detector)
    parser.add_argument("--hazards", type=Path, default=defaults.hazards)
    parser.add_argument("--review-dir", type=Path, help="Reopen a previously generated analysis")
    args = parser.parse_args(argv)
    root = tk.Tk()
    settings = DemoSettings(args.backend, args.calibration.resolve(), args.fusion_report.resolve(),
                            args.detector.resolve(), args.hazards.resolve())
    app = DemoApp(root, settings)
    if args.review_dir:
        try:
            app.load_review(args.review_dir)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("Could not open saved analysis", str(exc), parent=root)
    root.mainloop()


if __name__ == "__main__":
    main()
