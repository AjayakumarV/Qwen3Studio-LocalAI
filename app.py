import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional

import customtkinter as ctk
import requests

from file_tools import (
    AttachedFile,
    load_and_extract_file,
    load_folder_files,
    format_attachments_for_prompt,
)
from stt_engine import LocalSTTEngine
from verifier import run_symbolic_audit, build_critic_verification_messages


if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LocalAI.Qwen3Studio.1")
    except Exception:
        pass

APP_DIR = Path(__file__).resolve().parent
LOCALAI_HOME = Path(os.environ.get("LOCALAI_HOME", str(Path.home() / "LocalAI")))

ICON_PATH = APP_DIR / "app_icon.ico"
DRAFT_GGUF_PATH = LOCALAI_HOME / "models" / "Qwen3-0.6B-UD-Q4_K_XL.gguf"

LLAMA_SERVER_EXE = LOCALAI_HOME / "bin" / "llama-server.exe"
MODEL_GGUF_PATH = LOCALAI_HOME / "models" / "Qwen3-30B-A3B-UD-IQ3_XXS.gguf"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
API_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

SYSTEM_PROMPT = (
    "You are a helpful, accurate, and concise local AI assistant running 100% offline on the user's workstation. "
    "Format formulas and math in clean, readable plain text or Unicode (e.g., °C = (°F - 32) × 5 / 9) rather than raw LaTeX ($$ or \\frac)."
)


def clean_latex_and_mojibake(text: str) -> str:
    """Fixes any Latin-1 UTF-8 mojibake (e.g., Â° -> °) and converts raw LaTeX math into clean Unicode."""
    if not text:
        return text
    if "Â" in text or "â" in text or "ð" in text:
        try:
            text = text.encode("latin-1").decode("utf-8")
        except Exception:
            text = text.replace("Â°", "°")

    # Convert \frac{a}{b} -> (a) / (b) or a / b
    def _replace_frac(match: re.Match) -> str:
        num = match.group(1).strip()
        den = match.group(2).strip()
        if (" " in num or "+" in num or "-" in num) and not (num.startswith("(") and num.endswith(")")):
            num = f"({num})"
        if (" " in den or "+" in den or "-" in den) and not (den.startswith("(") and den.endswith(")")):
            den = f"({den})"
        return f"{num} / {den}"

    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", _replace_frac, text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", text)
    text = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]+)\}", r"\1", text)

    latex_replacements = {
        r"\times": "×",
        r"\div": "÷",
        r"\cdot": "·",
        r"\pm": "±",
        r"\approx": "≈",
        r"\leq": "≤",
        r"\le": "≤",
        r"\geq": "≥",
        r"\ge": "≥",
        r"\neq": "≠",
        r"\ne": "≠",
        r"\rightarrow": "→",
        r"\to": "→",
        r"^\circ": "°",
        r"\circ": "°",
        r"\degree": "°",
        "$$": "",
        r"\[": "",
        r"\]": "",
        r"\(": "",
        r"\)": "",
    }
    for k, v in latex_replacements.items():
        if k in text:
            text = text.replace(k, v)
    return text


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class LocalAIDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("LocalAI Studio — Qwen3-30B-A3B (Selective-Weight Turbo ~39 tok/s)")
        self.geometry("1260x840")
        self.minsize(980, 680)
        if ICON_PATH.exists():
            try:
                self.iconbitmap(str(ICON_PATH))
            except Exception:
                pass

        self.stt = LocalSTTEngine()
        self.attached_docs: List[AttachedFile] = []
        self.chat_history: List[Dict[str, str]] = []
        self.server_proc: Optional[subprocess.Popen] = None
        self.we_started_server = False
        self.is_generating = False
        self.stop_generation_flag = False
        self.active_experts = 4
        self._server_lock = threading.Lock()
        self._hw_poll_running = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start background threads for engine auto-start and hardware monitoring
        threading.Thread(target=self._init_backend_and_stt, daemon=True).start()
        self._schedule_hw_monitor()

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= LEFT SIDEBAR (Hardware, Attachments, Controls) =================
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)

        title_lbl = ctk.CTkLabel(
            self.sidebar,
            text="LocalAI Desktop",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title_lbl.grid(row=0, column=0, padx=16, pady=(16, 4), sticky="w")

        sub_lbl = ctk.CTkLabel(
            self.sidebar,
            text="Qwen3-30B-A3B (UD-IQ3_XXS)\n100% Offline • No Browser",
            font=ctk.CTkFont(size=12),
            text_color="#9CA3AF",
            justify="left",
        )
        sub_lbl.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")

        # --- Hardware & Engine Control Card ---
        hw_frame = ctk.CTkFrame(self.sidebar, fg_color="#1F2937")
        hw_frame.grid(row=2, column=0, padx=14, pady=6, sticky="ew")

        self.engine_status_lbl = ctk.CTkLabel(
            hw_frame,
            text="Engine: Starting...",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#FBBF24",
        )
        self.engine_status_lbl.pack(anchor="w", padx=12, pady=(10, 2))

        self.hw_stats_lbl = ctk.CTkLabel(
            hw_frame,
            text="VRAM: -- / 4096 MiB\nModel RAM: -- GB\nSpeed: -- tok/s",
            font=ctk.CTkFont(size=12),
            justify="left",
            text_color="#D1D5DB",
        )
        self.hw_stats_lbl.pack(anchor="w", padx=12, pady=(2, 8))

        btn_row = ctk.CTkFrame(hw_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(0, 10))

        self.toggle_engine_btn = ctk.CTkButton(
            btn_row,
            text="Unload (Free RAM)",
            height=28,
            fg_color="#B91C1C",
            hover_color="#991B1B",
            command=self._toggle_engine_power,
        )
        self.toggle_engine_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.expert_mode_var = ctk.StringVar(value="Top-4 Experts (~39.4 tok/s Recommended)")
        self.expert_mode_menu = ctk.CTkOptionMenu(
            hw_frame,
            values=[
                "Top-4 Experts (~39.4 tok/s Recommended)",
                "Top-5 Experts (~35.1 tok/s Balanced)",
                "Top-6 Experts (~31.0 tok/s Deep)",
                "All 8 Experts (~27.1 tok/s w/ Draft Speculative)",
            ],
            variable=self.expert_mode_var,
            height=28,
            fg_color="#111827",
            button_color="#374151",
            font=ctk.CTkFont(size=11),
            command=self._on_expert_mode_selected,
        )
        self.expert_mode_menu.pack(fill="x", padx=12, pady=(0, 6))

        self.spec_draft_var = ctk.BooleanVar(value=False)
        self.spec_draft_chk = ctk.CTkCheckBox(
            hw_frame,
            text="Qwen3-0.6B Draft Speculative (+45% on 8-Exp)",
            variable=self.spec_draft_var,
            font=ctk.CTkFont(size=11),
            command=self._on_spec_draft_toggled,
        )
        self.spec_draft_chk.pack(anchor="w", padx=12, pady=(0, 10))

        # --- Reasoning, Critic Verification & Voice Settings Card ---
        mode_frame = ctk.CTkFrame(self.sidebar, fg_color="#1F2937")
        mode_frame.grid(row=3, column=0, padx=14, pady=6, sticky="ew")

        self.critic_verify_var = ctk.BooleanVar(value=True)
        self.critic_switch = ctk.CTkSwitch(
            mode_frame,
            text="Auto-Critic Repair (Symbolic + Neural)",
            variable=self.critic_verify_var,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.critic_switch.pack(anchor="w", padx=12, pady=(10, 4))

        self.think_mode_var = ctk.BooleanVar(value=False)
        self.think_switch = ctk.CTkSwitch(
            mode_frame,
            text="Deep Thinking (/think CoT)",
            variable=self.think_mode_var,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.think_switch.pack(anchor="w", padx=12, pady=(4, 6))

        self.auto_send_voice_var = ctk.BooleanVar(value=False)
        self.auto_send_chk = ctk.CTkCheckBox(
            mode_frame,
            text="Auto-send after Mic STT",
            variable=self.auto_send_voice_var,
            font=ctk.CTkFont(size=12),
        )
        self.auto_send_chk.pack(anchor="w", padx=12, pady=(2, 10))

        # --- File Upload & Context Tools Section ---
        files_hdr = ctk.CTkLabel(
            self.sidebar,
            text="Uploaded Files & Context",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        files_hdr.grid(row=4, column=0, padx=16, pady=(12, 4), sticky="w")

        self.files_scroll = ctk.CTkScrollableFrame(
            self.sidebar, label_text="Attached Documents (0)"
        )
        self.files_scroll.grid(row=5, column=0, padx=14, pady=4, sticky="nsew")

        upload_btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        upload_btn_frame.grid(row=6, column=0, padx=14, pady=(6, 4), sticky="ew")

        self.upload_btn = ctk.CTkButton(
            upload_btn_frame,
            text="+ Upload Files (PDF/DOCX/XLSX/Code)",
            command=self._upload_files_dialog,
            fg_color="#2563EB",
            hover_color="#1D4ED8",
        )
        self.upload_btn.pack(fill="x", pady=(0, 6))

        self.clear_files_btn = ctk.CTkButton(
            upload_btn_frame,
            text="Clear Attached Files",
            height=28,
            fg_color="#374151",
            hover_color="#4B5563",
            command=self._clear_all_files,
        )
        self.clear_files_btn.pack(fill="x")

        # --- New Chat Button ---
        self.clear_chat_btn = ctk.CTkButton(
            self.sidebar,
            text="New Conversation",
            fg_color="#059669",
            hover_color="#047857",
            command=self._reset_conversation,
        )
        self.clear_chat_btn.grid(row=7, column=0, padx=14, pady=(8, 16), sticky="ew")

        # ================= RIGHT MAIN AREA (Chat Transcript + Voice/Prompt Bar) =================
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Chat display box
        self.chat_box = ctk.CTkTextbox(
            self.main_frame,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=14),
            fg_color="#111827",
            text_color="#F3F4F6",
            corner_radius=8,
        )
        self.chat_box.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.chat_box.insert(
            "end",
            "Welcome to LocalAI Desktop (Qwen3-30B-A3B).\n"
            "• 100% Offline Native App (No Browser Required)\n"
            "• Click [Record Mic (STT)] to dictate with local Faster-Whisper, or [Transcribe Audio File] for recordings.\n"
            "• Click [+ Upload Files] to attach PDF, Word (.docx), Excel (.xlsx), CSV, or code/text files.\n"
            "--------------------------------------------------------------------------------\n\n",
        )
        self.chat_box.configure(state="disabled")

        # Input & Voice Action Bar
        input_container = ctk.CTkFrame(self.main_frame, fg_color="#1F2937", corner_radius=8)
        input_container.grid(row=1, column=0, sticky="ew")
        input_container.grid_columnconfigure(0, weight=1)

        # Top toolbar inside input bar: STT buttons + status
        stt_bar = ctk.CTkFrame(input_container, fg_color="transparent")
        stt_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 4))

        self.mic_btn = ctk.CTkButton(
            stt_bar,
            text="🎤 Record Mic (STT)",
            width=165,
            height=32,
            fg_color="#7C3AED",
            hover_color="#6D28D9",
            command=self._toggle_mic_recording,
        )
        self.mic_btn.pack(side="left", padx=(0, 8))

        self.audio_file_btn = ctk.CTkButton(
            stt_bar,
            text="🎵 Transcribe Audio File",
            width=175,
            height=32,
            fg_color="#4B5563",
            hover_color="#374151",
            command=self._transcribe_audio_file_dialog,
        )
        self.audio_file_btn.pack(side="left", padx=(0, 8))

        self.quick_attach_btn = ctk.CTkButton(
            stt_bar,
            text="📎 Attach File",
            width=120,
            height=32,
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            command=self._upload_files_dialog,
        )
        self.quick_attach_btn.pack(side="left", padx=(0, 8))

        self.stt_status_lbl = ctk.CTkLabel(
            stt_bar,
            text="STT: Loading local Whisper...",
            font=ctk.CTkFont(size=12),
            text_color="#9CA3AF",
        )
        self.stt_status_lbl.pack(side="right", padx=6)

        # Prompt entry textbox
        self.prompt_entry = ctk.CTkTextbox(
            input_container,
            height=85,
            wrap="word",
            font=ctk.CTkFont(size=14),
            fg_color="#111827",
        )
        self.prompt_entry.grid(row=1, column=0, sticky="ew", padx=(10, 8), pady=(4, 10))
        self.prompt_entry.bind("<Control-Return>", lambda e: self._on_send_clicked())
        self.prompt_entry.bind("<Shift-Return>", lambda e: None)
        self.prompt_entry.bind("<Return>", self._on_enter_key)

        # Send / Stop button column
        send_col = ctk.CTkFrame(input_container, fg_color="transparent")
        send_col.grid(row=1, column=1, sticky="ns", padx=(0, 10), pady=(4, 10))

        self.send_btn = ctk.CTkButton(
            send_col,
            text="Send\n(Enter)",
            width=105,
            height=52,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            command=self._on_send_clicked,
        )
        self.send_btn.pack(side="top", pady=(0, 5))

        self.stop_btn = ctk.CTkButton(
            send_col,
            text="Stop",
            width=105,
            height=28,
            fg_color="#6B7280",
            hover_color="#4B5563",
            state="disabled",
            command=self._stop_generation,
        )
        self.stop_btn.pack(side="bottom")

    # ================= BACKEND & STT INITIALIZATION =================

    def _init_backend_and_stt(self):
        # 1. Pre-load local Whisper STT on CPU
        try:
            self.stt.ensure_loaded()
            self.after(
                0,
                lambda: self.stt_status_lbl.configure(
                    text="STT Ready (Whisper-Base Offline CPU)", text_color="#34D399"
                ),
            )
        except Exception as exc:
            self.after(
                0,
                lambda: self.stt_status_lbl.configure(
                    text=f"STT Error: {exc}", text_color="#F87171"
                ),
            )

        # 2. Ensure Qwen3-30B-A3B llama-server is running
        self._ensure_llama_server_running()

    def _is_server_healthy(self) -> bool:
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    def _ensure_llama_server_running(self):
        with self._server_lock:
            if self._is_server_healthy():
                self.after(0, self._set_engine_online_ui)
                return

            self.after(
                0,
                lambda: self.engine_status_lbl.configure(
                    text="Engine: Loading Model (~25s)...", text_color="#FBBF24"
                ),
            )

            use_spec = bool(self.spec_draft_var.get()) and DRAFT_GGUF_PATH.exists()
            ncmoe_layers = "44" if use_spec else "41"

            args = [
                str(LLAMA_SERVER_EXE),
                "-m", str(MODEL_GGUF_PATH),
                "--alias", "Qwen3-30B-A3B",
                "--load-mode", "none",
                "-ngl", "99",
                "-ncmoe", ncmoe_layers,
                "-fa", "on",
                "-ctk", "q8_0",
                "-ctv", "q8_0",
                "-np", "1",
                "--keep", "-1",
                "--cache-reuse", "256",
                "--no-webui",
                "-c", "8192",
                "-ub", "1024",
                "-t", "8",
                "-tb", "16",
                "-Cr", "0-15",
                "--prio", "2",
                "--prio-batch", "2",
                "--poll", "100",
                "--override-kv", f"qwen3moe.expert_used_count=int:{self.active_experts}",
                "--temp", "0.6",
                "--top-p", "0.95",
                "--top-k", "20",
                "--min-p", "0.06",
                "--repeat-penalty", "1.1",
                "--presence-penalty", "0.4",
                "--host", SERVER_HOST,
                "--port", str(SERVER_PORT),
            ]
            if use_spec:
                args.extend([
                    "-md", str(DRAFT_GGUF_PATH),
                    "-ngld", "99",
                    "-ctkd", "q8_0",
                    "-ctvd", "q8_0",
                    "--spec-type", "draft-simple",
                    "--spec-draft-sampling", "probabilistic",
                    "--spec-draft-n-max", "3",
                    "--spec-draft-n-min", "1",
                    "--spec-draft-p-min", "0.80",
                ])

            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.server_proc = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
            self.we_started_server = True

            for _ in range(90):
                if self._is_server_healthy():
                    # Run instant 1-token warmup so system prompt prefix is hot in VRAM
                    try:
                        requests.post(
                            f"{API_BASE_URL}/v1/chat/completions",
                            json={
                                "model": "Qwen3-30B-A3B",
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": SYSTEM_PROMPT,
                                    },
                                    {"role": "user", "content": "/no_think hi"},
                                ],
                                "chat_template_kwargs": {"enable_thinking": False},
                                "max_tokens": 1,
                            },
                            timeout=15,
                        )
                    except Exception:
                        pass
                    self.after(0, self._set_engine_online_ui)
                    return
                time.sleep(1.0)

            self.after(
                0,
                lambda: self.engine_status_lbl.configure(
                    text="Engine: Startup Timeout", text_color="#F87171"
                ),
            )

    def _set_engine_online_ui(self):
        spec_tag = "+0.6B Speculative" if self.spec_draft_var.get() else "Direct"
        self.engine_status_lbl.configure(
            text=f"Engine: Online (Top-{self.active_experts} {spec_tag})", text_color="#34D399"
        )
        self.toggle_engine_btn.configure(
            text="Unload (Free RAM)", fg_color="#B91C1C", hover_color="#991B1B"
        )

    def _on_spec_draft_toggled(self):
        if self._is_server_healthy():
            self.engine_status_lbl.configure(
                text="Engine: Restarting Speculative Mode...", text_color="#FBBF24"
            )
            def _restart():
                subprocess.run(
                    ["taskkill", "/F", "/IM", "llama-server.exe"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                time.sleep(1.0)
                self._ensure_llama_server_running()
            threading.Thread(target=_restart, daemon=True).start()

    def _toggle_engine_power(self):
        if self._is_server_healthy():
            subprocess.run(
                ["taskkill", "/F", "/IM", "llama-server.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.server_proc = None
            self.engine_status_lbl.configure(
                text="Engine: Unloaded (0 GB Used)", text_color="#9CA3AF"
            )
            self.toggle_engine_btn.configure(
                text="Load Model", fg_color="#059669", hover_color="#047857"
            )
        else:
            threading.Thread(target=self._ensure_llama_server_running, daemon=True).start()

    def _on_expert_mode_selected(self, choice: str):
        if "Top-4" in choice:
            self.active_experts = 4
            self.spec_draft_var.set(False)
        elif "Top-5" in choice:
            self.active_experts = 5
            self.spec_draft_var.set(False)
        elif "Top-6" in choice:
            self.active_experts = 6
            self.spec_draft_var.set(True)
        else:
            self.active_experts = 8
            self.spec_draft_var.set(True)

        if self._is_server_healthy():
            self.engine_status_lbl.configure(
                text=f"Engine: Switching to Top-{self.active_experts} Experts...",
                text_color="#FBBF24",
            )
            def _restart():
                subprocess.run(
                    ["taskkill", "/F", "/IM", "llama-server.exe"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                time.sleep(1.0)
                self._ensure_llama_server_running()

            threading.Thread(target=_restart, daemon=True).start()

    # ================= HARDWARE MONITOR =================

    def _schedule_hw_monitor(self):
        threading.Thread(target=self._poll_hw_stats, daemon=True).start()
        self.after(4000, self._schedule_hw_monitor)

    def _poll_hw_stats(self):
        if self._hw_poll_running:
            return
        self._hw_poll_running = True
        vram_str = "--"
        ram_str = "0.0 GB"
        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                creationflags=creation_flags,
                timeout=2,
            ).decode("utf-8", errors="ignore").strip()
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 2:
                vram_str = f"{parts[0]} / {parts[1]} MiB"
        except Exception:
            pass

        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            tl_out = subprocess.check_output(
                [
                    "tasklist",
                    "/FI",
                    "IMAGENAME eq llama-server.exe",
                    "/FO",
                    "CSV",
                    "/NH",
                ],
                creationflags=creation_flags,
                timeout=2,
            ).decode("utf-8", errors="ignore").strip()
            if "llama-server.exe" in tl_out:
                cols = [c.strip('"') for c in tl_out.splitlines()[0].split('","')]
                if len(cols) >= 5:
                    kb_str = cols[4].upper().replace("K", "").replace(",", "").replace(" ", "").strip()
                    ram_gb = float(kb_str) / (1024.0 * 1024.0)
                    ram_str = f"{ram_gb:.2f} GB"
        except Exception:
            pass

        last_speed = getattr(self, "last_tok_per_sec", None)
        speed_str = f"{last_speed:.1f} tok/s" if last_speed else "Idle"

        text = f"VRAM (T1000): {vram_str}\nModel RAM:    {ram_str}\nLast Speed:   {speed_str}"
        self.after(0, lambda: self.hw_stats_lbl.configure(text=text))
        self._hw_poll_running = False

    # ================= SPEECH-TO-TEXT (STT) HANDLERS =================

    def _toggle_mic_recording(self):
        if not self.stt.is_recording:
            try:
                self.stt.start_recording()
                self.mic_btn.configure(
                    text="⏹ Stop & Transcribe",
                    fg_color="#DC2626",
                    hover_color="#B91C1C",
                )
                self.stt_status_lbl.configure(
                    text="🔴 Recording Microphone... Speak now", text_color="#F87171"
                )
            except Exception as exc:
                messagebox.showerror("Microphone Error", f"Could not access microphone:\n{exc}")
        else:
            self.mic_btn.configure(state="disabled", text="Transcribing...")
            self.stt_status_lbl.configure(
                text="⏳ Transcribing voice locally...", text_color="#FBBF24"
            )
            threading.Thread(target=self._finish_mic_transcription, daemon=True).start()

    def _finish_mic_transcription(self):
        try:
            transcript = self.stt.stop_and_transcribe()
            self.after(0, lambda: self._on_transcript_ready(transcript))
        except Exception as exc:
            self.after(
                0,
                lambda: self._on_transcript_error(str(exc)),
            )

    def _on_transcript_ready(self, transcript: str):
        self.mic_btn.configure(
            state="normal",
            text="🎤 Record Mic (STT)",
            fg_color="#7C3AED",
            hover_color="#6D28D9",
        )
        if not transcript:
            self.stt_status_lbl.configure(
                text="STT: No speech detected", text_color="#9CA3AF"
            )
            return

        self.stt_status_lbl.configure(
            text=f"STT Transcribed ({len(transcript.split())} words)",
            text_color="#34D399",
        )
        current = self.prompt_entry.get("1.0", "end").strip()
        new_text = f"{current} {transcript}".strip() if current else transcript
        self.prompt_entry.delete("1.0", "end")
        self.prompt_entry.insert("1.0", new_text)

        if self.auto_send_voice_var.get():
            self._on_send_clicked()

    def _on_transcript_error(self, err_msg: str):
        self.mic_btn.configure(
            state="normal",
            text="🎤 Record Mic (STT)",
            fg_color="#7C3AED",
            hover_color="#6D28D9",
        )
        self.stt_status_lbl.configure(
            text=f"STT Error: {err_msg}", text_color="#F87171"
        )

    def _transcribe_audio_file_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File for Offline Transcription",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg *.webm *.mp4"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        self.stt_status_lbl.configure(
            text=f"⏳ Transcribing {Path(file_path).name}...", text_color="#FBBF24"
        )
        self.audio_file_btn.configure(state="disabled")

        def _worker():
            try:
                transcript = self.stt.transcribe_audio_file(file_path)
                from file_tools import build_chunks_for_text
                text_val = transcript or "[No speech detected in audio file]"
                doc = AttachedFile(
                    name=f"[Audio Transcript] {Path(file_path).name}",
                    path=file_path,
                    size_bytes=Path(file_path).stat().st_size,
                    file_type="audio",
                    content=text_val,
                    char_count=len(text_val),
                    summary_info=f"Audio STT ({len(text_val.split())} words)",
                    chunks=build_chunks_for_text(Path(file_path).name, text_val),
                )
                self.attached_docs.append(doc)
                self.after(0, self._refresh_files_list)
                self.after(
                    0,
                    lambda: self._on_audio_file_done(Path(file_path).name, transcript),
                )
            except Exception as exc:
                self.after(0, lambda: self._on_transcript_error(str(exc)))
            finally:
                self.after(0, lambda: self.audio_file_btn.configure(state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_audio_file_done(self, filename: str, transcript: str):
        self.stt_status_lbl.configure(
            text=f"Transcribed {filename} ({len(transcript.split())} words)",
            text_color="#34D399",
        )
        self._append_chat(
            f"🎵 [Offline STT Audio Attached: {filename}]\n"
            f"Transcript Preview: {transcript[:400]}{'...' if len(transcript) > 400 else ''}\n\n"
        )

    # ================= FILE UPLOAD & CONTEXT MANAGEMENT =================

    def _upload_files_dialog(self):
        paths = filedialog.askopenfilenames(
            title="Upload Files to Local AI Context",
            filetypes=[
                (
                    "Supported Documents & Code",
                    "*.pdf *.docx *.xlsx *.xlsm *.csv *.tsv *.txt *.md *.json *.yaml *.yml *.xml *.log *.py *.js *.ts *.cs *.cpp *.c *.h *.sql *.ps1 *.html *.css",
                ),
                ("PDF Files", "*.pdf"),
                ("Word Documents", "*.docx"),
                ("Excel & CSV", "*.xlsx *.xlsm *.csv *.tsv"),
                ("All Files", "*.*"),
            ],
        )
        if not paths:
            return

        for p in paths:
            try:
                doc = load_and_extract_file(p)
                self.attached_docs = [d for d in self.attached_docs if d.path != doc.path]
                self.attached_docs.append(doc)
                self._append_chat(
                    f"📎 [Attached File: {doc.name} | {doc.summary_info} ({doc.char_count:,} chars)]\n\n"
                )
            except Exception as exc:
                messagebox.showerror("File Parse Error", f"Failed to read {p}:\n{exc}")

        self._refresh_files_list()

    def _refresh_files_list(self):
        for widget in self.files_scroll.winfo_children():
            widget.destroy()

        total_chars = sum(d.char_count for d in self.attached_docs)
        mode_tag = "BM25 Top-5 RAG" if total_chars > 4500 else "Full Context"
        self.files_scroll.configure(
            label_text=f"Attached ({len(self.attached_docs)}) • {mode_tag}"
        )

        for idx, doc in enumerate(self.attached_docs):
            row = ctk.CTkFrame(self.files_scroll, fg_color="#111827")
            row.pack(fill="x", pady=3, padx=2)

            lbl = ctk.CTkLabel(
                row,
                text=f"{doc.name}\n{doc.summary_info}",
                font=ctk.CTkFont(size=11),
                justify="left",
                anchor="w",
            )
            lbl.pack(side="left", padx=6, pady=4, fill="x", expand=True)

            rm_btn = ctk.CTkButton(
                row,
                text="✕",
                width=26,
                height=26,
                fg_color="#7F1D1D",
                hover_color="#991B1B",
                command=lambda i=idx: self._remove_file_at(i),
            )
            rm_btn.pack(side="right", padx=4, pady=4)

    def _remove_file_at(self, idx: int):
        if 0 <= idx < len(self.attached_docs):
            removed = self.attached_docs.pop(idx)
            self._append_chat(f"🗑️ [Removed attachment: {removed.name}]\n\n")
            self._refresh_files_list()

    def _clear_all_files(self):
        if self.attached_docs:
            self.attached_docs.clear()
            self._refresh_files_list()
            self._append_chat("🗑️ [Cleared all attached files]\n\n")

    # ================= CHAT & STREAMING COMPLETION =================

    def _on_enter_key(self, event):
        if not (event.state & 0x0001):  # Not Shift+Enter
            self._on_send_clicked()
            return "break"
        return None

    def _reset_conversation(self):
        self.chat_history.clear()
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0", "end")
        self.chat_box.insert(
            "end",
            "Conversation cleared. Ready for new prompts, voice dictation, or file uploads.\n"
            "--------------------------------------------------------------------------------\n\n",
        )
        self.chat_box.configure(state="disabled")

    def _append_chat(self, text: str):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", text)
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def _stop_generation(self):
        self.stop_generation_flag = True

    def _on_send_clicked(self):
        if self.is_generating:
            return
        user_text = self.prompt_entry.get("1.0", "end").strip()
        if not user_text and not self.attached_docs:
            return
        if not user_text and self.attached_docs:
            user_text = "Please summarize and analyze the attached file(s)."

        self.prompt_entry.delete("1.0", "end")
        self.is_generating = True
        self.stop_generation_flag = False
        self.send_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal", fg_color="#DC2626")

        # Format visible chat message
        self._append_chat(f"🧑 You:\n{user_text}\n\n🤖 Qwen3-30B-A3B:\n")

        threading.Thread(
            target=self._stream_llm_response, args=(user_text,), daemon=True
        ).start()

    def _stream_llm_response(self, user_text: str):
        try:
            if not self._is_server_healthy():
                self.after(0, lambda: self._append_chat("[Starting local engine, please wait...]\n"))
                self._ensure_llama_server_running()

            think_enabled = bool(self.think_mode_var.get())
            think_prefix = "/think " if think_enabled else "/no_think "

            # Build focused prompt context using Smart Offline BM25 RAG (Top-5 chunks for large files)
            docs_block = ""
            if self.attached_docs:
                docs_block = format_attachments_for_prompt(self.attached_docs, user_query=user_text)
                full_user_content = (
                    f"{think_prefix}{docs_block}\n\n"
                    "STRICT GROUNDING RULE: Base your answer strictly on the attached file sections above. "
                    "Cite the file/section name for facts and numbers, and if an answer is not present in the files, explicitly state so.\n\n"
                    f"User Request: {user_text}"
                )
                # Lower temperature + stricter min_p when answering from documents to prevent hallucinations
                req_temp = 0.2
                req_min_p = 0.08
            else:
                full_user_content = f"{think_prefix}{user_text}"
                req_temp = 0.6
                req_min_p = 0.06

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ]
            messages.extend(self.chat_history[-10:])
            messages.append({"role": "user", "content": full_user_content})

            payload = {
                "model": "Qwen3-30B-A3B",
                "messages": messages,
                "stream": True,
                "temperature": req_temp,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": req_min_p,
                "repeat_penalty": 1.1,
                "presence_penalty": 0.4,
                "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": think_enabled},
            }

            t_start = time.perf_counter()
            think_start_time: Optional[float] = None
            token_count = 0
            think_tokens = 0
            assistant_reply: List[str] = []

            in_think_block = False
            answer_started = False
            think_badge_printed = False
            pre_answer_buf = ""
            think_buf = ""
            latex_buf = ""

            def _needs_more_latex_context(s: str) -> bool:
                if len(s) > 80:
                    return False
                if s.endswith("$") and not s.endswith("$$"):
                    return True
                if "\\" in s:
                    tail = s[s.rfind("\\"):]
                    if re.match(r"^\\[a-zA-Z]*$", tail):
                        return True
                    if tail.startswith("\\frac") and tail.count("{") < 2:
                        return True
                    if tail.count("{") > tail.count("}"):
                        return True
                return False

            def _emit_clean_answer_piece(raw_piece: str, flush: bool = False):
                nonlocal answer_started, think_badge_printed, latex_buf
                if raw_piece:
                    latex_buf += raw_piece
                if not latex_buf:
                    return
                if not flush and _needs_more_latex_context(latex_buf):
                    return
                clean_text = clean_latex_and_mojibake(latex_buf)
                latex_buf = ""
                if not clean_text:
                    return
                if not answer_started:
                    clean_text = clean_text.lstrip("\r\n")
                    if not clean_text:
                        return
                    answer_started = True
                    if think_enabled and think_tokens > 0 and not think_badge_printed:
                        think_elapsed = max(0.1, time.perf_counter() - (think_start_time or t_start))
                        badge = f"[🧠 Thought for {think_elapsed:.1f}s • {think_tokens} tokens]\n"
                        self.after(0, lambda b=badge: self._append_chat(b))
                        think_badge_printed = True
                    self.after(
                        0,
                        lambda: self.stt_status_lbl.configure(
                            text="⚡ Streaming response...", text_color="#60A5FA"
                        ),
                    )
                assistant_reply.append(clean_text)
                self.after(0, lambda p=clean_text: self._append_chat(p))

            with requests.post(
                f"{API_BASE_URL}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                resp.raise_for_status()
                resp.encoding = "utf-8"
                for raw_bytes in resp.iter_lines(decode_unicode=False):
                    if self.stop_generation_flag:
                        self.after(0, lambda: self._append_chat("\n[Generation stopped by user]"))
                        break
                    if not raw_bytes:
                        continue
                    raw_line = raw_bytes.decode("utf-8", errors="replace")
                    if not raw_line.startswith("data: "):
                        continue
                    data_str = raw_line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        reasoning_piece = delta.get("reasoning_content") or ""
                        content_piece = delta.get("content") or ""

                        if reasoning_piece:
                            token_count += 1
                            if reasoning_piece.strip():
                                if think_start_time is None:
                                    think_start_time = time.perf_counter()
                                think_tokens += 1
                                if think_enabled and (think_tokens == 1 or think_tokens % 4 == 0):
                                    self.after(
                                        0,
                                        lambda n=think_tokens: self.stt_status_lbl.configure(
                                            text=f"🧠 Reasoning... ({n} tokens)",
                                            text_color="#FBBF24",
                                        ),
                                    )

                        if content_piece:
                            token_count += 1
                            if in_think_block:
                                think_buf += content_piece
                                if "</think>" in think_buf:
                                    before_close, after_close = think_buf.split("</think>", 1)
                                    if before_close.strip():
                                        think_tokens += 1
                                    in_think_block = False
                                    think_buf = ""
                                    _emit_clean_answer_piece(after_close.replace("<think>", ""))
                                else:
                                    if content_piece.strip():
                                        if think_start_time is None:
                                            think_start_time = time.perf_counter()
                                        think_tokens += 1
                                        if think_enabled and (think_tokens == 1 or think_tokens % 4 == 0):
                                            self.after(
                                                0,
                                                lambda n=think_tokens: self.stt_status_lbl.configure(
                                                    text=f"🧠 Reasoning... ({n} tokens)",
                                                    text_color="#FBBF24",
                                                ),
                                            )
                            elif not answer_started:
                                pre_answer_buf += content_piece
                                stripped = pre_answer_buf.lstrip()
                                if not stripped:
                                    continue
                                if "<think>".startswith(stripped) and len(stripped) < len("<think>"):
                                    continue
                                if stripped.startswith("<think>"):
                                    in_think_block = True
                                    if think_start_time is None:
                                        think_start_time = time.perf_counter()
                                    think_buf = stripped[len("<think>"):]
                                    pre_answer_buf = ""
                                    if "</think>" in think_buf:
                                        before_close, after_close = think_buf.split("</think>", 1)
                                        if before_close.strip():
                                            think_tokens += max(1, len(before_close.split()))
                                        in_think_block = False
                                        think_buf = ""
                                        _emit_clean_answer_piece(after_close.replace("<think>", ""))
                                else:
                                    pre_answer_buf = ""
                                    _emit_clean_answer_piece(stripped.replace("</think>", ""))
                            else:
                                clean_piece = content_piece.replace("<think>", "").replace("</think>", "")
                                _emit_clean_answer_piece(clean_piece)
                    except Exception:
                        continue

            if pre_answer_buf.strip() and not in_think_block:
                _emit_clean_answer_piece(
                    pre_answer_buf.replace("<think>", "").replace("</think>", ""),
                    flush=True,
                )
            elif latex_buf:
                _emit_clean_answer_piece("", flush=True)

            elapsed = max(0.01, time.perf_counter() - t_start)
            self.last_tok_per_sec = token_count / elapsed

            draft_text = "".join(assistant_reply).strip()
            docs_context_str = docs_block if self.attached_docs else ""

            # Run <2ms Deterministic Symbolic & AST Grounding Audit on Pass-1 Draft
            audit = run_symbolic_audit(draft_text, docs_context_str, user_text)
            self.after(
                0,
                lambda a=audit, tc=token_count, el=elapsed, tps=self.last_tok_per_sec: self._append_chat(
                    f"\n\n🔍 [Verified in <2ms | {tc} tokens in {el:.1f}s (~{tps:.1f} tok/s) | Symbolic Audit: {a.audit_summary} | Grounding: {a.grounding_score_pct}%]\n"
                ),
            )

            # Pass-2 Neural Critic Verification Loop (only runs when Symbolic/AST Audit flags an issue < 100%)
            final_text = draft_text
            if (
                self.critic_verify_var.get()
                and audit.grounding_score_pct < 100.0
                and not self.stop_generation_flag
                and draft_text
            ):
                self.after(
                    0,
                    lambda: self._append_chat(
                        "\n🛡️ [Pass-2 Critic Auto-Repair (temp=0.10, min_p=0.10): Fixing flagged claims/syntax...]\n"
                    ),
                )
                critic_msgs = build_critic_verification_messages(
                    user_query=user_text,
                    source_context=docs_context_str,
                    draft_reply=draft_text,
                    audit=audit,
                )
                critic_payload = {
                    "model": "Qwen3-30B-A3B",
                    "messages": critic_msgs,
                    "stream": True,
                    "temperature": 0.10,
                    "top_p": 0.90,
                    "top_k": 10,
                    "min_p": 0.10,
                    "repeat_penalty": 1.1,
                    "presence_penalty": 0.4,
                    "max_tokens": 1536,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                c_start = time.perf_counter()
                c_tokens = 0
                critic_accum: List[str] = []
                c_in_think = False
                c_buf = ""
                with requests.post(
                    f"{API_BASE_URL}/v1/chat/completions",
                    json=critic_payload,
                    stream=True,
                    timeout=300,
                ) as c_resp:
                    c_resp.raise_for_status()
                    c_resp.encoding = "utf-8"
                    for raw_bytes in c_resp.iter_lines(decode_unicode=False):
                        if self.stop_generation_flag:
                            break
                        if not raw_bytes:
                            continue
                        raw_line = raw_bytes.decode("utf-8", errors="replace")
                        if not raw_line.startswith("data: "):
                            continue
                        data_str = raw_line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            c_piece = delta.get("content") or ""
                            if not c_piece:
                                continue
                            c_tokens += 1
                            if "<think>" in c_piece:
                                c_in_think = True
                                c_buf = c_piece.split("<think>", 1)[1]
                                continue
                            if c_in_think:
                                c_buf += c_piece
                                if "</think>" in c_buf:
                                    c_in_think = False
                                    c_piece = c_buf.split("</think>", 1)[1].lstrip("\r\n")
                                    c_buf = ""
                                else:
                                    continue
                            clean_c = clean_latex_and_mojibake(
                                c_piece.replace("<think>", "").replace("</think>", "")
                            )
                            if clean_c:
                                critic_accum.append(clean_c)
                                self.after(0, lambda p=clean_c: self._append_chat(p))
                        except Exception:
                            continue

                c_elapsed = max(0.01, time.perf_counter() - c_start)
                self.last_tok_per_sec = c_tokens / c_elapsed
                verified_reply = "".join(critic_accum).strip()
                if verified_reply:
                    final_text = verified_reply
                self.after(
                    0,
                    lambda ct=c_tokens, cel=c_elapsed, ctps=self.last_tok_per_sec: self._append_chat(
                        f"\n\n✅ [Pass-2 Verified Complete: {ct} tokens in {cel:.1f}s • ~{ctps:.1f} tok/s]\n"
                        "--------------------------------------------------------------------------------\n\n"
                    ),
                )
            else:
                self.after(
                    0,
                    lambda: self._append_chat(
                        "--------------------------------------------------------------------------------\n\n"
                    ),
                )

            self.chat_history.append({"role": "user", "content": user_text})
            self.chat_history.append({"role": "assistant", "content": final_text})
        except Exception as exc:
            self.after(0, lambda e=str(exc): self._append_chat(f"\n[Error: {e}]\n\n"))
        finally:
            self.is_generating = False
            self.after(
                0,
                lambda: (
                    self.send_btn.configure(state="normal"),
                    self.stop_btn.configure(state="disabled", fg_color="#6B7280"),
                    self.stt_status_lbl.configure(
                        text="STT Ready (Whisper-Base Offline CPU)", text_color="#34D399"
                    ),
                ),
            )

    def _on_close(self):
        if self.we_started_server and self.server_proc is not None:
            try:
                self.server_proc.terminate()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    app = LocalAIDesktopApp()
    app.mainloop()
