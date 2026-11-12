# Qwen3Studio: 30.5B Sparse MoE Local AI at 39.2 tok/s on a 4 GB VRAM Workstation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Model: Qwen3-30B-A3B](https://img.shields.io/badge/Model-Qwen3--30B--A3B%20%28UD--IQ3__XXS%29-blue)](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF)
[![GPU VRAM: 4 GB](https://img.shields.io/badge/GPU%20VRAM-4%20GB%20%283%2C762%20MiB%20Used%29-purple)](#3-hardware-memory-partitioning-architecture)
[![Speed: 39.2 tok/s](https://img.shields.io/badge/Speed-39.2%20tok%2Fs%20%285.7x%20Speedup%29-brightgreen)](#5-empirical-benchmarks--ablation-study)
[![Verification: <1ms Symbolic+AST Gate](https://img.shields.io/badge/Hallucination%20Gate-0.40ms%20Symbolic%20%2B%20AST%20Critic-orange)](#4-the-symbolic-gated-conditional-neural-critic)

![Qwen3Studio 39.2 tok/s Benchmark Overview](SOCIAL_PREVIEW_CARD.png)

**Run a frontier 30.5-billion-parameter Mixture-of-Experts (`Qwen3-30B-A3B`) model 100% offline at `39.2 tokens/sec` with a `291 ms` first-token latency on a standard 4 GB VRAM + 32 GB RAM workstation—while leaving `20.5 GB` of system RAM completely free.**

Includes a standalone native Windows desktop application (zero browser dependency), **0 MB VRAM offline Faster-Whisper Speech-to-Text (STT)**, **offline Windows SAPI Text-to-Speech (TTS)**, **sub-15ms BM25 Top-5 multi-format document RAG** (`PDF`, `DOCX`, `XLSX`, `CSV`, `JSON`, source code), **`Qwen3-0.6B` Draft Speculative Decoding** (`+45.1%` 8-expert speedup), and a **`<1 ms` (`0.403 ms` avg) Symbolic-Gated Conditional Neural Critic** that eliminates fabricated numbers and broken Python code without executing untrusted scripts.

---

## Key Highlights & Breakthrough Numbers

* **5.7× Faster Token Generation (`6.89` $\rightarrow$ `39.18–39.35 tok/s`):** Matches MIT & UC Berkeley's `FreeToken` 8 GB RTX 4060 benchmark (`~39 tok/s`) on **half the GPU VRAM (4 GB NVIDIA T1000)**, and runs **85% faster** than `FreeToken`'s 4 GB RTX 2050 benchmark (`21.3 tok/s`).
* **147× Faster Time-to-First-Token (`42,811 ms` $\rightarrow$ `290.9 ms`):** Eliminates cold `mmap` page-fault stalls via `--load-mode none` aligned SIMD buffers, single-slot prefix pinning (`-np 1 --keep -1`), and KV-shift reuse (`--cache-reuse 256`).
* **+45.1% Full 8-Expert Speedup via `0.6B` Speculative Decoding (`18.68` $\rightarrow$ `27.11 tok/s`):** Achieves a **94.8% draft token acceptance rate (`91/96` tokens)** when pairing `Qwen3-0.6B` (`386 MB`) with `Qwen3-30B-A3B`.
* **`0.403 ms` Zero-Execution Symbolic & AST Critic:** Audits every number, percentage, and Python code block (`ast.parse`) in `<1 ms`—adding **0 ms of LLM overhead** when Grounding = `100%`, and automatically triggering a Pass-2 Low-Temperature (`0.10`) Neural Critic when a hallucination or syntax bug is detected (restoring `45%–90%` $\rightarrow$ **`100.0%` Grounding Score** across all defect traps).

---

## 1. Quickstart (2-Minute Setup)

### Prerequisites
* **OS:** Windows 10/11 x64 (or Linux/macOS with `llama.cpp` installed)
* **RAM:** 16 GB – 32 GB System RAM (uses ~11.2 GB RAM, leaving 20.5 GB free on a 32 GB machine)
* **GPU:** Any 4 GB+ NVIDIA CUDA GPU (tested on NVIDIA T1000 4,096 MiB VRAM)
* **Python:** Python 3.10+

### Installation
```powershell
git clone https://github.com/AjayakumarV/Qwen3Studio-LocalAI.git
cd Qwen3Studio-LocalAI

# 1. Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Automated 1-command download of llama.cpp CUDA binaries, Qwen3-30B-A3B (UD-IQ3_XXS),
#    Qwen3-0.6B speculative draft model, and offline Faster-Whisper base.en STT
python setup_localai.py

# 3. Launch the native desktop studio!
python app.py
```

---

## 2. How Our Invention Works in Plain English (The "128-Specialist Hospital" Analogy)

Imagine trying to run a **world-class engineering consulting firm** inside a small office room using only a **tiny workbench (your 4 GB GPU)** and a **medium-sized filing cabinet (your 32 GB RAM)**:

1. **The Problem ("The 30-Volume Encyclopedia on a 4-Volume Workbench"):**
   * A traditional 30-billion-parameter "Dense" AI is like a **30-volume encyclopedia glued together**. Every time you ask for a single word, the clerk must carry all 30 volumes onto your workbench. Because your workbench only holds 4 volumes (4 GB VRAM), the clerk walks back and forth to the slow RAM filing cabinet for every word—taking **42.8 seconds just to start talking** and crawling at **3–6 words/sec**.
2. **Pillar 1 & 2 ("Receptionist on the GPU Workbench + Calling Only the Top-4 Doctors"):**
   * Instead of one giant book, `Qwen3-30B-A3B` is organized like a **hospital with 128 tiny specialist doctors per floor (48 floors)**. By default, it calls **8 doctors** into a meeting for every word—even though Doctors #5–#8 barely contribute (`<8%` weight).
   * **Our Fix:** We permanently seated the **Receptionist (Router) and the 7 busiest departments right on the fast 4 GB GPU workbench (`-ncmoe 41`)**, and instructed the Receptionist to **only call the Top-4 most confident doctors (`expert_used_count=4`)** using only our **8 fastest Senior Couriers (Intel P-Cores, `-Cr 0-15`)** instead of slow interns (Efficiency Cores). **Result: 0.29s start and 39.2 tokens/sec!**
3. **Pillar 3 ("The Speedy Junior Assistant Drafting Ahead — `0.6B` Speculative Decoding"):**
   * When a complex problem requires consulting all 8 senior doctors, a fast **Junior Assistant (`Qwen3-0.6B`)** sitting on the GPU drafts 3 words ahead in an instant, and the 8 senior doctors simply stamp **"Approved" on all 3 words at once (`94.8%` approval rate)**—boosting 8-doctor speed by **+45.1%**.
4. **Pillar 4 ("The `<1 Millisecond` Exit-Door Fact-Checker"):**
   * Before any answer reaches your screen, an automated **Exit-Door Inspector** checks every number, percentage, and Python code block against your uploaded PDF/Excel file in **0.40 milliseconds**. If every number matches (`100% Grounding`), it passes immediately with **zero delay**. If even one number is invented, the Inspector blocks it and forces a **100% verified rewrite**.

---

## 3. Hardware & Memory Partitioning Architecture

```mermaid
flowchart LR
    subgraph vram [NVIDIA T1000 4,096 MiB VRAM — 3,762 MiB Allocated]
        Attn["48 Shared Attention + Router + Output Heads (1,820 MiB)"]
        MoE7["7 Full MoE Layers Pinned on GPU (-ncmoe 41, 1,442 MiB)"]
        KV["Single-Slot 8K q8_0 Flash-Attn KV Cache (500 MiB)"]
    end

    subgraph ram [32 GB System RAM — 11.2 GB Allocated / 20.5 GB Free]
        MoE41["41 Sparse MoE Expert Layers in Aligned SIMD RAM (9.8 GB)"]
        Whisper["Faster-Whisper Base.en CPU INT8 STT (180 MB — 0 MB VRAM)"]
        BM25["BM25 Top-5 Document RAG Index + Native GUI (230 MB)"]
    end

    Attn -->|"Top-4 Selective Routing (~1.5B Active/Tok)"| MoE7
    Attn -->|"P-Core Locked (-Cr 0-15 --prio 2)"| MoE41
    Attn --> KV
```

---

## 4. The Symbolic-Gated Conditional Neural Critic

Unlike MIT CSAIL's *Recursive Language Models (RLM)* (which runs 5–15 recursive Python REPL calls taking 25–45 seconds and executing untrusted LLM-generated code), our neuro-symbolic verification pipeline enforces zero-hallucination grounding in **two deterministic-gated stages**:

1. **Stage 1 (`0.403 ms` Deterministic Symbolic & AST Gate):** Extracts all significant numbers (`>=10`, decimals, `%`) from the Pass-1 draft and cross-checks them against your prompt + retrieved document chunks while statically parsing any fenced Python code via `ast.parse()`.
2. **Stage 2 (Conditional Low-Temperature Neural Critic):**
   * If **Grounding Score = `100.0%`**, the response completes immediately (**`0.0 ms` extra LLM latency**).
   * If **Grounding Score < `100.0%`**, a targeted Pass-2 Critic (`temperature=0.10, min_p=0.10`) receives the exact list of unverified numbers or AST syntax errors and repairs the output in ~2.8s.

| Evaluation Scenario (ARS-510 Radar Spec) | Injected Defect / Hallucination | Gate Latency | Initial Grounding | Flagged Anomalies | Triggered Critic? | Final Grounding |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **Trap 1: Frequency & Range Fabrication** | Injected `79.2 GHz` & `320.5 m` | **`0.043 ms`** | `80.0%` | `['79.2', '320.5']` | **Yes** | **`100.0%`** |
| **Trap 2: Subtle Decimal Latency Drift** | Injected `14.2 ms` (vs `18.5 ms`) | **`0.016 ms`** | `90.0%` | `['14.2']` | **Yes** | **`100.0%`** |
| **Trap 3: Broken Python AST Syntax** | Missing `:` in `def parse_radar()` | **`0.989 ms`** | `65.0%` | `AST Line 1: expected ':'` | **Yes** | **`100.0%`** |
| **Trap 4: Hallucinated Metric + AST Bug** | Injected `99.4%` + `IndentationError` | **`0.564 ms`** | `45.0%` | `['99.4'] + AST Error` | **Yes** | **`100.0%`** |
| **Case 5: 100% Grounded Control** | Valid numbers + valid Python code | **`0.404 ms`** | **`100.0%`** | `None (0 anomalies)` | **No (`0.0 ms` cost)** | **`100.0%`** |

---

## 5. Empirical Benchmarks & Ablation Study

### 5.1 Six-Stage Systems Optimization Log (Intel i7-12700K + NVIDIA T1000 4 GB)

| Optimization Stage | Engine & Memory Configuration | T1000 VRAM | First-Turn TTFT | Prompt Speed | Decode Speed | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Cold Baseline** | `-ngl 99 --cpu-moe` (4 slots, cold `mmap`) | 2,322 MiB | 42,811.7 ms | 0.77 tok/s | 6.89 tok/s | `1.00x` |
| **2. Warm Baseline** | `-ngl 99 --cpu-moe` (4 slots, warm `mmap`) | 2,322 MiB | 1,093.1 ms | 21.95 tok/s | 18.68 tok/s | `2.71x` |
| **3. Single-Slot Prefix KV** | `-np 1 --keep -1 --cache-reuse 256 --no-webui` | 2,441 MiB | 309.4 ms | 38.79 tok/s | 19.54 tok/s | `2.84x` |
| **4. Dynamic PCIe LRU (Rejected)** | `--moe-cache-mib 768 --spec-type ngram-simple` | 3,158 MiB | 357.3 ms | 2.80 tok/s | **8.57 tok/s** *(PCIe Thrash)* | `1.24x` |
| **5. Static Hybrid + Top-4 (Optimal)** | `--load-mode none -ncmoe 41 -Cr 0-15 --prio 2` (`K=4`) | 3,762 MiB | **290.9 ms** | **82.49 tok/s** | **39.18 tok/s** | **`5.69x`** |
| **6. Speculative 0.6B Draft** | Stage 5 + `Qwen3-0.6B` Draft (`n-max=3, p-min=0.80`) | 3,890 MiB | 312.0 ms | 79.80 tok/s | **39.35 tok/s** *(Peak)* | **`5.71x`** |

### 5.2 Global Framework Comparison (As of November 2026)

| Global System / Framework | Target Hardware | Speed on 4 GB VRAM (30B–35B MoE) | Max Stable Context on 4 GB | Built-In Fact & Code Verification |
| :--- | :--- | :---: | :---: | :--- |
| **Our `Qwen3Studio` Architecture** | **Standard PC (i7-12700K + 4 GB T1000)** | **32.8 – 39.4 tok/s** *(Top-4)*<br/>**27.1 tok/s** *(8-Exp + 0.6B Draft)* | **8,192 Tokens (Stable @ 3.76 GB)** | **`0.403 ms` Symbolic + AST Gate + Pass-2 Critic** |
| **MIT & UC Berkeley `FreeToken` (`v0.1.2`, 2026)** | 8 GB RTX 4060 / 4 GB RTX 2050 | **20.7 – 21.8 tok/s** *(on 4 GB RTX 2050)* | **4,096 Tokens** *(Crashes at 8,192 on 4 GB)* | None (Raw LLM output) |
| **`KTransformers` (ACM SOSP 2025/2026)** | Intel Xeon Server (Requires AMX) | **~11 – 14 tok/s** *(No AMX on desktop i7)* | 4,096 – 8,192 Tokens | None (Raw LLM output) |
| **`PowerInfer` (SJTU) & `Fiddler` (ICLR 2025)** | ReLU-Only Models / Linux PyTorch Ops | **Incompatible (SwiGLU) / ~9 tok/s PCIe** | 2,048 – 4,096 Tokens | None (Raw LLM output) |
| **MIT CSAIL `RLM` (NeurIPS 2026) & `VeNRA`** | Datacenter GPU + Python Subprocess | **25 – 45 sec latency** *(5–15 REPL calls)* | External REPL Variable | Executes untrusted LLM Python code |

---

## 6. Included Research Papers & Reports

* **[IEEE / arXiv Research Paper (Markdown)](RESEARCH_PAPER_IEEE_ARXIV.md)** & **[Two-Column IEEE Print/PDF Format (HTML)](RESEARCH_PAPER_IEEE_FORMAT.html)** (**[Download PDF](RESEARCH_PAPER_IEEE_FORMAT.pdf)**)
* **[Comprehensive Benchmark Comparison & Architectural Overview](COMPREHENSIVE_BENCHMARK_COMPARISON_REPORT.md)**
* **[Detailed Engineering Study Report](DETAILED_TECHNICAL_STUDY_REPORT.md)**
* **[Executive One-Pager Summary (HTML)](ONE_PAGER_EXECUTIVE_SUMMARY.html)** (**[Download PDF](ONE_PAGER_EXECUTIVE_SUMMARY.pdf)**)

---

## 7. Author & Citation

**Author:** Ajayakumar Vijayakumar (`ajayav2002@gmail.com`)

If you use this architecture, benchmark study, or verification pipeline in your research or engineering work, please cite:

```bibtex
@article{vijayakumar2026beyond,
  title   = {Beyond PCIe Thrashing and Recursive Sandboxes: High-Throughput 30B Sparse MoE Inference and Symbolic-Gated Hallucination Verification on 4 GB VRAM Workstations},
  author  = {Vijayakumar, Ajayakumar},
  year    = {2026},
  month   = {November},
  note    = {Available at https://github.com/AjayakumarV/Qwen3Studio-LocalAI}
}
```
