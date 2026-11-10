"""
Local Document & File Extraction + Smart Offline BM25 RAG Engine for Qwen3-30B-A3B Desktop Studio.
Extracts structured text from PDF, DOCX, XLSX, CSV, JSON, Markdown, and source code files
entirely offline without external services.
Automatically uses full-document injection for small files and BM25 Top-K chunk retrieval
for large documents or folders so prompt processing stays fast and fits comfortably in VRAM.
"""

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pypdf import PdfReader
import docx
import openpyxl


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".xlsm", ".csv", ".json",
    ".txt", ".md", ".markdown", ".rst", ".log", ".ini", ".cfg", ".conf",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".html", ".css", ".scss", ".sql",
    ".sh", ".ps1", ".bat", ".cmd", ".yaml", ".yml", ".toml", ".xml", ".env"
}


@dataclass
class DocumentChunk:
    file_name: str
    chunk_id: int
    header: str
    text: str
    tokens: List[str] = field(default_factory=list)


@dataclass
class AttachedFile:
    path: str
    name: str
    size_bytes: int
    file_type: str
    content: str
    char_count: int
    summary_info: str
    chunks: List[DocumentChunk] = field(default_factory=list)


def _tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]{2,}", text.lower())


def build_chunks_for_text(file_name: str, full_text: str, chunk_size: int = 1100, overlap: int = 150) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    if not full_text.strip():
        return chunks

    # Split naturally by paragraph/section first
    paragraphs = re.split(r"\n{2,}", full_text)
    current_buf: List[str] = []
    current_len = 0
    chunk_idx = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > chunk_size and current_buf:
            combined = "\n\n".join(current_buf)
            chunks.append(
                DocumentChunk(
                    file_name=file_name,
                    chunk_id=chunk_idx,
                    header=f"{file_name} [Section {chunk_idx}]",
                    text=combined,
                    tokens=_tokenize_for_bm25(combined),
                )
            )
            chunk_idx += 1
            # Keep overlap from tail of previous buffer
            tail = combined[-overlap:] if len(combined) > overlap else combined
            current_buf = [tail, para]
            current_len = len(tail) + len(para)
        else:
            current_buf.append(para)
            current_len += len(para)

    if current_buf:
        combined = "\n\n".join(current_buf)
        chunks.append(
            DocumentChunk(
                file_name=file_name,
                chunk_id=chunk_idx,
                header=f"{file_name} [Section {chunk_idx}]",
                text=combined,
                tokens=_tokenize_for_bm25(combined),
            )
        )

    return chunks


def extract_pdf(path: Path, max_chars: int = 250000) -> tuple[str, str]:
    reader = PdfReader(str(path))
    num_pages = len(reader.pages)
    pages_text: List[str] = []
    total_len = 0

    for idx, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        if txt:
            block = f"--- [Page {idx}/{num_pages}] ---\n{txt}"
            pages_text.append(block)
            total_len += len(block)
            if total_len >= max_chars:
                pages_text.append(f"\n[...Truncated at {max_chars:,} characters out of {num_pages} pages...]")
                break

    combined = "\n\n".join(pages_text) if pages_text else "[No extractable text found in PDF]"
    return combined[:max_chars], f"PDF ({num_pages} pages)"


def extract_docx(path: Path, max_chars: int = 250000) -> tuple[str, str]:
    doc = docx.Document(str(path))
    parts: List[str] = []

    for p in doc.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())

    for t_idx, table in enumerate(doc.tables, start=1):
        rows_str = [f"--- [Table {t_idx}] ---"]
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                rows_str.append(" | ".join(cells))
        if len(rows_str) > 1:
            parts.append("\n".join(rows_str))

    combined = "\n\n".join(parts) if parts else "[Empty Word Document]"
    if len(combined) > max_chars:
        combined = combined[:max_chars] + f"\n[...Truncated at {max_chars:,} chars...]"
    return combined, f"Word DOCX ({len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables)"


def extract_xlsx(path: Path, max_chars: int = 250000) -> tuple[str, str]:
    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    sheet_names = wb.sheetnames
    out_blocks: List[str] = []
    total_rows = 0

    for name in sheet_names:
        ws = wb[name]
        sheet_lines = [f"=== Sheet: {name} ==="]
        for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if any(cell is not None and str(cell).strip() != "" for cell in row):
                vals = ["" if cell is None else str(cell).strip() for cell in row]
                sheet_lines.append(" | ".join(vals))
                total_rows += 1
                if row_idx >= 1500:
                    sheet_lines.append("[...Row limit reached for sheet preview...]")
                    break
        out_blocks.append("\n".join(sheet_lines))

    wb.close()
    combined = "\n\n".join(out_blocks)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + f"\n[...Truncated at {max_chars:,} chars...]"
    return combined, f"Excel Workbook ({len(sheet_names)} sheets, {total_rows} rows)"


def extract_csv(path: Path, max_chars: int = 250000) -> tuple[str, str]:
    lines: List[str] = []
    row_count = 0
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            row_count += 1
            lines.append(" | ".join(row))
            if sum(len(x) for x in lines) >= max_chars or row_count >= 2500:
                lines.append(f"[...Truncated at row {row_count}...]")
                break
    return "\n".join(lines), f"CSV Data ({row_count} rows)"


def load_and_extract_file(file_path: str, max_chars: int = 250000) -> AttachedFile:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    size_bytes = p.stat().st_size
    ext = p.suffix.lower()

    if ext == ".pdf":
        content, summary = extract_pdf(p, max_chars=max_chars)
    elif ext == ".docx":
        content, summary = extract_docx(p, max_chars=max_chars)
    elif ext in {".xlsx", ".xlsm"}:
        content, summary = extract_xlsx(p, max_chars=max_chars)
    elif ext == ".csv":
        content, summary = extract_csv(p, max_chars=max_chars)
    elif ext == ".json":
        raw = p.read_text(encoding="utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
            content = json.dumps(parsed, indent=2)[:max_chars]
        except Exception:
            content = raw[:max_chars]
        summary = f"JSON File ({size_bytes / 1024:.1f} KB)"
    else:
        raw = p.read_text(encoding="utf-8", errors="replace")
        if len(raw) > max_chars:
            content = raw[:max_chars] + f"\n[...Truncated at {max_chars:,} chars...]"
        else:
            content = raw
        line_count = raw.count("\n") + 1
        summary = f"Text/Code ({ext or 'file'}, {line_count} lines)"

    chunks = build_chunks_for_text(p.name, content)
    return AttachedFile(
        path=str(p.resolve()),
        name=p.name,
        size_bytes=size_bytes,
        file_type=ext or "txt",
        content=content,
        char_count=len(content),
        summary_info=f"{summary} • {len(chunks)} chunks",
        chunks=chunks,
    )


def load_folder_files(folder_path: str, max_files: int = 25) -> List[AttachedFile]:
    root = Path(folder_path)
    if not root.is_dir():
        return []
    results: List[AttachedFile] = []
    skip_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}

    for path in root.rglob("*"):
        if len(results) >= max_files:
            break
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                if path.stat().st_size <= 15 * 1024 * 1024:  # up to 15 MB per file
                    results.append(load_and_extract_file(str(path)))
            except Exception:
                continue
    return results


def retrieve_top_bm25_chunks(
    attachments: List[AttachedFile],
    query: str,
    top_k: int = 8,
) -> List[DocumentChunk]:
    all_chunks: List[DocumentChunk] = []
    for att in attachments:
        all_chunks.extend(att.chunks)

    if len(all_chunks) <= top_k:
        return all_chunks

    query_tokens = _tokenize_for_bm25(query)
    if not query_tokens:
        # Return first chunks + evenly spaced summary chunks
        return all_chunks[:top_k]

    N = len(all_chunks)
    avgdl = sum(len(c.tokens) for c in all_chunks) / max(1, N)
    k1 = 1.5
    b = 0.75

    # Document frequencies
    df = Counter()
    for c in all_chunks:
        for tok in set(c.tokens):
            df[tok] += 1

    scored: List[tuple[float, int, DocumentChunk]] = []
    for idx, c in enumerate(all_chunks):
        tf = Counter(c.tokens)
        dl = len(c.tokens)
        score = 0.0
        for q in query_tokens:
            if q not in tf:
                continue
            n_q = df[q]
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            freq = tf[q]
            denom = freq + k1 * (1 - b + b * (dl / max(1.0, avgdl)))
            score += idf * ((freq * (k1 + 1)) / denom)
        # Boost chunk 1 of each file slightly so document headers/overviews are included
        if c.chunk_id == 1:
            score += 0.35
        scored.append((score, -idx, c))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = [item[2] for item in scored[:top_k]]
    # Sort selected chunks back into natural reading order
    selected.sort(key=lambda c: (c.file_name, c.chunk_id))
    return selected


def format_attachments_for_prompt(
    attachments: List[AttachedFile],
    user_query: str = "",
    direct_char_threshold: int = 4500,
) -> str:
    if not attachments:
        return ""

    total_chars = sum(a.char_count for a in attachments)

    # Small attachments (<4,500 chars / ~1,100 tokens): inject full contents directly
    if total_chars <= direct_char_threshold:
        sections = [
            "### ATTACHED LOCAL FILES (FULL CONTENT) ###",
            "The user has attached the following local file(s):\n",
        ]
        for idx, att in enumerate(attachments, start=1):
            sections.append(
                f"--- FILE {idx}: {att.name} ({att.summary_info}) ---\n"
                f"{att.content}\n"
                f"--- END OF FILE {idx}: {att.name} ---\n"
            )
        return "\n".join(sections)

    # Larger attachments: use Smart Offline BM25 RAG retrieval (Top-5 focused chunks, ~1,100 tokens total)
    top_chunks = retrieve_top_bm25_chunks(attachments, user_query, top_k=5)
    file_manifest = ", ".join(f"{a.name} ({a.summary_info})" for a in attachments)
    sections = [
        "### ATTACHED LOCAL FILES (SMART RAG RETRIEVAL) ###",
        f"Attached Files Inventory: {file_manifest}",
        f"Showing the {len(top_chunks)} highest-scoring sections matching the user's query:\n",
    ]
    for ch in top_chunks:
        sections.append(
            f"--- [{ch.header}] ---\n"
            f"{ch.text}\n"
        )
    return "\n".join(sections)
