import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import docx
import openpyxl
from pypdf import PdfReader


@dataclass
class AttachedDocument:
    name: str
    path: str
    size_bytes: int
    content: str
    char_count: int
    est_tokens: int


SUPPORTED_EXTENSIONS = (
    ".pdf", ".docx", ".xlsx", ".xlsm", ".csv", ".tsv",
    ".txt", ".md", ".json", ".yaml", ".yml", ".xml", ".log", ".ini", ".cfg",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".sql",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".java", ".go", ".rs", ".ps1", ".sh", ".bat", ".cmd"
)


def extract_file_content(file_path: str, max_chars: int = 48000) -> AttachedDocument:
    path = Path(file_path)
    ext = path.suffix.lower()
    size_bytes = path.stat().st_size if path.exists() else 0

    if ext == ".pdf":
        text = _extract_pdf(path)
    elif ext == ".docx":
        text = _extract_docx(path)
    elif ext in (".xlsx", ".xlsm"):
        text = _extract_xlsx(path)
    elif ext in (".csv", ".tsv"):
        text = _extract_csv(path, delimiter="\t" if ext == ".tsv" else ",")
    else:
        text = _extract_text(path)

    text = text.strip()
    if len(text) > max_chars:
        text = (
            text[:max_chars]
            + f"\n\n... [Truncated at {max_chars:,} characters to fit local context window] ..."
        )

    char_count = len(text)
    est_tokens = max(1, char_count // 4)
    return AttachedDocument(
        name=path.name,
        path=str(path),
        size_bytes=size_bytes,
        content=text,
        char_count=char_count,
        est_tokens=est_tokens,
    )


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages_out: List[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages_out.append(f"--- Page {idx} ---\n{page_text}")
    return "\n\n".join(pages_out) if pages_out else "[No extractable text found in PDF]"


def _extract_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    parts: List[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for t_idx, table in enumerate(doc.tables, start=1):
        rows_text = []
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                rows_text.append(" | ".join(cells))
        if rows_text:
            parts.append(f"\n[Table {t_idx}]\n" + "\n".join(rows_text))
    return "\n".join(parts) if parts else "[Empty DOCX document]"


def _extract_xlsx(path: Path) -> str:
    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    sheets_out: List[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows_out = [f"=== Sheet: {sheet_name} ==="]
        row_count = 0
        for row in ws.iter_rows(values_only=True):
            vals = ["" if v is None else str(v).strip() for v in row]
            if any(vals):
                rows_out.append(" | ".join(vals))
                row_count += 1
                if row_count >= 500:
                    rows_out.append("... [Truncated after 500 rows] ...")
                    break
        if row_count > 0:
            sheets_out.append("\n".join(rows_out))
    wb.close()
    return "\n\n".join(sheets_out) if sheets_out else "[Empty Excel Workbook]"


def _extract_csv(path: Path, delimiter: str = ",") -> str:
    rows_out: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for idx, row in enumerate(reader):
            rows_out.append(" | ".join(row))
            if idx >= 500:
                rows_out.append("... [Truncated after 500 rows] ...")
                break
    return "\n".join(rows_out)


def _extract_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")
