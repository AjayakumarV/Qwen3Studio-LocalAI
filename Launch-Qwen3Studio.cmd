@echo off
title Qwen3-30B-A3B Local AI Desktop Studio
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0qwen3_studio.py"
) else (
    python "%~dp0qwen3_studio.py"
)
