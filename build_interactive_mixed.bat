@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\build_interactive_mixed.ps1"
