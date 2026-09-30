@echo off
REM Minimal GUI launcher for the Andor iXon 897 control package.
REM Usage examples:
REM   run_minimal_gui.bat --mock
REM   run_minimal_gui.bat --dll "C:\Path\To\atmcd64d.dll"
py -3.11 scripts\minimal_gui.py %*
