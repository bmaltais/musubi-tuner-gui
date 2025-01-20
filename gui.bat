@echo off
set Path=.\uv\windows;%Path%

:: If the exit code is 0, run the kohya_gui.py script with the command-line arguments
if %errorlevel% equ 0 (
    REM Check if the batch was started via double-click
    IF /i "%comspec% /c %~0 " equ "%cmdcmdline:"=%" (
        REM echo This script was started by double clicking.
        cmd /k uv run gui.py %*
    ) ELSE (
        REM echo This script was started from a command prompt.
        uv run gui.py %*
    )
)