@echo off

if "%1"=="" (
    echo Usage: build_optimized.bat [RELEASE^|DEV]
    exit /b 1
)

set COMMON_FLAGS=--standalone --enable-plugin=pyside6 --include-data-dir=assets=assets

if /I "%1"=="RELEASE" (
    echo Building RELEASE version...
    python -m nuitka %COMMON_FLAGS% --windows-console-mode=disable --output-dir=build\release src\main.py --lto=yes
) else if /I "%1"=="DEV" (
    echo Building DEV version...
    python -m nuitka %COMMON_FLAGS% --output-dir=build\dev src\main.py
) else (
    echo Invalid argument. Use RELEASE or DEV.
    exit /b 1
)

if errorlevel 1 (
    echo Build failed.
    exit /b 1
) else (
    echo Build completed successfully.
)
