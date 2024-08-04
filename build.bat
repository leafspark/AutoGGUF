@echo off

if "%1"=="" (
    echo Usage: build.bat [RELEASE^|DEV]
    exit /b 1
)

if /I "%1"=="RELEASE" (
    echo Building RELEASE version...
    pyinstaller --windowed --onefile --name=AutoGGUF --icon=../../assets/favicon_light.ico --distpath=build\release\dist --workpath=build\release\build --specpath=build\release src\main.py
) else if /I "%1"=="DEV" (
    echo Building DEV version...
    pyinstaller --onefile --name=AutoGGUF --icon=../../assets/favicon.ico --distpath=build\dev\dist --workpath=build\dev\build --specpath=build\dev src\main.py
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
