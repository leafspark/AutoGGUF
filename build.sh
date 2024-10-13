#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: build.sh [RELEASE|DEV]"
    exit 1
fi

if [ "${1,,}" = "release" ]; then
    echo "Building RELEASE version..."
    pyinstaller --windowed --onefile --name=AutoGGUF --icon=../../assets/favicon_large.png --add-data "../../assets:assets" --distpath=build/release/dist --workpath=build/release/build --specpath=build/release src/main.py
elif [ "${1,,}" = "dev" ]; then
    echo "Building DEV version..."
    pyinstaller --onefile --name=AutoGGUF --icon=../../assets/favicon_large.png --add-data "../../assets:assets" --distpath=build/dev/dist --workpath=build/dev/build --specpath=build/dev src/main.py
else
    echo "Invalid argument. Use RELEASE or DEV."
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
else
    echo "Build completed successfully."
fi
