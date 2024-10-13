#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: build_fast.sh [RELEASE|DEV]"
    exit 1
fi

COMMON_FLAGS="--standalone --enable-plugin=pyside6 --include-data-dir=assets=assets"

if [ "$1" == "RELEASE" ]; then
    echo "Building RELEASE version..."
    python -m nuitka $COMMON_FLAGS --windows-console-mode=disable --output-dir=build/release src/main.py --lto=yes
elif [ "$1" == "DEV" ]; then
    echo "Building DEV version..."
    python -m nuitka $COMMON_FLAGS --output-dir=build/dev src/main.py
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
