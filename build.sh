#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 [RELEASE|DEV]"
    exit 1
fi

BUILD_TYPE=$1
ICON_PATH="../../assets/favicon_large.png"
ASSETS_PATH="../../assets"
SRC_PATH="src/main.py"

case $BUILD_TYPE in
    RELEASE)
        OUTPUT_DIR="build/release"
        EXTRA_ARGS="--windowed"
        ;;
    DEV)
        OUTPUT_DIR="build/dev"
        EXTRA_ARGS=""
        ;;
    *)
        echo "Invalid build type. Use RELEASE or DEV."
        exit 1
        ;;
esac

echo "Building $BUILD_TYPE version..."

pyinstaller $EXTRA_ARGS --onefile --name=AutoGGUF --icon=$ICON_PATH --add-data "$ASSETS_PATH:assets" --distpath=$OUTPUT_DIR/dist --workpath=$OUTPUT_DIR/build --specpath=$OUTPUT_DIR $SRC_PATH

if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
else
    echo "Build completed successfully."
fi
