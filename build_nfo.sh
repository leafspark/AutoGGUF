#!/bin/bash

clear
echo ""
echo "                      .     ."
echo "                    .  |  ."
echo "                 .    .  .    ."
echo "              .   . |. |..  ."
echo "             .   |    |  | ."
echo "            . | | |  |  | ."
echo "           .\"    |...| .\""
echo "          .\"   .\" |  |."
echo "         .\"   .\"  |  | ."
echo "        .\"   .\"    | .\""
echo "      .\"   .\"     .\""
echo "______.\"   .\"     .\""
echo "       .\"     .\""
echo "      .\"   .\""
echo "     .\"  .\"         AutoGGUF Builder v1.337"
echo "    .\" .\"        ~~~ Cracked by CODEX Team ~~~"
echo "   .\".\" "
echo "  .\".\" "
echo ".\". "
echo ""
echo "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
echo ":                         Release Notes                             :"
echo ": - Now with 100% less Python dependency!                           :"
echo ": - Added quantum entanglement for faster builds                    :"
echo ": - Integrated AI to write better code than you                     :"
echo ": - Free pizza with every successful compile (while stocks last)    :"
echo "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
echo ":                         Installation                              :"
echo ": 1. Run this totally legit .sh file                                :"
echo ": 2. Choose your poison: RELEASE or DEV                             :"
echo ": 3. ???                                                            :"
echo ": 4. Profit!                                                        :"
echo "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
echo ":                     System Requirements                           :"
echo ": - A computer (duh)                                                :"
echo ": - Electricity (optional but recommended)                          :"
echo ": - At least 3 brain cells                                          :"
echo "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
echo ""

if [ $# -eq 0 ]; then
    echo "[!] ERROR: No build type specified. RTFM, n00b!"
    echo "    Usage: $0 [RELEASE|DEV]"
    exit 1
fi

if [ "$1" = "RELEASE" ]; then
    echo "[+] Initiating RELEASE build sequence..."
    echo "[+] Compressing code until it becomes a singularity..."
    pyinstaller --windowed --onefile --name=AutoGGUF --icon=../../assets/favicon_large.png --add-data "../../assets:assets" --distpath=build/release/dist --workpath=build/release/build --specpath=build/release src/main.py
elif [ "$1" = "DEV" ]; then
    echo "[+] Launching DEV build missiles..."
    echo "[+] Obfuscating code to confuse even its creator..."
    pyinstaller --onefile --name=AutoGGUF --icon=../../assets/favicon_large.png --add-data "../../assets:assets" --distpath=build/dev/dist --workpath=build/dev/build --specpath=build/dev src/main.py
else
    echo "[!] FATAL ERROR: Invalid build type. Are you even trying?"
    echo "    Use RELEASE or DEV, genius."
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "[-] Build failed. Blame the intern."
    exit 1
else
    echo "[+] Build completed successfully. Time to take credit for someone else's work!"
fi

echo ""
echo "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
echo ":   Remember: Piracy is wrong. Unless you're really good at it.     :"
echo "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
