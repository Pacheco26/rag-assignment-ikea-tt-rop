"""CLI: Launch Streamlit demo application."""

import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def main():
    print("Launching FurnishRAG Demo...")
    print(f"App: {APP_PATH}")
    print("Open http://localhost:8501 in your browser")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH)],
        cwd=str(APP_PATH.parent.parent),
    )


if __name__ == "__main__":
    main()
