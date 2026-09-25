import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.app import demo, custom_theme

if __name__ == "__main__":
    print("===================================================")
    print("      Starting UniFace Studio (Local Web UI)")
    print("===================================================")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=custom_theme)
