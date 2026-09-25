import os
import sys

# Ensure uniface-app root and src are on sys.path
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)
SRC_DIR = os.path.join(APP_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.app import demo, custom_theme

if __name__ == "__main__":
    print("===================================================")
    print("      Starting UniFace Studio (Local Web UI)")
    print("===================================================")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=custom_theme)
