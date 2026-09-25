import os
import sys

# Ensure directories are on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, os.path.join(CURRENT_DIR, "src"), PARENT_DIR, os.path.join(PARENT_DIR, "src")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from src.app import demo, custom_theme

if __name__ == "__main__":
    print("===================================================")
    print("      Starting UniFace Studio (Local Web UI)")
    print("===================================================")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=custom_theme)
