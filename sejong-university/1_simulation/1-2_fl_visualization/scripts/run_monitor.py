import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "common", "src")))

from defai_addons.monitor.server import main

if __name__ == "__main__":
    main()
