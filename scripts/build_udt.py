"""Build the Java UDT adapter; equivalent to defai udt build."""

import argparse
from pathlib import Path

from defai.udt import build

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(build(parser.parse_args().output))
