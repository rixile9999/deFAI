import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("FLWR_LOG_LEVEL", "WARNING")
