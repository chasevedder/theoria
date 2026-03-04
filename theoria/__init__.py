"""theoria — Multimodal AI subtitle generator."""

__version__ = "0.1.0"


def suppress_warnings():
    """Suppress noisy library warnings before heavy imports."""
    import os
    import warnings
    import logging

    os.environ["KMP_WARNINGS"] = "0"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    warnings.filterwarnings("ignore")
    logging.getLogger("pyannote").setLevel(logging.ERROR)
