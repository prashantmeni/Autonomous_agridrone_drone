"""Crop disease AI - standalone offline inference for Raspberry Pi.

Observation-only. This module contains no flight-control code and no MAVLink;
it cannot arm, change mode, or write parameters.

Quick start (from this directory):
    pip install -r requirements.txt
    python scripts/image_inference.py --image leaf.jpg

No trained model ships with this module. See "Model status" below.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]