"""Shared parsing for the image and training commands."""

import argparse


def _anchor(value, special, *, default=None):
    if value == special:
        return value
    if value == "auto" and default == "auto":
        return None
    try:
        return float(value)
    except ValueError:
        options = f"a number, '{special}'"
        if default == "auto":
            options += ", or 'auto'"
        raise argparse.ArgumentTypeError(f"Expected {options}") from None


def infant_anchor(value):
    return _anchor(value, "fundamental")


def adult_anchor(value):
    return _anchor(value, "nyquist", default="auto")
