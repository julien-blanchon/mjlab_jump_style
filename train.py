"""Registers the custom spinkick task before running mjlab's training pipeline."""

import mjlab_jump_style  # noqa: F401

from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
