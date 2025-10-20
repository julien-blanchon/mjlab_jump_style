# /// script
# dependencies = [
#   "mjlab",
#   "mjlab_jump_style",
# ]
# ///


"""Registers the custom spinkick task before running mjlab's play pipeline."""

import mjlab_jump_style  # noqa: F401

from mjlab.scripts.play import main

if __name__ == "__main__":
    main()
