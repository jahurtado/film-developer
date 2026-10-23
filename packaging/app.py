"""Entry point for the macOS bundle.

`cli.main` already treats a missing command as the editor and a bare path as a
roll to open, so a bundle is the same program with two extra chores:

- macOS hands some launches a Process Serial Number, which is neither a
  command nor a path and must not be taken for one;
- **`freeze_support()`, first thing.** Detecting and developing run on a
  `spawn` pool of workers, and a spawned worker of a frozen app is the app's
  own executable started again. Without this call each worker would open a
  second editor instead of doing its share of the roll.
"""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from film_developer.cli import main
    sys.exit(main([a for a in sys.argv[1:] if not a.startswith("-psn_")]))
