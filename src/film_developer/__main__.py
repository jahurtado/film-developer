"""`python -m film_developer` -- the same entry point as the installed command."""

import sys

from .cli import main

# Guarded, although `multiprocessing` already declines to re-import a
# package's `__main__` in the workers it spawns: the guard is what says so.
if __name__ == "__main__":
    sys.exit(main())
