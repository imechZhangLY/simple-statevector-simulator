"""Exit 0 only when every named module can be located.

`importlib.util.find_spec` locates a module without executing it, which keeps
the check far cheaper than importing torch just to prove it exists.
"""

import importlib.util
import sys


def main() -> int:
    for name in sys.argv[1:]:
        try:
            if importlib.util.find_spec(name) is None:
                return 1
        except (ImportError, ValueError):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
