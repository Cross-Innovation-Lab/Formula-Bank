"""Runtime-only compatibility glue for PaSST's pinned kk-sacred stack."""

from __future__ import annotations

import sys
from pathlib import Path


def prepare_passt_import(repo_root):
    """Expose the legacy docopt symbol required by kk-sacred on modern Python.

    It is only used by Sacred's command-line help path, which these adapters do
    not call.  The upstream PaSST checkout is intentionally left untouched.
    """
    import docopt

    if not hasattr(docopt, "printable_usage"):
        docopt.printable_usage = docopt.formal_usage
    passt_root = str(Path(repo_root) / "external" / "PaSST")
    if passt_root not in sys.path:
        sys.path.insert(0, passt_root)
