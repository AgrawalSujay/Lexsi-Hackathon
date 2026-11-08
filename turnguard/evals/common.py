"""Small shared helpers for external evaluation records."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

_CACHE_LOCK = RLock()


@contextmanager
def auditkit_cache(cache_dir: Path) -> Iterator[None]:
    """Scope AuditKIT 1.1.6's environment-only cache setting to this run."""
    with _CACHE_LOCK:
        prior = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = str(cache_dir)
        try:
            yield
        finally:
            if prior is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = prior
