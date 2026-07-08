"""
rag-toolkit / core / utils.py

Standalone utility functions: text processing, timeout control, ID generation.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import queue
import random
import re
import string
from typing import Any, Callable, Optional


class TimeoutError(Exception):
    """Raised when a function exceeds its allowed execution time."""
    pass


def run_with_timeout(
    func: Callable[..., Any],
    timeout_seconds: int = 10,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute *func(*args, **kwargs)* in a subprocess with a timeout.

    Uses ``multiprocessing.Process`` + ``Queue`` so it can kill runaway code.

    Raises:
        TimeoutError: if the function does not finish within *timeout_seconds*.
        Any exception raised by *func* is re-raised in the caller.
    """
    result_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)

    def target(q: multiprocessing.Queue) -> None:
        try:
            r = func(*args, **kwargs)
            q.put(r, timeout=timeout_seconds)
        except Exception as e:
            q.put(e, timeout=timeout_seconds)

    p = multiprocessing.Process(target=target, args=(result_queue,))
    p.daemon = True
    p.start()

    try:
        result = result_queue.get(timeout=timeout_seconds)
        p.join(timeout=0)
    except (queue.Empty, multiprocessing.TimeoutError):
        p.terminate()
        p.join()
        raise TimeoutError(
            f"Function execution timed out after {timeout_seconds} seconds."
        )
    finally:
        if p.is_alive():
            p.terminate()
            p.join()

    if isinstance(result, Exception):
        raise result
    return result


def text_to_md5(text: str) -> str:
    """Return the MD5 hex digest of *text*."""
    return hashlib.md5(text.encode()).hexdigest()


def generate_table_name(length: int = 5) -> str:
    """Generate a random ASCII-letter table/collection name."""
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


def remove_html(text: str) -> str:
    """Strip all HTML tags from *text*."""
    return re.sub(r"<[^>]+>", "", text)


def cut_sentences(text: str) -> list[str]:
    """Split Chinese text into sentences on punctuation boundaries.

    Handles ``。！？?…＂＂`` and keeps quotation marks attached.
    """
    text = re.sub(r"([。！？\?])([^”’])", r"\1\n\2", text)
    text = re.sub(r"(\…{2})([^”’])", r"\1\n\2", text)
    text = re.sub(r"([。！？\?][”’])([^。！？\?])", r"\1\n\2", text)
    return [s.strip() for s in text.split("\n") if s.strip()]
