"""Content-based cache for the standalone Linux C++ test builds."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def compile_cached(command, output, *, inputs, identity, timeout):
    """Reuse successful builds only; callers supply all project dependencies."""
    import fcntl

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps([command, identity]).encode())
    for path in sorted(map(Path, inputs)):
        content = path.read_bytes()
        digest.update(str(path).encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    key = digest.hexdigest()
    stamp = output.with_suffix(output.suffix + ".sha256")
    with output.with_suffix(output.suffix + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.is_file() and stamp.is_file() and stamp.read_text() == key:
            return
        # A failed or interrupted build must never publish a valid cache entry.
        stamp.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            target = Path(temporary) / output.name
            subprocess.run([*command, "-o", str(target)], check=True,
                           capture_output=True, text=True, timeout=timeout)
            os.replace(target, output)
        stamp.write_text(key)
