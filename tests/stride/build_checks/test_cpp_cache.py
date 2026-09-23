"""Cache invalidation without compiling native contracts."""

import subprocess
import sys

import pytest

from tests.stride.support.cpp_cache import compile_cached


pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='native build cache requires Linux')


@pytest.fixture
def build(tmp_path):
    compiler = tmp_path / "compiler.py"
    compiler.write_text('''import pathlib, sys
root = pathlib.Path(__file__).parent
with (root / "calls").open("a") as log:
    log.write("compile\\n")
pathlib.Path(sys.argv[-1]).write_text("binary")
if (root / "fail").exists():
    sys.exit(1)
''')
    header = tmp_path / "header.h"
    header.write_text("original")
    output = tmp_path / "contract"

    def run(*flags, identity="compiler-v1"):
        compile_cached([sys.executable, str(compiler), *flags], output,
                       inputs=[compiler, header], identity=identity, timeout=10)
        return (tmp_path / "calls").read_text().count("compile")

    return run, header, output


@pytest.mark.parametrize("change", ["header", "command", "compiler", "environment", "output", "stamp"])
def test_build_cache_invalidation(build, change):
    run, header, output = build
    assert run() == 1
    assert run() == 1
    flags, identity = (), "compiler-v1"
    if change == "header":
        # Content changes, even at the same size, invalidate the cache.
        header.write_text("modified")
    elif change == "command":
        flags = ("-O2",)
    elif change in ("compiler", "environment"):
        identity = {change: "changed"}
    elif change == "output":
        output.unlink()
    else:
        output.with_suffix(".sha256").unlink()
    assert run(*flags, identity=identity) == 2
    assert run(*flags, identity=identity) == 2


def test_failed_build_is_not_cached(build):
    run, header, output = build
    assert run() == 1
    header.write_text("changed")
    failure = output.parent / "fail"
    failure.touch()
    with pytest.raises(subprocess.CalledProcessError):
        run()
    assert output.read_text() == "binary"
    assert not output.with_suffix(".sha256").exists()
    failure.unlink()
    assert run() == 3
    assert run() == 3


def test_concurrent_builds_share_one_result(build):
    from concurrent.futures import ThreadPoolExecutor

    run, _, _ = build
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: run(), range(4))) == [1] * 4
