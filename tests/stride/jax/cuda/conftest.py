"""Explicit device requirement for GPU integration tests."""

import pytest

from tests.stride.support.availability import cuda_device_or_skip


@pytest.fixture
def cuda_device():
    return cuda_device_or_skip()
