"""Common utilities for publishing benchmark results."""

import platform
import sys
from datetime import datetime
from pathlib import Path

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from lance_bench_db.models import TestBed


def get_test_bed(name: str | None = None) -> TestBed:
    """Create a TestBed instance from the current system information.

    Args:
        name: Optional testbed name. If not provided, uses platform.node()

    Returns:
        TestBed instance with current system information
    """
    import psutil

    return TestBed(
        name=name or platform.node(),
        cpu=platform.processor() or platform.machine(),
        memory_gb=int(psutil.virtual_memory().total / (1024**3)),
        os=f"{platform.system()} {platform.release()}",
        created_at=int(datetime.now().timestamp()),
    )
