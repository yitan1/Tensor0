from .layout_cache import LAYOUT_CACHE_SCENARIO_SPECS
from .strided_indices import STRIDED_INDICES_SCENARIO_SPECS


DIAGNOSTIC_SCENARIO_SPECS = (
    LAYOUT_CACHE_SCENARIO_SPECS
    + STRIDED_INDICES_SCENARIO_SPECS
)


def diagnostic_metadata() -> dict[str, object]:
    return {
        "scenario_count": len(DIAGNOSTIC_SCENARIO_SPECS),
        "layout_cache_scenario_count": len(LAYOUT_CACHE_SCENARIO_SPECS),
        "strided_indices_scenario_count": len(
            STRIDED_INDICES_SCENARIO_SPECS
        ),
    }


__all__ = [
    "DIAGNOSTIC_SCENARIO_SPECS",
    "diagnostic_metadata",
]
