from .api_networks import (
    API_NETWORK_SCENARIO_SPECS,
    api_network_metadata,
)
from .primitives import PRIMITIVE_SCENARIO_SPECS


CONTRACTION_SCENARIO_SPECS = (
    PRIMITIVE_SCENARIO_SPECS + API_NETWORK_SCENARIO_SPECS
)


def contraction_metadata() -> dict[str, object]:
    api_metadata = api_network_metadata()
    return {
        "scenario_count": len(CONTRACTION_SCENARIO_SPECS),
        "primitive_scenario_count": len(PRIMITIVE_SCENARIO_SPECS),
        "api_network_scenario_count": len(API_NETWORK_SCENARIO_SPECS),
        "chain_dimensions": api_metadata["chain_dimensions"],
    }

__all__ = [
    "CONTRACTION_SCENARIO_SPECS",
    "contraction_metadata",
]
