from typing import Protocol, Any, List
from paper2.common.types import AircraftState, TargetTruthState, NoFlyZoneState


class Paper2EnvProtocol(Protocol):
    def reset(self, seed: int | None = None) -> Any:
        ...

    def step(self, action: Any) -> Any:
        ...

    def get_aircraft_state(self) -> AircraftState:
        ...

    def get_target_truth(self) -> TargetTruthState:
        ...

    def get_no_fly_zones(self) -> List[NoFlyZoneState]:
        ...