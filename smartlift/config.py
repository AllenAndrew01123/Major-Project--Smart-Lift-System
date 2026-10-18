from dataclasses import dataclass


@dataclass(frozen=True)
class BuildingConfig:
    n_floors: int = 16            # floors 0..n_floors-1, floor 0 is the lobby
    n_lifts: int = 4
    capacity: int = 12            # passengers per car
    pop_per_floor: int = 60       # people working on each upper floor

    # timing (seconds / ticks)
    ticks_per_floor: int = 2      # ~1.75 m/s with 3.5 m floors
    stop_ticks: int = 6           # decelerate + door open + door close + accelerate
    ticks_per_passenger: int = 1  # boarding / alighting time per person
    wake_ticks: int = 8           # time to come back from standby


@dataclass(frozen=True)
class EnergyConfig:
    """Simplified physical model following the ISO 25745-2 split into
    running energy (moving), and non-running energy (base load / standby)."""
    floor_height_m: float = 3.5
    car_mass_kg: float = 1000.0
    person_kg: float = 75.0
    balance: float = 0.5          # counterweight = car + 50% of rated load
    efficiency: float = 0.75      # motor + drive efficiency
    regen: float = 0.0            # fraction of surplus energy recovered (0 = no regenerative drive)
    friction_n: float = 1000.0    # rope/guide friction force
    speed_ms: float = 1.75
    door_cycle_j: float = 600.0   # one open+close cycle
    base_w: float = 300.0         # controller, lights, fan: drawn whenever not in standby (moving too)
    standby_w: float = 50.0       # powered-down standby mode
