"""Energy accounting for one lift.

Running energy comes from simple mechanics of a counterweighted traction lift:
  * the counterweight balances the empty car + 50% of rated load, so the motor only
    lifts the *imbalance*:  m_net = (car + load) - counterweight
  * moving up with m_net > 0 (heavy car) costs  m_net*g*h,  moving down costs -m_net*g*h;
    when that number is negative gravity does the work (recovered only with a regenerative drive)
  * friction costs F*h on every floor regardless of direction
  * each start accelerates the whole moving mass to rated speed: 1/2 m v^2
  * each door cycle costs a fixed amount
Non-running energy is the base electrical load (or the lower standby power) times time.
"""
from dataclasses import dataclass

from ..config import BuildingConfig, EnergyConfig

G = 9.81


@dataclass
class EnergyMeter:
    cfg: BuildingConfig
    ecfg: EnergyConfig
    running_j: float = 0.0
    base_j: float = 0.0
    standby_j: float = 0.0

    def _masses(self, n_people: int):
        rated = self.cfg.capacity * self.ecfg.person_kg
        car = self.ecfg.car_mass_kg + n_people * self.ecfg.person_kg
        counterweight = self.ecfg.car_mass_kg + self.ecfg.balance * rated
        return car, counterweight

    def move_floor(self, n_people: int, direction: int):
        car, cw = self._masses(n_people)
        h = self.ecfg.floor_height_m
        potential = direction * (car - cw) * G * h  # >0: motor must supply it
        if potential > 0:
            e = potential / self.ecfg.efficiency
        else:
            e = potential * self.ecfg.efficiency * self.ecfg.regen  # negative = recovered
        self.running_j += e + self.ecfg.friction_n * h / self.ecfg.efficiency

    def start(self, n_people: int):
        car, cw = self._masses(n_people)
        self.running_j += 0.5 * (car + cw) * self.ecfg.speed_ms ** 2 / self.ecfg.efficiency

    def door_cycle(self):
        self.running_j += self.ecfg.door_cycle_j

    def tick(self, standby: bool):
        if standby:
            self.standby_j += self.ecfg.standby_w
        else:
            self.base_j += self.ecfg.base_w

    @property
    def total_kwh(self) -> float:
        return (self.running_j + self.base_j + self.standby_j) / 3.6e6
