import pandas as pd
from dataclasses import dataclass

from modtox.modtox.utils.enums import StandardTypes, Database


@dataclass(frozen=True, order=True)
class Standard:
    std_type: StandardTypes
    std_val: float or int
    std_unit: str
    std_rel: str

    def __str__(self):
        return f"{self.std_type.name} {self.std_rel} {self.std_val} {self.std_unit}"

    def __repr__(self):
        return f"{self.std_type.name} {self.std_rel} {self.std_val} {self.std_unit}"


@dataclass
class Activity:
    inchi: str
    standard: Standard
    database: Database
    target: str

    def is_active(self, std_type, cutoff) -> bool or None:
        self.activity = False
        if self.std_type != std_type:
            self.activity = None
        elif self.std_type == std_type and self.std_val <= cutoff:
            self.activity = True
        return self.activity

    def __repr__(self) -> str:
        return f"{self.standard} ({self.database.name})"

    def __str__(self) -> str:
        return f"{self.standard} ({self.database.name})"
