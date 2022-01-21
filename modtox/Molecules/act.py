import pandas as pd
from dataclasses import dataclass

from modtox.modtox.utils.enums import StandardTypes, Database


@dataclass(frozen=True, order=True)
class Standard:
    """Represents a standard. 1 to 1 relationship with Activity.
    """
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
    """Represents a activity. It is related to a molecule by InChI."""
    inchi: str
    standard: Standard
    database: Database
    target: str

    def __repr__(self) -> str:
        return f"{self.standard} ({self.database.name})"

    def __str__(self) -> str:
        return f"{self.standard} ({self.database.name})"
