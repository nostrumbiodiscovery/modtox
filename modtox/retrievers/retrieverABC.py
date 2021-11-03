from abc import ABC, abstractmethod
from dataclasses import dataclass

class Retriever(ABC):
    def __init__(self) -> None:
        self.ids = dict()
        self.activities = list()
    @abstractmethod
    def retrieve_by_target(self):
        """Accepts UniProt accession code as argument. Sets self.ids
        and self.activities attributes."""
