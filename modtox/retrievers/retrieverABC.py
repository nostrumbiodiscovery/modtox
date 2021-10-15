from abc import ABC, abstractmethod
from dataclasses import dataclass

class Retriever(ABC):
    @abstractmethod
    def retrieve_by_target(self):
        """Must accept UniProt accession code as argument."""

@dataclass
class RetSum:
    request: str
    retrieved_molecules: int
    retrieved_activities: int