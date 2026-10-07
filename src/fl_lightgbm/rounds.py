"""The fixed round budget each site and the aggregator hold themselves to (ADR 0005)."""


class RoundCountError(RuntimeError):
    """A site or the aggregator was asked for more rounds than its budget, or the run ended before it was spent."""


class RoundCounter:
    def __init__(self, budget: int, owner: str):
        self.left = budget
        self.owner = owner

    def take(self) -> None:
        if self.left == 0:
            raise RoundCountError(f"{self.owner} was asked for a round beyond its budget")
        self.left -= 1
