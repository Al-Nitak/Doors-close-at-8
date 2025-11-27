"""Type definitions and dataclasses for the Ludo environment."""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CaptureEvent:
    """Event representing a pawn capture."""

    player: int
    pawn: int


@dataclass
class SpecialSquareEvent:
    """Event representing landing on a special square (shortcut or construction zone)."""

    player: int
    pawn: int
    square: int
    square_type: str  # "shortcut" or "construction"
    steps: int  # positive for shortcut, negative for construction
    from_position: int
    to_position: int


@dataclass
class StepInfo:
    """Information about a single step in the game."""

    roll: int
    bonus_primary: int
    bonus_secondary: int
    turns: int
    action_key: Tuple
    player: int
    primary: int
    secondary: Optional[int]
    primary_steps: int
    secondary_steps: int
    positions_before: Tuple[int, ...]
    positions_after: Tuple[int, ...]
    captures: List[CaptureEvent]
    special_square_events: List[SpecialSquareEvent]


@dataclass
class ActionOption:
    """Represents a possible action the agent can take."""

    key: Tuple
    player: int
    primary: int
    primary_steps: int
    primary_entry: bool = False
    secondary: Optional[int] = None
    secondary_steps: int = 0
    secondary_entry: bool = False

