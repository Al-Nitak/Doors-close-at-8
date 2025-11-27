"""Movement and special square effect handlers."""

from typing import List, Tuple

from .board_utils import BoardUtils
from .types import CaptureEvent, SpecialSquareEvent


class MovementHandler:
    """Handles pawn movement and special square effects."""

    def __init__(self, env):
        """Initialize with reference to environment."""
        self.env = env
        self.board_utils = BoardUtils()

    def advance_pawn(self, player: int, pawn: int, steps: int) -> Tuple[List[CaptureEvent], List[SpecialSquareEvent]]:
        """Move a pawn forward by the given number of steps.

        Continues moving if the pawn lands on a special square (shortcut or construction zone),
        as pawns cannot stop on these squares.

        Returns:
            Tuple of (captures, special_square_events)
        """
        captures: List[CaptureEvent] = []
        special_events: List[SpecialSquareEvent] = []
        progress = self.env.positions[player][pawn]
        if progress < 0:
            return captures, special_events

        new_progress = progress + steps
        if new_progress >= self.env.board_size:
            slot = self.board_utils.next_finish_slot(
                self.env.positions[player], self.env.board_size
            )
            self.env.positions[player][pawn] = self.env.board_size + slot
            return captures, special_events

        self.env.positions[player][pawn] = new_progress
        captures.extend(self.resolve_captures(player, pawn))

        # Continue moving if landed on a special square (cannot stop on special squares)
        additional_captures, additional_events = self._continue_until_non_special_square(player, pawn)
        captures.extend(additional_captures)
        special_events.extend(additional_events)
        return captures, special_events

    def _is_special_square(self, player: int, progress: int) -> bool:
        """Check if a position is on a special square (shortcut or construction zone)."""
        if progress < 0 or progress >= self.env.board_size:
            return False
        square = self.board_utils.absolute_square_from_progress(
            player,
            progress,
            self.env.start_offsets,
            self.env.board_size,
            self.env.num_pawns,
        )
        return square in self.env.shortcut_squares or square in self.env.construction_zones

    def _continue_until_non_special_square(
        self, player: int, pawn: int, max_iterations: int = 20
    ) -> Tuple[List[CaptureEvent], List[SpecialSquareEvent]]:
        """Continue moving a pawn until it lands on a non-special square.

        This implements the rule that pawns cannot stop on shortcut or construction squares.

        Returns:
            Tuple of (captures, special_square_events)
        """
        captures: List[CaptureEvent] = []
        special_events: List[SpecialSquareEvent] = []
        iterations = 0

        while iterations < max_iterations:
            progress = self.env.positions[player][pawn]
            if progress < 0 or progress >= self.env.board_size:
                break

            if not self._is_special_square(player, progress):
                break  # Landed on a non-special square, can stop here

            # Must continue moving - apply the special square effect
            square = self.board_utils.absolute_square_from_progress(
                player,
                progress,
                self.env.start_offsets,
                self.env.board_size,
                self.env.num_pawns,
            )

            delta = 0
            square_type = ""
            if square in self.env.shortcut_squares:
                delta = self.env.shortcut_squares[square]
                square_type = "shortcut"
            elif square in self.env.construction_zones:
                delta = -self.env.construction_zones[square]
                square_type = "construction"

            if delta == 0:
                break  # No effect, should not happen but safety check

            from_position = progress
            new_progress = progress + delta
            if new_progress < 0:
                self.env.positions[player][pawn] = -1
                # Record the event even if pawn goes back to base
                special_events.append(SpecialSquareEvent(
                    player=player,
                    pawn=pawn,
                    square=square,
                    square_type=square_type,
                    steps=delta,
                    from_position=from_position,
                    to_position=-1
                ))
                break
            if new_progress >= self.env.board_size:
                slot = self.board_utils.next_finish_slot(
                    self.env.positions[player], self.env.board_size
                )
                self.env.positions[player][pawn] = self.env.board_size + slot
                # Record the event
                special_events.append(SpecialSquareEvent(
                    player=player,
                    pawn=pawn,
                    square=square,
                    square_type=square_type,
                    steps=delta,
                    from_position=from_position,
                    to_position=self.env.board_size + slot
                ))
                break

            target_square = self.board_utils.absolute_square_from_progress(
                player,
                new_progress,
                self.env.start_offsets,
                self.env.board_size,
                self.env.num_pawns,
            )

            if self._has_friendly_on_square(player, target_square, exclude={pawn}):
                break  # Blocked by friendly pawn, must stop

            self.env.positions[player][pawn] = new_progress
            captures.extend(self.resolve_captures(player, pawn))

            # Record the special square event
            special_events.append(SpecialSquareEvent(
                player=player,
                pawn=pawn,
                square=square,
                square_type=square_type,
                steps=delta,
                from_position=from_position,
                to_position=new_progress
            ))

            iterations += 1

        return captures, special_events

    def apply_square_effect(self, player: int, pawn: int) -> List[CaptureEvent]:
        """Apply shortcut or construction zone effects when landing on special squares.

        DEPRECATED: Use _continue_until_non_special_square instead.
        Kept for backward compatibility.
        """
        return self._continue_until_non_special_square(player, pawn)

    def resolve_captures(self, player: int, pawn: int) -> List[CaptureEvent]:
        """Resolve captures when a pawn lands on an opponent's square."""
        if not self.env.enable_capture:
            return []
        progress = self.env.positions[player][pawn]
        if progress < 0 or progress >= self.env.board_size:
            return []

        square = self.board_utils.absolute_square_from_progress(
            player,
            progress,
            self.env.start_offsets,
            self.env.board_size,
            self.env.num_pawns,
        )

        captures: List[CaptureEvent] = []
        for other_player in range(self.env.num_players):
            if other_player == player:
                continue
            for other_pawn, other_progress in enumerate(self.env.positions[other_player]):
                if other_progress < 0 or other_progress >= self.env.board_size:
                    continue
                other_square = self.board_utils.absolute_square_from_progress(
                    other_player,
                    other_progress,
                    self.env.start_offsets,
                    self.env.board_size,
                    self.env.num_pawns,
                )
                if other_square == square:
                    self.env.positions[other_player][other_pawn] = -1
                    captures.append(CaptureEvent(player=other_player, pawn=other_pawn))
        return captures

    def _has_friendly_on_square(
        self, player: int, square: int, exclude: set[int] | None = None
    ) -> bool:
        """Check if a friendly pawn is on the given square."""
        excludes = exclude or set()
        for pawn_index, progress in enumerate(self.env.positions[player]):
            if pawn_index in excludes:
                continue
            if progress < 0 or progress >= self.env.board_size:
                continue
            pawn_square = self.board_utils.absolute_square_from_progress(
                player,
                progress,
                self.env.start_offsets,
                self.env.board_size,
                self.env.num_pawns,
            )
            if pawn_square == square:
                return True
        return False

