"""Move validation logic."""

from typing import Optional

from .board_utils import BoardUtils


class MoveValidator:
    """Validates whether moves are legal."""

    def __init__(self, env):
        """Initialize with reference to environment."""
        self.env = env
        self.board_utils = BoardUtils()

    def can_land_without_friend(
        self, player: int, pawn: int, steps: int, exclude: Optional[int] = None
    ) -> bool:
        """Check if a pawn can land on a square without a friendly pawn blocking."""
        progress = self.env.positions[player][pawn]
        if progress < 0:
            return True
        new_progress = progress + steps
        if new_progress >= self.env.board_size:
            return True  # entering finish queue

        target_square = self.board_utils.absolute_square_from_progress(
            player,
            new_progress,
            self.env.start_offsets,
            self.env.board_size,
            self.env.num_pawns,
        )

        excludes = {pawn}
        if exclude is not None:
            excludes.add(exclude)
        return not self.has_friendly_on_square(player, target_square, exclude=excludes)

    def can_enter_base(self, player: int) -> bool:
        """Check if a pawn can enter the base (starting square)."""
        start_square = self.board_utils.start_square(
            player, self.env.start_offsets, self.env.board_size
        )
        return not self.has_friendly_on_square(player, start_square)

    def has_friendly_on_square(
        self, player: int, square: int, exclude: Optional[set[int]] = None
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

