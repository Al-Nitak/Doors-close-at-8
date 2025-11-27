"""Action option building logic."""

from typing import List, Optional

from .board_utils import BoardUtils
from .types import ActionOption


class ActionBuilder:
    """Builds available action options for the environment."""

    def __init__(self, env):
        """Initialize with reference to environment."""
        self.env = env
        self.board_utils = BoardUtils()

    def build_single_option(
        self, player: int, pawn: int, roll: int
    ) -> Optional[ActionOption]:
        """Build a single move option for a pawn."""
        progress = self.env.positions[player][pawn]
        if progress < 0:
            if roll != 6 or not self._can_enter_base(player):
                return None
            return ActionOption(
                key=("entry", player, pawn),
                player=player,
                primary=pawn,
                primary_steps=0,
                primary_entry=True,
            )

        steps = self._calc_steps(player, pawn, roll)
        if steps is None or steps <= 0:
            return None

        if not self._can_land_without_friend(player, pawn, steps):
            return None

        return ActionOption(
            key=("single", player, pawn, roll),
            player=player,
            primary=pawn,
            primary_steps=steps,
        )

    def _calc_steps(self, player: int, pawn: int, base_roll: int) -> Optional[int]:
        """Calculate the number of steps a pawn can move."""
        progress = self.env.positions[player][pawn]
        if progress < 0 or progress >= self.env.board_size:
            return None
        bonus = self._bonus(player, pawn)
        max_steps = self.env.board_size - progress
        return min(base_roll + bonus, max_steps)

    def _bonus(self, player: int, pawn: int) -> int:
        """Calculate bonus steps from power squares."""
        progress = self.env.positions[player][pawn]
        if progress < 0 or progress >= self.env.board_size:
            return 0
        square = self.board_utils.absolute_square_from_progress(
            player,
            progress,
            self.env.start_offsets,
            self.env.board_size,
            self.env.num_pawns,
        )
        return self.env.power_bonus if square in self.env.power_squares else 0

    def _can_land_without_friend(self, player: int, pawn: int, steps: int) -> bool:
        """Check if pawn can land without friendly blocking."""
        # Use validator from environment if available, otherwise create one
        if hasattr(self.env, '_validator'):
            return self.env._validator.can_land_without_friend(player, pawn, steps)
        from .validation import MoveValidator
        validator = MoveValidator(self.env)
        return validator.can_land_without_friend(player, pawn, steps)

    def _can_enter_base(self, player: int) -> bool:
        """Check if pawn can enter base."""
        # Use validator from environment if available, otherwise create one
        if hasattr(self.env, '_validator'):
            return self.env._validator.can_enter_base(player)
        from .validation import MoveValidator
        validator = MoveValidator(self.env)
        return validator.can_enter_base(player)

