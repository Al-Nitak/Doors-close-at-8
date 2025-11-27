"""Board utility functions for square and position calculations."""

from typing import List


class BoardUtils:
    """Utility class for board-related calculations."""

    @staticmethod
    def calculate_start_offsets(board_size: int, num_players: int) -> List[int]:
        """Calculate the starting square offset for each player."""
        segment = board_size // num_players
        return [player * segment for player in range(num_players)]

    @staticmethod
    def start_square(player: int, start_offsets: List[int], board_size: int) -> int:
        """Get the starting square for a player."""
        return start_offsets[player] % board_size

    @staticmethod
    def finish_square(
        player: int, slot: int, start_offsets: List[int], board_size: int, num_pawns: int
    ) -> int:
        """Calculate the finish square for a pawn in a given slot."""
        return (start_offsets[player] - num_pawns + slot + 1) % board_size

    @staticmethod
    def absolute_square_from_progress(
        player: int,
        progress: int,
        start_offsets: List[int],
        board_size: int,
        num_pawns: int,
    ) -> int:
        """Convert player progress to absolute board square number."""
        if progress < 0:
            return -1
        if progress >= board_size:
            slot = progress - board_size
            return BoardUtils.finish_square(player, slot, start_offsets, board_size, num_pawns)
        return (start_offsets[player] + progress) % board_size

    @staticmethod
    def next_finish_slot(player_positions: List[int], board_size: int) -> int:
        """Calculate the next available finish slot for a player."""
        return sum(1 for progress in player_positions if progress >= board_size)

    @staticmethod
    def normalize_effects(raw: dict, board_size: int) -> dict:
        """Normalize effect squares to be within board bounds."""
        normalized: dict = {}
        if not raw:
            return normalized
        for square, steps in raw.items():
            try:
                square_int = int(square)
                steps_int = int(steps)
            except (TypeError, ValueError):
                continue
            if steps_int == 0:
                continue
            normalized[square_int % board_size] = abs(steps_int)
        return normalized

