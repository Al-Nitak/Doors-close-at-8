from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .action_builder import ActionBuilder
from .board_utils import BoardUtils
from .movement import MovementHandler
from .types import ActionOption, CaptureEvent, SpecialSquareEvent, StepInfo
from .validation import MoveValidator


@dataclass
class LudoEnvironment:
    board_size: int = 32
    num_players: int = 4
    num_pawns: int = 4
    dice_sides: int = 6
    power_squares: List[int] = field(default_factory=list)
    power_bonus: int = 1
    shortcut_squares: Dict[int, int] = field(default_factory=dict)
    construction_zones: Dict[int, int] = field(default_factory=dict)
    max_turns: int = 500
    seed: int | None = None
    allow_odd_split: bool = False
    enable_capture: bool = True

    def __post_init__(self) -> None:
        if self.board_size <= 0:
            raise ValueError("board_size must be positive")
        if self.num_players < 1 or self.num_players > 4:
            raise ValueError("num_players must be between 1 and 4")
        if self.board_size % self.num_players != 0:
            raise ValueError("board_size must be divisible by num_players")
        if self.num_pawns <= 0:
            raise ValueError("num_pawns must be positive")
        if self.dice_sides < 2:
            raise ValueError("dice_sides must be at least 2")
        self.segment = self.board_size // self.num_players
        self.start_offsets = BoardUtils.calculate_start_offsets(self.board_size, self.num_players)
        self.rng = random.Random(self.seed)
        self.power_squares = [p % self.board_size for p in self.power_squares]
        self.shortcut_squares = BoardUtils.normalize_effects(self.shortcut_squares, self.board_size)
        self.construction_zones = BoardUtils.normalize_effects(self.construction_zones, self.board_size)

        # Initialize helper handlers
        self._movement = MovementHandler(self)
        self._validator = MoveValidator(self)
        self._action_builder = ActionBuilder(self)
        self._board_utils = BoardUtils()

        self.reset()

    def reset(self) -> Tuple[int, ...]:
        self.positions = [[-1 for _ in range(self.num_pawns)] for _ in range(self.num_players)]
        self.turns = 0
        self.current_player = 0
        self.pending_roll = self.rng.randint(1, self.dice_sides)
        return self.state

    @property
    def state(self) -> Tuple[int, ...]:
        flat = [pos for player in self.positions for pos in player]
        flat.append(self.current_player)
        return tuple(flat)

    def available_actions(self) -> List[ActionOption]:
        roll = getattr(self, "pending_roll", None)
        if roll is None:
            return []

        player = self.current_player
        unfinished = [idx for idx, progress in enumerate(self.positions[player]) if progress < self.board_size]
        options: List[ActionOption] = []

        for pawn in unfinished:
            option = self._action_builder.build_single_option(player, pawn, roll)
            if option:
                options.append(option)

        movable_for_split = [pawn for pawn in unfinished if self.positions[player][pawn] >= 0]

        if (
            self.allow_odd_split
            and roll % 2 == 1
            and len(movable_for_split) >= 2
            and roll > 1
        ):
            for primary in movable_for_split:
                for secondary in movable_for_split:
                    if primary == secondary:
                        continue
                    for first_allocation in range(1, roll):
                        second_allocation = roll - first_allocation
                        steps_primary = self._action_builder._calc_steps(player, primary, first_allocation)
                        steps_secondary = self._action_builder._calc_steps(player, secondary, second_allocation)
                        if steps_primary is None or steps_secondary is None:
                            continue
                        if steps_primary <= 0 and steps_secondary <= 0:
                            continue
                        if not self._validator.can_land_without_friend(player, primary, steps_primary):
                            continue
                        if not self._validator.can_land_without_friend(player, secondary, steps_secondary, exclude=primary):
                            continue
                        key = ("split", player, roll, primary, secondary, first_allocation)
                        options.append(
                            ActionOption(
                                key=key,
                                player=player,
                                primary=primary,
                                primary_steps=steps_primary,
                                secondary=secondary,
                                secondary_steps=steps_secondary,
                            )
                        )

        if not options:
            options.append(
                ActionOption(
                    key=("pass", player, roll),
                    player=player,
                    primary=-1,
                    primary_steps=0,
                )
            )

        return options

    def is_finished(self, player_index: int, pawn_index: int) -> bool:
        return self.positions[player_index][pawn_index] >= self.board_size

    def step(self, option: ActionOption) -> Tuple[Tuple[int, ...], float, bool, StepInfo]:
        self.turns += 1
        reward = -0.1  # small time penalty per turn (reduced from -0.5)
        roll = getattr(self, "pending_roll", None)

        if roll is None:
            return self.state, reward - 5.0, False, self._invalid_info()

        if option.player != self.current_player:
            return self.state, reward - 5.0, False, self._invalid_info()

        if option.key[0] == "pass":
            info = StepInfo(
                roll=roll,
                bonus_primary=0,
                bonus_secondary=0,
                turns=self.turns,
                action_key=option.key,
                player=option.player,
                primary=-1,
                secondary=None,
                primary_steps=0,
                secondary_steps=0,
                positions_before=self.state,
                positions_after=self.state,
                captures=[],
                special_square_events=[],
            )
            done = self._check_global_end()
            self._advance_turn(done)
            return self.state, reward - 0.1, done, info  # reduced penalty for pass action

        if option.primary not in range(self.num_pawns):
            return self.state, reward - 5.0, False, self._invalid_info()

        positions_before = self.state
        total_moved = 0
        captures: List[CaptureEvent] = []
        special_square_events: List[SpecialSquareEvent] = []

        bonus_primary = 0 if option.primary_entry else self._action_builder._bonus(option.player, option.primary)
        bonus_secondary = 0
        primary_steps = 0
        secondary_steps = 0

        if option.primary_entry:
            if roll != 6 or self.positions[option.player][option.primary] != -1:
                return self.state, reward - 2.0, False, self._invalid_info()
            if not self._validator.can_enter_base(option.player):
                return self.state, reward - 2.0, False, self._invalid_info()
            # Enter at position 0 (start square for the player)
            # Start squares are evenly distributed around the board
            self.positions[option.player][option.primary] = 0
            capture_events = self._movement.resolve_captures(option.player, option.primary)
            captures.extend(capture_events)
            # Continue moving if start square is a special square (cannot stop on special squares)
            additional_captures, additional_events = self._movement._continue_until_non_special_square(option.player, option.primary)
            captures.extend(additional_captures)
            special_square_events.extend(additional_events)
        else:
            primary_steps = option.primary_steps
            capture_events, special_events = self._movement.advance_pawn(option.player, option.primary, primary_steps)
            total_moved += primary_steps
            captures.extend(capture_events)
            special_square_events.extend(special_events)

        if option.secondary is not None:
            if option.secondary not in range(self.num_pawns):
                return self.state, reward - 5.0, False, self._invalid_info()
            if option.secondary_entry:
                if roll != 6 or self.positions[option.player][option.secondary] != -1:
                    return self.state, reward - 2.0, False, self._invalid_info()
                if not self._validator.can_enter_base(option.player):
                    return self.state, reward - 2.0, False, self._invalid_info()
                # Enter at position 0 (start square for the player)
                # Start squares are evenly distributed around the board
                self.positions[option.player][option.secondary] = 0
                capture_events = self._movement.resolve_captures(option.player, option.secondary)
                captures.extend(capture_events)
                # Continue moving if start square is a special square (cannot stop on special squares)
                additional_captures, additional_events = self._movement._continue_until_non_special_square(option.player, option.secondary)
                captures.extend(additional_captures)
                special_square_events.extend(additional_events)
            else:
                bonus_secondary = self._action_builder._bonus(option.player, option.secondary)
                secondary_steps = option.secondary_steps
                capture_events, special_events = self._movement.advance_pawn(option.player, option.secondary, secondary_steps)
                total_moved += secondary_steps
                captures.extend(capture_events)
                special_square_events.extend(special_events)

        reward += total_moved * 0.2  # increased from 0.1 to make movement more rewarding
        reward += len(captures) * 2.0  # increased from 1.5 to make captures more rewarding

        if self._player_finished(option.player):
            reward += 5.0

        player_zero_done = self._player_finished(0)
        other_winner = any(
            self._player_finished(idx) for idx in range(1, self.num_players)
        )

        done = False
        if player_zero_done:
            reward += 25.0
            done = True
        elif other_winner:
            reward -= 15.0
            done = True

        if self.turns >= self.max_turns:
            done = True
            reward -= 5.0

        positions_after = self.state
        info = StepInfo(
            roll=roll,
            bonus_primary=bonus_primary,
            bonus_secondary=bonus_secondary,
            turns=self.turns,
            action_key=option.key,
            player=option.player,
            primary=option.primary,
            secondary=option.secondary,
            primary_steps=primary_steps,
            secondary_steps=secondary_steps,
            positions_before=positions_before,
            positions_after=positions_after,
            captures=captures,
            special_square_events=special_square_events,
        )

        self._advance_turn(done)
        return self.state, reward, done, info

    def copy(self) -> "LudoEnvironment":
        return LudoEnvironment(
            board_size=self.board_size,
            num_players=self.num_players,
            num_pawns=self.num_pawns,
            dice_sides=self.dice_sides,
            power_squares=list(self.power_squares),
            power_bonus=self.power_bonus,
            shortcut_squares=dict(self.shortcut_squares),
            construction_zones=dict(self.construction_zones),
            max_turns=self.max_turns,
            seed=self.rng.randint(0, 9999999),
            allow_odd_split=self.allow_odd_split,
            enable_capture=self.enable_capture,
        )

    # --- Helpers -----------------------------------------------------------------

    def _player_finished(self, player: int) -> bool:
        """Check if a player has finished all their pawns."""
        return all(progress >= self.board_size for progress in self.positions[player])

    # Convenience methods for backward compatibility and internal use
    def _absolute_square_from_progress(self, player: int, progress: int) -> int:
        """Convert player progress to absolute board square number."""
        return self._board_utils.absolute_square_from_progress(
            player, progress, self.start_offsets, self.board_size, self.num_pawns
        )

    def _advance_turn(self, done: bool) -> None:
        if done:
            self.pending_roll = None
            return
        self.current_player = (self.current_player + 1) % self.num_players
        self.pending_roll = self.rng.randint(1, self.dice_sides)

    def _check_global_end(self) -> bool:
        player_zero_done = self._player_finished(0)
        if player_zero_done:
            return True
        return self.turns >= self.max_turns

    def _invalid_info(self) -> StepInfo:
        return StepInfo(
            roll=0,
            bonus_primary=0,
            bonus_secondary=0,
            turns=self.turns,
            action_key=("invalid",),
            player=-1,
            primary=-1,
            secondary=None,
            primary_steps=0,
            secondary_steps=0,
            positions_before=self.state,
            positions_after=self.state,
            captures=[],
            special_square_events=[],
        )
