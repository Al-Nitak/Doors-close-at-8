from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CaptureEvent:
    player: int
    pawn: int


@dataclass
class StepInfo:
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


@dataclass
class ActionOption:
    key: Tuple
    player: int
    primary: int
    primary_steps: int
    primary_entry: bool = False
    secondary: Optional[int] = None
    secondary_steps: int = 0
    secondary_entry: bool = False


@dataclass
class LudoEnvironment:
    board_size: int = 52
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
        self.start_offsets = [player * self.segment for player in range(self.num_players)]
        self.rng = random.Random(self.seed)
        self.power_squares = [p % self.board_size for p in self.power_squares]
        self.shortcut_squares = self._normalize_effects(self.shortcut_squares)
        self.construction_zones = self._normalize_effects(self.construction_zones)
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
            option = self._build_single_option(player, pawn, roll)
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
                        steps_primary = self._calc_steps(player, primary, first_allocation)
                        steps_secondary = self._calc_steps(player, secondary, second_allocation)
                        if steps_primary is None or steps_secondary is None:
                            continue
                        if steps_primary <= 0 and steps_secondary <= 0:
                            continue
                        if not self._can_land_without_friend(player, primary, steps_primary):
                            continue
                        if not self._can_land_without_friend(player, secondary, steps_secondary, exclude=primary):
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
        reward = -0.5  # time penalty per turn
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
            )
            done = self._check_global_end()
            self._advance_turn(done)
            return self.state, reward - 0.2, done, info

        if option.primary not in range(self.num_pawns):
            return self.state, reward - 5.0, False, self._invalid_info()

        positions_before = self.state
        total_moved = 0
        captures: List[CaptureEvent] = []

        bonus_primary = 0 if option.primary_entry else self._bonus(option.player, option.primary)
        bonus_secondary = 0
        primary_steps = 0
        secondary_steps = 0

        if option.primary_entry:
            if roll != 6 or self.positions[option.player][option.primary] != -1:
                return self.state, reward - 2.0, False, self._invalid_info()
            if not self._can_enter_base(option.player):
                return self.state, reward - 2.0, False, self._invalid_info()
            self.positions[option.player][option.primary] = 0
            capture_events = self._resolve_captures(option.player, option.primary)
            captures.extend(capture_events)
            captures.extend(self._apply_square_effect(option.player, option.primary))
        else:
            primary_steps = option.primary_steps
            capture_events = self._advance_pawn(option.player, option.primary, primary_steps)
            total_moved += primary_steps
            captures.extend(capture_events)

        if option.secondary is not None:
            if option.secondary not in range(self.num_pawns):
                return self.state, reward - 5.0, False, self._invalid_info()
            if option.secondary_entry:
                if roll != 6 or self.positions[option.player][option.secondary] != -1:
                    return self.state, reward - 2.0, False, self._invalid_info()
                if not self._can_enter_base(option.player):
                    return self.state, reward - 2.0, False, self._invalid_info()
                self.positions[option.player][option.secondary] = 0
                capture_events = self._resolve_captures(option.player, option.secondary)
                captures.extend(capture_events)
                captures.extend(self._apply_square_effect(option.player, option.secondary))
            else:
                bonus_secondary = self._bonus(option.player, option.secondary)
                secondary_steps = option.secondary_steps
                capture_events = self._advance_pawn(option.player, option.secondary, secondary_steps)
                total_moved += secondary_steps
                captures.extend(capture_events)

        reward += total_moved * 0.1
        reward += len(captures) * 1.5

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

    def _normalize_effects(self, raw: Optional[Dict[int, int]]) -> Dict[int, int]:
        normalized: Dict[int, int] = {}
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
            normalized[square_int % self.board_size] = abs(steps_int)
        return normalized

    def _player_finished(self, player: int) -> bool:
        return all(progress >= self.board_size for progress in self.positions[player])

    def _calc_steps(self, player: int, pawn: int, base_roll: int) -> Optional[int]:
        progress = self.positions[player][pawn]
        if progress < 0 or progress >= self.board_size:
            return None
        bonus = self._bonus(player, pawn)
        max_steps = self.board_size - progress
        return min(base_roll + bonus, max_steps)

    def _bonus(self, player: int, pawn: int) -> int:
        progress = self.positions[player][pawn]
        if progress < 0 or progress >= self.board_size:
            return 0
        square = self._absolute_square_from_progress(player, progress)
        return self.power_bonus if square in self.power_squares else 0

    def _build_single_option(self, player: int, pawn: int, roll: int) -> Optional[ActionOption]:
        progress = self.positions[player][pawn]
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

    def _advance_pawn(self, player: int, pawn: int, steps: int) -> List[CaptureEvent]:
        captures: List[CaptureEvent] = []
        progress = self.positions[player][pawn]
        if progress < 0:
            return captures

        new_progress = progress + steps
        if new_progress >= self.board_size:
            slot = self._next_finish_slot(player)
            self.positions[player][pawn] = self.board_size + slot
            return captures

        self.positions[player][pawn] = new_progress
        captures.extend(self._resolve_captures(player, pawn))
        captures.extend(self._apply_square_effect(player, pawn))
        return captures

    def _apply_square_effect(self, player: int, pawn: int) -> List[CaptureEvent]:
        captures: List[CaptureEvent] = []
        progress = self.positions[player][pawn]
        if progress < 0 or progress >= self.board_size:
            return captures
        square = self._absolute_square_from_progress(player, progress)
        delta = 0
        if square in self.shortcut_squares:
            delta = self.shortcut_squares[square]
        elif square in self.construction_zones:
            delta = -self.construction_zones[square]
        if delta == 0:
            return captures

        new_progress = progress + delta
        if new_progress < 0:
            self.positions[player][pawn] = -1
            return captures
        if new_progress >= self.board_size:
            slot = self._next_finish_slot(player)
            self.positions[player][pawn] = self.board_size + slot
            return captures

        target_square = self._absolute_square_from_progress(player, new_progress)
        if self._has_friendly_on_square(player, target_square, exclude={pawn}):
            return captures

        self.positions[player][pawn] = new_progress
        captures.extend(self._resolve_captures(player, pawn))
        return captures

    def _resolve_captures(self, player: int, pawn: int) -> List[CaptureEvent]:
        if not self.enable_capture:
            return []
        progress = self.positions[player][pawn]
        if progress < 0 or progress >= self.board_size:
            return []
        square = self._absolute_square_from_progress(player, progress)
        captures: List[CaptureEvent] = []
        for other_player in range(self.num_players):
            if other_player == player:
                continue
            for other_pawn, other_progress in enumerate(self.positions[other_player]):
                if other_progress < 0 or other_progress >= self.board_size:
                    continue
                if self._absolute_square_from_progress(other_player, other_progress) == square:
                    self.positions[other_player][other_pawn] = -1
                    captures.append(CaptureEvent(player=other_player, pawn=other_pawn))
        return captures

    def _can_land_without_friend(self, player: int, pawn: int, steps: int, exclude: Optional[int] = None) -> bool:
        progress = self.positions[player][pawn]
        if progress < 0:
            return True
        new_progress = progress + steps
        if new_progress >= self.board_size:
            return True  # entering finish queue
        target_square = self._absolute_square_from_progress(player, new_progress)
        excludes = {pawn}
        if exclude is not None:
            excludes.add(exclude)
        return not self._has_friendly_on_square(player, target_square, exclude=excludes)

    def _can_enter_base(self, player: int) -> bool:
        start_square = self._start_square(player)
        return not self._has_friendly_on_square(player, start_square)

    def _has_friendly_on_square(self, player: int, square: int, exclude: Optional[set[int]] = None) -> bool:
        excludes = exclude or set()
        for pawn_index, progress in enumerate(self.positions[player]):
            if pawn_index in excludes:
                continue
            if progress < 0 or progress >= self.board_size:
                continue
            if self._absolute_square_from_progress(player, progress) == square:
                return True
        return False

    def _start_square(self, player: int) -> int:
        return self.start_offsets[player] % self.board_size

    def _next_finish_slot(self, player: int) -> int:
        return sum(1 for progress in self.positions[player] if progress >= self.board_size)

    def _finish_square(self, player: int, slot: int) -> int:
        return (self.start_offsets[player] - self.num_pawns + slot + 1) % self.board_size

    def _absolute_square_from_progress(self, player: int, progress: int) -> int:
        if progress < 0:
            return -1
        if progress >= self.board_size:
            slot = progress - self.board_size
            return self._finish_square(player, slot)
        return (self.start_offsets[player] + progress) % self.board_size

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
        )
