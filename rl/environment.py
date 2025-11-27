import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class StepInfo:
    roll: int
    bonus_primary: int
    bonus_secondary: int
    turns: int
    action_key: Tuple


@dataclass
class ActionOption:
    key: Tuple
    primary: int
    primary_steps: int
    secondary: Optional[int] = None
    secondary_steps: int = 0


@dataclass
class LudoEnvironment:
    board_size: int = 30
    num_pawns: int = 4
    dice_sides: int = 6
    power_squares: List[int] = field(default_factory=list)
    power_bonus: int = 1
    max_turns: int = 500
    seed: int | None = None
    allow_odd_split: bool = False

    def __post_init__(self) -> None:
        if self.board_size <= 0:
            raise ValueError("board_size must be positive")
        if self.num_pawns <= 0:
            raise ValueError("num_pawns must be positive")
        if self.dice_sides < 2:
            raise ValueError("dice_sides must be at least 2")
        self.rng = random.Random(self.seed)
        self.power_squares = [p for p in self.power_squares if 0 <= p <= self.board_size]
        self.reset()

    def reset(self) -> Tuple[int, ...]:
        self.positions = [0 for _ in range(self.num_pawns)]
        self.turns = 0
        self.pending_roll = self.rng.randint(1, self.dice_sides)
        return self.state

    @property
    def state(self) -> Tuple[int, ...]:
        return tuple(self.positions)

    def available_actions(self) -> List[ActionOption]:
        roll = getattr(self, "pending_roll", None)
        if roll is None:
            return []

        unfinished = [idx for idx, pos in enumerate(self.positions) if pos < self.board_size]
        options: List[ActionOption] = []

        for pawn in unfinished:
            steps = self._calc_steps(pawn, roll)
            options.append(
                ActionOption(
                    key=("single", pawn, roll),
                    primary=pawn,
                    primary_steps=steps,
                )
            )

        if (
            self.allow_odd_split
            and roll % 2 == 1
            and len(unfinished) >= 2
            and roll > 1
        ):
            for primary in unfinished:
                for secondary in unfinished:
                    if primary == secondary:
                        continue
                    for first_allocation in range(1, roll):
                        second_allocation = roll - first_allocation
                        steps_primary = self._calc_steps(primary, first_allocation)
                        steps_secondary = self._calc_steps(secondary, second_allocation)
                        key = ("split", roll, primary, secondary, first_allocation)
                        options.append(
                            ActionOption(
                                key=key,
                                primary=primary,
                                primary_steps=steps_primary,
                                secondary=secondary,
                                secondary_steps=steps_secondary,
                            )
                        )

        return options

    def is_finished(self, pawn_index: int) -> bool:
        return self.positions[pawn_index] >= self.board_size

    def step(self, option: ActionOption) -> Tuple[Tuple[int, ...], float, bool, StepInfo]:
        self.turns += 1
        reward = -0.5  # time penalty
        roll = getattr(self, "pending_roll", None)

        if roll is None:
            return self.state, reward - 5.0, False, StepInfo(roll=0, bonus_primary=0, bonus_secondary=0, turns=self.turns, action_key=("invalid",))

        if option.primary not in range(self.num_pawns) or self.is_finished(option.primary):
            return self.state, reward - 2.5, False, StepInfo(
                roll=roll,
                bonus_primary=0,
                bonus_secondary=0,
                turns=self.turns,
                action_key=("invalid",),
            )

        total_moved = 0
        bonus_primary = self._bonus(option.primary)
        move_primary = min(option.primary_steps, self.board_size - self.positions[option.primary])
        self.positions[option.primary] += move_primary
        total_moved += move_primary

        bonus_secondary = 0
        if option.secondary is not None and not self.is_finished(option.secondary):
            bonus_secondary = self._bonus(option.secondary)
            move_secondary = min(option.secondary_steps, self.board_size - self.positions[option.secondary])
            self.positions[option.secondary] += move_secondary
            total_moved += move_secondary

        reward += total_moved * 0.1  # reward progress

        if self.is_finished(option.primary):
            reward += 5.0

        if option.secondary is not None and self.is_finished(option.secondary):
            reward += 5.0

        done = all(self.is_finished(idx) for idx in range(self.num_pawns))

        if done:
            reward += 25.0

        if self.turns >= self.max_turns:
            done = True
            reward -= 5.0

        info = StepInfo(
            roll=roll,
            bonus_primary=bonus_primary,
            bonus_secondary=bonus_secondary,
            turns=self.turns,
            action_key=option.key,
        )

        self.pending_roll = None if done else self.rng.randint(1, self.dice_sides)
        return self.state, reward, done, info

    def copy(self) -> "LudoEnvironment":
        return LudoEnvironment(
            board_size=self.board_size,
            num_pawns=self.num_pawns,
            dice_sides=self.dice_sides,
            power_squares=list(self.power_squares),
            power_bonus=self.power_bonus,
            max_turns=self.max_turns,
            seed=self.rng.randint(0, 9999999),
            allow_odd_split=self.allow_odd_split,
        )

    def _calc_steps(self, pawn_index: int, base_roll: int) -> int:
        bonus = self._bonus(pawn_index)
        total = base_roll + bonus
        return min(total, self.board_size - self.positions[pawn_index])

    def _bonus(self, pawn_index: int) -> int:
        return self.power_bonus if self.positions[pawn_index] in self.power_squares else 0
