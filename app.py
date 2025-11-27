from __future__ import annotations

from typing import Dict, List

from flask import Flask, render_template, request

from rl.simulation import run_simulation

app = Flask(__name__)

DEFAULT_CONFIG: Dict = {
    "board_size": 52,
    "num_players": 4,
    "num_pawns": 4,
    "dice_sides": 6,
    "power_squares": [5, 12, 18, 24],
    "power_bonus": 1,
    "shortcut_squares": {},
    "construction_zones": {},
    "episodes": 200,
    "evaluation_runs": 30,
    "alpha": 0.1,
    "gamma": 0.95,
    "epsilon": 0.3,
    "min_epsilon": 0.05,
    "epsilon_decay": 0.99,
    "max_turns": 400,
    "allow_odd_split": False,
    "enable_capture": True,
}


def parse_power_squares(raw: str) -> List[int]:
    squares: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            squares.append(int(chunk))
        except ValueError:
            raise ValueError(f"Invalid power square value: {chunk}")
    return squares


def parse_square_map(raw: str, label: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    raw = raw.strip()
    if not raw:
        return mapping
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"{label} must use the format square:steps")
        square_part, step_part = chunk.split(":", 1)
        try:
            square = int(square_part.strip())
            steps = int(step_part.strip())
        except ValueError:
            raise ValueError(f"{label} values must be integers (got '{chunk}')")
        if steps == 0:
            continue
        mapping[square] = abs(steps)
    return mapping


def format_square_map(mapping: Dict[int, int]) -> str:
    if not mapping:
        return ""
    return ", ".join(f"{square}:{steps}" for square, steps in mapping.items())


def parse_form(form) -> Dict:
    config = DEFAULT_CONFIG.copy()
    config["power_squares"] = list(DEFAULT_CONFIG["power_squares"])
    config["shortcut_squares"] = dict(DEFAULT_CONFIG["shortcut_squares"])
    config["construction_zones"] = dict(DEFAULT_CONFIG["construction_zones"])
    config.update(
        {
            "board_size": int(form.get("board_size", config["board_size"])),
            "num_players": int(form.get("num_players", config["num_players"])),
            "num_pawns": int(form.get("num_pawns", config["num_pawns"])),
            "dice_sides": int(form.get("dice_sides", config["dice_sides"])),
            "power_bonus": int(form.get("power_bonus", config["power_bonus"])),
            "episodes": int(form.get("episodes", config["episodes"])),
            "evaluation_runs": int(form.get("evaluation_runs", config["evaluation_runs"])),
            "alpha": float(form.get("alpha", config["alpha"])),
            "gamma": float(form.get("gamma", config["gamma"])),
            "epsilon": float(form.get("epsilon", config["epsilon"])),
            "min_epsilon": float(form.get("min_epsilon", config["min_epsilon"])),
            "epsilon_decay": float(form.get("epsilon_decay", config["epsilon_decay"])),
            "max_turns": int(form.get("max_turns", config["max_turns"])),
            "allow_odd_split": form.get("allow_odd_split") == "on",
            "enable_capture": form.get("enable_capture") == "on",
        }
    )

    power_square_raw = form.get("power_squares", ",".join(str(s) for s in config["power_squares"]))
    config["power_squares"] = parse_power_squares(power_square_raw)
    shortcut_raw = form.get("shortcut_squares", format_square_map(config["shortcut_squares"]))
    construction_raw = form.get("construction_zones", format_square_map(config["construction_zones"]))
    config["shortcut_squares"] = parse_square_map(shortcut_raw, "Shortcut squares")
    config["construction_zones"] = parse_square_map(construction_raw, "Construction zones")
    return config


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        config=DEFAULT_CONFIG,
        result=None,
        error=None,
        shortcut_string=format_square_map(DEFAULT_CONFIG["shortcut_squares"]),
        construction_string=format_square_map(DEFAULT_CONFIG["construction_zones"]),
    )


@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        config = parse_form(request.form)
        result = run_simulation(config)
        shortcut_str = format_square_map(config["shortcut_squares"])
        construction_str = format_square_map(config["construction_zones"])
        return render_template(
            "results.html",
            config=config,
            result=result,
            training_history=result.training_history[-20:],
            evaluation=result.evaluation,
            shortcut_string=shortcut_str,
            construction_string=construction_str,
        )
    except ValueError as exc:
        return render_template(
            "index.html",
            config=DEFAULT_CONFIG,
            result=None,
            error=str(exc),
            shortcut_string=format_square_map(DEFAULT_CONFIG["shortcut_squares"]),
            construction_string=format_square_map(DEFAULT_CONFIG["construction_zones"]),
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
