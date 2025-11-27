# Ludo Reinforcement Learning Lab

A lightweight Flask application for experimenting with new Ludo rule sets. Tune dice sides, board size, and power squares, then run reinforcement learning simulations to gauge how quickly a trained agent can finish the board.

## Features
- Adjustable board size, dice sides, number of players (up to 4), pawns per player, and power-square layout.
- Optional odd-roll split rule: when a die shows an odd value, you can divide it into two moves for different pawns.
- Authentic Ludo constraints: pawns must roll a six to leave base and can capture opponent pawns (configurable toggle).
- Board topology that enforces unique starting offsets per player (board size must be divisible by player count) and home queues where finished pawns line up on consecutive squares.
- Q-learning agent with tunable hyper-parameters (learning rate, discount, epsilon schedule).
- Rich episode viewer that lists every die roll and pawn movement for the most recent training games.
- Summary dashboard with training history and evaluation runs.

## Getting Started
1. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Launch the Flask app**
   ```bash
   flask --app app run --debug
   ```
   Then open http://127.0.0.1:5000/ in your browser.

## How it Works
- The environment models a simplified single-player Ludo track. Each pawn advances by the dice roll and gains +1 whenever it starts a turn on a designated power square.
- Rewards encourage finishing pawns quickly: small time penalty per move, bonus for finishing pawns and clearing the board.
- After training for the requested episodes, the app runs evaluation games with a near-greedy policy and reports average reward/turn counts.

## Customization Ideas
- Tweak reward shaping or add penalties for overcrowding squares.
- Extend the environment to multiple agents competing for the same track.
- Log raw trajectories to feed downstream analytics.
