# Self-Play Game AI

A small, inspectable reinforcement-learning project: train a tic-tac-toe player
from its own games, then test whether it can actually be beaten.

**Beginner instructions in Slovak: [START_HERE_SK.md](START_HERE_SK.md).**

This first milestone uses **tabular Q-learning, not a neural network**. The
learner receives the rules and terminal rewards, not expert moves. A separate
minimax solver is used only as an evaluation opponent and reference checker.
The trained playing policy performs table lookups; it does not search the game
tree or call an external AI service.

## Quick start

Python 3.10+; no third-party packages, API keys, GPU setup, or downloads required
after obtaining the repository. Executed locally on Python 3.13.5 / Linux;
a GitHub Actions workflow for Python 3.12 and 3.13 on Linux and Windows is included
but has not been run on GitHub as part of creating this starter.

From the directory containing this README:

```powershell
# Windows: play immediately against the bundled learned model
py -m tictactoe play

# Play second instead
py -m tictactoe play --human O

# Check the implementation and measure the bundled model
py -m unittest discover -s tests -v
py -m tictactoe evaluate

# Train a NEW model, starting from an empty table
py -m tictactoe train --episodes 200000 --seed 42

# Explicitly select YOUR new model when testing or playing
py -m tictactoe evaluate --model models/my-agent.json
py -m tictactoe play --model models/my-agent.json
```

On macOS/Linux, use `python3` instead of `py`. On Windows, `python` is also
suitable when it is the configured Python command. Human moves are numbered 1–9;
`q` quits. The first version intentionally has a terminal interface, not a GUI.

## Measured results

The bundled `models/demo.json` was trained for **200,000 self-play games**, seed
42. Evaluation used seed 7, no exploration or learning, and alternated which side
the agent played. Each row below uses 10,000 games against the named opponent.

| Model | Opponent | Wins | Draws | Losses |
|---|---|---:|---:|---:|
| Empty table | Random | 6,100 | 435 | 3,465 |
| Trained table | Random | 8,992 | 1,008 | 0 |
| Empty table | Optimal minimax | 0 | 0 | 10,000 |
| Trained table | Optimal minimax | 0 | 10,000 | 0 |

The empty-table baseline deterministically selects the first legal square. It is
**not a random player** and is not representative of all possible untrained
policies. Opponent tie-breaking in minimax games is randomized among optimal
moves. A draw is the correct outcome when both players play optimally here.

The sample matches are not the whole evidence:

- An exhaustive best-response search against the frozen learned policy found
  **no possible loss from the empty board, for either side**.
- An all-state audit found **0 suboptimal actions in 4,520 reachable nonterminal
  positions**, measured by exact minimax win/draw/loss value.

These checks certify the particular checkpoint and deterministic tie-break under
the implemented rules and evaluator. They do not guarantee the result of every
training run, optimal exploitation of a weak opponent, or the correctness of
all possible modifications. See [MODEL_CARD.md](MODEL_CARD.md) for scope.

Raw results, per-side splits, training logs, and learning-curve data are in
[reports/](reports/). All-state audits are evaluation only; no solver information
is used to update the model.

## What is where?

| File | Purpose |
|---|---|
| `tictactoe/game.py` | Legal moves, turn switching, wins, draws, board encoding |
| `tictactoe/agent.py` | Q table, exploration, value updates, JSON save/load |
| `tictactoe/training.py` | Self-play training from scratch |
| `tictactoe/evaluation.py` | Frozen-policy benchmarks and exact verification |
| `tictactoe/__main__.py` | Commands and human-vs-AI terminal game |
| `tests/` | Rule, learning, reproducibility, and demo-model regression tests |
| `experiments.py` | Reproduce checkpoint, benchmarks, and learning-curve data |
| `AGENTS.md` | Project-specific instructions for coding assistants |

To understand the algorithm, read [docs/LEARNING_SK.md](docs/LEARNING_SK.md).
The important two-player detail is that the next state belongs to the opponent:
its continuation value must be **negated**. Illegal actions are masked in both
action selection and learning targets.

## Reproducibility and experiments

```powershell
# Different training budget and a separate checkpoint/log
py -m tictactoe train --episodes 10000 --seed 1 --output models/short-run.json --log runs/short-run.csv
py -m tictactoe evaluate --model models/short-run.json --games 10000 --seed 7 --output runs/short-evaluation.json

# Reproduce the entire bundled experiment into runs/reproduction/
py experiments.py
```

`train` always starts from scratch. Existing model or log paths are protected;
use new paths or explicitly pass `--overwrite` to replace them. This is not a
resume-training flag. The self-play log records exploratory training outcomes,
not independent playing strength. The separate learning curve measures frozen
snapshots against evaluation opponents.

New models and `runs/` are ignored by Git by default. The small bundled demo
checkpoint is an explicit exception so a fresh clone is immediately playable.
The reproduction script refuses to overwrite an existing reproduction directory.

## Next milestones

This is a learning starter, not a claim to state-of-the-art game AI. Planned,
**not implemented**: Connect Four and stronger search. A Tkinter GUI and a
separate PyTorch DQN pipeline are available. See [docs/ROADMAP.md](docs/ROADMAP.md).

For a portfolio, document experiments you actually ran and can explain. This
starter was generated and tested with AI assistance. Independent modifications,
analysis, failures, and measured improvements should be described accurately.

## References

- Sutton and Barto, *Reinforcement Learning: An Introduction*:
  https://incompleteideas.net/book/the-book-2nd.html
- PyTorch's official DQN tutorial (for a later neural-network milestone):
  https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- Silver et al., *Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm* (advanced direction, not this implementation):
  https://arxiv.org/abs/1712.01815
- GitHub Desktop first-repository guide:
  https://docs.github.com/en/desktop/overview/creating-your-first-repository-using-github-desktop
- Codex project instructions:
  https://developers.openai.com/codex/agent-configuration/agents-md

## Graphical interface

On a desktop Python installation with Tkinter, run:

```bash
python play_gui.py
```

Choose **Play as X** or **Play as O**, click a free cell, and play against the trained model in `models/demo.json`. The GUI does not retrain the model and does not use minimax. A graphical display is required; on headless servers use the terminal interface instead. Slovak implementation explanation: [`docs/EXPLAINED_SK.md`](docs/EXPLAINED_SK.md).

## Optional neural agent and DQN training

`tictactoe/neural_agent.py` adds a separate PyTorch network with architecture
9 -> 64 -> 64 -> 9 and ReLU between linear layers. Only this optional module
and the neural training/evaluation tools and tests require PyTorch; the existing tabular agent, CLI, and GUI keep
using the original model. Install PyTorch in your environment to run the full
suite including the neural tests.

The network receives the current player's board (own pieces = 1, opponent
pieces = -1, empty = 0). `NeuralAgent.choose(position)` selects the highest
predicted Q-value among legal moves, breaking ties by lowest square index.
A Q-value represents a predicted value of choosing that square. The initial
weights are random until trained. Neural playing strength must be measured separately.

```python
from tictactoe.game import Position
from tictactoe.neural_agent import NeuralAgent

agent = NeuralAgent()
move = agent.choose(Position())  # zero-based legal square index
```

On Windows, run all tests with `.venv/Scripts/python.exe -m unittest discover -s tests -v`.



### Train and evaluate DQN

```powershell
.venv/Scripts/python.exe -m tictactoe train-neural --episodes 20000 --seed 42 --run-dir runs/my-dqn
.venv/Scripts/python.exe -m tictactoe evaluate-neural --model runs/my-dqn/weights.pt --games 1000 --seed 7 --output runs/my-dqn/evaluation.json
.venv/Scripts/python.exe -m tictactoe train-neural --resume runs/my-dqn/checkpoint-010000.pt --run-dir runs/my-dqn-resumed
```

Use a new run directory each time. Output directories must be under `runs/`;
existing artifacts are never overwritten. `weights.pt` is the inference model;
`checkpoint-*.pt` additionally stores the online/target networks, Adam state,
replay buffer, random-generator state, configuration, counters, and metric
history. Resume continues to the original configured episode budget and uses
the saved configuration, ignoring training flags supplied with `--resume`.
Checkpoints are written at report boundaries and at completion. `metrics.jsonl`
contains interval training outcomes/losses; `training-summary.json` includes
measured elapsed time. Training outcomes include exploration and are not an
independent strength measurement.

Both sides share the network and generate only their own self-play experience.
Replay samples are uniform; Adam minimizes Huber loss on minibatches. A frozen
target network is copied from the online network every `--target-update`
gradient updates. Nonterminal targets are **negative gamma times the best legal
next-player target-network value**. Terminal targets are the final reward from
the acting player's perspective (win 1, draw 0), with no bootstrap. The previous
player learns about a loss through the negative continuation value. No minimax
labels, table values, symmetry augmentation, or expert moves enter training.

`train-neural --help` lists all hyperparameters, including replay capacity,
warmup moves, batch size, update frequency, target-copy frequency, learning
rate, gradient clipping, gamma, epsilon schedule, report interval, and CPU
thread count. Epsilon decreases by episode to its floor and controls only
training exploration. The implementation uses CPU and defaults to one thread.
Reproducibility is tested on the same runtime; cross-version/hardware bitwise
identity is not promised.

Evaluation uses deterministic greedy moves without exploration or updates.
Games alternate X/O; `--games` is the total per opponent. Reports include each
side separately, exhaustive best replies from the empty board, and minimax
comparison in all 4,520 reachable nonterminal states. A no-loss result from the
empty board differs from all-state optimality, and both assume correct game
rules and evaluation code. The original `train`, `evaluate`, terminal play,
GUI, and tabular demo checkpoint keep their original behavior.

Slovak explanation and executed experiment results: [docs/DQN_SK.md](docs/DQN_SK.md).


### Reproducible DQN experiments

```powershell
.venv/Scripts/python.exe -m tictactoe.dqn_experiments --output runs/my-dqn-search --games 1000 --evaluation-seed 7 --budget-seconds 1800
```

This runs a fixed 12-configuration screening plan (training seed 42), then
repeats the two best eligible configurations with seeds 43 and 44. It tests
learning rates, discount factors, exploration schedules, target-copy intervals,
hidden-layer widths, and episode budgets. The original 64-wide model remains
the default; `train-neural --width` configures both hidden layers. Old weights
and checkpoints remain readable.

Every trained model receives random/minimax matches as X/O, an all-state audit,
and exhaustive best replies from the empty board. Selection first requires no
forced loss as either side, then minimizes all-state audit errors. Minimax is
used for evaluation and hyperparameter/model selection only, never as training
labels. All candidates start from scratch. This is a targeted small search,
not a full factorial study or an unbiased held-out performance estimate.

The output includes `manifest.json` (fixed plan, seeds, environment, baseline
and source SHA-256 hashes), a source snapshot, each candidate's complete
checkpoint/weights/metrics/report, an incrementally updated `leaderboard.csv`,
and `best-checkpoint.pt` / `best-weights.pt`. Baseline contents are hashed again
at completion to verify preservation. The output directory must be new and
under `runs/`. The time budget is a soft wall-clock limit checked before each
trial; an active trial finishes. `summary.json` records whether all planned
screening and followup trials completed.

Executed search and Slovak interpretation: [docs/DQN_EXPERIMENTS_SK.md](docs/DQN_EXPERIMENTS_SK.md).
