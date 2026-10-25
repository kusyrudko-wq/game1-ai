# Project instructions for coding assistants

## Context
This is a beginner's AI-learning project, starting with 3x3 tic-tac-toe and
later aiming for Connect Four with a trained neural network. The user knows
some Python and mathematics but is new to Git and machine learning.
Explain work to the user in Slovak, intuitively, defining symbols before using
them. Keep code identifiers and the main README in English.

## Scope and architecture
- Keep this milestone dependency-free: Python standard library only.
- game.py contains rules; agent.py the learned table; training.py self-play;
  evaluation.py the independent minimax reference and frozen-policy audits.
- The current model is TABULAR reinforcement learning, NOT a neural network.
- Never put minimax values or expert moves into training without explicitly
  changing and documenting the experimental design. The learner must not call
  the solver at inference time either.
- Current-player perspective switches each turn: negate the opponent's
  continuation value. Mask illegal actions in training AND evaluation.
- Preserve deterministic inference tie-breaking used by exhaustive checks.
- Do not introduce a neural network, web framework, Docker, cloud services, or
  dependencies until a task explicitly calls for that new milestone.

## Workflow
Read README.md and START_HERE_SK.md. Before changing code, run:

    python -m unittest discover -s tests -v
    python -m tictactoe evaluate --games 1000

Make one small understandable change at a time. Add tests and run them again.
Explain changed files and the result. Do not weaken tests to hide a regression.
Distinguish actual executed results from proposed commands and predictions.

## Reproducibility and honesty
- Use explicit training and evaluation seeds; never evaluate with exploration.
- Evaluating a QAgent must not mutate its table or train it.
- Keep the demo checkpoint and its reports reproducible. Do not overwrite them
  as part of unrelated tasks. Save new experiments in models/ and runs/.
- A certificate from the empty board differs from optimality in every reachable
  position. State its scope and the assumptions about engine/evaluator correctness.
- Never invent results, Elo, compute usage, user contributions, or a successful push.
- Source generation was AI-assisted. Help the user understand and extend it rather
  than claim they independently authored work that they have not reviewed.
- Never stage secrets, credentials, .env, or virtual environments. Do not publish,
  force-push, delete files, or make a repository public without an explicit request.
