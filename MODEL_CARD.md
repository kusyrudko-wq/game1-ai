# Model card: bundled tic-tac-toe agent

## Identity and scope

File: `models/demo.json`  
Algorithm: zero-sum alternating-turn tabular Q-learning  
Neural network: **no**  
Task: 3x3 tic-tac-toe, X starts, stop immediately on win or full board  
Model SHA-256: `d9aff8583e0704790e4e61d9d6c1946c0ce32640bebd75808d42f61e20b4c573`

This is a learning example generated with AI assistance, not a novel algorithm,
commercial benchmark, or evidence of unassisted authorship by the repository owner.

## Training

200,000 self-play episodes from an empty table; training seed 42; learning rate
0.2; discount 1.0. Both sides share a current-player-relative table. Exploration
falls linearly from 1.0 to 0.1 over the first 80% of episodes and then stays at
0.1. During training, exact value ties are broken randomly. During evaluation
and normal play, exact ties use the lowest legal board index.

Immediate own win has target +1, draw has target 0. Nonterminal targets negate
the opponent's best legal continuation estimate. Expert moves, game databases,
minimax labels, external models, and search during inference are not used.
The table contains 4520 decision-state entries.

## Actual evaluation

Seed 7; 10,000 games per opponent; 5,000 with the agent as X and 5,000 as O.
Model frozen, no exploration, deterministic tie-break. Random opponents choose
uniformly among legal moves; minimax opponents randomly choose among exact
value-optimal moves.

| Opponent | Agent as X: W / D / L | Agent as O: W / D / L |
|---|---|---|
| Random | 4,974 / 26 / 0 | 4,018 / 982 / 0 |
| Optimal minimax | 0 / 5,000 / 0 | 0 / 5,000 / 0 |

An exhaustive best-response check found worst-case score 0 from the empty board
for each side. A separate all-state audit found 0 value-suboptimal decisions
across all 4,520 reachable nonterminal positions. The engine enumerated 5,478
reachable positions including terminal ones.

Thus this specific policy preserves the optimal win/draw/loss value at every
reachable decision state in the implemented game. It is not a guarantee of the
fastest possible win, best win rate against every weak opponent, convergence of
every random training run, or formally verified correctness of the engine.

## Limitations and evaluation assumptions

The reference solver and learner share the rule engine; rule errors could
therefore affect both. Rule invariants, wins/draws, illegal moves, turn
perspective, learning updates, and evaluation immutability have regression tests.
The certificate depends on the exact checkpoint, tie-break, and legal rules.
Re-run all checks after changing these. Finite sample loss rates alone do not
certify non-exploitability. Evaluation intentionally covers the complete game
state space; this is not a held-out-position generalization benchmark.

Only seed 42's main training run is presented here; a second identical-seed run
reproduced the model byte-for-byte. This checks reproducibility, not robustness
across different seeds. The empty-table comparison is a fixed first-legal-move
policy, not a uniform-random baseline.

## Reproduce

```powershell
py experiments.py
```

This writes a fresh result bundle under `runs/reproduction/`, leaving the bundled
model untouched, and prints whether its SHA-256 matches. It refuses to overwrite
an existing reproduction directory. `reports/provenance.json` records the actual
Python version and platform of the executed reproduction.

The learning curve uses 1,000 games per opponent at each 10,000-episode
checkpoint; final headline numbers use 10,000 games per opponent. Self-play
outcome logs are not the same as the independent evaluation curve.
