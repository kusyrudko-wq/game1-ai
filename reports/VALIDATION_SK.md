# Overenie priloženého projektu

Projekt bol vytvorený a otestovaný 15. októbra 2026 v prostredí Linux / Python 3.13.5.

**27 automatických testov prešlo.** Boli spustené cez:

```text
python -m unittest discover -s tests -v
```

Overené bolo aj interaktívne spustenie, odmietnutie nesprávneho vstupu, hranie za
druhú stranu, tréning nového modelu, jeho načítanie, uloženie vyhodnotenia a ochrana
pred nechceným prepísaním modelu. Windows príkazy sú pripravené, ale beh na tvojom
počítači ani GitHub Actions tu neboli spustené. Na GitHub nebolo nič publikované.

## Namerané výsledky

Tréning: 200 000 partií vlastnej hry, seed 42. Vyhodnotenie: seed 7, model bez
ďalšieho učenia; 5 000 hier za prvého a 5 000 za druhého hráča proti každému súperovi.

| Súper | Výhry | Remízy | Prehry |
|---|---:|---:|---:|
| Náhodný hráč | 8 992 | 1 008 | 0 |
| Optimálny minimax | 0 | 10 000 | 0 |

Úplné prehľadanie všetkých súperových odpovedí proti pevnému správaniu modelu
nenašlo možnosť prehry z prázdnej dosky za žiadnu stranu. Audit všetkých 4 520
legálne dosiahnuteľných neukončených pozícií našiel 0 neoptimálnych rozhodnutí
z hľadiska hodnoty výhra/remíza/prehra proti optimálnemu pokračovaniu.

Výsledok platí pre konkrétny priložený model a jeho rozhodovanie pri zhode hodnôt,
za predpokladu správnosti enginu a kontrolného programu. Nie je to záruka každého
ďalšieho tréningu, najrýchlejšej výhry či maximálnej miery výhier proti slabému súperovi.

Celý tréning bol zopakovaný s rovnakým seedom; výsledný model sa zhodoval bajt po
bajte a zhodovali sa aj oba konečné vyhodnocovacie reporty. To overuje
reprodukovateľnosť tohto nastavenia, nie úspešnosť všetkých možných seedov.

## Výstup testov

```text
test_all_reachable_position_invariants (test_game.GameTests.test_all_reachable_position_invariants) ... ok
test_draw (test_game.GameTests.test_draw) ... ok
test_illegal_move (test_game.GameTests.test_illegal_move) ... ok
test_key_uses_current_player_perspective (test_game.GameTests.test_key_uses_current_player_perspective) ... ok
test_new_board (test_game.GameTests.test_new_board) ... ok
test_o_can_win (test_game.GameTests.test_o_can_win) ... ok
test_play_does_not_mutate (test_game.GameTests.test_play_does_not_mutate) ... ok
test_rows_columns_diagonals (test_game.GameTests.test_rows_columns_diagonals) ... ok
test_bad_policy_is_exploitable (test_game.SolverTests.test_bad_policy_is_exploitable) ... ok
test_forced_block (test_game.SolverTests.test_forced_block) ... ok
test_immediate_win (test_game.SolverTests.test_immediate_win) ... ok
test_perfect_play_draws (test_game.SolverTests.test_perfect_play_draws) ... ok
test_perfect_policy_cannot_lose_either_side (test_game.SolverTests.test_perfect_policy_cannot_lose_either_side) ... ok
test_demo_cannot_lose_from_empty_board (test_learning.BundledModelTests.test_demo_cannot_lose_from_empty_board) ... ok
test_demo_optimal_at_every_reachable_state (test_learning.BundledModelTests.test_demo_optimal_at_every_reachable_state) ... ok
test_bad_training_parameters_rejected (test_learning.LearningTests.test_bad_training_parameters_rejected) ... ok
test_deterministic_tie_break (test_learning.LearningTests.test_deterministic_tie_break) ... ok
test_evaluation_does_not_train (test_learning.LearningTests.test_evaluation_does_not_train) ... ok
test_invalid_model_rejected (test_learning.LearningTests.test_invalid_model_rejected) ... ok
test_json_round_trip (test_learning.LearningTests.test_json_round_trip) ... ok
test_learning_rate (test_learning.LearningTests.test_learning_rate) ... ok
test_legal_move_mask (test_learning.LearningTests.test_legal_move_mask) ... ok
test_opponent_value_must_be_negated (test_learning.LearningTests.test_opponent_value_must_be_negated) ... ok
test_terminal_draw_reward (test_learning.LearningTests.test_terminal_draw_reward) ... ok
test_terminal_win_reward_for_x_and_o (test_learning.LearningTests.test_terminal_win_reward_for_x_and_o) ... ok
test_training_is_reproducible (test_learning.LearningTests.test_training_is_reproducible) ... ok
test_unseen_state_is_not_inserted_during_play (test_learning.LearningTests.test_unseen_state_is_not_inserted_during_play) ... ok

----------------------------------------------------------------------
Ran 27 tests in 0.345s

OK
```
