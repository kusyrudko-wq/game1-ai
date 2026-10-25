# Neurónové učenie DQN: postup a namerané výsledky

DQN znamená Deep Q-learning: namiesto tabuľky odhaduje hodnotu každého ťahu
neurónová sieť. Používa existujúcu architektúru 9 → 64 → 64 → 9.
Vlastný kameň má na vstupe hodnotu 1, súperov -1 a prázdne políčko 0.
Po každom ťahu sa perspektíva zmení na nasledujúceho hráča.

## Ako tréning funguje

Obe strany používajú tú istú učiacu sa sieť. Parameter epsilon je
pravdepodobnosť náhodného legálneho ťahu; postupne klesá. Inak agent vyberie
legálny ťah s najvyšším odhadom. Počas nezávislého vyhodnocovania sa náhodné
skúšanie nepoužíva. Pri zhode odhadov vyhráva najnižší index políčka.

Pamäť skúseností (experience replay) uchováva posledné odohrané prechody.
Každý obsahuje pôvodnú dosku, ťah, odmenu, nasledujúcu dosku, príznak konca
a masku legálnych nasledujúcich ťahov. Tréning náhodne vyberá malé dávky
z tejto pamäte. Optimalizátor Adam upravuje váhy siete podľa Huberovej chyby.

Cieľová sieť je oddelená kópia siete; nemení sa pri každom tréningovom kroku.
Pravidelne sa do nej skopírujú aktuálne váhy. Tým sa spomaľujú zmeny hodnôt,
ku ktorým sa učiaca sieť približuje.

Symbol gamma označuje váhu budúceho výsledku. Pri nekoncovom prechode je
tréningovým cieľom záporný súčin gamma a najvyššieho odhadu cieľovej siete
medzi legálnymi nasledujúcimi ťahmi. Záporné znamienko je dôležité: nasledujúca
doska patrí súperovi, takže jeho dobrý výsledok je pre nás zlý.
Pri koncovom prechode je cieľom iba odmena: 1 za výhru a 0 za remízu.
Hodnoty ďalších ťahov sa pri konci vôbec nepočítajú, ani keď zostali voľné
políčka. Prehra predchádzajúceho hráča sa prenáša späť cez zápornú hodnotu
súperovej odpovede.

Tréning nevolá minimax, nečíta Q-tabuľku a nedostáva expertné ťahy.
Minimax slúži iba na nezávislé vyhodnotenie.

## Súbory a používanie

- `tictactoe/dqn_training.py`: konfigurácia, self-play, pamäť, učenie a checkpointy.
- `tictactoe/neural_cli.py`: samostatné neurónové príkazy a vyhodnotenie.
- `tictactoe/__main__.py`: registrácia nových príkazov; pôvodné vetvy zostali zachované.
- `tests/test_dqn_training.py`: 12 nových testov učenia, checkpointov a ochrany súborov.
- `tictactoe/neural_agent.py`: iba aktualizovaný úvodný popis; výber ťahu sa nemenil.

```powershell
.venv/Scripts/python.exe -m tictactoe train-neural --episodes 20000 --seed 42 --report-every 2000 --run-dir runs/novy-pokus
.venv/Scripts/python.exe -m tictactoe evaluate-neural --model runs/novy-pokus/weights.pt --games 1000 --seed 7 --output runs/novy-pokus/evaluation.json
```

Priečinok pokusu musí byť nový a nachádzať sa pod `runs/`. `--help` uvádza
všetky nastaviteľné parametre. Model na hranie je `weights.pt`; úplný stav
tréningu je v súboroch `checkpoint-*.pt`. Tie obsahujú aj stav optimalizátora,
cieľovú sieť, pamäť skúseností a stav generátora náhodnosti.
Pokračovanie z checkpointu používa jeho pôvodnú konfiguráciu a skončí pri
pôvodne zadanom celkovom počte hier:

```powershell
.venv/Scripts/python.exe -m tictactoe train-neural --resume runs/novy-pokus/checkpoint-010000.pt --run-dir runs/pokracovanie
```

Metriky intervalov sú v `metrics.jsonl`, konfigurácia v `config.json` a súhrn
vrátane nameraného času v `training-summary.json`. Tréningové výhry zahŕňajú
náhodné skúšanie a nie sú meraním sily zmrazeného modelu.

## Skutočne vykonané overenie — 24. októbra 2026

Pred úpravami prešlo všetkých 33 testov a pôvodné vyhodnotenie. Po úpravách
prešlo všetkých **45 testov** cez `.venv/Scripts/python.exe` (záverečný beh
2,917 s). Testy overujú okrem iného znamienko cieľa, ignorovanie obsadených
políčok aj pri záporných hodnotách, koncové odmeny pre X aj O, cieľovú sieť,
obmedzenie pamäte, nemennosť modelu počas vyhodnocovania a presnú zhodu
pokračovania z checkpointu s neprerušeným tréningom na tomto runtime.
Pôvodné tabuľkové vyhodnotenie funguje aj pri zablokovanom importe PyTorch.

Krátky pokus: 100 hier, seed 42, dávka 32, začiatok učenia po 32 prechodoch,
pokles epsilon počas 80 hier. Výstupy sú v `runs/dqn-smoke-20261024/`.
Vyhodnotenie so seedom 7: proti náhodnému súperovi 59 výhier, 6 remíz,
35 prehier zo 100 hier; proti minimaxu 9 remíz a 91 prehier. Úplný audit našiel
1 449 neoptimálnych rozhodnutí a možnosť prehry z prázdnej dosky za obe strany.
Tento krátky pokus overoval spustiteľnosť celého postupu.

Väčší pokus: **20 000 self-play hier**, seed 42, CPU, jedno vlákno,
PyTorch 2.14.1+cpu. Použité boli predvolené parametre okrem intervalu hlásení
2 000 hier: dávka 128, pamäť 20 000 prechodov, začiatok učenia po 512 prechodoch,
aktualizácia každé 4 ťahy, kopírovanie cieľovej siete každých 250 aktualizácií,
rýchlosť učenia 0,001, gamma 1, epsilon od 1 po 0,1 počas prvých 16 000 hier.

Tréning vykonal 152 453 ťahov a 37 986 gradientových aktualizácií.
Nameraný čas tréningu vrátane ukladania bol **90,48 s**; nezahŕňa následné
vyhodnotenie. Ide o jediný väčší pokus s jedným seedom, bez ladenia podľa
minimaxového auditu.

Vyhodnotenie používalo seed 7, zmrazený model a spolu 1 000 hier proti každému
súperovi, z toho 500 za X a 500 za O:

| Súper | Strana modelu | Výhry | Remízy | Prehry |
|---|---|---:|---:|---:|
| Náhodný | X | 491 | 9 | 0 |
| Náhodný | O | 427 | 73 | 0 |
| Minimax | X | 0 | 500 | 0 |
| Minimax | O | 0 | 500 | 0 |

Úplné prehľadanie všetkých súperových odpovedí proti tejto pevnej stratégii
z prázdnej dosky nenašlo možnosť prehry za X ani O; najhorší výsledok bol
remíza. Audit všetkých 4 520 dosiahnuteľných nekoncových pozícií však našiel
**114 neoptimálnych rozhodnutí**. Model teda nie je optimálny v každej pozícii.
Tieto dva výsledky sa nevylučujú: pri vlastnom rozhodovaní sa model nemusí
dostať do všetkých pozícií, ktoré sú podľa pravidiel dosiahnuteľné.
Kontrola platí pre konkrétne váhy a pevné rozhodovanie pri zhode a predpokladá
správnosť pravidiel a vyhodnocovania; nie je zárukou pre iný tréningový pokus.

Váhy, úplné checkpointy, metriky a surové výsledky sú v
`runs/dqn-20000-20261024/`. Pôvodný `models/demo.json` nebol prepísaný;
jeho SHA-256 pred aj po práci bol
`D9AFF8583E0704790E4E61D9D6C1946C0CE32640BEBD75808D42F61E20B4C573`.
Opakované tabuľkové vyhodnotenie dalo pôvodné výsledky: proti náhodnému súperovi
895 výhier a 105 remíz; proti minimaxu 1 000 remíz; audit 0 chýb zo 4 520.

Implementácia a dokumentácia vznikli s pomocou AI. Nič sa neposielalo na GitHub.
