# Ako funguje Game AI — technický výklad

## 1. Architektúra

- `tictactoe/game.py`: pravidlá, legálne ťahy, koncové stavy; žiadna AI.
- `tictactoe/agent.py`: Q-tabuľka, výber ťahu a učenie.
- `tictactoe/training.py`: tréning hraním proti sebe.
- `tictactoe/evaluation.py`: meranie výkonu a kontrola voči optimálnej stratégii.
- `tictactoe/gui.py`: grafické rozhranie Tkinter, ktoré načíta naučený model.
- `tests/`: regresné testy pravidiel a učenia.
- `models/demo.json`: uložené naučené hodnoty, nie ručne zadané pravidlá stratégie.

## 2. Reprezentácia

Doska má deväť čísel: `0` prázdne, `1` X, `-1` O. Stav obsahuje aj informáciu, kto je na ťahu. Prázdne políčka sú legálne akcie. Výsledok: `1` výhra X, `-1` výhra O, `0` remíza, `None` hra pokračuje. Indexy v kóde sú 0 až 8; používateľ vidí 1 až 9.

Stav sa kóduje **z perspektívy hráča na ťahu**: `0` prázdne, `1` moje, `2` súperove. Vďaka tomu jeden model hrá X aj O.

## 3. Čo sa model učí

`Q(s,a)` je odhad hodnoty vykonania ťahu `a` v stave `s` z pohľadu hráča na ťahu. V tabuľke sú pre každý stav hodnoty deviatich možných ťahov; pri výbere sa berú do úvahy iba legálne.

Keď sa hra skončí, cieľová hodnota `y` je `+1` za výhru hráča, ktorý ťah vykonal, `-1` za prehru a `0` za remízu. Keď hra pokračuje, cieľ je `y = -γ max_b Q(s',b)`, kde `s'` je nasledujúci stav **súpera**, `b` jeho legálny ťah a `γ` diskontný faktor. Znamienko mínus je zásadné: súperov zisk je naša strata.

Aktualizácia:

`Q(s,a) ← Q(s,a) + α [y − Q(s,a)]`

`α` je rýchlosť učenia. Zápis znamená: starý odhad posunieme časť cesty k novému cieľu. Napríklad pri `Q=0.2`, `y=1`, `α=0.25` je nová hodnota `0.4`.

## 4. Tréning hraním proti sebe

Dve strany používajú tú istú tabuľku. V každom kroku hráč buď skúša náhodný legálny ťah (explorácia s pravdepodobnosťou `ε`), alebo vyberá ťah s najvyššou naučenou hodnotou. Po ťahu sa Q-hodnota aktualizuje. V priebehu tréningu sa explorácia postupne znižuje, aby model najprv objavoval a neskôr využíval naučené skúsenosti.

Tréning a vyhodnocovanie sú oddelené: pri teste sa tabuľka nemení. Výsledky sú závislé od nastavení tréningu a náhodného seedu. Priložený `demo.json` bol trénovaný na 200 000 hrách; pozri `reports/VALIDATION_SK.md` pre rozsah a obmedzenia testovania.

## 5. Ako sa rozhoduje pri hre

GUI načíta `models/demo.json`, zistí aktuálnu pozíciu, zavolá `agent.choose(position)` a vykoná vybraný ťah. **GUI nepoužíva minimax**. Minimax je iba nezávislý kontrolný súper vo vyhodnocovaní. V prípade rovnosti hodnôt sa v hre vyberie prvý najlepší legálny ťah.

## 6. Prečo nejde o neurónovú sieť

Je to **tabuľkový reinforcement learning**, nie deep learning. Pre 3×3 piškvorky je počet stavov malý, takže Q-tabuľka je jednoduchšia a interpretovateľnejšia. Neurónová sieť by namiesto ukladania každého stavu samostatne aproximovala funkciu `Q(s,a)` pomocou váh a umožnila by generalizovať medzi pozíciami. Pri väčších hrách je to zaujímavejšie, ale aj náročnejšie na stabilitu tréningu.

## 7. Čo povedať na pohovore

Presný opis: „Vybudoval som vzdelávací projekt pre učenie herného agenta hraním proti sebe. Implementácia využíva zero-sum tabular Q-learning, oddelený engine, evaluáciu voči minimaxu a GUI. Pri implementácii som používal AI asistenciu a následne som študoval architektúru a experimenty.“ Ak si kód nepísal sám, netvrď opak. Najlepšie je doplniť vlastný experiment a vedieť vysvetliť, čo si zmenil.

### Kontrolné otázky

1. Prečo má cieľ pri pokračujúcej hre mínus pred maximom?
2. Prečo nie je správne vyberať ťahy z obsadených políčok?
3. Aký je rozdiel medzi `None` a `0` vo výsledku?
4. Čo sa stane, ak je `ε` počas celého tréningu nulové?
5. Prečo je porovnanie s náhodným hráčom nedostatočné?
6. Ako by sa model zmenil, keby Q-tabuľku nahradila neurónová sieť?
