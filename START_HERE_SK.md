# Začni tu: prvá vlastná herná AI

Táto verzia je rozcvička: piškvorky 3 × 3. Hrá sa v termináli. Obsahuje skutočný
tréning z odohraných hier, ale modelom je zatiaľ tabuľka čísel, nie neurónová sieť.
Nemusíš čítať všetky súbory naraz. Najprv projekt spusti.

## 1. Rozbaľ projekt a otvor správny priečinok

Na Windows rozbaľ ZIP cez „Extrahovať všetko“. Vo VS Code vyber **File → Open
Folder** a otvor priečinok `selfplay-game-ai`. Správny priečinok obsahuje priamo
`README.md`, `AGENTS.md`, `models`, `tests` a `tictactoe`.

Vyber **Terminal → New Terminal**. Príkazy píš do terminálu, nie do súboru Pythonu.
Najprv over:

```powershell
py --version
```

Potrebný je Python 3.10 alebo novší; balík bol tu odskúšaný na Pythone 3.13.5 na
Linuxe. Windows beh overíš nasledujúcimi príkazmi u seba. Ak `py` neexistuje, ale
funguje `python --version`, vo všetkých príkazoch nižšie používaj `python` namiesto
`py`. Ak nefunguje ani jedno, treba nainštalovať Python. Pre tento krok neinštaluj
PyTorch, CUDA ani nové ovládače. Projekt nepoužíva žiadny externý Python balík.

## 2. Zahraj si proti priloženému modelu

```powershell
py -m tictactoe play
```

Na doske uvidíš čísla 1 až 9. Napíš číslo voľného políčka a stlač Enter. Hráš X
a začínaš. `q` hru ukončí. Druhý v poradí hráš takto:

```powershell
py -m tictactoe play --human O
```

Za `py -m` nasleduje názov balíka, ktorý Python spustí. Za ním je činnosť (`play`).
Parameter `--human O` určuje, ktorou značkou budeš hrať.

## 3. Over, že projekt funguje

```powershell
py -m unittest discover -s tests -v
py -m tictactoe evaluate
```

Prvý príkaz spustí automatické testy. Druhý otestuje priložený model proti náhodnému
a optimálnemu súperovi a potom vykoná dve úplné kontroly. V tejto priloženej verzii
má byť výsledok `CANNOT LOSE` za X aj O a nula neoptimálnych rozhodnutí v audite.
Nie je to záruka výsledku ľubovoľného budúceho tréningu.

## 4. Natrénuj si nový model od nuly

```powershell
py -m tictactoe train --episodes 200000 --seed 42
```

`episodes` je počet celých odohraných hier. `seed` určuje náhodnosť pokusu, aby sa
dal zopakovať. Výsledok bude v `models/my-agent.json`. Záznam priebehu bude v
`runs/training.csv`. Model sa neberie z priloženého demo súboru: začne s prázdnou
tabuľkou a učí sa vlastnými partiami. Nepoužíva internet, API ani platený tréning.

Potom ho výslovne vyber:

```powershell
py -m tictactoe evaluate --model models/my-agent.json
py -m tictactoe play --model models/my-agent.json
```

**Pozor:** bez `--model` sa hra aj vyhodnotenie vrátia k priloženému `demo.json`.

Pri ďalšom pokuse použi iné cesty, aby si výsledky nepomiešal:

```powershell
py -m tictactoe train --episodes 10000 --seed 1 --output models/short-run.json --log runs/short-run.csv
py -m tictactoe evaluate --model models/short-run.json --output runs/short-evaluation.json
```

Opätovné použitie existujúcich výstupov tréning odmietne. Prepínač `--overwrite`
ich úmyselne nahradí; nepridáva ďalší tréning k starému modelu. Každé spustenie
`train` začína od nuly. CSV obsahuje výsledky prieskumných self-play hier; tieto
výsledky samy osebe nie sú meradlom sily proti nezávislému súperovi.

## 5. Čo je Git a GitHub

**VS Code** je miesto, kde kód upravuješ. **Python** ho spúšťa. **Git** uchováva
históriu zmien. **GitHub** je online miesto pre repozitár, teda projekt s touto
históriou. **GitHub Desktop** je aplikácia, cez ktorú používaš Git tlačidlami.

**Commit** je pomenovaný bod v histórii uložený lokálne. **Push** ho odošle na
GitHub. Samotné uloženie súboru vo VS Code preto ešte nič nenahráva na GitHub.

## 6. Prvé nahratie cez GitHub Desktop

Nainštaluj GitHub Desktop a prihlás sa do svojho GitHub účtu. Nesnaž sa uploadovať
ZIP ako jediný súbor: na GitHube majú byť normálne jednotlivé súbory projektu.

1. V GitHub Desktop vyber **File → New repository**. Názov:
   `selfplay-game-ai`. Local path zvoľ napríklad `Documents/GitHub`, teda iné
   miesto než dočasne rozbalený projekt. Nevyberaj nový README ani licenciu.
   Klikni **Create repository**.
2. Aplikácia vytvorí nový priečinok `selfplay-game-ai` v zvolenom umiestnení.
   Skopíruj doň **obsah** rozbaleného priečinka: `tictactoe`, `tests`, `models`,
   `docs`, `.github` a ostatné súbory. Nekopíruj celý vonkajší priečinok do ďalšieho
   rovnako pomenovaného priečinka. Nemaž skrytý priečinok `.git`, ktorý vytvoril Git.
3. V GitHub Desktop sa zobrazia zmeny. Skontroluj súbory. Do Summary napíš
   `Add self-play tic-tac-toe starter and tests` a klikni **Commit to main**
   (alebo názov svojej predvolenej vetvy).
4. Klikni **Publish repository**. Na prvé pokusy nechaj **Keep this code private**
   zapnuté. Verejné zverejnenie urob až po kontrole obsahu. Potvrď publikovanie.

Odteraz vo VS Code otváraj **nový priečinok vytvorený GitHub Desktop**, nie starú
rozbalenú kópiu. Bežný cyklus: upravíš kód → spustíš testy → commit → **Push origin**.

Nové modely a `runs/` sa vďaka `.gitignore` automaticky nepridávajú. Priložené malé
`models/demo.json` je výnimka, aby projekt fungoval hneď po získaní z GitHubu.
Vybraný budúci checkpoint pridaj zámerne spolu s jeho nastavením a vyhodnotením,
nie so všetkými pokusmi. `.env`, heslá a prístupové tokeny nikdy nenahrávaj.

Oficiálne návody:
- https://docs.github.com/en/desktop/overview/creating-your-first-repository-using-github-desktop
- https://docs.github.com/en/desktop/making-changes-in-a-branch/pushing-changes-to-github-from-github-desktop

## 7. Ako zapojiť Codex, ktorý už používaš vo VS Code

V otvorenom projekte pošli Codexu tento text:

> Prečítaj AGENTS.md a START_HERE_SK.md. Zatiaľ nič neprepisuj. Over, že som v
> správnom priečinku, zisti dostupný Python a spusti testy aj vyhodnotenie modelu.
> Potom mi po slovensky vysvetli, čo robia game.py, agent.py a training.py, bez
> predpokladu znalosti machine learningu. Navrhni jednu malú úlohu, nie celý ďalší
> projekt. Nevykonávaj publish ani push.

Prvý vhodný experiment: porovnaj 1 000, 10 000 a 200 000 hier s rovnakým seedom,
ulož výsledky pod odlišnými názvami a vysvetli, prečo vysoká úspešnosť proti
náhodnému súperovi ešte nedokazuje neporaziteľnosť.

## 8. Časté chyby

**No module named tictactoe:** terminál nie je v priečinku obsahujúcom `tictactoe`.
Otvor správny priečinok cez File → Open Folder a nový terminál.

**Program čaká:** počas hry očakáva číslo políčka. Príkazy Pythonu zadávaj až po
ukončení hry, keď sa znovu ukáže PowerShell prompt `PS ...>`.

**Nový model prehráva:** tréning nie je dôkaz kvality. Spusti `evaluate`; hlásenie
ukáže aj konkrétnu prehrávajúcu postupnosť ťahov. Neupravuj testy, aby chybu zakryli.

**Codex zmenil priveľa súborov:** pozri Changes v GitHub Desktop a necommituj
nepochopené zmeny. Pracuj po jednej funkcii či experimente.
