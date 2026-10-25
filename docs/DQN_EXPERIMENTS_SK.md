# Hľadanie lepšieho neurónového modelu

Základný model v `runs/dqn-20000-20261024/` mal 114 neoptimálnych rozhodnutí
zo 4 520 dosiahnuteľných nekoncových pozícií. Tento experiment ho nemení.
Kontrolné súčty všetkých jeho súborov sa ukladajú pred experimentom a porovnajú
po skončení. Menšie číslo chýb je lepšie, ale kandidát musí zároveň odolať
všetkým súperovým odpovediam z prázdnej dosky za obe strany.

## Čo sa menilo

Rýchlosť učenia určuje veľkosť úpravy váh. Gamma určuje váhu budúceho výsledku.
Epsilon je pravdepodobnosť náhodného legálneho ťahu počas tréningu. Interval
cieľovej siete je počet gradientových aktualizácií medzi kopírovaniami váh.
Šírka je počet neurónov v každej z dvoch skrytých vrstiev.
Seed určuje opakovateľnú náhodnosť inicializácie, hry a výberu tréningových dávok.

Kontrolná konfigurácia skráteného porovnania má 10 000 hier, rýchlosť učenia 0,001,
gamma 1, epsilon od 1 po 0,1 počas 8 000 hier, interval cieľovej siete 250
a šírku 64. Pamäť má 20 000 prechodov, dávka 128 prechodov, učenie začína
po 512 prechodoch a aktualizuje sieť každé 4 ťahy. Všetky pokusy začínajú
od náhodných váh, bez predtrénovania základným modelom.

Plán obsahuje 12 konfigurácií so seedom 42:

| Názov | Rozdiel oproti kontrolnej konfigurácii |
|---|---|
| control | Referenčné nastavenie skráteného plánu |
| lr-low | Rýchlosť učenia 0,0003 |
| lr-high | Rýchlosť učenia 0,003 |
| gamma-095 | Gamma 0,95 |
| gamma-099 | Gamma 0,99 |
| explore-more | Epsilon končí na 0,3 po 10 000 hrách |
| explore-fast | Epsilon končí na 0,05 po 4 000 hrách |
| target-fast | Interval cieľovej siete 100 aktualizácií |
| target-slow | Interval cieľovej siete 1 000 aktualizácií |
| width-128-g095 | Šírka 128 a gamma 0,95 |
| short-width-32 | Šírka 32, 5 000 hier, pokles epsilon počas 4 000 hier |
| long-g095-explore | 20 000 hier, gamma 0,95, epsilon končí na 0,2 po 15 000 hrách |

Prvé porovnania menia jeden parameter; posledné kombinujú viac parametrov.
Nie je to úplné preskúšanie všetkých kombinácií. Dve najlepšie prípustné
konfigurácie sa automaticky zopakujú so seedmi 43 a 44.

## Výber a vyhodnotenie

Každý model hrá 1 000 hier proti náhodnému súperovi a 1 000 proti minimaxu,
vždy 500 za X a 500 za O; vyhodnocovací seed je 7. Navyše sa kontroluje
všetkých 4 520 dosiahnuteľných nekoncových pozícií a všetky súperove odpovede
z prázdnej dosky. Vyhodnocovanie nemení váhy a nepoužíva prieskum.

Najprv sa vyžaduje nemožnosť vynútiť prehru z prázdnej dosky za X aj O.
Potom sa minimalizuje počet chýb úplného auditu. Pri zhode rozhodne počet
prehier s minimaxom, skóre proti náhodnému súperovi a pevné poradie názvu
konfigurácie a seedu. Čas tréningu nie je rozhodovacím kritériom, aby zmeny
zaťaženia počítača neovplyvnili výber. Aj pôvodný model môže zostať víťazom,
ak by ho žiadny kandidát neprekonal.

Minimaxové výsledky sa používajú na výber konfigurácie a checkpointu, nikdy
ako cieľové hodnoty pri učení. Ide o ladenie podľa známeho konečného priestoru
stavov, nie o nezávislý test zovšeobecnenia na iné hry. Výsledok konkrétnych
váh možno úplne skontrolovať; neznamená to, že každý tréningový seed dopadne
rovnako. Kontroly predpokladajú správnosť herných pravidiel a vyhodnocovania.

## Ako experiment zopakovať

```powershell
.venv/Scripts/python.exe -m tictactoe.dqn_experiments --output runs/moje-hladanie --games 1000 --evaluation-seed 7 --budget-seconds 1800
```

Predpokladom je zachovaný pôvodný experiment v `runs/dqn-20000-20261024/`.
Použi nový výstupný priečinok. Výpočty bežia na CPU s jedným vláknom.
Rozpočet 1 800 sekúnd je mäkký limit nameraného uplynutého času: pred ďalším
pokusom sa overí, či ešte nevypršal; práve spustený pokus sa dokončí.
Súhrn uvádza, či sa vykonal celý plán. Reprodukcia je testovaná v rovnakom
prostredí; bitovo rovnaké výsledky medzi verziami PyTorch nie sú zaručené.

`manifest.json` uchováva plán, seedy, prostredie a kontrolné súčty zdrojov
aj všetkých súborov základného experimentu. V `source/` je kópia spusteného
Python kódu. Každý kandidát má konfiguráciu, metriky, úplný checkpoint, váhy
a vyhodnotenie. `leaderboard.csv` obsahuje aj výsledky podľa strán a časy.
`best-checkpoint.pt` umožňuje obnoviť úplný tréningový stav; `best-weights.pt`
slúži na načítanie agenta. `best.json` identifikuje pôvod najlepších váh.


## Pilot a úprava rozpočtu

Pôvodný plán skúšal prevažne 20 000 hier na konfiguráciu. Jeho hotové pokusy
sú zachované v `runs/dqn-search-20261024/`: kontrolná konfigurácia zopakovala
114 chýb a nemožnosť vynútiť prehru (170,46 s); rýchlosť učenia 0,0003 dosiahla
119 chýb a umožnila vynútenú prehru (177,88 s). Nasledujúci pokus s rýchlosťou
učenia 0,003 bol prerušený a nemá dokončený model ani vyhodnotenie; nezapočítava
sa do dokončených konfigurácií ani do poradia.

Po zmeraní nákladov sa plán skrátil na prevažne 10 000 hier, aby zostal priestor
na všetkých 12 konfigurácií a opakovania seedov. Nový plán sa pred spustením
uložil do samostatného manifestu v `runs/dqn-search-compact-20261024/`.
Pilot sa neprepisoval a základný model sa nemenil. Výsledky a časy oboch sérií
sa uvádzajú oddelene; neúplný pokus nemá vymyslený výsledok ani čas tréningu.

## Namerané výsledky — 24. októbra 2026

Skrátený plán dokončil všetkých 12 konfigurácií a štyri opakovania. Tréning hotových 16 modelov trval spolu 842.59 s; celá séria vrátane vyhodnotení a ukladania 861.41 s (14,36 min). Ide o nameraný uplynutý čas na CPU s jedným vláknom, nie o účtovanie procesorového času. Dva dokončené pilotné tréningy trvali navyše spolu 348.33 s. Tieto súčty nezahŕňajú prerušený pilotný pokus, písanie kódu ani testy.

Skrátená séria odohrala 185 000 tréningových hier; dva dokončené pilotné pokusy ďalších 40 000. Prostredie: Python 3.13.14, PyTorch 2.14.1+cpu, Windows 11. Rýchlosť behu sa počas série menila; z časov rôzne spustených pokusov preto nevyvodzujeme poradie výpočtovej efektivity.

Nasledujúca tabuľka obsahuje všetkých 16 dokončených tréningov skrátenej série a pôvodný model. V/R/P znamená počet výhier, remíz a prehier zo spolu 1 000 hier proti danému súperovi. Prípustný model nemá vynútiteľnú prehru z prázdnej dosky za X ani O. Čas pôvodného modelu je historický údaj a nezapočítava sa do novej série.

| Konfigurácia | Seed | Chyby / 4 520 | Prípustný | Tréning (s) | Náhodný V/R/P | Minimax V/R/P |
|---|---:|---:|---|---:|---|---|
| long-g095-explore | 43 | 72 | áno | 54.27 | 926/74/0 | 0/1000/0 |
| gamma-095 | 42 | 84 | áno | 86.73 | 948/52/0 | 0/1000/0 |
| long-g095-explore | 44 | 86 | áno | 54.16 | 931/69/0 | 0/1000/0 |
| gamma-095 | 43 | 90 | áno | 27.11 | 915/85/0 | 0/1000/0 |
| long-g095-explore | 42 | 94 | áno | 52.81 | 939/61/0 | 0/1000/0 |
| target-fast | 42 | 111 | áno | 26.21 | 919/81/0 | 0/1000/0 |
| baseline | 42 | 114 | áno | 90.48 | 918/82/0 | 0/1000/0 |
| width-128-g095 | 42 | 54 | nie | 29.73 | 930/68/2 | 0/980/20 |
| explore-more | 42 | 82 | nie | 51.86 | 934/65/1 | 0/980/20 |
| lr-high | 42 | 84 | nie | 86.77 | 938/61/1 | 0/980/20 |
| gamma-095 | 44 | 95 | nie | 27.05 | 933/65/2 | 0/934/66 |
| target-slow | 42 | 104 | nie | 26.76 | 889/110/1 | 0/904/96 |
| control | 42 | 126 | nie | 88.88 | 926/72/2 | 0/982/18 |
| lr-low | 42 | 126 | nie | 95.53 | 952/45/3 | 0/949/51 |
| gamma-099 | 42 | 133 | nie | 89.33 | 927/72/1 | 0/977/23 |
| explore-fast | 42 | 168 | nie | 32.49 | 936/63/1 | 0/987/13 |
| short-width-32 | 42 | 169 | nie | 12.91 | 958/42/0 | 0/966/34 |

Pri nastavení gamma-095 mali seedy 42, 43, 44 postupne 84, 90 a 95 chýb (medián 90); kontrolou z prázdnej dosky prešli iba seedy 42 a 43. Pri long-g095-explore mali rovnaké seedy 94, 72 a 86 chýb (medián 86), pričom prešli všetky tri. Dlhšie nastavenie preto v tejto malej vzorke preukázalo aj väčšiu spoľahlivosť. Tri seedy však nie sú zárukou každého budúceho behu.

Najlepší prípustný model má **72 chýb**, teda o 42 chýb (**36,8 %**) menej než baseline. Používa šírku 64, gamma 0,95, rýchlosť učenia 0,001, epsilon od 1 po 0,2 počas 15 000 hier, cieľovú sieť každých 250 aktualizácií a 20 000 tréningových hier, seed **43**. Jeho tréning trval **54,27 s**.

| Súper | Strana víťazného modelu | Výhry | Remízy | Prehry |
|---|---|---:|---:|---:|
| Náhodný | X | 492 | 8 | 0 |
| Náhodný | O | 434 | 66 | 0 |
| Minimax | X | 0 | 500 | 0 |
| Minimax | O | 0 | 500 | 0 |

Úplná kontrola z prázdnej dosky má za obe strany najhorší výsledok remízu. Model však **nie je optimálny v každej pozícii**: 72 chýb v úplnom audite zostáva. Širšia sieť so 54 chybami bola vyradená, pretože proti nej možno vynútiť prehru. Viac neurónov teda v tomto pokuse samo osebe nesplnilo požadované kritérium.

Tabuľka hodnotí už vybrané modely; vplyv jednotlivých parametrov nemožno z kombinovaných variantov oddeliť. Vyšší podiel výhier proti náhodnému súperovi nebol hlavným cieľom výberu.

Všetkých **49 automatických testov prešlo** v záverečnom behu (2,300 s). Samostatne sa overilo načítanie skopírovaných víťazných váh aj úplného checkpointu, zhodnosť ich sietí a kontrolných súčtov, opätovné dosiahnutie 72 chýb a nemennosť všetkých baseline súborov. Zhodujú sa aj kontrolné súčty spustených zdrojov s manifestom. Pôvodné tabuľkové vyhodnotenie zostalo bez zmeny.

Výstupy sú v `runs/dqn-search-compact-20261024/`: `leaderboard.csv`, `leaderboard-all-completed.csv` (aj pilotné výsledky), `seed-aggregates.json`, `summary.json`, `verification.json`, `best-checkpoint.pt` a `best-weights.pt`. Každý kandidát má aj samostatné metriky a vyhodnotenie so stranovými výsledkami.

Implementácia a analýza vznikli s pomocou AI. Baseline ani jeho metriky sa neprepísali. Nič sa neposielalo na GitHub.

## Opätovné overenie — 25. októbra 2026

Existujúcu dokončenú sériu sme netrénovali znova. Nanovo sa vyhodnotil baseline
a všetkých 16 kandidátov: náhodný súper a minimax po 1 000 hier so seedom 7,
úplný audit aj všetky súperove odpovede z prázdnej dosky. Všetky nové výsledky
sa presne zhodovali s uloženými reportmi. Víťazné váhy aj úplný checkpoint
sa načítali a ich siete sa zhodovali; kontrolné súčty baseline, zdrojov
a víťazných artefaktov tiež súhlasili. Zostáva 72 chýb a žiadna vynútiteľná
prehra z prázdnej dosky za X ani O.

V tomto overovacom behu prešlo všetkých 49 testov za 7,500 s. Pôvodné tabuľkové
vyhodnotenie tiež prešlo: nula chýb v 4 520 pozíciách. Nové reporty sú
v `runs/dqn-reverification-20261025/`; overovací skript je
`runs/reverify_dqn_20261025.py`. Vyššie uvedené časy tréningu sú historické
merania pôvodnej série, nie nové tréningové merania. Baseline sa nezmenil.

