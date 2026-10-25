# Čo sa tu vlastne učí

## Najprv myšlienka bez vzorcov

V každej pozícii si hráč môže vybrať z niekoľkých ťahov. Ku každému z nich si uloží
číslo: odhad, či tento ťah vedie k dobrému výsledku. Na začiatku sú všetky čísla
nulové. Hráč teda nevie, čo je dobré.

Za výhru dostane cieľovú hodnotu +1, za remízu 0. Prehra zodpovedá -1. Počas
tréningu občas vyberie náhodný legálny ťah, aby objavoval aj zatiaľ neznáme možnosti.
Po každom ťahu posunie svoj odhad smerom k novému cieľu. Hrajú obe strany a obe
používajú tú istú tabuľku. Tréningové dáta preto vznikajú samotným hraním.

Pozor: naše čísla nie sú kalibrované pravdepodobnosti výhry. Sú to odhady hodnoty
ťahu v hre s výsledkami +1, 0, -1, orientované na optimálne pokračovanie súpera.

## Čo obsahuje vstup a výstup

Doska je deväť čísel: 0 znamená prázdno, 1 je X, -1 je O. Pred uložením do tabuľky
sa prepíše z pohľadu aktuálneho hráča. V kľúči tabuľky znamená 1 „moja značka“ a 2
„súperova značka“. Je to len kódovanie dát, nie stratégia.

Výstupom tabuľky je deväť odhadov. Vybrať sa však môžu iba prázdne políčka.
Obsadený štvorec sa nikdy nesmie zvoliť, ani keby mal omylom najvyššie číslo.
Tento filter sa používa aj pri výpočte tréningového cieľa.

## Jedna aktualizácia

Označme starý odhad písmenom **q**. Nový cieľ označme **t**. Písmeno **a** tu
znamená rýchlosť učenia: číslo väčšie než 0 a najviac 1. V programe sa volá
`alpha`, aby sa neplietlo s označením ťahu.

Nový odhad = q + a × (t − q).

Ak bol odhad 0, nový cieľ je 1 a rýchlosť učenia je 0,2, nový odhad bude 0,2.
Ďalšie potvrdenie toho istého výsledku ho posunie na 0,36. Číslo sa postupne
približuje k pozorovanému cieľu; neskočí k nemu hneď.

## Prečo je v kóde mínus

Po našom ťahu hrá súper. Jeho tabuľka hodnotí situáciu z JEHO pohľadu. Ak má súper
z ďalšej pozície hodnotu +1, znamená to pre nás -1. Ak má -1, znamená to pre nás +1.
Preto má neukončený ťah cieľ **mínus najlepšia dostupná hodnota ďalšieho hráča**.

V programe uvidíš `-gamma * max(...)`. `gamma` je faktor, ktorý môže tlmiť význam
vzdialenejšieho výsledku. Tu je rovný 1: výsledok neslabne len preto, že nastane
o niekoľko ťahov neskôr. Hra má konečný počet ťahov.

Keď náš ťah priamo ukončí partiu, neberieme hodnotu ďalšieho hráča. Použijeme
skutočný výsledok: +1 za vlastnú výhru alebo 0 za remízu. Hodnota -1 sa k skorším
ťahom prenáša cez odhad súperovho víťazného pokračovania.

## Tréning verzus hranie

Pri tréningu sa menia čísla a časť ťahov je prieskumná. Pri vyhodnocovaní sú čísla
zmrazené a vyberá sa najlepšie hodnotený legálny ťah. Pri presnej zhode vyhrá
políčko s najnižším indexom. Tento detail je súčasťou testovaného správania.

Výsledný JSON je samotná naučená tabuľka plus záznam nastavení. Program pri hraní
nečíta správne odpovede z minimaxu a nevolá jazykový model.

## Dva veľmi odlišné „AI“ prístupy

**Minimax** v `evaluation.py` vypočíta hodnotu pozície prehľadaním možných
pokračovaní. Nepotrebuje tréning. Je to tu rozhodca a referenčný súper.

**Q-learning** v `agent.py` a `training.py` upravuje odhady pomocou odohraných
ťahov. Pri samotnom hraní použije naučené čísla. Je to strojové učenie, zatiaľ však
nie neurónová sieť.

V tomto malom svete sa dá uložiť tabuľka. Pre väčší projekt chceme odhadnúť
hodnoty aj v obrovskom množstve nenavštívených pozícií. Tam budeme skúšať
neurónovú sieť. Viac vrstiev ani dlhší tréning samy osebe nezaručujú lepšiu hru.

## Čo presne dokazujú testy

Náhodný súper je len základné porovnanie. Aj slabý hráč môže proti nemu často
vyhrať. Dokonca neprehrávať vo vzorke hier proti minimaxu nie je úplný dôkaz:
jeho náhodný výber rovnako dobrých ťahov nemusí odhaliť každú slabinu nášho hráča.

`worst_case_policy` preto z prázdnej dosky prejde všetky súperove odpovede proti
nášmu pevnému správaniu. `audit_all_positions` navyše porovná vybraný ťah s presnou
hodnotou v každej legálne dosiahnuteľnej neukončenej pozícii. Ani tento programový
certifikát nie je formálnym dôkazom správnosti celého zdrojového kódu: spolieha sa
na správny herný engine a vyhodnocovač. Ich základné vlastnosti kontrolujú testy.

## Čo si skúsiť vysvetliť vlastnými slovami

Prečo pri učení potrebujeme skúšať aj ťahy, ktoré teraz nevyzerajú dobre?
Prečo nesmieme do maxima zahrnúť obsadené políčka?
Prečo sa pri striedaní hráčov mení znamienko?
Prečo výsledky tréningových partií nie sú dostatočný benchmark?
Prečo sa neoplatí trénovať väčší model, keď už v tejto hre hrá optimálne?

Na tieto otázky nemusíš vedieť odpovedať pred prvým spustením. Sú cieľom prvej
etapy učenia, nie podmienkou, aby si vôbec mohol začať.
