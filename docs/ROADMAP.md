# Od prvej hry k väčšiemu AI projektu

Nasledujúce etapy sú návrh ďalšieho vývoja, nie už hotové funkcie. Piškvorky 3 × 3
sú rozcvička. Väčším cieľom projektu je Connect Four, teda hra „štyri v rade“.

## Etapa 1 — Pochopiť a reprodukovať túto verziu

Spusti hru, testy a tréning. Vysvetli, ako sa ukladá doska, vyberajú legálne ťahy,
udeľuje odmena a prenáša hodnota od súpera späť. Porovnaj krátky a dlhší tréning.
Urob prvý vlastný commit a push. Nauč sa načítať práve ten model, ktorý testuješ.

Hotovo vtedy, keď dokážeš vysvetliť rozdiel medzi engine, agentom, tréningom a
evaluáciou a zopakovať meranie bez zamieňania checkpointov.

## Etapa 2 — Malé samostatné experimenty

Skús viac seedov pri rovnakej konfigurácii. Porovnaj rôznu mieru náhodného
skúšania ťahov. Zisti, ako skoro model prestane prehrávať a či to znamená aj
optimálny ťah vo všetkých pozíciách. Zmeň spôsob rozhodovania pri zhode hodnôt
iba ako samostatný experiment a znovu spusti úplnú kontrolu.

Možné programátorské rozšírenie je spoločné kódovanie otočení a zrkadlení dosky.
Treba pritom správne prepočítať aj indexy ťahov; samotné otočenie kľúča nestačí.

Hotovo vtedy, keď máš vlastnú zmenu, testy a stručný záznam: otázka → postup →
výsledok → interpretácia. Nárast počtu riadkov nie je cieľ.

## Etapa 3 — Skutočná neurónová sieť

Ako kontrolovaný medzikrok možno nahradiť tabuľku malou sieťou v PyTorch ešte na
piškvorkách. Vstup: doska z pohľadu hráča na ťahu; napríklad tri kanály „moja
značka, súperova značka, prázdno“, spolu 27 vstupných čísel. Výstup: 9 odhadov,
jeden pre každé políčko. Výstupy pre obsadené políčka sú zakázané.

Začať s neurónovou Q-learning verziou so zásobníkom minulých prechodov (experience
replay) a oddelenou cieľovou sieťou (target network). Obe časti pomáhajú oddeliť
vznik skúseností od aktualizácií odhadov. Rovnaká kontrola znamienka pri zmene
hráča zostáva nevyhnutná. Toto nie je len zmena `dict` na jednu funkciu.

Cieľom nie je prekonať už optimálnu tabuľku v tejto malej hre, ale naučiť sa
neurónový tréning vo svete, kde vieme presne skontrolovať výsledok.

Oficiálny výklad DQN používa inú, jednoagentovú úlohu; pri prenose treba výslovne
upraviť perspektívu hráčov a legálne akcie:
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

## Etapa 4 — Hlavný projekt: Connect Four

Použi dosku 6 × 7, spúšťanie kameňov do stĺpcov a štyri rovnaké kamene v rade.
Napíš engine a testy pre vodorovné, zvislé aj diagonálne výhry a plné stĺpce.
Pridaj náhodného hráča a prehľadávajúceho hráča s obmedzenou hĺbkou ako referencie.

Neurónový model bude vyberať medzi 7 stĺpcami. Vyhodnocuj zmrazené modely proti
nezávislým súperom a starším uloženým verziám, za obe strany a pri rovnakom
rozpočte prehľadávania. Zaznamenaj nastavenia, seed a čas/počet krokov meraný na
svojom stroji. Netrénuj a netestuj len proti jednej práve sa meniacej kópii.

Až potom porovnaj čisto neurónovú hru s kombináciou siete a prehľadávania stromu.
Pokročilá možnosť je Monte Carlo Tree Search a učenie hodnoty pozície aj vhodných
ťahov v štýle AlphaZero. Nie je to sľub sily AlphaZero ani tréningu na rovnakú
úroveň na osobnom notebooku.

Výskumná referencia:
https://arxiv.org/abs/1712.01815

## Čo ukázať na GitHube

README s presným cieľom a spustením, pravidlá a testy, oddelený tréning a
vyhodnotenie, reprodukovateľné nastavenia, výsledky proti pomenovaným súperom a
opis obmedzení. Jednoduché hrateľné rozhranie príde až po správnej logike.

Prínos projektu môže byť napríklad poctivé porovnanie rôznych metód pri rovnakom
počte odohraných hier či pri rovnakom čase na rozhodnutie. Nemusíš vymyslieť nový
algoritmus. Musíš vedieť vysvetliť, čo si skúšal, ako si to meral a čo z toho naozaj
vyplýva. Neuvádzaj neodmerané Elo ani neoverené tvrdenia o „najlepšej AI“.

## Aktualizácia: DQN je implementované

Neurónová verzia pre piškvorky už obsahuje self-play, pamäť skúseností,
cieľovú sieť a pokračovanie z checkpointov. Skutočný vstup má 9 čísel
(vlastné kamene 1, súperove -1, prázdno 0), nie navrhovaných 27.
Postup, hranice overenia a namerané výsledky sú v [DQN_SK.md](DQN_SK.md).
Connect Four zostáva ďalšou etapou.
