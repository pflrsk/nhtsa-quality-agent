# Agent analityczny jakości: skargi i akcje serwisowe NHTSA na lokalnym modelu językowym

Asystent dla inżyniera jakości, któremu zadaje się pytania po polsku, np. *„Który komponent Jeepa Compass miał najwięcej skarg w 2025 roku i czy była na to akcja serwisowa?”*.
Agent sam wybiera, co policzyć, liczy na prawdziwych danych NHTSA (skargi klientów i akcje serwisowe marek Stellantis) i odpowiada liczbami oraz numerami akcji.

Model językowy (Qwen3 8B przez Ollamę) działa **lokalnie na laptopie**: dane nie opuszczają komputera i nie ma limitów zapytań.
Zasada projektu: **model językowy decyduje, co zrobić, a kod gwarantuje, że liczby są dokładne.**

**Wyniki:**
- pytania z szablonów, modele aut nieużywane przy rozwoju: **102/102 (100%)**,
- pytania sformułowane swobodnie, jak przez użytkownika (bez polskich znaków, z literówkami, porównania, miesiące, rankingi): **około 90%** na trzech nowych zestawach (94%, 91%, 88%).

Szczegóły, historia poprawek i ograniczenia są niżej.

---

## Przykład

```
> python agent.py "Wrangler czy Gladiator - który miał więcej skarg na silnik w 2025?"
  -> count_complaints({"component": "ENGINE", "make": "JEEP", "model": "WRANGLER", "year_from": 2025, "year_to": 2025})
  -> count_complaints({"component": "ENGINE", "make": "JEEP", "model": "GLADIATOR", "year_from": 2025, "year_to": 2025})

Wrangler miał 447 skarg na silnik w 2025, podczas gdy Gladiator miał 34 skarg.

Zakres: Jeep Wrangler, silnik, skargi z 2025; Jeep Gladiator, silnik, skargi z 2025
```

Linie ze strzałką to narzędzia, które agent sam wybrał i wywołał. Ostatnią linię, **„Zakres”**, pisze kod, a nie model: pokazuje, jak pytanie zostało zrozumiane (marka, model, komponent, okres). Jeśli agent źle zrozumie pytanie, użytkownik widzi to od razu, zamiast dostać pewną siebie złą liczbę.

---

## Jak to działa

```
pytanie po polsku
      │
      ▼
model językowy (Qwen3 8B, Ollama)  ──►  wybiera narzędzie i parametry (function calling)
      │                                         │
      │                          kod poprawia parametry na podstawie pytania (bezpiecznik 1)
      │                                         ▼
      │                               narzędzie liczy na danych (pandas)
      │                                         │
      ◄─────────────── wynik (JSON) ────────────┘
      │   (może wywołać kolejne narzędzia)
      ▼
odpowiedź  ──►  kod sprawdza odpowiedź (bezpieczniki 2–7)  ──►  odpowiedź + linia „Zakres”
```

### Narzędzia (`tools.py`)

| Narzędzie | Co robi |
|---|---|
| `count_complaints` | liczba skarg z filtrami (marka, model, rocznik lub zakres roczników, komponent, lata, miesiąc, kwartał), rankingi komponentów, modeli i marek z udziałem procentowym, rozkład po latach i miesiącach z gotowym maksimum i minimum, liczba skarg z wypadkiem, pożarem, rannymi |
| `complaint_trend` | liczba skarg rok po roku, kwartał lub miesiąc, zmiana w %, okres z największą i najmniejszą liczbą skarg |
| `find_campaigns` | akcje serwisowe (recalle) po modelu, roczniku, komponencie lub słowie kluczowym; liczba wszystkich pasujących akcji oraz akcja najnowsza, najstarsza i obejmująca najwięcej pojazdów |
| `campaign_details` | szczegóły akcji: wada, skutek, naprawa, objęte pojazdy, części |
| `search_complaints` | wyszukiwanie skarg o podobnej treści (TF-IDF) |

Każde narzędzie zwraca pole `filters` z warunkami, których faktycznie użyło. Narzędzia dopasowują nazwy („Jeep Wrangler” → `WRANGLER`, „hamulce” → `SERVICE BRAKES`) i poprawiają literówki w nazwach modeli.
Zasada przy rozbudowie narzędzi: wszystko, co da się policzyć (maksimum, procent, najstarsza akcja), liczy narzędzie. Model nie musi wtedy niczego wyliczać ani wybierać z długiej listy liczb, bo przy tym najczęściej się mylił.

### Bezpieczniki (`agent.py`)

Każdy powstał z konkretnego błędu zaobserwowanego w testach:

| # | Bezpiecznik | Błąd, który naprawia | Jak działa |
|---|---|---|---|
| 1 | parametry z pytania | model wpisywał „PACIFICA” zamiast „PACIFICA PLUG-IN HYBRID”, gubił komponent, mylił rok skargi z rocznikiem auta, liczył cały rok zamiast miesiąca | kod sam rozpoznaje w pytaniu model, markę, komponent, rocznik, rok, miesiąc i kwartał (także w odmianie i bez polskich znaków) i **poprawia** parametry przed uruchomieniem narzędzia |
| 2 | brakujący model | „Ram 1500” policzony jako wszystkie Ramy; przy porównaniu dwóch modeli jeden pominięty | jeśli żadne wywołanie nie użyło modelu z pytania, kod sam wywołuje narzędzie dla tego modelu i oddaje wynik modelowi |
| 3 | liczby bez pokrycia w danych | zmyślona albo przekręcona liczba | każda liczba w odpowiedzi musi występować w wynikach narzędzi albo w pytaniu; jeśli nie, kod prosi model o poprawkę |
| 4 | niepełna lista akcji | z 11 akcji o poduszkach powietrznych w Ram 1500 model przepisał 7 | jeśli odpowiedź wymienia co najmniej dwie akcje z wyniku, musi wymienić wszystkie; gdy druga odpowiedź też jest niepełna, brakujące numery dopisuje kod |
| 5 | złe narzędzie | pytanie o liczbę akcji serwisowych, a model liczył skargi | pytanie o akcje bez użycia narzędzia do akcji, więc kod prosi o poprawkę |
| 6 | model spoza danych | pytanie o Fiata Pandę, a odpowiedź o Fiacie 500 | jeśli narzędzie odrzuciło model z pytania, odpowiedź nie może dotyczyć innego modelu |
| 7 | pusta odpowiedź | model nie napisał nic | jedna prośba o odpowiedź, potem komunikat, że się nie udało |

Gdzie się da, poprawia kod (bezpieczniki 1, 2, 4), bo jest pewniejszy niż prośba do modelu. Każda poprawka zapisuje się w logu (pole `fixes`).

---

## Dane

Publiczne dane amerykańskiego urzędu NHTSA, ograniczone do producenta FCA US (Stellantis):

- **skargi klientów** wpływające od 01.2020 do 10.2026: 60 366 skarg (Jeep, Ram, Dodge, Chrysler, Fiat, Alfa Romeo),
- **akcje serwisowe** ogłoszone od 2020 roku: 270 akcji, 1 403 kombinacje model/rocznik/komponent, 372 części.

`prepare_data.py` czyści surowe pliki: zostawia tylko Stellantis i tylko pojazdy (bez części zamiennych Mopar), usuwa duplikaty, ujednolica nazwy komponentów i dzieli plik akcji na trzy tabele (`campaigns`, `campaign_vehicles`, `campaign_parts`), bo w surowym pliku jeden wiersz to nie jedna akcja, a np. liczba objętych pojazdów powtarza się w wielu wierszach.

---

## Wyniki

Poprawne odpowiedzi we wszystkich testach liczy osobny kod (pandas), niezależnie od narzędzi agenta. Zbiory pytań dotyczą **różnych modeli aut**. Każdy zbiór testowy powstawał przed uruchomieniem agenta na nim, a jego błędy służyły potem do kolejnych poprawek. Dlatego w tabelach jest **pierwszy, niezależny wynik** każdego zbioru.

### 1. Pytania z szablonów (`eval_agent.py`)

Typy pytań: liczba skarg (także dla rocznika), pożary, ranking komponentów, trend rok do roku, lista akcji serwisowych, szczegóły akcji, pytania wieloetapowe oraz pytania, na które w danych **nie ma odpowiedzi** (koszty, sprzedaż, fabryki, inne marki), gdzie agent ma odmówić, a nie zmyślać.

| Zbiór | Wersja agenta | Auta | Pytania | Poprawne |
|---|---|---|---|---|
| dev | v7 | Cherokee, Ram 1500, Pacifica, Durango | 45 | 45 (100%) |
| test | v5 | Wrangler, Grand Cherokee, Compass, Ram 2500, Grand Caravan, Gladiator, Charger, Renegade | 93 | 89 (96%) |
| test2 | v6 | Ram 3500, Journey, Town and Country, Pacifica Plug-in Hybrid, Wagoneer, Grand Cherokee L, Wrangler 4xe, Chrysler 300 | 95 | 88 (93%) |
| test3 | v7 | ProMaster, Challenger, Dart, Chrysler 200, Patriot, Grand Cherokee 4xe, Giulia, Grand Wagoneer | 94 | 92 (98%) |
| **test4** | v9 | Stelvio, Liberty, Avenger, 124 Spider, Hornet, Wagoneer S, Ram 5500, Voyager | 102 | **102 (100%)** |

Wersja końcowa (v13) na `test4`: 102/102.

### 2. Pytania swobodne (`eval_free.py`)

Testy z szablonów zawyżały obraz: przy ręcznym zadawaniu pytań agent mylił się częściej. Dlatego dopisałem pytania pisane tak, jak pisze człowiek: bez polskich znaków („ile skarg na swiatla w dodge charger 2025”), z literówkami („Grand Cherokke”, „Pacifca”), potocznie („jak tam z pożarami w chryslerach 300”), z synonimami („reklamacje”, „zażalenia”, „recalle”), po angielsku, z porównaniami kilku modeli, zakresami lat i roczników, miesiącami, kwartałami, rankingami modeli i marek, procentami, pytaniami o największą i najstarszą akcję oraz pytania o marki Stellantis, których nie ma w danych z USA (Opel, Peugeot, Citroën).

| Zbiór | Wersja agenta | Pytania | Poprawne | Co poprawiłem po tym zbiorze |
|---|---|---|---|---|
| free | v9 | 35 | 24 (69%) | polskie znaki i odmiana nazw, skrócone nazwy modeli, brakujący komponent, rok i zakres roczników z pytania, maksimum i procent liczone przez narzędzie, kontrole złego narzędzia i podmienionego modelu |
| free2 | v10 | 35 | 33 (94%) | informacja, że Opel i Peugeot należą do Stellantis, ale nie ma ich w danych; linia „Zakres” |
| free3 | v11 | 34 | 31 (91%) | filtr miesiąca i kwartału, najstarsza akcja, wywołania dla brakujących modeli wykonywane przez kod, kontrola pustej odpowiedzi |
| free4 | v12 | 32 | 28 (88%) | maksimum w rozkładzie po miesiącach, komponent przenoszony do kolejnego kroku, „Jeep Avenger” (jest tylko Dodge Avenger), lata poza zakresem danych |

- Po pierwszej rundzie poprawek wynik na nowych pytaniach wynosi około 90% i dalej utrzymuje się na tym poziomie. Każdy nowy zbiór znajduje kilka nowych, coraz rzadszych przypadków. To realna dokładność tej architektury z modelem 8B na swobodnych pytaniach.
- Wszystkie odpowiedzi przejrzałem ręcznie. W `free2` i `free3` dwie odpowiedzi poprawne merytorycznie zostały źle ocenione przez skrypt („w kwartale III”, „spadła o 2,9%”); poprawiłem ocenianie i przeliczyłem zapisane odpowiedzi (`--rescore`). W `free2` jedną odpowiedź oceniłem ręcznie jako błędną, choć skrypt ją zaliczył (agent twierdził, że Opel nie należy do Stellantis; to samo napisał o Peugeocie). W `free4` dwie zaliczone odpowiedzi zawierały obok poprawnej liczby drobne fałszywe zdanie; przy ostrej ocenie byłoby 26/32.
- Poprawki po `free4` (wersja v13) sprawdziłem tylko na zapisanych wywołaniach i na `test4`, a nie na nowym zbiorze swobodnych pytań.

### Uwagi do testów

- Model: Qwen3 8B (Ollama), bez trybu „myślenia”, temperatura 0. Średnio około 8 s na pytanie na laptopowym RTX 5070 (8 GB).
- Ocena sprawdza **główną odpowiedź** na pytanie (np. przy rankingu: czy jako pierwszy wymieniony jest właściwy element), a nie każde zdanie w odpowiedzi.
- Przy testach `test` i `test2` część „błędów” okazała się błędami skryptu oceniającego (np. „2.135 pojazdów” nie było rozpoznawane jako 2135). Po poprawce oceniania zmieniły się wyłącznie te oceny. Automatyczne wyniki przed poprawką: 86/93 i 87/95.
- Przed każdą zmianą sprawdzałem nowe reguły na wszystkich zapisanych wywołaniach z wcześniejszych testów, żeby zobaczyć, czy nie zmieniają tego, co działało.
- Wszystkie odpowiedzi agenta, wywołania narzędzi, poprawki kodu i oceny są w `results/`.

---

## Ograniczenia

- Dane NHTSA to skargi zgłaszane do urzędu w USA, a nie wewnętrzne dane reklamacyjne producenta. To mała, nastawiona na bezpieczeństwo próbka wszystkich usterek. Nie ma w nich kosztów napraw ani danych o sprzedaży.
- Na swobodnych pytaniach agent myli się w około 1 na 10 przypadków. Główne źródło błędów to rozumienie pytania przez mały model, a nie liczenie. Linia „Zakres” pozwala wychwycić błędną interpretację.
- Bezpiecznik 1 opiera się na ręcznie napisanych regułach (lista początków słów dla komponentów, wzorce dla lat, miesięcy i roczników). Każdy nowy sposób zadania pytania może trafić w lukę.
- Bezpiecznik liczb sprawdza, czy liczba występuje w wynikach, a nie czy stoi przy właściwej rzeczy. Nie wyłapuje też fałszywych zdań bez liczb (np. błędnej informacji o marce).
- Agent odpowiada tylko na pytania, które da się złożyć z pięciu narzędzi.
- Niejednoznaczne pytania („Dodge Charger 2025”: rocznik czy rok skargi?) agent rozstrzyga sam, zamiast dopytać.

**Następne kroki:**
- model zwraca tylko strukturalne zapytanie (JSON według schematu), a kod je sprawdza, wykonuje i sam składa odpowiedź z szablonu, więc model w ogóle nie przepisuje liczb;
- dopytywanie przy niejednoznacznych pytaniach;
- pytania testowe pisane przez innych ludzi, nie przeze mnie;
- porównanie z większym modelem.

---

## Co sprawdziłem wcześniej i odrzuciłem (`experiments/`)

Zanim powstał agent, sprawdziłem trzy inne podejścia i każde zmierzyłem na danych z lat 2024–2026, nieużywanych przy strojeniu:

1. **Wczesne ostrzeganie o akcjach serwisowych ze wzrostów liczby skarg** (karta kontrolna Shewharta): po 12% alarmów w ciągu pół roku pojawiała się akcja serwisowa na ten model i komponent. To 4 razy częściej niż przy losowym wyborze (2,9%), ale nie lepiej niż prosta reguła „co najmniej 10 skarg w 4 tygodnie” (16%). Sam wzrost liczby skarg nie wnosił informacji ponad ich liczbę.
2. **Przypisywanie skargi do znanej akcji serwisowej wyszukiwaniem tekstu (TF-IDF):** 79% trafień przy 31% losowo (1 266 skarg), ale słabe rozpoznawanie, że skarga dotyczy nowego, nieznanego problemu.
3. **Weryfikacja dopasowania przez model językowy:** Gemini (chmura) wyraźnie pomagał. Przy tym samym poziomie rozpoznawania nowych problemów (86%) wskazywał właściwą znaną wadę w 67% przypadków, a samo wyszukiwanie w 42% (próbka 72 przypadków). Lokalny Qwen3 8B, także w trybie „myślenia”, nie był lepszy od samego wyszukiwania z progiem, a limity chmury blokowały pracę.

Wniosek, który ukształtował projekt: mały lokalny model słabo radzi sobie z subtelną oceną tekstu, ale dobrze wybiera narzędzia. Dlatego w agencie model tylko decyduje, a liczy kod.

---

## Uruchomienie

1. Pobierz z [NHTSA Datasets and APIs](https://www.nhtsa.gov/nhtsa-datasets-and-apis) pliki `COMPLAINTS_RECEIVED_2020-2024.zip`, `COMPLAINTS_RECEIVED_2025-2026.zip` (skargi) i `FLAT_RCL_POST_2010.zip` (akcje serwisowe) do folderu `data/raw/`.
2. Zainstaluj biblioteki i przygotuj dane:
   ```
   python -m pip install -r requirements.txt
   python prepare_data.py
   ```
3. Zainstaluj [Ollamę](https://ollama.com) i pobierz model:
   ```
   ollama pull qwen3:8b
   ```
4. Zadaj pytanie:
   ```
   python agent.py "Który komponent w Dodge Challenger miał najwięcej skarg w 2025 roku?"
   ```
5. Testy (pytania i poprawne odpowiedzi możesz najpierw wypisać z `--list`):
   ```
   python eval_agent.py --list test4
   python eval_agent.py test4
   python eval_free.py --list free4
   python eval_free.py free4
   ```

## Struktura

```
prepare_data.py   przygotowanie danych NHTSA
tools.py          narzędzia (funkcje liczące na danych) i ich opisy dla modelu
agent.py          pętla agenta, bezpieczniki i linia „Zakres”
eval_agent.py     test na pytaniach z szablonów (zbiory dev, test-test4)
eval_free.py      test na pytaniach swobodnych (zbiory free-free4)
results/          zapisane odpowiedzi agenta z testów
experiments/      wcześniejsze, odrzucone podejścia (opis wyżej)
```

## Jak powstał projekt

Projekt zbudowałem metodą AI-assisted development: kod generował asystent AI na podstawie wymagań, które określiłem.
Ja wybierałem kierunek projektu i decydowałem o odrzuceniu podejść, które nie dawały dobrych wyników, sprawdzałem dane (m.in. wychwyciłem błędy w strukturze tabel akcji serwisowych), uruchamiałem eksperymenty i testy na swoim komputerze z lokalnym modelem, analizowałem wyniki i zdecydowałem o dodaniu testu na pytaniach swobodnych, gdy testy z szablonów okazały się zbyt łatwe.
