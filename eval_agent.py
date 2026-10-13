"""
Test agenta na pytaniach z szablonów: poprawne odpowiedzi liczy osobny, prosty kod (pandas), niezależny od narzędzi agenta.
Pytania są w kilku zbiorach z RÓŻNYMI modelami samochodów:
  dev                - na nim poprawiałem instrukcje, narzędzia i bezpieczniki,
  test, test2, test3 - każdy uruchomiony po kolejnej rundzie poprawek (wersje v5, v6, v7) na autach,
                       których wcześniej nie używałem; ich błędy posłużyły do kolejnych poprawek,
  test4              - nowe auta, dodatkowo pytania o rocznik i o pełne listy akcji serwisowych.
Pytania sformułowane swobodnie, jak przez użytkownika, są w eval_free.py.

Uruchomienie:
  python eval_agent.py --list test4      (tylko wypisuje pytania i poprawne odpowiedzi)
  python eval_agent.py test4             (uruchamia agenta)
  python eval_agent.py test4 --rescore   (ponowna ocena zapisanych odpowiedzi, bez uruchamiania agenta)
Wyniki zapisują się w results/agent_<zbiór>_<wersja>_<model>.jsonl; można przerwać i wznowić.
"""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

VERSION = "v13"   # zmieniam po każdej poprawce agenta/narzędzi, żeby wyniki kolejnych wersji się nie mieszały
DATA = Path("data/processed")
C = pd.read_csv(DATA / "complaints.csv", parse_dates=["received_date"])
C["year"] = C["received_date"].dt.year
K = pd.read_csv(DATA / "campaigns.csv", parse_dates=["report_date"])
V = pd.read_csv(DATA / "campaign_vehicles.csv")

# Polskie nazwy komponentów użyte w pytaniach (i akceptowane w odpowiedziach).
PL = {
    "ELECTRICAL SYSTEM": ["układ elektryczny", "elektryczn", "elektryk", "elektronik", "electrical"],
    "ENGINE": ["silnik", "engine"],
    "POWER TRAIN": ["układ napędowy", "skrzyni", "skrzynia", "napęd", "power train", "powertrain"],
    "SERVICE BRAKES": ["hamulc", "hamulec", "brakes"],
    "STEERING": ["układ kierowniczy", "kierowni", "steering"],
    "FUEL SYSTEM": ["układ paliwowy", "paliw", "fuel system"],
    "AIR BAGS": ["poduszk", "air bag", "airbag"],
    "SUSPENSION": ["zawieszen", "suspension"],
    "STRUCTURE": ["nadwozi", "konstrukcj", "structure"],
    "VEHICLE SPEED CONTROL": ["tempomat", "prędkości", "speed control"],
    "FORWARD COLLISION AVOIDANCE": ["kolizj", "collision"],
    "SEATS": ["fotel", "seats"],
    "EXTERIOR LIGHTING": ["oświetleni", "lighting"],
    "VISIBILITY": ["widoczno", "wycieracz", "visibility"],
    "SEAT BELTS": ["pas", "seat belt"],
    "BACK OVER PREVENTION": ["cofania", "back over"],
    "WHEELS": ["koła", "wheels"],
}
QUESTION_NAME = {  # forma użyta w pytaniu
    "ELECTRICAL SYSTEM": "układ elektryczny", "ENGINE": "silnik", "POWER TRAIN": "układ napędowy",
    "SERVICE BRAKES": "hamulce", "STEERING": "układ kierowniczy", "FUEL SYSTEM": "układ paliwowy",
    "AIR BAGS": "poduszki powietrzne", "SUSPENSION": "zawieszenie", "STRUCTURE": "nadwozie",
    "VEHICLE SPEED CONTROL": "tempomat", "FORWARD COLLISION AVOIDANCE": "system zapobiegania kolizjom",
    "SEATS": "fotele", "EXTERIOR LIGHTING": "oświetlenie zewnętrzne", "VISIBILITY": "widoczność i wycieraczki",
    "SEAT BELTS": "pasy bezpieczeństwa", "BACK OVER PREVENTION": "kamerę cofania", "WHEELS": "koła",
}

SPLITS = {
    "dev": [("JEEP", "CHEROKEE"), ("RAM", "1500"), ("CHRYSLER", "PACIFICA"), ("DODGE", "DURANGO")],
    "test": [("JEEP", "WRANGLER"), ("JEEP", "GRAND CHEROKEE"), ("JEEP", "COMPASS"), ("RAM", "2500"),
             ("DODGE", "GRAND CARAVAN"), ("JEEP", "GLADIATOR"), ("DODGE", "CHARGER"), ("JEEP", "RENEGADE")],
    # test2: świeży zbiór po poprawkach z wersji v6 - inne auta niż w dev i test, nieużywane przy poprawkach
    "test2": [("RAM", "3500"), ("DODGE", "JOURNEY"), ("CHRYSLER", "TOWN AND COUNTRY"),
              ("CHRYSLER", "PACIFICA PLUG-IN HYBRID"), ("JEEP", "WAGONEER"), ("JEEP", "GRAND CHEROKEE L"),
              ("JEEP", "WRANGLER 4XE"), ("CHRYSLER", "300")],
    # test3: świeży zbiór po poprawkach z wersji v7 (kod przypina model z pytania) - kolejne nowe auta
    "test3": [("RAM", "PROMASTER"), ("DODGE", "CHALLENGER"), ("DODGE", "DART"), ("CHRYSLER", "200"),
              ("JEEP", "PATRIOT"), ("JEEP", "GRAND CHEROKEE 4XE"), ("ALFA ROMEO", "GIULIA"), ("JEEP", "GRAND WAGONEER")],
    # test4: świeży zbiór po poprawkach z wersji v9 (filtr rocznika, pełne listy akcji) - kolejne nowe auta,
    # dodatkowo pytania o rocznik i o dłuższe listy akcji
    "test4": [("ALFA ROMEO", "STELVIO"), ("JEEP", "LIBERTY"), ("DODGE", "AVENGER"), ("FIAT", "124 SPIDER"),
              ("DODGE", "HORNET"), ("JEEP", "WAGONEER S"), ("RAM", "5500"), ("CHRYSLER", "VOYAGER")],
}
DISPLAY = {"1500": "Ram 1500", "2500": "Ram 2500", "3500": "Ram 3500", "300": "Chrysler 300",
           "TOWN AND COUNTRY": "Chrysler Town and Country", "WRANGLER 4XE": "Jeep Wrangler 4xe",
           "PACIFICA PLUG-IN HYBRID": "Chrysler Pacifica Plug-in Hybrid", "GRAND CHEROKEE L": "Jeep Grand Cherokee L",
           "PROMASTER": "Ram ProMaster", "200": "Chrysler 200", "GRAND CHEROKEE 4XE": "Jeep Grand Cherokee 4xe"}

UNANSWERABLE = {
    "dev": ["Ile kosztowała Stellantis akcja serwisowa 24V896000?",
            "Ile Jeepów Cherokee sprzedano w 2024 roku?",
            "Ile skarg na hamulce miała Toyota Corolla w 2024 roku?"],
    "test": ["Jaki był koszt napraw gwarancyjnych skrzyń biegów w Jeepie Wrangler w 2025 roku?",
             "Który dostawca wyprodukował wadliwe części objęte akcją 22V406000?",
             "Ile skarg na silnik miał Volkswagen Golf w 2025 roku?",
             "W której fabryce wyprodukowano Jeepy Compass objęte akcjami serwisowymi?",
             "Ile Dodge Charger zostało sprzedanych w Europie w 2024 roku?"],
    "test2": ["Jaka była średnia cena naprawy pompy paliwa w Ram 3500?",
              "Ile Jeepów Wagoneer wyprodukowano w 2025 roku?",
              "Który dostawca dostarczał poduszki powietrzne do Chryslera 300?",
              "Ile skarg na skrzynię biegów miała Toyota RAV4 w 2025 roku?",
              "Jaki był zysk Stellantis w 2025 roku?"],
    "test3": ["Ile kosztuje nowy Dodge Challenger w Polsce?",
              "Jaka jest średnia ocena klientów dla Alfa Romeo Giulia?",
              "Ile reklamacji gwarancyjnych zgłoszono w fabryce w Tychach w 2025 roku?",
              "Ile skarg na hamulce miał Ford F-150 w 2025 roku?",
              "Ilu pracowników zatrudnia Stellantis w Gliwicach?"],
    "test4": ["Ile kosztuje wymiana skrzyni biegów w Jeepie Liberty?",
              "Który model Stellantis sprzedawał się najlepiej w Polsce w 2025 roku?",
              "Ile skarg na zawieszenie miała Honda Civic w 2024 roku?",
              "Jaki jest czas oczekiwania na części do Alfa Romeo Stelvio?",
              "Ile wyniosły koszty gwarancyjne Stellantis w 2025 roku?"],
}


def name(make, model):
    return DISPLAY.get(model, f"{make.title()} {model.title()}")


def n_complaints(df):
    return int(df["odino"].nunique())


def build(split):
    qs = []
    for make, model in SPLITS[split]:
        m = C[(C.make == make) & (C.model == model)]
        nm = name(make, model)
        comps = [x for x in m[m.year.isin([2024, 2025])].component_group.value_counts().index
                 if x in QUESTION_NAME][:3]
        # 1) liczba skarg na komponent w roku
        for comp in comps[:2]:
            for year in [2025]:
                qs.append(dict(type="liczba", q=f"Ile skarg na {QUESTION_NAME[comp]} w {nm} wpłynęło w {year} roku?",
                               expected=n_complaints(m[(m.component_group == comp) & (m.year == year)])))
        # 2) komponent z największą liczbą skarg
        for year in [2024, 2025]:
            g = m[(m.year == year) & (m.component_group != "UNKNOWN OR OTHER")]
            top = g.drop_duplicates(["odino", "component_group"]).component_group.value_counts()
            if len(top) > 1 and top.iloc[0] > top.iloc[1]:
                qs.append(dict(type="ranking komponentów",
                               q=f"Który komponent w {nm} miał najwięcej skarg w {year} roku (nie licząc kategorii 'inne')?",
                               expected=top.index[0]))
        # 3) trend rok do roku
        comp = comps[0]
        a = n_complaints(m[(m.component_group == comp) & (m.year == 2024)])
        b = n_complaints(m[(m.component_group == comp) & (m.year == 2025)])
        if a > 0:
            qs.append(dict(type="trend",
                           q=f"O ile procent zmieniła się liczba skarg na {QUESTION_NAME[comp]} w {nm} w 2025 roku w porównaniu z 2024?",
                           expected=round(100 * (b - a) / a, 1)))
        # 4) pożary
        qs.append(dict(type="liczba (pożar)", q=f"Ile skarg na {nm} wpłyniętych w 2025 roku dotyczyło pożaru?",
                       expected=int(m[m.year == 2025].drop_duplicates("odino").fire.eq("Y").sum())))
        # 5) akcje serwisowe na komponent
        for comp in comps:
            ids = sorted(V[(V.make == make) & (V.model == model) & (V.component_group == comp)].campaign.unique())
            if 1 <= len(ids) <= 4:
                qs.append(dict(type="lista akcji",
                               q=f"Jakie akcje serwisowe dotyczące komponentu {QUESTION_NAME[comp]} obejmowały {nm}? Podaj numery.",
                               expected=ids))
                break
        # 6) liczba pojazdów w najnowszej akcji tego modelu + wieloetapowe
        ids = V[(V.make == make) & (V.model == model)].campaign.unique()
        kk = K[K.campaign.isin(ids)].sort_values("report_date")
        if len(kk) and (len(kk) == 1 or kk.iloc[-1].report_date > kk.iloc[-2].report_date):  # bez remisów
            last = kk.iloc[-1]
            qs.append(dict(type="szczegóły akcji", q=f"Ilu pojazdów dotyczyła akcja serwisowa {last.campaign}?",
                           expected=int(last.vehicles_affected)))
            qs.append(dict(type="wieloetapowe",
                           q=f"Jaka jest najnowsza akcja serwisowa obejmująca {nm} i ilu pojazdów dotyczyła?",
                           expected=[last.campaign, int(last.vehicles_affected)]))
    # Pytania sformułowane inaczej niż szablony powyżej - osobne sformułowania dla dev i dla test,
    # żeby sprawdzić, czy agent radzi sobie z innym językiem pytania, a nie tylko z jednym szablonem.
    for make, model in SPLITS[split]:
        m = C[(C.make == make) & (C.model == model)]
        nm = name(make, model)
        g = m[(m.year == 2025) & (m.component_group != "UNKNOWN OR OTHER")]
        top = g.drop_duplicates(["odino", "component_group"]).component_group.value_counts()
        comp = next(x for x in top.index if x in QUESTION_NAME)
        cnt = n_complaints(m[(m.component_group == comp) & (m.year == 2025)])
        a = n_complaints(m[(m.component_group == comp) & (m.year == 2024)])
        if split == "dev":
            qs.append(dict(type="liczba", expected=cnt,
                           q=f"Ile zgłoszeń dotyczących komponentu {QUESTION_NAME[comp]} odnotowano dla {nm} w 2025?"))
            if len(top) > 1 and top.iloc[0] > top.iloc[1]:
                qs.append(dict(type="ranking komponentów", expected=top.index[0],
                               q=f"Na jaki element {nm} kierowcy skarżyli się najczęściej w 2025 roku? Pomiń kategorię innych/nieznanych."))
        elif split == "test3":
            qs.append(dict(type="liczba", expected=cnt,
                           q=f"Policz skargi na {QUESTION_NAME[comp]} zgłoszone dla {nm} w roku 2025."))
            if len(top) > 1 and top.iloc[0] > top.iloc[1]:
                qs.append(dict(type="ranking komponentów", expected=top.index[0],
                               q=f"Jaki obszar pojazdu {nm} był najczęściej reklamowany w 2025 roku (bez kategorii inne)?"))
            if a > 0:
                qs.append(dict(type="trend", expected=round(100 * (cnt - a) / a, 1),
                               q=f"{nm}: jaka była procentowa zmiana liczby skarg na {QUESTION_NAME[comp]} między 2024 a 2025 rokiem?"))
        elif split == "test4":
            qs.append(dict(type="liczba", expected=cnt,
                           q=f"Ile skarg dotyczących komponentu {QUESTION_NAME[comp]} zarejestrowano w 2025 roku dla {nm}?"))
            if len(top) > 1 and top.iloc[0] > top.iloc[1]:
                qs.append(dict(type="ranking komponentów", expected=top.index[0],
                               q=f"Który układ w {nm} sprawiał klientom najwięcej problemów w 2025 roku? Kategorię inne pomiń."))
            if a > 0:
                qs.append(dict(type="trend", expected=round(100 * (cnt - a) / a, 1),
                               q=f"O ile procent wzrosła lub spadła liczba skarg na {QUESTION_NAME[comp]} w {nm} w 2025 roku względem 2024?"))
            # pytanie o rocznik: tu filtr rocznika jest potrzebny (wszystkie lata wpływu skarg)
            years = m.drop_duplicates("odino").model_year.value_counts()
            if len(years) > 1 and years.iloc[0] > years.iloc[1]:
                qs.append(dict(type="liczba", expected=int(years.iloc[0]),
                               q=f"Ile skarg wpłynęło na {nm} z rocznika {int(years.index[0])}?"))
            # pełna lista akcji modelu, dłuższa niż w szablonie (do 15, tyle pokazuje narzędzie)
            ids = sorted(V[(V.make == make) & (V.model == model)].campaign.unique())
            if 5 <= len(ids) <= 15:
                qs.append(dict(type="lista akcji", expected=ids,
                               q=f"Wymień wszystkie akcje serwisowe obejmujące {nm}. Podaj numery."))
        elif split == "test2":
            qs.append(dict(type="liczba", expected=cnt,
                           q=f"Ile razy w 2025 roku klienci zgłaszali problem z komponentem {QUESTION_NAME[comp]} w {nm}?"))
            if len(top) > 1 and top.iloc[0] > top.iloc[1]:
                qs.append(dict(type="ranking komponentów", expected=top.index[0],
                               q=f"Która grupa komponentów generowała najwięcej skarg dla {nm} w 2025? Pomiń 'inne'."))
            if a > 0:
                qs.append(dict(type="trend", expected=round(100 * (cnt - a) / a, 1),
                               q=f"Jak w 2025 roku w porównaniu z 2024 zmieniła się liczba skarg dotyczących komponentu {QUESTION_NAME[comp]} w {nm}? Podaj zmianę w procentach."))
        else:
            qs.append(dict(type="liczba", expected=cnt,
                           q=f"Podaj liczbę skarg klientów na {QUESTION_NAME[comp]} w modelu {nm} za rok 2025."))
            if len(top) > 1 and top.iloc[0] > top.iloc[1]:
                qs.append(dict(type="ranking komponentów", expected=top.index[0],
                               q=f"Co było najczęstszą przyczyną skarg w {nm} w 2025 roku? Nie uwzględniaj kategorii inne."))
            if a > 0:
                qs.append(dict(type="trend", expected=round(100 * (cnt - a) / a, 1),
                               q=f"Czy w 2025 roku skarg na {QUESTION_NAME[comp]} w {nm} było więcej czy mniej niż w 2024 i o ile procent?"))
    seen, unique = set(), []
    for q in qs:                       # bez powtórzeń tego samego pytania
        if q["q"] not in seen:
            seen.add(q["q"]); unique.append(q)
    qs = unique
    for q in UNANSWERABLE[split]:
        qs.append(dict(type="brak danych", q=q, expected="BRAK"))
    for i, q in enumerate(qs):
        q["id"] = f"{split}-{i:02d}"
    return qs


# ---------- ocena odpowiedzi ----------

def numbers(text):
    # separator tysięcy tylko w poprawnym zapisie: 1-3 cyfry, potem grupy po 3 (1 234, 1,234, 2.135, 844 027);
    # dzięki temu "Ram 3500 155 razy" to dwie liczby (3500 i 155), a nie 3500155
    t = re.sub(r"(?<![\d.,])\d{1,3}(?:[   .,]\d{3})+(?![\d]|[.,]\d)",
               lambda m: re.sub(r"\D", "", m.group()), text)
    found = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", t)]
    if re.search(r"\b(żadn|zero\b|brak skarg|nie wpłynęł)", text.lower()):  # "żadnej skargi" = 0
        found.append(0.0)
    return found


def first_component(text):
    """Pierwszy komponent wymieniony w odpowiedzi (alias musi zaczynać słowo)."""
    low = text.lower()
    pos = []
    for comp, aliases in PL.items():
        for a in aliases + [comp.lower()]:
            m = re.search(r"(?<![a-ząćęłńóśźż])" + re.escape(a.lower()), low)
            if m:
                pos.append((m.start(), comp))
    return min(pos)[1] if pos else None


def score(q, answer):
    answer = answer.split("\n\nZakres:")[0]   # linię zakresu dopisuje kod, nie oceniamy jej
    t, e = q["type"], q["expected"]
    if t in ("liczba", "liczba (pożar)", "szczegóły akcji"):
        return float(e) in numbers(answer)
    if t == "ranking komponentów":
        return first_component(answer) == e
    if t == "trend":
        return any(abs(abs(x) - abs(e)) <= 0.6 for x in numbers(answer)) and \
            (("spad" in answer.lower() or "mniej" in answer.lower() or "-" in answer) if e < 0 else True)
    if t == "lista akcji":
        found = {x[:6] for x in re.findall(r"\d{2}V\d{3}", answer)}
        return found == {x[:6] for x in e}
    if t == "wieloetapowe":
        return e[0][:6] in answer and float(e[1]) in numbers(answer)
    if t == "brak danych":
        return bool(re.search(r"brak|nie ma |nie zawiera|nie posiad|nie mam|niedostęp|nie obejmuj|nie są dostępne|"
                              r"nie dysponuj|nie znajduj|nie pozwalaj|tylko .*stellantis", answer.lower()))
    raise ValueError(t)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split = args[0] if args else "dev"
    qs = build(split)
    if "--list" in sys.argv:
        for q in qs:
            print(q["id"], q["type"], "|", q["q"], "->", q["expected"])
        print(len(qs), "pytań")
        sys.exit()

    import agent
    think = "--think" in sys.argv
    if "--rescore" in sys.argv:
        # ponowna ocena zapisanych odpowiedzi (bez uruchamiania agenta) - np. po poprawce błędu w ocenianiu
        out = Path("results") / f"agent_{split}_{VERSION}_{agent.MODEL.replace(':', '-')}{'-think' if think else ''}.jsonl"
        res = pd.DataFrame([json.loads(l) for l in out.open(encoding="utf-8")])
        res["correct_old"] = res["correct"]
        res["correct"] = [score(r, r["answer"]) for r in res.to_dict("records")]
        for r in res[res.correct != res.correct_old].itertuples():
            print(f"zmiana oceny {r.id}: {r.correct_old} -> {r.correct} | {r.answer[:150]}")
        print(f"\nWYNIK po ponownej ocenie ({split}): {res.correct.sum()}/{len(res)} = {100 * res.correct.mean():.0f}%")
        print(res.groupby("type").correct.agg(["sum", "count"]).to_string())
        sys.exit()
    out = Path("results") / f"agent_{split}_{VERSION}_{agent.MODEL.replace(':', '-')}{'-think' if think else ''}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    for q in qs:
        if q["id"] in done:
            continue
        t0 = time.time()
        try:
            answer, calls = agent.ask(q["q"], think=think, verbose=False)
        except Exception as ex:  # np. błąd połączenia z Ollamą - zapisujemy i lecimy dalej
            answer, calls = f"BŁĄD: {ex}", []
        ok = score(q, answer)
        rec = {**q, "answer": answer, "correct": ok, "calls": calls, "seconds": round(time.time() - t0, 1)}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"[{q['id']}] {'OK ' if ok else 'ŹLE'} ({rec['seconds']} s) {q['q']}")
    res = pd.DataFrame([json.loads(l) for l in out.open(encoding="utf-8")])
    print(f"\nWYNIK ({split}, {agent.MODEL}): {res.correct.sum()}/{len(res)} = {100 * res.correct.mean():.0f}%")
    print(res.groupby("type").correct.agg(["sum", "count"]).to_string())
