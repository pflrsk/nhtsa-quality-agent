"""
Test agenta na pytaniach pisanych "jak przez użytkownika", a nie z szablonu:
bez polskich znaków, z literówkami, potocznie, z synonimami, porównania, zakresy lat, rankingi modeli,
procenty, jedno pytanie po angielsku.

Pytania i poprawne odpowiedzi zostały zapisane przed uruchomieniem agenta. Poprawne odpowiedzi liczy
osobny kod (pandas) - funkcja expected() niżej. Ocena automatyczna sprawdza główną odpowiedź;
wszystkie odpowiedzi warto też przejrzeć ręcznie (zapisują się w results/).

Trzy zbiory, każdy napisany przed uruchomieniem agenta na nim:
  free  - pierwszy; jego błędy posłużyły do poprawek v10,
  free2 - napisany po poprawkach v10 (inne auta i sformułowania); jego błędy posłużyły do poprawki v11,
  free3 - napisany po poprawce v11; jego błędy (luki w narzędziach) posłużyły do poprawek v12,
  free4 - napisany po poprawkach v12, uruchomiony raz na wersji końcowej.

Uruchomienie:
  python eval_free.py free4 --list   (pytania i poprawne odpowiedzi)
  python eval_free.py free4          (uruchamia agenta)
"""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

import eval_agent as E

C, K, V = E.C, E.K, E.V
C["month"] = C["received_date"].dt.month
C["quarter"] = C["received_date"].dt.quarter


def n(df):
    return int(df["odino"].nunique())


def model(make, mod):
    return C[(C.make == make) & (C.model == mod)]


def flag(df, col):
    d = df.drop_duplicates("odino")
    return int((d[col] == "Y").sum()) if col != "injured" else int((d["injured"].fillna(0) > 0).sum())


def campaigns_of(make, mod, comp=None):
    v = V[(V.make == make) & (V.model == mod)]
    if comp:
        v = v[v.component_group == comp]
    return K[K.campaign.isin(v.campaign)]


def expected():
    """Lista (typ, pytanie, poprawna odpowiedź)."""
    q = []
    add = lambda t, text, e: q.append(dict(type=t, q=text, expected=e))
    # zwykłe liczby, ale innymi słowami, bez znaków, z literówką, z synonimem
    add("liczba", "ile było skarg na kierownicę w jeepie cherokee w 2024?",
        n(model("JEEP", "CHEROKEE").query("component_group == 'STEERING' and year == 2024")))
    add("liczba", "Ile skarg na układ hamulcowy dostał Dodge Durango w 2023 roku?",
        n(model("DODGE", "DURANGO").query("component_group == 'SERVICE BRAKES' and year == 2023")))
    add("liczba", "ile skarg na silnik mial jeep compass w 2022",
        n(model("JEEP", "COMPASS").query("component_group == 'ENGINE' and year == 2022")))
    add("liczba", "Ile skarg na zawieszenie miał Jeep Grand Cherokke w 2025?",
        n(model("JEEP", "GRAND CHEROKEE").query("component_group == 'SUSPENSION' and year == 2025")))
    add("liczba", "ile skarg na swiatla w dodge charger 2025",
        n(model("DODGE", "CHARGER").query("component_group == 'EXTERIOR LIGHTING' and year == 2025")))
    add("liczba", "W 2025 ile było skarg na Pacifica Hybrid?",
        n(model("CHRYSLER", "PACIFICA PLUG-IN HYBRID").query("year == 2025")))
    add("liczba", "How many engine complaints did the Ram 1500 get in 2025?",
        n(model("RAM", "1500").query("component_group == 'ENGINE' and year == 2025")))
    # zakresy czasu, miesiąc, rocznik
    add("liczba", "Ile skarg na elektrykę Chryslera Pacifica wpłynęło łącznie w latach 2021-2023?",
        n(model("CHRYSLER", "PACIFICA").query("component_group == 'ELECTRICAL SYSTEM' and 2021 <= year <= 2023")))
    add("liczba", "Ile skarg na Jeep Compass wpłynęło w marcu 2025?",
        n(model("JEEP", "COMPASS").query("year == 2025 and month == 3")))
    add("liczba", "Ile skarg dotyczyło Jeep Wrangler z roczników 2018-2020?",
        n(model("JEEP", "WRANGLER").query("2018 <= model_year <= 2020")))
    add("liczba", "Jeep Compass rocznik 2017 - ile skarg na silnik?",
        n(model("JEEP", "COMPASS").query("model_year == 2017 and component_group == 'ENGINE'")))
    add("liczba", "Ile w sumie skarg na Jeepy wpłynęło w 2024 roku?", n(C[(C.make == "JEEP") & (C.year == 2024)]))
    # wypadki, pożary, ranni - potocznie
    add("liczba", "W ilu skargach na Dodge Charger zgłoszonych w 2024 roku był wypadek?",
        flag(model("DODGE", "CHARGER").query("year == 2024"), "crash"))
    add("liczba", "jak tam z pożarami w chryslerach 300 w 2025?", flag(model("CHRYSLER", "300").query("year == 2025"), "fire"))
    add("liczba", "Ile skarg z rannymi wpłynęło na Ram 2500 w 2024?", flag(model("RAM", "2500").query("year == 2024"), "injured"))
    add("liczba", "Ile skarg na Wranglera w 2025 dotyczyło wypadków?", flag(model("JEEP", "WRANGLER").query("year == 2025"), "crash"))
    # porównania i trend
    add("liczby", "Który model miał więcej skarg w 2025: Jeep Compass czy Jeep Renegade? Podaj obie liczby.",
        [n(model("JEEP", "COMPASS").query("year == 2025")), n(model("JEEP", "RENEGADE").query("year == 2025"))])
    add("liczby", "Porównaj liczbę skarg na hamulce w Jeep Grand Cherokee w 2023 i 2024.",
        [n(model("JEEP", "GRAND CHEROKEE").query(f"component_group == 'SERVICE BRAKES' and year == {y}")) for y in (2023, 2024)])
    a, b = (n(model("JEEP", "GLADIATOR").query(f"year == {y}")) for y in (2024, 2025))
    add("trend lub liczby", "Czy na Jeepa Gladiatora w 2025 skarżono się częściej niż rok wcześniej?",
        [a, b, round(100 * (b - a) / a, 1)])
    # rankingi modeli, kilka komponentów, miesiąc
    top = lambda df: df.groupby(["make", "model"]).odino.nunique().sort_values(ascending=False)
    add("ranking modeli", "Który model Jeepa miał najwięcej skarg w 2025 roku?", top(C[(C.make == "JEEP") & (C.year == 2025)]).index[0][1])
    add("ranking modeli", "Który samochód Stellantis miał najwięcej skarg na poduszki powietrzne w 2024?",
        top(C[(C.component_group == "AIR BAGS") & (C.year == 2024)]).index[0][1])
    add("ranking modeli", "Które auto Stellantis zebrało najwięcej skarg w 2025 roku?", top(C[C.year == 2025]).index[0][1])
    g = model("RAM", "1500").query("year == 2024 and component_group != 'UNKNOWN OR OTHER'")
    add("komponenty", "Podaj trzy komponenty z największą liczbą skarg dla Ram 1500 w 2024 (bez kategorii inne).",
        list(g.drop_duplicates(["odino", "component_group"]).component_group.value_counts().index[:3]))
    add("procent", "Jaki procent skarg na Jeep Cherokee w 2024 roku dotyczył układu elektrycznego?",
        round(100 * n(model("JEEP", "CHEROKEE").query("year == 2024 and component_group == 'ELECTRICAL SYSTEM'"))
              / n(model("JEEP", "CHEROKEE").query("year == 2024")), 1))
    add("miesiąc", "W którym miesiącu 2025 roku Ram 1500 miał najwięcej skarg?",
        int(model("RAM", "1500").query("year == 2025").groupby("month").odino.nunique().idxmax()))
    # akcje serwisowe
    add("lista akcji", "Jakie były akcje serwisowe dotyczące kamery cofania w Jeep Grand Cherokee? Podaj numery.",
        sorted(campaigns_of("JEEP", "GRAND CHEROKEE", "BACK OVER PREVENTION").campaign))
    add("liczba", "Ile pojazdów objęła największa akcja serwisowa dotycząca Jeepa Wranglera?",
        int(campaigns_of("JEEP", "WRANGLER").vehicles_affected.max()))
    last = campaigns_of("DODGE", "DURANGO").sort_values("report_date").iloc[-1]
    add("data i liczba", "Kiedy ogłoszono ostatnią akcję serwisową dla Dodge Durango i ile aut obejmowała?",
        [last.report_date.date().isoformat(), int(last.vehicles_affected)])
    add("liczba", "Ile akcji serwisowych ogłoszono dla Ram 1500 w 2024 roku?",
        int((campaigns_of("RAM", "1500").report_date.dt.year == 2024).sum()))
    air = K[K.campaign.isin(V[V.component_group == "AIR BAGS"].campaign) & (K.report_date.dt.year == 2023)]
    add("liczba", "Ile akcji serwisowych dotyczących poduszek powietrznych ogłosił Stellantis w 2023?", len(air))
    big = K[K.report_date.dt.year == 2025].sort_values("vehicles_affected").iloc[-1]
    add("wieloetapowe", "Która akcja serwisowa ogłoszona w 2025 roku objęła najwięcej pojazdów?",
        [big.campaign, int(big.vehicles_affected)])
    d = model("DODGE", "DURANGO").query("year == 2025 and component_group != 'UNKNOWN OR OTHER'")
    comp = d.drop_duplicates(["odino", "component_group"]).component_group.value_counts().index[0]
    assert campaigns_of("DODGE", "DURANGO", comp).empty
    add("komponent bez akcji", "Który komponent Dodge Durango miał najwięcej skarg w 2025 i czy była na niego akcja serwisowa?", comp)
    # brak danych
    for text in ["Ile kosztuje naprawa hamulców w Durango?",
                 "Jaki jest wskaźnik awaryjności na 1000 sprzedanych Jeepów Compass?",
                 "Ile skarg miał Fiat Panda w 2025?"]:
        add("brak danych", text, "BRAK")
    for i, x in enumerate(q):
        x["id"] = f"free-{i:02d}"
    return q


def expected_free4():
    """Czwarty zbiór pytań swobodnych: napisany po poprawkach v12, przed uruchomieniem agenta."""
    q = []
    add = lambda t, text, e: q.append(dict(type=t, q=text, expected=e))
    comp = lambda mk, md, c, cond: n(model(mk, md).query(f"component_group == '{c}' and {cond}"))
    top = lambda df, col: df.groupby(col).odino.nunique().sort_values(ascending=False)
    add("liczba", "Ile skarg na Fiata 500 wpłynęło w pierwszym kwartale 2025?", n(model("FIAT", "500").query("year == 2025 and quarter == 1")))
    add("liczba", "Ile skarg na Dodge Charger przyszło w Q4 2024?", n(model("DODGE", "CHARGER").query("year == 2024 and quarter == 4")))
    add("liczba", "ile skarg na jeepa cherokee w lipcu 2025", n(model("JEEP", "CHEROKEE").query("year == 2025 and month == 7")))
    add("liczba", "Ile skarg na układ kierowniczy Ram 1500 wpłynęło w styczniu 2024?",
        comp("RAM", "1500", "STEERING", "year == 2024 and month == 1"))
    add("liczba", "How many complaints did the Dodge Durango receive in March 2025?",
        n(model("DODGE", "DURANGO").query("year == 2025 and month == 3")))
    add("liczby", "Porównaj liczbę skarg na Ram 1500, Ram 2500 i Ram 3500 w 2024.",
        [n(model("RAM", m).query("year == 2024")) for m in ("1500", "2500", "3500")])
    add("liczby", "Wrangler czy Gladiator - który miał więcej skarg na silnik w 2025?",
        [comp("JEEP", m, "ENGINE", "year == 2025") for m in ("WRANGLER", "GLADIATOR")])
    assert campaigns_of("DODGE", "CHARGER", "SEAT BELTS").empty
    add("nie było akcji", "Czy była jakaś akcja serwisowa dotycząca pasów bezpieczeństwa w Dodge Charger?", None)
    add("akcja", "Która akcja serwisowa dla Jeep Wrangler była najwcześniejsza w tych danych?",
        campaigns_of("JEEP", "WRANGLER").sort_values("report_date").iloc[0].campaign)
    add("data", "Kiedy ogłoszono pierwszą akcję serwisową dla Jeep Compass w tych danych?",
        campaigns_of("JEEP", "COMPASS").sort_values("report_date").iloc[0].report_date.date().isoformat())
    add("liczba", "Ile wszystkich skarg na Alfa Romeo jest w danych?", n(C[C.make == "ALFA ROMEO"]))
    g = model("JEEP", "COMPASS").query("year == 2026 and component_group != 'UNKNOWN OR OTHER'")
    add("komponenty", "Jakie trzy komponenty Jeepa Compass zbierają najwięcej skarg w 2026 roku (bez inne)?",
        list(g.drop_duplicates(["odino", "component_group"]).component_group.value_counts().index[:3]))
    add("liczba", "Ile skarg na napęd hybrydowy miała Pacifica Plug-in Hybrid w 2024?",
        comp("CHRYSLER", "PACIFICA PLUG-IN HYBRID", "HYBRID PROPULSION SYSTEM", "year == 2024"))
    add("liczba", "Grand Cherokee 4xe - ile skarg na elektrykę w 2024?", comp("JEEP", "GRAND CHEROKEE 4XE", "ELECTRICAL SYSTEM", "year == 2024"))
    add("liczba", "Ile skarg dotyczyło Dodge Journey rocznik 2017?", n(model("DODGE", "JOURNEY").query("model_year == 2017")))
    add("liczba", "Ile skarg z pożarem dotyczyło Jeep Grand Cherokee w 2024?", flag(model("JEEP", "GRAND CHEROKEE").query("year == 2024"), "fire"))
    add("liczba", "Ile skarg z obrażeniami zgłoszono dla wszystkich modeli Ram w 2024?", flag(C[(C.make == "RAM") & (C.year == 2024)], "injured"))
    y = model("DODGE", "DURANGO").groupby("year").odino.nunique()
    add("rok", "W którym roku Dodge Durango miał najwięcej skarg?", [int(y.idxmax()), int(y.max())])
    add("miesiąc", "W którym miesiącu 2024 Jeep Grand Cherokee miał najwięcej skarg?",
        int(model("JEEP", "GRAND CHEROKEE").query("year == 2024").groupby("month").odino.nunique().idxmax()))
    add("liczba", "Ile akcji serwisowych ogłoszono w 2023 roku łącznie?", int((K.report_date.dt.year == 2023).sum()))
    add("liczba", "Ile akcji serwisowych obejmujących Jeepy ogłoszono w 2026?",
        int(K[K.campaign.isin(V[V.make == "JEEP"].campaign) & (K.report_date.dt.year == 2026)].campaign.nunique()))
    big = campaigns_of("DODGE", "DURANGO").sort_values("vehicles_affected").iloc[-1]
    add("wieloetapowe", "Która akcja dla Dodge Durango objęła najwięcej pojazdów?", [big.campaign, int(big.vehicles_affected)])
    a, b = (n(model("JEEP", "GLADIATOR").query(f"year == {y}")) for y in (2023, 2025))
    add("trend lub liczby", "Jak zmieniła się liczba skarg na Jeep Gladiator między 2023 a 2025?", [a, b, round(100 * (b - a) / a, 1)])
    add("ranking modeli", "Który model miał najwięcej skarg na silnik w 2024?", top(C[(C.component_group == "ENGINE") & (C.year == 2024)], "model").index[0])
    add("ranking modeli", "Który Ram miał najwięcej skarg w 2025?", top(C[(C.make == "RAM") & (C.year == 2025)], "model").index[0])
    r = model("JEEP", "COMPASS").query("year == 2025")
    add("procent", "Ile procent skarg na Jeep Compass w 2025 dotyczyło poduszek powietrznych?",
        round(100 * n(r.query("component_group == 'AIR BAGS'")) / n(r), 1))
    add("liczba", "Ile skarg na skrzynię biegów Jeep Cherokee wpłynęło w latach 2021 i 2022 razem?",
        comp("JEEP", "CHEROKEE", "POWER TRAIN", "2021 <= year <= 2022"))
    g = model("JEEP", "COMPASS").query("year == 2025 and component_group != 'UNKNOWN OR OTHER'")
    c0 = g.drop_duplicates(["odino", "component_group"]).component_group.value_counts().index[0]
    add("komponent i akcja", "Który komponent Jeep Compass miał najwięcej skarg w 2025 i jaka jest najnowsza akcja serwisowa na ten komponent?",
        [c0, campaigns_of("JEEP", "COMPASS", c0).sort_values("report_date").iloc[-1].campaign])
    for text in ["Ile skarg miał Jeep Avenger w 2025?",
                 "Jaka jest średnia cena naprawy skrzyni biegów w Jeepie Compass?",
                 "Ile skarg miał Opel Corsa w 2024?",
                 "Ile skarg na Jeep Wrangler wpłynęło w 2019 roku?"]:
        add("brak danych", text, "BRAK")
    for i, x in enumerate(q):
        x["id"] = f"free4-{i:02d}"
    return q


def expected_free3():
    """Trzeci zbiór pytań swobodnych: napisany po poprawce v11, przed uruchomieniem agenta."""
    q = []
    add = lambda t, text, e: q.append(dict(type=t, q=text, expected=e))
    comp = lambda mk, md, c, cond: n(model(mk, md).query(f"component_group == '{c}' and {cond}"))
    top = lambda df, col: df.groupby(col).odino.nunique().sort_values(ascending=False)
    add("liczba", "ProMaster od Rama - ile skarg na elektryke bylo w 2024?", comp("RAM", "PROMASTER", "ELECTRICAL SYSTEM", "year == 2024"))
    add("liczba", "Ile zażaleń na układ paliwowy złożono na Jeep Grand Cherokee w 2025 r.?",
        comp("JEEP", "GRAND CHEROKEE", "FUEL SYSTEM", "year == 2025"))
    add("liczba", "Ile skarg dostał Fiat 500X w całym 2023 roku?", n(model("FIAT", "500X").query("year == 2023")))
    add("liczba", "grand caravan - ile skarg na zamki i klamki w 2023?",
        comp("DODGE", "GRAND CARAVAN", "LATCHES/LOCKS/LINKAGES", "year == 2023"))
    add("liczba", "Ile skarg na silnik w Jeep Patriot zgłoszono od początku 2020 do końca 2022?",
        comp("JEEP", "PATRIOT", "ENGINE", "2020 <= year <= 2022"))
    add("liczba", "What was the number of transmission complaints for the Jeep Gladiator in 2024?",
        comp("JEEP", "GLADIATOR", "POWER TRAIN", "year == 2024"))
    add("liczba", "Ile skarg w październiku 2024 dotyczyło Jeep Wrangler?", n(model("JEEP", "WRANGLER").query("year == 2024 and month == 10")))
    add("liczba", "Ile skarg dotyczyło Chryslerów 300 z rocznika 2015?", n(model("CHRYSLER", "300").query("model_year == 2015")))
    add("liczba", "Ile skarg na zawieszenie dotyczyło Ram 2500 z lat modelowych 2019-2021?",
        comp("RAM", "2500", "SUSPENSION", "2019 <= model_year <= 2021"))
    add("liczba", "ile skarg na kamere cofania w ram 1500 w 2025", comp("RAM", "1500", "BACK OVER PREVENTION", "year == 2025"))
    add("liczba", "Ile skarg na oświetlenie Dodge Durango wpłynęło w 2024?", comp("DODGE", "DURANGO", "EXTERIOR LIGHTING", "year == 2024"))
    add("liczba", "Ile skarg na Jeep Wrangler 4xe w 2025?", n(model("JEEP", "WRANGLER 4XE").query("year == 2025")))
    add("liczba", "Ile skarg z pożarem dotyczyło wszystkich Jeepów w 2025?", flag(C[(C.make == "JEEP") & (C.year == 2025)], "fire"))
    add("liczba", "W ilu skargach na Dodge Durango w 2024 zgłoszono obrażenia?", flag(model("DODGE", "DURANGO").query("year == 2024"), "injured"))
    add("liczba", "Ile skarg z wypadkami miał Jeep Gladiator w 2025?", flag(model("JEEP", "GLADIATOR").query("year == 2025"), "crash"))
    add("liczby", "Czy więcej skarg na hamulce miał w 2025 Ram 1500 czy Ram 2500?",
        [comp("RAM", m, "SERVICE BRAKES", "year == 2025") for m in ("1500", "2500")])
    a, b = (n(model("DODGE", "JOURNEY").query(f"year == {y}")) for y in (2023, 2024))
    add("trend lub liczby", "O ile wzrosła lub spadła liczba skarg na Dodge Journey między 2023 a 2024?", [a, b, round(100 * (b - a) / a, 1)])
    add("liczby", "Jeep Compass vs Jeep Cherokee w 2023 - który miał więcej skarg na układ elektryczny?",
        [comp("JEEP", m, "ELECTRICAL SYSTEM", "year == 2023") for m in ("COMPASS", "CHEROKEE")])
    add("ranking modeli", "Który model Chryslera miał najwięcej skarg w 2023?", top(C[(C.make == "CHRYSLER") & (C.year == 2023)], "model").index[0])
    g = model("DODGE", "GRAND CARAVAN").query("year == 2024 and component_group != 'UNKNOWN OR OTHER'")
    add("ranking komponentów", "Na jaki komponent najczęściej skarżyli się właściciele Dodge Grand Caravan w 2024 (bez kategorii inne)?",
        g.drop_duplicates(["odino", "component_group"]).component_group.value_counts().index[0])
    add("ranking marek", "Która marka miała najwięcej skarg na poduszki powietrzne w 2025?",
        top(C[(C.component_group == "AIR BAGS") & (C.year == 2025)], "make").index[0])
    r = model("JEEP", "GRAND CHEROKEE").query("year == 2025")
    add("procent", "Jaki procent skarg na Jeep Grand Cherokee w 2025 dotyczył układu elektrycznego?",
        round(100 * n(r.query("component_group == 'ELECTRICAL SYSTEM'")) / n(r), 1))
    y = model("JEEP", "COMPASS").groupby("year").odino.nunique()
    add("rok", "W którym roku Jeep Compass miał najwięcej skarg?", [int(y.idxmax()), int(y.max())])
    add("lista akcji", "Jakie akcje serwisowe dotyczyły poduszek powietrznych w Chrysler Pacifica? Podaj numery.",
        sorted(campaigns_of("CHRYSLER", "PACIFICA", "AIR BAGS").campaign))
    add("liczba", "Ile recalli ogłoszono dla Jeep Wrangler w 2025?", int((campaigns_of("JEEP", "WRANGLER").report_date.dt.year == 2025).sum()))
    big = campaigns_of("JEEP", "GRAND CHEROKEE").sort_values("vehicles_affected").iloc[-1]
    add("wieloetapowe", "Który recall Jeepa Grand Cherokee objął najwięcej samochodów?", [big.campaign, int(big.vehicles_affected)])
    old = campaigns_of("RAM", "1500").sort_values("report_date").iloc[0]
    add("akcja", "Jaka była najstarsza akcja serwisowa dla Ram 1500 w danych?", old.campaign)
    steer = K[K.campaign.isin(V[V.component_group == "STEERING"].campaign) & (K.report_date.dt.year == 2025)]
    add("liczba", "Ile akcji serwisowych dotyczyło układu kierowniczego w 2025 roku (wszystkie marki)?", int(steer.campaign.nunique()))
    add("liczba", "Ile pojazdów i jakich modeli obejmuje akcja 25V741000?", int(K[K.campaign == "25V741000"].vehicles_affected.iloc[0]))
    g = model("JEEP", "GLADIATOR").query("year == 2025 and component_group != 'UNKNOWN OR OTHER'")
    c0 = g.drop_duplicates(["odino", "component_group"]).component_group.value_counts().index[0]
    add("komponent i liczba akcji", "Jaki komponent miał najwięcej skarg w Jeep Gladiator w 2025 i ile było na niego akcji serwisowych?",
        [c0, int(campaigns_of("JEEP", "GLADIATOR", c0).campaign.nunique())])
    for text in ["Ile skarg miał Citroen C3 w 2025?",
                 "Który dostawca części odpowiada za najwięcej akcji serwisowych?",
                 "Ile Jeepów Avenger sprzedano w Europie w 2024?",
                 "Ile kosztowała akcja serwisowa 24V436000?"]:
        add("brak danych", text, "BRAK")
    for i, x in enumerate(q):
        x["id"] = f"free3-{i:02d}"
    return q


def expected_free2():
    """Drugi zbiór pytań swobodnych: napisany po poprawkach v10, przed uruchomieniem agenta."""
    q = []
    add = lambda t, text, e: q.append(dict(type=t, q=text, expected=e))
    comp = lambda mk, md, c, cond: n(model(mk, md).query(f"component_group == '{c}' and {cond}"))
    add("liczba", "ile skarg na hamulce mial ram 3500 w 2024", comp("RAM", "3500", "SERVICE BRAKES", "year == 2024"))
    add("liczba", "Ile reklamacji dotyczących skrzyni biegów zgłosili właściciele Jeepa Cherokee w 2023?",
        comp("JEEP", "CHEROKEE", "POWER TRAIN", "year == 2023"))
    add("liczba", "Chrysler Pacifca - ile skarg na elektronikę w 2024?", comp("CHRYSLER", "PACIFICA", "ELECTRICAL SYSTEM", "year == 2024"))
    add("liczba", "How many complaints about air bags did the Jeep Wrangler get in 2023?",
        comp("JEEP", "WRANGLER", "AIR BAGS", "year == 2023"))
    add("liczba", "jakie problemy z silnikiem zglaszali klienci dodge durango w 2025 i ile ich bylo?",
        comp("DODGE", "DURANGO", "ENGINE", "year == 2025"))
    add("liczba", "Ile skarg wpłynęło na Jeep Renegade w drugim kwartale 2024?",
        n(model("JEEP", "RENEGADE").query("year == 2024 and quarter == 2")))
    add("liczba", "Ile skarg na Durango wpłynęło od 2022 do 2024 roku łącznie?", n(model("DODGE", "DURANGO").query("2022 <= year <= 2024")))
    add("liczba", "Ile skarg dotyczyło Ram 1500 z rocznika 2019?", n(model("RAM", "1500").query("model_year == 2019")))
    add("liczba", "Ile skarg na układ kierowniczy dotyczyło Jeepów Grand Cherokee z roczników 2014-2016?",
        comp("JEEP", "GRAND CHEROKEE", "STEERING", "2014 <= model_year <= 2016"))
    add("liczba", "Ile skarg na wszystkie Dodge'e wpłynęło w 2025?", n(C[(C.make == "DODGE") & (C.year == 2025)]))
    add("liczba", "W ilu skargach na Jeep Grand Cherokee w 2025 roku ktoś został ranny?",
        flag(model("JEEP", "GRAND CHEROKEE").query("year == 2025"), "injured"))
    add("liczba", "Ile zgłoszeń o pożarach dotyczyło Ram 1500 w 2024?", flag(model("RAM", "1500").query("year == 2024"), "fire"))
    add("liczba", "Ile skarg z wypadkami miał Chrysler Pacifica w 2023?", flag(model("CHRYSLER", "PACIFICA").query("year == 2023"), "crash"))
    add("liczby", "Co miało więcej skarg w 2024: Dodge Charger czy Dodge Durango? Podaj liczby.",
        [n(model("DODGE", m).query("year == 2024")) for m in ("CHARGER", "DURANGO")])
    add("liczby", "Porównaj skargi na silnik w Jeep Compass w latach 2022 i 2023.",
        [comp("JEEP", "COMPASS", "ENGINE", f"year == {y}") for y in (2022, 2023)])
    a, b = (n(model("JEEP", "CHEROKEE").query(f"year == {y}")) for y in (2024, 2025))
    add("trend lub liczby", "Czy w 2025 skarg na Jeepa Cherokee było mniej niż w 2024?", [a, b, round(100 * (b - a) / a, 1)])
    top = lambda df, col: df.groupby(col).odino.nunique().sort_values(ascending=False)
    add("ranking modeli", "Który model Dodge'a miał najwięcej skarg w 2024?", top(C[(C.make == "DODGE") & (C.year == 2024)], "model").index[0])
    add("ranking modeli", "Który model miał najwięcej skarg na układ kierowniczy w 2025 roku?",
        top(C[(C.component_group == "STEERING") & (C.year == 2025)], "model").index[0])
    add("ranking marek", "Która marka Stellantis miała najwięcej skarg w 2025?", top(C[C.year == 2025], "make").index[0])
    g = model("JEEP", "WRANGLER").query("year == 2025 and component_group != 'UNKNOWN OR OTHER'")
    order = g.drop_duplicates(["odino", "component_group"]).component_group.value_counts()
    add("komponenty", "Wymień 3 najczęściej reklamowane komponenty Jeep Wrangler w 2025 (pomiń inne).", list(order.index[:3]))
    r = model("RAM", "1500").query("year == 2025")
    add("procent", "Jaki odsetek skarg na Ram 1500 w 2025 dotyczył silnika?",
        round(100 * n(r.query("component_group == 'ENGINE'")) / n(r), 1))
    add("miesiąc", "W którym miesiącu 2024 roku Jeep Wrangler miał najmniej skarg?",
        int(model("JEEP", "WRANGLER").query("year == 2024").groupby("month").odino.nunique().idxmin()))
    add("kwartał", "W którym kwartale 2025 Jeep Cherokee miał najwięcej skarg?",
        int(model("JEEP", "CHEROKEE").query("year == 2025").groupby("quarter").odino.nunique().idxmax()))
    add("lista akcji", "Jakie akcje serwisowe dotyczyły hamulców w Jeep Cherokee? Podaj numery.",
        sorted(campaigns_of("JEEP", "CHEROKEE", "SERVICE BRAKES").campaign))
    add("liczba", "Ile akcji serwisowych obejmowało Jeep Compass?", int(campaigns_of("JEEP", "COMPASS").campaign.nunique()))
    big = campaigns_of("RAM", "1500").sort_values("vehicles_affected").iloc[-1]
    add("wieloetapowe", "Jaka akcja serwisowa dla Ram 1500 objęła najwięcej aut i ile?", [big.campaign, int(big.vehicles_affected)])
    last = campaigns_of("JEEP", "GLADIATOR").sort_values("report_date").iloc[-1]
    add("data", "Kiedy była ostatnia akcja serwisowa dla Jeep Gladiator i czego dotyczyła?", last.report_date.date().isoformat())
    add("liczba", "Ile akcji serwisowych Stellantis ogłosił w 2024 roku?", int((K.report_date.dt.year == 2024).sum()))
    add("liczba", "Czego dotyczyła akcja serwisowa 24V720000 i ilu aut dotyczyła?",
        int(K[K.campaign == "24V720000"].vehicles_affected.iloc[0]))
    big = K[K.report_date.dt.year == 2024].sort_values("vehicles_affected").iloc[-1]
    add("wieloetapowe", "Która akcja z 2024 roku objęła najwięcej pojazdów?", [big.campaign, int(big.vehicles_affected)])
    wc = campaigns_of("JEEP", "WRANGLER", order.index[0]).sort_values("report_date")
    add("komponent i akcja", "Jaki komponent Jeepa Wranglera ma najwięcej skarg w 2025 i czy była na niego akcja serwisowa? "
        "Jeśli tak, podaj najnowszą.", [order.index[0], wc.iloc[-1].campaign])
    for text in ["Ile kosztuje wymiana skrzyni biegów w Ram 1500?",
                 "Ile skarg miał Opel Astra w 2024?",
                 "Jaki procent Jeepów Compass sprzedanych w 2024 trafił na naprawę gwarancyjną?",
                 "Ile skarg miał Peugeot 208 w 2025?"]:
        add("brak danych", text, "BRAK")
    for i, x in enumerate(q):
        x["id"] = f"free2-{i:02d}"
    return q


MODEL_NAMES = {"WRANGLER": r"wrangler", "CHEROKEE": r"(?<!grand )cherokee", "GRAND CHEROKEE": r"grand cherokee",
               "COMPASS": r"compass", "1500": r"1500", "CHARGER": r"charger", "RENEGADE": r"renegade",
               "GLADIATOR": r"gladiator", "DURANGO": r"durango", "PACIFICA": r"pacifica"}
MAKE_NAMES = {"JEEP": r"jeep", "RAM": r"\bram\b", "DODGE": r"dodge", "CHRYSLER": r"chrysler", "FIAT": r"fiat",
              "ALFA ROMEO": r"alfa"}
MODEL_NAMES.update({"GRAND CARAVAN": r"grand caravan", "JOURNEY": r"journey", "DURANGO": r"durango"})
QUARTERS = {1: r"q1\b|kwarta\w* i\b|\bi kwarta|pierwsz\w* kwarta|1\. kwarta|kwarta\w* 1\b|styczeń.{0,3}marz|styczen.{0,3}marz",
            2: r"q2\b|kwarta\w* ii\b|\bii kwarta|drug\w* kwarta|2\. kwarta|kwarta\w* 2\b",
            3: r"q3\b|kwarta\w* iii\b|\biii kwarta|trzeci\w* kwarta|3\. kwarta|kwarta\w* 3\b|lipiec.{0,3}wrzes",
            4: r"q4\b|kwarta\w* iv\b|\biv kwarta|czwart\w* kwarta|4\. kwarta|kwarta\w* 4\b"}
MONTHS = ["stycz", "lut", "mar", "kwie", "maj", "czerw", "lip", "sierp", "wrze", "pa[zź]dz", "listop", "grud"]


def first_match(answer, patterns):
    hits = [(m.start(), key) for key, p in patterns.items() for m in [re.search(p, answer.lower())] if m]
    return min(hits)[1] if hits else None


def score(q, answer):
    answer = answer.split("\n\nZakres:")[0]   # linię zakresu dopisuje kod, nie oceniamy jej
    t, e, nums = q["type"], q["expected"], E.numbers(answer)
    if t == "liczba":
        return float(e) in nums
    if t == "liczby":
        return all(float(x) in nums for x in e)
    if t == "trend lub liczby":
        return (float(e[0]) in nums and float(e[1]) in nums) or any(abs(abs(x) - abs(e[2])) <= 0.6 for x in nums)
    if t == "procent":
        return any(abs(x - e) <= 1.0 for x in nums)
    if t == "ranking modeli":
        return first_match(answer, MODEL_NAMES) == e
    if t == "komponenty":
        low = answer.lower()
        return all(any(a in low for a in E.PL[c]) for c in e)
    if t == "miesiąc":
        return first_match(answer, {i + 1: p for i, p in enumerate(MONTHS)}) == e
    if t == "data i liczba":
        y, mth, d = e[0].split("-")
        date_ok = e[0] in answer or f"{d}.{mth}.{y}" in answer or \
            re.search(rf"{int(d)}\s+{MONTHS[int(mth) - 1]}\w*\s+{y}", answer.lower())
        return bool(date_ok) and float(e[1]) in nums
    if t == "ranking marek":
        return first_match(answer, MAKE_NAMES) == e
    if t == "kwartał":
        return first_match(answer, QUARTERS) == e
    if t == "data":
        y, mth, d = e.split("-")
        return bool(e in answer or f"{d}.{mth}.{y}" in answer or
                    re.search(rf"{int(d)}\s+{MONTHS[int(mth) - 1]}\w*\s+{y}", answer.lower()))
    if t == "komponent i akcja":
        return E.first_component(answer) == e[0] and e[1][:6] in answer
    if t == "nie było akcji":
        return bool(re.search(r"nie (było|by[łl]a|ma |znaleziono)|brak|żadn", answer.lower())) and not re.search(r"\d{2}V\d{3}", answer)
    if t == "rok":
        return str(e[0]) in answer and float(e[1]) in nums
    if t == "akcja":
        return e[:6] in answer
    if t == "komponent i liczba akcji":
        return E.first_component(answer) == e[0] and float(e[1]) in nums
    if t in ("lista akcji", "wieloetapowe", "brak danych", "ranking komponentów"):
        return E.score(dict(type=t, expected=e), answer)
    if t == "komponent bez akcji":
        return E.first_component(answer) == e and \
            bool(re.search(r"nie (było|by[łl]a|ma |znaleziono)|brak|żadn", answer.lower())) and not re.search(r"\d{2}V\d{3}", answer)
    raise ValueError(t)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split = args[0] if args else "free"
    qs = {"free": expected, "free2": expected_free2, "free3": expected_free3, "free4": expected_free4}[split]()
    if "--list" in sys.argv:
        for q in qs:
            print(q["id"], q["type"], "|", q["q"], "->", q["expected"])
        print(len(qs), "pytań")
        sys.exit()
    import agent
    if "--rescore" in sys.argv:        # ponowna ocena zapisanych odpowiedzi, bez uruchamiania agenta
        out = Path("results") / f"agent_{split}_{E.VERSION}_{agent.MODEL.replace(':', '-')}.jsonl"
        rows = [json.loads(l) for l in out.open(encoding="utf-8")]
        for r in rows:
            ok = bool(score(r, r["answer"]))
            if ok != r["correct"]:
                print(f"zmiana oceny {r['id']}: {r['correct']} -> {ok} | {r['answer'][:120]}")
            r["correct"] = ok
        out.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")
        print(f"WYNIK po ponownej ocenie ({split}): {sum(r['correct'] for r in rows)}/{len(rows)}")
        sys.exit()
    out = Path("results") / f"agent_{split}_{E.VERSION}_{agent.MODEL.replace(':', '-')}.jsonl"
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    with out.open("a", encoding="utf-8") as f:
        for q in qs:
            if q["id"] in done:
                continue
            t0 = time.time()
            answer, calls = agent.ask(q["q"], verbose=False)
            q.update(answer=answer, calls=calls, seconds=round(time.time() - t0, 1), correct=bool(score(q, answer)))
            f.write(json.dumps(q, ensure_ascii=False, default=str) + "\n")
            f.flush()
            print(f"[{q['id']}] {'OK ' if q['correct'] else 'ŹLE'} ({q['seconds']} s) {q['q']}")
    res = pd.DataFrame([json.loads(l) for l in out.open(encoding="utf-8")])
    print(f"\nWYNIK ({split}, {agent.MODEL}): {res.correct.sum()}/{len(res)} = {100 * res.correct.mean():.0f}%")
