"""
Narzędzia, z których korzysta agent. Każde narzędzie to zwykła funkcja w Pythonie:
liczy coś na danych NHTSA i zwraca krótki wynik tekstowy (JSON), który model językowy czyta.

Zasada: model językowy NIE liczy sam. Wszystkie liczby pochodzą z tych funkcji,
model tylko wybiera narzędzie, podaje parametry i opisuje wynik.
"""

import difflib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path("data/processed")

# Grupy komponentów NHTSA z polskimi nazwami (agent dostaje tę listę w opisie narzędzi).
COMPONENTS_PL = {
    "ELECTRICAL SYSTEM": "układ elektryczny / elektronika",
    "ENGINE": "silnik",
    "POWER TRAIN": "układ napędowy / skrzynia biegów",
    "SERVICE BRAKES": "hamulce",
    "STEERING": "układ kierowniczy",
    "FUEL SYSTEM": "układ paliwowy",
    "AIR BAGS": "poduszki powietrzne",
    "VEHICLE SPEED CONTROL": "tempomat / sterowanie prędkością",
    "SUSPENSION": "zawieszenie",
    "FORWARD COLLISION AVOIDANCE": "system zapobiegania kolizjom",
    "SEATS": "fotele",
    "STRUCTURE": "nadwozie / konstrukcja",
    "EXTERIOR LIGHTING": "oświetlenie zewnętrzne",
    "VISIBILITY": "widoczność / wycieraczki / lusterka",
    "SEAT BELTS": "pasy bezpieczeństwa",
    "WHEELS": "koła",
    "BACK OVER PREVENTION": "kamera cofania / czujniki cofania",
    "ELECTRONIC STABILITY CONTROL (ESC)": "kontrola stabilności ESC",
    "LANE DEPARTURE": "asystent pasa ruchu",
    "TIRES": "opony",
    "LATCHES/LOCKS/LINKAGES": "zamki / klamki",
    "HYBRID PROPULSION SYSTEM": "napęd hybrydowy",
    "PARKING BRAKE": "hamulec postojowy",
    "UNKNOWN OR OTHER": "inne / nieznane",
}
MAKES = ["JEEP", "RAM", "DODGE", "CHRYSLER", "ALFA ROMEO", "FIAT"]

# Początki słów, po których rozpoznajemy, że pytanie dotyczy danego komponentu (po polsku i angielsku).
COMPONENT_WORDS = {
    "ELECTRICAL SYSTEM": ["elektr", "electric", "akumulator", "bateri", "battery"],
    "ENGINE": ["silnik", "engine", "olej"],
    "POWER TRAIN": ["napęd", "skrzyni", "skrzyń", "skrzynia", "przekładni", "transmis", "power train", "powertrain"],
    "SERVICE BRAKES": ["hamul", "hamow", "brake"],
    "STEERING": ["kierown", "kierow", "wspomagani", "steering"],
    "FUEL SYSTEM": ["paliw", "fuel"],
    "AIR BAGS": ["podusz", "airbag", "air bag"],
    "VEHICLE SPEED CONTROL": ["tempomat", "prędkoś", "speed control", "cruise"],
    "SUSPENSION": ["zawiesz", "amortyzator", "suspension"],
    "FORWARD COLLISION AVOIDANCE": ["kolizj", "antykolizyj", "hamowanie awaryjne", "awaryjnego hamowania", "collision"],
    "SEATS": ["fotel", "siedze", "seats"],
    "STRUCTURE": ["nadwozi", "konstrukc", "structure"],
    "EXTERIOR LIGHTING": ["oświetl", "świateł", "światła", "lamp", "reflektor", "kierunkowskaz", "lighting"],
    "VISIBILITY": ["widoczn", "wycieracz", "luster", "szyba", "szyby", "szybie", "szybą", "szyb ", "visibility", "wiper"],
    "SEAT BELTS": ["pas ", "pasy", "pasów", "pasa ", "pasami", "seat belt"],
    "WHEELS": ["koła", "kół ", "kół?", "kół,", "koło", "kołach", "kołami", "felg", "wheel"],
    "BACK OVER PREVENTION": ["cofani", "kamer", "parkowani", "back over"],
    "ELECTRONIC STABILITY CONTROL (ESC)": ["stabilnoś", "trakcj", "esc"],
    "LANE DEPARTURE": ["pasa ruchu", "pasie ruchu", "lane"],
    "TIRES": ["opon", "ogumien", "tire"],
    "LATCHES/LOCKS/LINKAGES": ["zamk", "zamek", "klamk", "drzwi", "latch", "lock"],
    "HYBRID PROPULSION SYSTEM": ["hybryd", "hybrid"],
    "PARKING BRAKE": ["postojow", "ręczn", "parking brake"],
    "UNKNOWN OR OTHER": ["inne", "nieznan"],
}
GROUP_BY = {"component": "component_group", "model": "model", "make": "make",
            "model_year": "model_year", "year": "year", "month": "month"}


def _load():
    c = pd.read_csv(DATA / "complaints.csv", parse_dates=["received_date"])
    c["year"] = c["received_date"].dt.year
    c["month"] = c["received_date"].dt.to_period("M").astype(str)
    k = pd.read_csv(DATA / "campaigns.csv", parse_dates=["report_date"])
    v = pd.read_csv(DATA / "campaign_vehicles.csv")
    p = pd.read_csv(DATA / "campaign_parts.csv")
    return c, k, v, p


COMPLAINTS, CAMPAIGNS, VEHICLES, PARTS = _load()
MODELS = sorted(COMPLAINTS["model"].dropna().unique())


# ---------- dopasowanie nazw (model językowy może napisać "Grand Cherokee", "hamulce", "Ram") ----------

def _norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def match_make(make):
    if not make:
        return None
    m = _norm(make)
    for x in MAKES:
        if _norm(x) == m:
            return x
    raise ValueError(f"Nieznana marka '{make}'. Dostępne: {', '.join(MAKES)}")


def match_model(model, make=None):
    if not model:
        return None
    pool = COMPLAINTS if make is None else COMPLAINTS[COMPLAINTS["make"] == make]
    names = sorted(pool["model"].dropna().unique())
    m = _norm(model)
    for x in names:                      # 1) dokładnie
        if _norm(x) == m:
            return x
    for x in names:                      # 2) bez marki na początku, np. "Jeep Wrangler" -> "WRANGLER"
        for mk in MAKES:
            if m == _norm(mk) + _norm(x):
                return x
    close = difflib.get_close_matches(m, [_norm(x) for x in names], n=1, cutoff=0.8)   # 3) literówka
    if close:
        return next(x for x in names if _norm(x) == close[0])
    raise ValueError(f"Nieznany model '{model}'. Przykłady modeli: {', '.join(names[:30])}")


def match_component(component):
    if not component:
        return None
    m = _norm(component)
    for en, pl in COMPONENTS_PL.items():
        if _norm(en) == m or any(_norm(w) == m for w in pl.split(" / ")):
            return en
    for en, pl in COMPONENTS_PL.items():   # częściowe dopasowanie, np. "brakes", "hamulc"
        if m and (m in _norm(en) or any(m in _norm(w) or _norm(w) in m for w in pl.split(" / "))):
            return en
    raise ValueError(f"Nieznany komponent '{component}'. Dostępne: {', '.join(COMPONENTS_PL)}")


def _filter_complaints(make=None, model=None, model_year=None, component=None, year_from=None, year_to=None,
                       model_year_from=None, model_year_to=None, month=None, quarter=None):
    make = match_make(make)
    model = match_model(model, make)
    component = match_component(component)
    c = COMPLAINTS
    if make:
        c = c[c["make"] == make]
    if model:
        c = c[c["model"] == model]
    if model_year:
        c = c[c["model_year"] == int(model_year)]
    if model_year_from:
        c = c[c["model_year"] >= int(model_year_from)]
    if model_year_to:
        c = c[c["model_year"] <= int(model_year_to)]
    if component:
        c = c[c["component_group"] == component]
    if year_from:
        c = c[c["year"] >= int(year_from)]
    if year_to:
        c = c[c["year"] <= int(year_to)]
    if month:
        c = c[c["received_date"].dt.month == int(month)]
    if quarter:
        c = c[c["received_date"].dt.quarter == int(quarter)]
    used = {k: v for k, v in dict(make=make, model=model, model_year=model_year, model_year_from=model_year_from,
                                  model_year_to=model_year_to, component=component,
                                  year_from=year_from, year_to=year_to, month=month, quarter=quarter).items() if v}
    return c, used


OUTSIDE_NOTE = "Wybrany okres jest poza danymi: skargi i akcje w danych są od stycznia 2020 do października 2026."


def _outside_data(year_from, year_to):
    return bool((year_to and int(year_to) < 2020) or (year_from and int(year_from) > 2026))


# ---------- narzędzia ----------

def count_complaints(make=None, model=None, model_year=None, component=None,
                     year_from=None, year_to=None, group_by=None, top_n=10, include_unknown=False,
                     model_year_from=None, model_year_to=None, month=None, quarter=None):
    """Liczba skarg (unikalnych zgłoszeń) z filtrami, opcjonalnie pogrupowana."""
    note = None
    if group_by == "component" and component:
        note = f"Pominięto filtr component='{component}', bo wynik jest grupowany po komponentach."
        component = None
    c, used = _filter_complaints(make, model, model_year, component, year_from, year_to, model_year_from, model_year_to,
                                 month, quarter)
    out = {"filters": used}
    if note:
        out["note"] = note
    if group_by:
        col = GROUP_BY[group_by]
        total = c["odino"].nunique()
        g = c.drop_duplicates(["odino", col]).groupby(col)["odino"].nunique()
        if group_by == "component" and not include_unknown and "UNKNOWN OR OTHER" in g.index:
            # kategoria "inne/nieznane" nic nie mówi inżynierowi, więc w rankingu ją pomijamy (i mówimy o tym)
            out["skipped_UNKNOWN_OR_OTHER"] = int(g.pop("UNKNOWN OR OTHER"))
        if group_by in ("year", "month", "model_year"):
            out["counts"] = {str(k): int(v) for k, v in g.sort_index().items()}
            if len(g):
                out["highest"] = {"period": str(g.idxmax()), "complaints": int(g.max())}
                out["lowest"] = {"period": str(g.idxmin()), "complaints": int(g.min())}
        else:
            g = g.sort_values(ascending=False).head(int(top_n))
            out["ranking"] = [{"rank": i + 1, "name": str(k), **({"name_pl": COMPONENTS_PL[k]} if k in COMPONENTS_PL else {}),
                               "complaints": int(v), "share_pct": round(100 * v / total, 1) if total else None}
                              for i, (k, v) in enumerate(g.items())]
            out["top"] = out["ranking"][0] if out["ranking"] else None
    if _outside_data(year_from, year_to):
        out["note"] = OUTSIDE_NOTE
    u = c.drop_duplicates("odino")
    out["total_complaints"] = int(u["odino"].nunique())
    out["with_crash"] = int((u["crash"] == "Y").sum())
    out["with_fire"] = int((u["fire"] == "Y").sum())
    out["with_injuries"] = int((u["injured"].fillna(0) > 0).sum())
    return out


def complaint_trend(make=None, model=None, component=None, year_from=2020, year_to=2026, period="year"):
    """Liczba skarg w kolejnych okresach i zmiana procentowa względem poprzedniego okresu."""
    c, used = _filter_complaints(make, model, None, component, year_from, year_to)
    key = {"year": "year", "quarter": None, "month": "month"}[period]
    if period == "quarter":
        c = c.assign(quarter=c["received_date"].dt.to_period("Q").astype(str))
        key = "quarter"
    s = c.drop_duplicates("odino").groupby(key)["odino"].nunique().sort_index()
    rows = []
    prev = None
    for k, v in s.items():
        change = None if not prev else round(100 * (v - prev) / prev, 1)
        rows.append({"period": str(k), "complaints": int(v), "change_pct_vs_previous": change})
        prev = v
    note = OUTSIDE_NOTE if _outside_data(year_from, year_to) else \
        "Dane za rok 2026 obejmują tylko styczeń-październik." if (year_to or 2026) >= 2026 else None
    out = {"filters": used, "series": rows, "note": note}
    if rows:
        out["highest"] = max(rows, key=lambda r: r["complaints"])
        out["lowest"] = min(rows, key=lambda r: r["complaints"])
    return out


def find_campaigns(make=None, model=None, model_year=None, component=None, keyword=None,
                   year_from=None, year_to=None, limit=15, model_year_from=None, model_year_to=None):
    """Akcje serwisowe (recalle) pasujące do filtrów."""
    make = match_make(make)
    model = match_model(model, make)
    component = match_component(component)
    v = VEHICLES
    if make:
        v = v[v["make"] == make]
    if model:
        v = v[v["model"] == model]
    if model_year:
        v = v[v["model_year"] == int(model_year)]
    if model_year_from:
        v = v[v["model_year"] >= int(model_year_from)]
    if model_year_to:
        v = v[v["model_year"] <= int(model_year_to)]
    if component:
        v = v[v["component_group"] == component]
    k = CAMPAIGNS[CAMPAIGNS["campaign"].isin(v["campaign"])]
    if keyword:
        kw = str(keyword).lower()
        hit_parts = PARTS[PARTS["part_name"].astype(str).str.lower().str.contains(kw, regex=False)]["campaign"]
        k = k[k["defect"].str.lower().str.contains(kw, regex=False) | k["campaign"].isin(hit_parts)]
    if year_from:
        k = k[k["report_date"].dt.year >= int(year_from)]
    if year_to:
        k = k[k["report_date"].dt.year <= int(year_to)]
    k = k.sort_values("report_date", ascending=False)
    comps = VEHICLES.groupby("campaign")["component_group"].apply(lambda s: ", ".join(sorted(set(s))))
    row = lambda r: {
        "campaign": r.campaign, "date": r.report_date.date().isoformat(),
        "vehicles_affected": None if pd.isna(r.vehicles_affected) else int(r.vehicles_affected),
        "component": comps.get(r.campaign, ""), "defect": _short_defect(r.defect)}
    rows = [row(r) for r in k.head(int(limit)).itertuples()]
    biggest = k.sort_values("vehicles_affected", ascending=False).head(1)
    oldest = k.sort_values("report_date").head(1)
    used = {key: val for key, val in dict(make=make, model=model, model_year=model_year, model_year_from=model_year_from,
                                         model_year_to=model_year_to, component=component, keyword=keyword,
                                         year_from=year_from, year_to=year_to).items() if val}
    return {"filters": used, **({"note": OUTSIDE_NOTE} if _outside_data(year_from, year_to) else {}),
            "found": int(len(k)), "shown": len(rows), "order": "od najnowszej do najstarszej",
            "newest": rows[0] if rows else None,
            "largest": row(next(biggest.itertuples())) if len(biggest) else None,
            "oldest": row(next(oldest.itertuples())) if len(oldest) else None,
            "campaigns": rows}


def campaign_details(campaign_id):
    """Szczegóły jednej akcji serwisowej: wada, skutek, naprawa, objęte pojazdy, części."""
    cid = str(campaign_id).upper().strip()
    if re.fullmatch(r"\d{2}V\d{3}", cid):
        cid += "000"
    k = CAMPAIGNS[CAMPAIGNS["campaign"] == cid]
    if k.empty:
        raise ValueError(f"Nie ma akcji {campaign_id} w danych (dane obejmują akcje Stellantis od 2020 roku).")
    r = k.iloc[0]
    veh = VEHICLES[VEHICLES["campaign"] == cid]
    covered = [f"{mk} {md} {int(veh[(veh.make == mk) & (veh.model == md)].model_year.min())}-"
               f"{int(veh[(veh.make == mk) & (veh.model == md)].model_year.max())}"
               for mk, md in veh[["make", "model"]].drop_duplicates().itertuples(index=False)
               if veh[(veh.make == mk) & (veh.model == md)].model_year.notna().any()]
    parts = PARTS[PARTS["campaign"] == cid]
    return {
        "campaign": cid, "date": r.report_date.date().isoformat(),
        "vehicles_affected": None if pd.isna(r.vehicles_affected) else int(r.vehicles_affected),
        "components": sorted(set(veh["component_group"])),
        "vehicles_covered": covered,
        "defect": r.defect, "consequence": r.consequence, "remedy": r.remedy,
        "parts": [{"name": p.part_name, "number": p.part_number} for p in parts.head(10).itertuples()],
        "parts_total": int(len(parts)),
    }


_VEC = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
_UNIQUE = COMPLAINTS.drop_duplicates("odino").reset_index(drop=True)
_X = _VEC.fit_transform(_UNIQUE["description"].fillna(""))


def search_complaints(query, make=None, model=None, component=None, year_from=None, year_to=None, top_k=5):
    """Najbardziej podobne skargi do zapytania (po angielsku, bo skargi są po angielsku)."""
    c, used = _filter_complaints(make, model, None, component, year_from, year_to)
    idx = _UNIQUE.index[_UNIQUE["odino"].isin(c["odino"])]
    if len(idx) == 0:
        return {"filters": used, "results": []}
    scores = (_X[idx] @ _VEC.transform([query]).T).toarray().ravel()
    best = scores.argsort()[::-1][: int(top_k)]
    res = []
    for i in best:
        r = _UNIQUE.loc[idx[i]]
        res.append({"odino": int(r.odino), "date": r.received_date.date().isoformat(),
                    "vehicle": f"{int(r.model_year) if pd.notna(r.model_year) else ''} {r.make} {r.model}".strip(),
                    "component": r.component_group, "text": str(r.description)[:300]})
    return {"filters": used, "query": query, "results": res}


def _short_defect(text, n=220):
    t = re.sub(r"^.*?is recalling certain .*?\.\s+(?=[A-Z])", "", str(text))
    return t[:n] + ("..." if len(t) > n else "")


# ---------- opis narzędzi dla modelu (format "function calling") ----------

_COMP_LIST = "; ".join(f"{en} = {pl}" for en, pl in COMPONENTS_PL.items())
_COMMON = {
    "make": {"type": "string", "description": f"Marka: {', '.join(MAKES)}"},
    "model": {"type": "string", "description": "Model bez marki, np. WRANGLER, GRAND CHEROKEE, CHEROKEE, PACIFICA, DURANGO. "
                                                "Dla 'Ram 1500' podaj make=RAM i model=1500 (tak samo 2500, 3500)."},
    "model_year": {"type": "integer", "description": "Rocznik pojazdu, np. 2019"},
    "model_year_from": {"type": "integer", "description": "Od rocznika pojazdu (zakres roczników, np. 2018-2020)"},
    "model_year_to": {"type": "integer", "description": "Do rocznika pojazdu włącznie"},
    "component": {"type": "string", "description": f"Grupa komponentów (podaj nazwę angielską): {_COMP_LIST}"},
    "year_from": {"type": "integer", "description": "Od roku wpłynięcia skargi / ogłoszenia akcji (dane 2020-2026)"},
    "year_to": {"type": "integer", "description": "Do roku włącznie"},
}


def _schema(name, description, props, required=()):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props, "required": list(required)}}}


TOOL_SCHEMAS = [
    _schema("count_complaints",
            "Liczy skargi klientów (NHTSA) z filtrami. Z group_by zwraca ranking/rozkład, np. które komponenty "
            "albo modele mają najwięcej skarg (wtedy NIE podawaj parametru component). W rankingu komponentów "
            "kategoria UNKNOWN OR OTHER (inne) jest pomijana. Pozycje rankingu mają share_pct (procent wszystkich skarg). "
            "Zwraca też, ile skarg dotyczyło wypadku, pożaru, obrażeń.",
            {**_COMMON, "month": {"type": "integer", "description": "Miesiąc wpłynięcia skargi 1-12 (razem z rokiem)"},
             "quarter": {"type": "integer", "description": "Kwartał wpłynięcia skargi 1-4 (razem z rokiem)"},
             "group_by": {"type": "string", "enum": list(GROUP_BY),
                                     "description": "Po czym grupować (opcjonalnie)"},
             "top_n": {"type": "integer", "description": "Ile pozycji rankingu zwrócić (domyślnie 10)"},
             "include_unknown": {"type": "boolean", "description": "Czy w rankingu komponentów uwzględnić kategorię UNKNOWN OR OTHER (domyślnie nie)"}}),
    _schema("complaint_trend",
            "Trend liczby skarg w czasie (rok/kwartał/miesiąc) ze zmianą procentową względem poprzedniego okresu. "
            "Pola 'highest' i 'lowest' to okres z największą i najmniejszą liczbą skarg.",
            {k: _COMMON[k] for k in ["make", "model", "component", "year_from", "year_to"]} |
            {"period": {"type": "string", "enum": ["year", "quarter", "month"]}}),
    _schema("find_campaigns",
            "Wyszukuje akcje serwisowe (recalle) Stellantis od 2020 r. po modelu, roczniku, komponencie "
            "lub słowie kluczowym (po angielsku, szukane w opisie wady i nazwach części, np. 'fuel pump'). "
            "Wyniki są posortowane od najnowszej; pole 'newest' to najnowsza akcja, 'largest' to akcja obejmująca "
            "najwięcej pojazdów, 'oldest' najstarsza, 'found' to liczba wszystkich pasujących akcji. Rok (year_from/year_to) to rok ogłoszenia akcji.",
            {**_COMMON, "keyword": {"type": "string", "description": "Słowo kluczowe po angielsku"},
             "limit": {"type": "integer"}}),
    _schema("campaign_details",
            "Szczegóły jednej akcji serwisowej: opis wady, skutek, naprawa, objęte pojazdy, liczba pojazdów, części.",
            {"campaign_id": {"type": "string", "description": "Numer akcji, np. 24V896000"}}, ["campaign_id"]),
    _schema("search_complaints",
            "Wyszukuje treść skarg najbardziej podobnych do zapytania (zapytanie po angielsku), "
            "żeby zobaczyć, co klienci piszą o problemie.",
            {"query": {"type": "string", "description": "Opis problemu po angielsku"},
             **{k: _COMMON[k] for k in ["make", "model", "component", "year_from", "year_to"]},
             "top_k": {"type": "integer"}}, ["query"]),
]
TOOLS = {"count_complaints": count_complaints, "complaint_trend": complaint_trend, "find_campaigns": find_campaigns,
         "campaign_details": campaign_details, "search_complaints": search_complaints}


def run_tool(name, args):
    """Wywołuje narzędzie i zwraca wynik jako tekst JSON. Błędy też zwraca jako tekst, żeby model mógł się poprawić."""
    try:
        if name not in TOOLS:
            raise ValueError(f"Nie ma narzędzia {name}")
        args = {k: v for k, v in (args or {}).items() if v not in (None, "", "null")}
        return json.dumps(TOOLS[name](**args), ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
