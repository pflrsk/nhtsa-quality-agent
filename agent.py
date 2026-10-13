"""
Agent analityczny jakości: lokalny model językowy (Ollama) + narzędzia z tools.py.

Jak działa (pętla agenta):
  1. Model dostaje pytanie i listę narzędzi.
  2. Model decyduje, które narzędzie wywołać i z jakimi parametrami.
  3. Program wywołuje narzędzie (zwykła funkcja w Pythonie) i oddaje wynik modelowi.
  4. Kroki 2-3 powtarzają się, aż model ma wszystko, i wtedy pisze odpowiedź.

Uruchomienie:  python agent.py "Które komponenty Jeepa Cherokee miały najwięcej skarg w 2025?"
Wymaga uruchomionej Ollamy z modelem (domyślnie qwen3:8b).
"""

import difflib
import json
import os
import re
import sys
import time
import urllib.request

import tools

MODEL = os.environ.get("AGENT_MODEL", "qwen3:8b")
URL = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
MAX_STEPS = 8

SYSTEM = """Jesteś asystentem inżyniera jakości w koncernie Stellantis. Odpowiadasz po polsku.
Masz dostęp do publicznych danych NHTSA (rynek USA) o markach Stellantis sprzedawanych w USA (Jeep, Ram, Dodge, Chrysler,
Fiat, Alfa Romeo). Inne marki Stellantis, np. Peugeot, Citroën, Opel, DS, Lancia, też należą do koncernu, ale nie ma ich
w tych danych. Dane obejmują:
- skargi klientów na pojazdy, wpływające od stycznia 2020 do października 2026,
- akcje serwisowe (recalle) ogłoszone od 2020 roku, z opisem wady, naprawy i części.

Zasady:
1. Każdą liczbę bierz z narzędzi. Nigdy nie zgaduj i nie licz z pamięci.
2. Jeśli pytanie wymaga kilku informacji, wywołaj kilka narzędzi po kolei.
3. Nazwy komponentów podawaj narzędziom po angielsku (lista jest w opisie narzędzi), w odpowiedzi możesz użyć polskiej nazwy.
4. Jeśli narzędzie zwróci błąd (np. nieznana nazwa modelu), popraw parametry i spróbuj ponownie.
5. Jeśli dane nie pozwalają odpowiedzieć (np. pytanie o koszty napraw, ceny, gwarancję, sprzedaż, produkcję,
   fabryki, dostawców części, wyniki finansowe, inne marki), powiedz wprost, że tych informacji nie ma w danych.
   Nie wymyślaj. Uwaga: "Chrysler (FCA US, LLC)" w opisie akcji to producent ogłaszający akcję, a nie dostawca części.
6. Odpowiedź ma być krótka i konkretna: najpierw liczby i fakty, potem jedno zdanie komentarza.
   Podawaj numery akcji serwisowych, jeśli o nich mowa.
7. Wynik każdego narzędzia zawiera pole "filters". Sprawdź, czy są w nim wszystkie warunki z pytania
   (marka, model, rok, komponent). Jeśli czegoś brakuje, wywołaj narzędzie ponownie z poprawnymi parametrami.
8. W rankingu odpowiedzią jest pozycja "top" (rank 1).
9. Pytania o akcje serwisowe (recalle) - find_campaigns (liczba akcji to pole "found"); pytania o skargi - count_complaints.
10. Jeśli modelu z pytania nie ma w danych, napisz to. Nie podawaj zamiast niego danych innego modelu."""


def chat(messages, think=False):
    body = {"model": MODEL, "messages": messages, "tools": tools.TOOL_SCHEMAS, "stream": False,
            "think": think, "options": {"temperature": 0, "num_ctx": 12288}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]


def ungrounded_numbers(answer, question, calls):
    """Liczby z odpowiedzi, których nie ma ani w wynikach narzędzi, ani w pytaniu (możliwe zmyślenie).

    To prosty, deterministyczny bezpiecznik: model może pomylić się przy przepisywaniu liczb,
    więc sprawdzamy, czy każda liczba w odpowiedzi ma pokrycie w danych."""
    def nums(text):
        t = re.sub(r"(?<![\d.,])\d{1,3}(?:[ \u00a0\u202f.,]\d{3})+(?![\d]|[.,]\d)",
                   lambda m: re.sub(r"\D", "", m.group()), text)                     # 1 234 / 2.135 -> 1234 / 2135
        return {x.replace(",", ".").rstrip("0").rstrip(".") for x in re.findall(r"\d+(?:[.,]\d+)?", t)}
    known = nums(question) | set().union(*(nums(c["result"]) for c in calls)) if calls else nums(question)
    return sorted(n for n in nums(answer) if n not in known and not (n.isdigit() and int(n) < 10))


PL_CHARS = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
SUFFIX = r"(?:A|U|EM|OWI|ACH|AMI|OM|Y|OW|IE)?"   # odmiana nazw: Jeepa, Wranglera, Chryslerach


def plain(text):
    """Małe litery bez polskich znaków - użytkownicy często piszą 'swiatla' zamiast 'światła'."""
    return str(text).lower().translate(PL_CHARS)


def _model_patterns():
    """Wzorce nazw modeli: 'MARKA MODEL' zawsze, sam 'MODEL' tylko gdy nie jest liczbą ani nazwą marki.
    Dodatkowo skrócona nazwa bez 'PLUG-IN' (np. 'Pacifica Hybrid')."""
    counts = tools.COMPLAINTS.groupby(["make", "model"])["odino"].nunique()
    pats = []
    for (make, model), n in counts.items():
        if n < 20 or model in tools.MAKES:
            continue
        for name in {model, model.replace("PLUG-IN ", "")}:
            pats.append((re.escape(make) + SUFFIX + r"\s+" + re.escape(name), len(make) + 1 + len(name), make, model, n))
            if not name[0].isdigit() and name.split()[0] not in tools.MAKES:
                pats.append((re.escape(name), len(name), make, model, n))
    return sorted(pats, key=lambda x: (-x[1], -x[4]))      # najpierw dłuższe, potem popularniejsze


_PATTERNS = _model_patterns()
_WORDS = {c: [plain(w) for w in words] for c, words in tools.COMPONENT_WORDS.items()}


def entities_in_question(question):
    """Modele aut i komponenty wymienione w pytaniu.

    Zwraca (lista (marka, model), zbiór komponentów). Nazwy modeli wycinamy z tekstu, zanim szukamy
    komponentów, żeby 'Pacifica Plug-in Hybrid' nie liczyła się jako pytanie o napęd hybrydowy."""
    q = " " + plain(question).upper() + " "
    found = []
    for pattern, _, make, model, _ in _PATTERNS:
        hit = re.search(r"(?<![A-Z0-9])" + pattern + SUFFIX + r"(?![A-Z0-9])", q)
        if hit:
            before = re.search(_MAKE_BEFORE, q[:hit.start()])
            if before and before.group(1) != make:        # np. 'Jeep Avenger' - model innej marki, nie przypinamy
                q = q[:hit.start()] + " " * (hit.end() - hit.start()) + q[hit.end():]
                continue
            if (make, model) not in found:
                found.append((make, model))
            q = q[:hit.start()] + " " * (hit.end() - hit.start()) + q[hit.end():]   # zużyty fragment
    low = q.lower()
    comps = {c for c, words in _WORDS.items() if any(re.search(r"(?<![a-z])" + re.escape(w), low) for w in words)}
    return found, comps


_MAKE_BEFORE = r"(" + "|".join(re.escape(m) for m in tools.MAKES) + r")" + SUFFIX + r"\s+$"


def mismatched_models(question):
    """Pary (marka z pytania, model), gdy pytanie łączy model z inną marką niż w danych, np. 'Jeep Avenger'
    (w danych jest tylko Dodge Avenger). Takiego modelu nie podmieniamy na model innej marki."""
    q = " " + plain(question).upper() + " "
    out = []
    for pattern, _, make, model, _ in _PATTERNS:
        hit = re.search(r"(?<![A-Z0-9])" + pattern + SUFFIX + r"(?![A-Z0-9])", q)
        if hit:
            before = re.search(_MAKE_BEFORE, q[:hit.start()])
            if before and before.group(1) != make and not any(m == model and mk == before.group(1) for mk, m in _KNOWN):
                out.append((before.group(1), model))
    return out


_KNOWN = set(tools.COMPLAINTS[["make", "model"]].drop_duplicates().itertuples(index=False, name=None))


def models_in_question(question):
    return [m for _, m in entities_in_question(question)[0]]


def missing_campaigns(answer, calls):
    """Akcje z ostatniego wyniku find_campaigns, których brakuje w odpowiedzi.

    Sprawdzane tylko wtedy, gdy odpowiedź wymienia co najmniej dwie akcje z wyniku (czyli podaje listę).
    Przy pytaniu o jedną akcję, np. najnowszą, nie trzeba wymieniać pozostałych."""
    last = next((c for c in reversed(calls) if c["tool"] == "find_campaigns"), None)
    if last is None:
        return []
    try:
        listed = json.loads(last["result"]).get("campaigns", [])
    except ValueError:
        return []
    named = {x[:6] for x in re.findall(r"\d{2}V\d{3}", answer)}
    if sum(c["campaign"][:6] in named for c in listed) < 2:
        return []
    return [c for c in listed if c["campaign"][:6] not in named]


MODEL_YEAR_WORDS = r"rocznik|modelow|produkcj|wyprodukowan|model year"
NOT_SINGLE_YEAR = r"\b(od|do|po|przed|since|until|between|miedzy)\b|wczesniej|poprzedni|porown|zmian|trend|wzros|spad|kolejn"
SHARE_WORDS = r"procent|udzial|odsetek|%"


def asks_about_model_year(question, models):
    """Czy pytanie dotyczy rocznika: pada słowo 'rocznik' (lub podobne) albo rok sprzed 2020 stoi tuż po nazwie
    modelu ('Compass 2017'). 'W 2025 roku' to rok wpłynięcia skargi, a nie rocznik; skargi w danych są od 2020,
    więc '... Charger 2025' traktujemy jako rok skargi."""
    if re.search(MODEL_YEAR_WORDS, plain(question)):
        return True
    q = plain(question).upper()
    return any(re.search(re.escape(model) + SUFFIX + r"\s*\(?(?:19\d{2}|201\d)\b", q) for _, model in models)


def model_year_range(question):
    """Rocznik albo zakres roczników podany przy słowie 'rocznik': (od, do) albo None."""
    p = plain(question)
    m = re.search(r"rocznik\w*\s+((?:19|20)\d{2})\s*(?:-|–|do|i)\s*((?:19|20)\d{2})", p)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"rocznik\w*\s+((?:19|20)\d{2})\b", p) or re.search(r"\b((?:19|20)\d{2})\s+rocznik", p)
    return (int(m.group(1)), int(m.group(1))) if m else None


def single_year(question):
    """Jedyny rok (2010-2029) w pytaniu, jeśli pytanie nie dotyczy zakresu ani porównania lat."""
    years = set(re.findall(r"(?<!\d)(20[12]\d)(?!\d)", question))
    if len(years) != 1 or re.search(NOT_SINGLE_YEAR, plain(question)):
        return None
    return int(years.pop())


MONTHS_IN = ["styczniu", "lutym", "marcu", "kwietniu", "maju", "czerwcu", "lipcu", "sierpniu", "wrzesniu",
             "pazdzierniku", "listopadzie", "grudniu"]
MONTHS_OF = ["stycznia|styczen", "lutego|luty", "marca|marzec", "kwietnia|kwiecien", "maja|maj", "czerwca|czerwiec",
             "lipca|lipiec", "sierpnia|sierpien", "wrzesnia|wrzesien", "pazdziernika|pazdziernik", "listopada|listopad",
             "grudnia|grudzien"]
QUARTER_WORDS = [r"pierwsz\w*|\bi|1", r"drug\w*|\bii|2", r"trzeci\w*|\biii|3", r"czwart\w*|\biv|4"]


def month_in_question(question):
    """Miesiąc (1-12), jeśli pytanie podaje jeden konkretny miesiąc: 'w październiku', 'marca 2025'."""
    p = plain(question)
    found = {i + 1 for i, w in enumerate(MONTHS_IN) if re.search(rf"\b{w}\b", p)}
    found |= {i + 1 for i, w in enumerate(MONTHS_OF) if re.search(rf"\b(?:{w})\s+(?:roku\s+)?20\d\d", p)}
    return found.pop() if len(found) == 1 else None


def quarter_in_question(question):
    """Kwartał (1-4), jeśli pytanie podaje jeden konkretny kwartał: 'w drugim kwartale', 'Q3'."""
    p = plain(question)
    found = {i + 1 for i, w in enumerate(QUARTER_WORDS)
             if re.search(rf"(?:{w})\.?\s+kwarta|kwarta\w*\s+(?:{w})\b|\bq{i + 1}\b", p)}
    return found.pop() if len(found) == 1 else None


TOOLS_WITH_MODEL = {"count_complaints", "complaint_trend", "find_campaigns", "search_complaints"}


def pin_arguments(question, name, args):
    """Kod pilnuje tego, co musi być dokładne:
    - jeśli pytanie wymienia dokładnie jeden model, to ten model (i marka) trafia do narzędzia,
      nawet jeśli model językowy wpisał inny albo żaden (np. 'PACIFICA' zamiast 'PACIFICA PLUG-IN HYBRID'),
    - model, którego marka w pytaniu nie zgadza się z danymi ('Jeep Avenger'), nie jest podmieniany na inny,
    - filtr komponentu zostaje tylko wtedy, gdy pytanie mówi o komponencie, a gdy mówi o dokładnie jednym, kod go dodaje,
    - filtr rocznika zostaje tylko wtedy, gdy pytanie mówi o roczniku; rocznik i zakres roczników bierze kod,
    - rok, miesiąc i kwartał wpłynięcia skargi kod bierze z pytania, jeśli pytanie podaje jeden konkretny okres.
    Zwraca (poprawione argumenty, lista poprawek do zapisania w logu)."""
    if name not in TOOLS_WITH_MODEL:
        return args, []
    args, fixes = dict(args), []
    models, comps = entities_in_question(question)
    for make, model in mismatched_models(question):
        try:
            same = tools.match_model(args.get("model")) == model if args.get("model") else False
        except ValueError:
            same = False
        if same and str(args.get("make", "")).upper() != make:
            fixes.append(f"marka z pytania: {args.get('make')} {model} -> {make} {model}")
            args["make"] = make
    if len(models) == 1:
        make, model = models[0]
        try:
            given = tools.match_model(args.get("model"), tools.match_make(args.get("make"))) if args.get("model") else None
        except ValueError:
            given = None
        if given != model or (args.get("make") and str(args["make"]).upper() != make):
            fixes.append(f"model: {args.get('make')} {args.get('model')} -> {make} {model}")
            args["make"], args["model"] = make, model
    real = comps - {"UNKNOWN OR OTHER"}
    if args.get("component") and not comps:
        fixes.append(f"usunięto filtr component={args['component']} (pytanie nie dotyczy komponentu)")
        args.pop("component")
    elif (not args.get("component") and len(real) == 1 and args.get("group_by") != "component"
          and name != "search_complaints" and not re.search(SHARE_WORDS, plain(question))):
        args["component"] = next(iter(real))
        fixes.append(f"dodano filtr component={args['component']} (pytanie dotyczy komponentu)")
    asks_year = asks_about_model_year(question, models)
    if not asks_year:
        for key in ("model_year", "model_year_from", "model_year_to"):
            if args.get(key) is not None:
                fixes.append(f"usunięto filtr {key}={args[key]} (pytanie nie dotyczy rocznika)")
                args.pop(key)
    rng = model_year_range(question)
    if rng and name in ("count_complaints", "find_campaigns"):
        lo, hi = rng
        want = {"model_year": lo} if lo == hi else {"model_year_from": lo, "model_year_to": hi}
        have = {k: args.get(k) for k in ("model_year", "model_year_from", "model_year_to") if args.get(k) is not None}
        if have != want:
            for k in have:
                args.pop(k)
            if (args.get("year_from"), args.get("year_to")) == (lo, hi):   # model wpisał rocznik jako rok skargi
                args.pop("year_from"), args.pop("year_to")
            args.update(want)
            fixes.append(f"rocznik z pytania: {want}")
    year = single_year(question)
    if year and not asks_year and (name in ("count_complaints", "complaint_trend")
                                   or (name == "find_campaigns" and "skarg" not in plain(question))):
        if (args.get("year_from"), args.get("year_to")) != (year, year):
            fixes.append(f"rok z pytania: {args.get('year_from')}-{args.get('year_to')} -> {year}")
            args["year_from"] = args["year_to"] = year
    if year and name == "count_complaints" and args.get("group_by") not in ("month", "year"):
        for key, value in (("month", month_in_question(question)), ("quarter", quarter_in_question(question))):
            if value and args.get(key) != value:
                fixes.append(f"{key} z pytania: {value}")
                args[key] = value
    return args, fixes


THIS_COMPONENT = r"\bten komponent|\bna niego\b|\bna ten\b|\btego komponentu|\bthis component"


def carry_component(question, name, args, calls):
    """W pytaniu wieloetapowym ('który komponent... i jaka akcja na ten komponent') komponent z rankingu
    w poprzednim kroku trafia do wyszukiwania akcji, jeśli model go pominął."""
    if name != "find_campaigns" or args.get("component") or not re.search(THIS_COMPONENT, plain(question)):
        return args, []
    for c in reversed(calls):
        if c["tool"] == "count_complaints" and c["args"].get("group_by") == "component":
            top = (json.loads(c["result"]).get("top") or {}).get("name")
            if top:
                return {**args, "component": top}, [f"komponent z poprzedniego kroku: {top}"]
    return args, []


def wrong_tool(question, calls):
    """Pytanie o akcje serwisowe, a agent użył tylko narzędzi do skarg."""
    p = plain(question)
    return bool(calls) and bool(re.search(r"akcj|recall|kampani", p)) and "skarg" not in p and \
        not any(c["tool"] in ("find_campaigns", "campaign_details") for c in calls)


def substituted_model(question, calls):
    """Nazwa modelu, którego nie ma w danych, jeśli agent podał zamiast niego dane innego modelu."""
    rejected = [m.group(1) for c in calls for m in [re.search(r"Nieznany model '([^']+)'", c["result"])] if m]
    if not rejected:
        return None
    asked, q = set(models_in_question(question)), plain(question).upper()
    for c in calls:
        if not c["args"].get("model") or '"error"' in c["result"][:20]:
            continue
        try:
            used = tools.match_model(c["args"]["model"], tools.match_make(c["args"].get("make")))
        except ValueError:
            continue
        similar = any(difflib.SequenceMatcher(None, tools._norm(r), tools._norm(used)).ratio() >= 0.6 for r in rejected)
        if used not in asked and used not in q and not similar:
            return rejected[0]
    return None


def describe_scope(calls):
    """Opis zakresu danych, z którego powstała odpowiedź, zbudowany z pola 'filters' wyników narzędzi.
    Pisze go kod, nie model, więc użytkownik widzi, jak pytanie zostało zrozumiane."""
    parts = []
    for c in calls:
        try:
            f = json.loads(c["result"]).get("filters")
        except (ValueError, AttributeError):
            continue
        if f is None:
            continue
        car = " ".join(x if x[0].isdigit() else x.title() for x in (f.get("make"), f.get("model")) if x)
        bits = [car or "wszystkie marki i modele"]
        if f.get("component"):
            bits.append(tools.COMPONENTS_PL.get(f["component"], f["component"]))
        if f.get("model_year"):
            bits.append(f"rocznik {f['model_year']}")
        elif f.get("model_year_from") or f.get("model_year_to"):
            bits.append(f"roczniki {f.get('model_year_from', '')}-{f.get('model_year_to', '')}")
        yf, yt = f.get("year_from"), f.get("year_to")
        years = None if not (yf or yt) else str(yf) if yf == yt else f"{yf or 2020}-{yt or 2026}"
        if years and f.get("month"):
            years = f"{f['month']:02d}.{years}"
        elif years and f.get("quarter"):
            years = f"Q{f['quarter']} {years}"
        if c["tool"] == "find_campaigns":
            bits.append(f"akcje ogłoszone w {years}" if years else "akcje od 2020")
        else:
            bits.append(f"skargi z {years}" if years else "skargi z lat 2020-2026")
        if f.get("keyword"):
            bits.append(f"słowo kluczowe '{f['keyword']}'")
        text = ", ".join(bits)
        if text not in parts:
            parts.append(text)
    return "Zakres: " + "; ".join(parts) if parts else ""


def calls_for_missing_models(question, calls, lost):
    """Wywołania tego samego narzędzia co ostatnie, ale osobno dla każdego modelu z pytania, którego
    agent nie użył (np. przy porównaniu dwóch modeli). Wykonuje je kod, nie model językowy."""
    base = next((c for c in reversed(calls) if c["tool"] in TOOLS_WITH_MODEL and '"error"' not in c["result"][:20]), None)
    if base is None:
        return []
    makes = dict((m, mk) for mk, m in entities_in_question(question)[0])
    new = []
    for model in lost:
        args = {k: v for k, v in base["args"].items() if not (k == "group_by" and v in ("model", "make"))}
        args.update(make=makes[model], model=model)
        if args.get("group_by") is None:
            args.pop("top_n", None)
        args, fixes = pin_arguments(question, base["tool"], args)
        new.append({"tool": base["tool"], "args": args, "result": tools.run_tool(base["tool"], args),
                    "fixes": fixes + [f"wywołanie dodane przez kod dla modelu {model}"]})
    return new


def missing_models(question, calls):
    """Modele z pytania, których żadne wywołanie narzędzia nie użyło jako filtra."""
    used = set()
    for c in calls:
        if c["args"].get("model"):
            try:
                used.add(tools.match_model(c["args"]["model"], tools.match_make(c["args"].get("make"))))
            except ValueError:
                pass
    uses_model_param = any(c["tool"] != "campaign_details" for c in calls)
    return [m for m in models_in_question(question) if m not in used] if uses_model_param else []


def ask(question, think=False, verbose=True):
    """Zadaje pytanie agentowi. Zwraca (odpowiedź, lista wywołań narzędzi)."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    calls = []
    checked = False
    model_checked = False
    list_checked = False
    tool_checked = False
    swap_checked = False
    empty_checked = False
    for _ in range(MAX_STEPS):
        msg = chat(messages, think)
        messages.append({"role": "assistant", "content": msg.get("content", ""),
                         "tool_calls": msg.get("tool_calls", [])})
        if not msg.get("tool_calls"):
            answer = msg.get("content", "").strip()
            swapped = substituted_model(question, calls)
            if swapped and not swap_checked:
                swap_checked = True  # jedna szansa na poprawkę
                if verbose:
                    print(f"  ! modelu {swapped} nie ma w danych, a odpowiedź dotyczy innego modelu - prosimy o poprawkę")
                messages.append({"role": "user", "content":
                                 f"Modelu {swapped} nie ma w danych. Nie podawaj danych innego modelu - "
                                 f"odpowiedz, że w danych nie ma tego modelu."})
                continue
            if swapped:
                return f"W danych nie ma modelu {swapped}.", calls
            if wrong_tool(question, calls) and not tool_checked:
                tool_checked = True  # jedna szansa na poprawkę
                if verbose:
                    print("  ! pytanie o akcje serwisowe, a użyto tylko narzędzi do skarg - prosimy o poprawkę")
                messages.append({"role": "user", "content":
                                 "Pytanie dotyczy akcji serwisowych. Użyj narzędzia find_campaigns "
                                 "(liczba akcji to pole 'found') i popraw odpowiedź."})
                continue
            lost = missing_models(question, calls)
            if lost and not model_checked:
                model_checked = True  # jedna szansa na poprawkę
                extra = calls_for_missing_models(question, calls, lost)
                if verbose:
                    print(f"  ! narzędzia wywołano bez modelu z pytania: {lost} - kod wywołuje je dla każdego modelu")
                if extra:
                    calls.extend(extra)
                    messages.append({"role": "assistant", "content": "", "tool_calls": [
                        {"function": {"name": c["tool"], "arguments": c["args"]}} for c in extra]})
                    for c in extra:
                        messages.append({"role": "tool", "tool_name": c["tool"], "content": c["result"]})
                    messages.append({"role": "user", "content":
                                     f"Powyżej są wyniki osobno dla modeli: {', '.join(lost)}. "
                                     f"Odpowiedz na pytanie na ich podstawie."})
                else:
                    messages.append({"role": "user", "content":
                                     f"Pytanie dotyczy modelu {', '.join(lost)}, a narzędzia wywołano bez filtra model. "
                                     f"Wywołaj narzędzie ponownie z parametrem model i popraw odpowiedź."})
                continue
            if not answer and calls:
                if not empty_checked:
                    empty_checked = True  # jedna szansa na poprawkę
                    messages.append({"role": "user", "content":
                                     "Odpowiedź jest pusta. Odpowiedz na pytanie na podstawie wyników narzędzi."})
                    continue
                answer = "Nie udało się sformułować odpowiedzi na podstawie wyników narzędzi."
            bad = ungrounded_numbers(answer, question, calls)
            if bad and not checked:
                checked = True  # jedna szansa na poprawkę
                if verbose:
                    print(f"  ! liczby bez pokrycia w danych: {bad} - prosimy model o poprawkę")
                messages.append({"role": "user", "content":
                                 f"Liczby {', '.join(bad)} nie występują w wynikach narzędzi. Popraw odpowiedź, "
                                 f"używając wyłącznie liczb z wyników narzędzi."})
                continue
            lacking = missing_campaigns(answer, calls)
            if lacking and not list_checked:
                list_checked = True  # jedna szansa na poprawkę
                ids = [c["campaign"] for c in lacking]
                if verbose:
                    print(f"  ! w odpowiedzi brakuje akcji z wyniku narzędzia: {ids} - prosimy model o poprawkę")
                messages.append({"role": "user", "content":
                                 f"W odpowiedzi brakuje akcji {', '.join(ids)} z wyniku narzędzia. "
                                 f"Wymień wszystkie akcje z wyniku i popraw odpowiedź."})
                continue
            if lacking:  # druga odpowiedź też niepełna: brakujące akcje dopisuje kod
                answer += "\n\nPozostałe akcje z wyniku narzędzia (dopisane przez kod): " + "; ".join(
                    f"{c['campaign']} ({c['date']}, {c['component']})" for c in lacking) + "."
            scope = describe_scope(calls)
            return (answer + "\n\n" + scope if scope else answer), calls
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            args = tc["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            args, fixes = pin_arguments(question, name, args)
            args, carried = carry_component(question, name, args, calls)
            fixes += carried
            result = tools.run_tool(name, args)
            calls.append({"tool": name, "args": args, "result": result, "fixes": fixes})
            if verbose:
                print(f"  -> {name}({json.dumps(args, ensure_ascii=False)})" + (f"  [poprawki kodu: {'; '.join(fixes)}]" if fixes else ""))
            messages.append({"role": "tool", "tool_name": name, "content": result})
    return "Nie udało się ustalić odpowiedzi w dozwolonej liczbie kroków.", calls


if __name__ == "__main__":
    q = " ".join(a for a in sys.argv[1:] if a != "--think") or "Które komponenty Jeepa Cherokee miały najwięcej skarg w 2025 roku?"
    t0 = time.time()
    answer, _ = ask(q, think="--think" in sys.argv)
    print(f"\n{answer}\n\n({time.time() - t0:.1f} s, model {MODEL})")
