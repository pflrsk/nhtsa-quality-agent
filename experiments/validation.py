"""
Walidacja metody alarmów na prawdziwych akcjach serwisowych Stellantis (backtest).

Pytania:
  1. Czułość: przed jakim procentem akcji serwisowych pojawił się alarm (w oknie 26 tygodni)?
  2. Wyprzedzenie: ile tygodni przed akcją był pierwszy alarm?
  3. Trafność: jaki procent nowych alarmów skończył się akcją serwisową w ciągu 26 tygodni?

Zasady uczciwego testu:
  - narzędzie obserwuje cały model (marka + model + komponent, wszystkie roczniki), bo przed
    akcją nie wiadomo, których roczników dotyczy problem (inaczej: wyciek informacji z przyszłości),
  - oceniamy tylko akcje, przed którymi mamy co najmniej rok skarg,
  - alarm do 26 tygodni po akcji na ten sam model i komponent to skutek akcji (właściciele dostają
    listy i zgłaszają usterki), a nie fałszywy alarm, więc liczymy go osobno,
  - podział w czasie: ustawienia dobieramy na latach 2021-2023 (STROJENIE), a wynik podajemy
    na latach 2024-2026 (TEST), których przy dobieraniu nie oglądamy. Tak działa narzędzie
    w praktyce: ustawione na przeszłości, pracuje na przyszłości.

Uruchomienie: python validation.py  (zamknij wcześniej pliki CSV w Excelu)
"""

import pandas as pd
import analysis

WINDOW = pd.Timedelta(weeks=analysis.LEAD_WINDOW_WEEKS)
NEW_EPISODE_GAP_WEEKS = 8   # alarm po tylu tygodniach spokoju liczymy jako nowy epizod
SIGNAL_MIN = 20             # akcja "z sygnałem" = co najmniej tyle skarg na objęte roczniki w oknie przed nią
SERIES_COLS = ["make", "model", "component_group"]   # jeden szereg = jeden model i komponent

TUNE = (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31"))
TEST = (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31"))

# Sprawdzane ustawienia: okno (1 tydzień / suma 4 tyg.), potwierdzenie, minimalna liczba skarg.
VARIANTS = [
    {"window": w, "confirm": c, "min": m}
    for w in [1, 4] for c in [1, 2] for m in [5, 10]
]


def load():
    complaints = analysis.load_complaints()
    campaigns = pd.read_csv("data/processed/campaigns.csv", parse_dates=["report_date"])
    vehicles = pd.read_csv("data/processed/campaign_vehicles.csv").merge(
        campaigns[["campaign", "report_date"]], on="campaign"
    )
    return complaints, campaigns, vehicles


def signal_strength(complaints, vehicles):
    """Ile skarg na dokładnie objęte modele i roczniki wpłynęło w oknie przed każdą akcją.

    Używane tylko do opisu wyników (czy akcja w ogóle była poprzedzona skargami),
    nigdy do samego wykrywania.
    """
    c = complaints.drop_duplicates(["odino", "component_group"])
    key = ["make", "model", "model_year", "component_group"]
    m = c.merge(vehicles[key + ["campaign", "report_date"]], on=key)
    before = m[(m["received_date"] < m["report_date"]) & (m["received_date"] >= m["report_date"] - WINDOW)]
    return before.groupby("campaign")["odino"].nunique()


def all_weekly_series(complaints):
    """Tygodniowa liczba skarg dla każdego modelu i komponentu (tygodnie bez skarg = 0)."""
    start, end = complaints["received_date"].min(), complaints["received_date"].max()
    weeks = pd.date_range(start, end, freq="W-MON")
    series = {}
    for key, g in complaints.drop_duplicates(["odino", "component_group"]).groupby(SERIES_COLS):
        w = g.set_index("received_date")["odino"].resample("W-MON").count()
        series[key] = w.reindex(weeks, fill_value=0)
    return series


def evaluate(series, vehicles, variant, period):
    """Czułość, wyprzedzenie i trafność dla jednego ustawienia w jednym okresie (period = (od, do))."""
    targets = vehicles[["campaign", "report_date"] + SERIES_COLS].drop_duplicates()
    results = {
        key: analysis.detect_spikes(w, variant["confirm"], variant["window"], variant["min"])
        for key, w in series.items()
    }
    first_week = min(w.index.min() for w in series.values())
    start, end = period

    # --- 1 i 2: czułość i wyprzedzenie, na poziomie całej akcji ---
    rows = []
    in_period = targets[(targets["report_date"] >= start) & (targets["report_date"] <= end)]
    for camp, g in in_period.groupby("campaign"):
        report = g["report_date"].iloc[0]
        if report < first_week + pd.Timedelta(weeks=52):
            continue  # za mało historii skarg przed akcją
        firsts = []
        for _, t in g.iterrows():
            key = tuple(t[SERIES_COLS])
            if key in results:
                f = analysis.first_alarm_before(results[key], report)
                if f is not None:
                    firsts.append(f)
        lead = (report - min(firsts)).days // 7 if firsts else None
        rows.append({"campaign": camp, "report_date": report, "detected": bool(firsts), "lead_weeks": lead})
    per_campaign = pd.DataFrame(rows)

    # --- 3: trafność alarmów (alarmy z tego okresu) ---
    # Alarm oceniamy tylko wtedy, gdy całe 26 tygodni po nim mieści się w danych tego okresu,
    # żeby przy strojeniu nie podglądać akcji z lat testowych.
    known_until = min(end, targets["report_date"].max())
    dates_by_key = {k: g["report_date"] for k, g in targets.groupby(SERIES_COLS)}
    ep_rows = []
    for key, r in results.items():
        prev = None
        for w in r.index[r["confirmed"]]:
            new_episode = prev is None or (w - prev) > pd.Timedelta(weeks=NEW_EPISODE_GAP_WEEKS)
            prev = w
            if not new_episode or w < start or w + WINDOW > known_until:
                continue
            dates = dates_by_key.get(key, pd.Series(dtype="datetime64[ns]"))
            if ((dates > w) & (dates <= w + WINDOW)).any():
                outcome = "trafny"
            elif ((dates <= w) & (dates > w - WINDOW)).any():
                outcome = "po akcji"
            else:
                outcome = "fałszywy"
            ep_rows.append({"series": key, "week": w, "outcome": outcome})
    per_episode = pd.DataFrame(ep_rows, columns=["series", "week", "outcome"])
    return per_campaign, per_episode


def summarize(pc, pe, signal):
    pc = pc.copy()
    pc["signal"] = pc["campaign"].map(signal).fillna(0)
    strong = pc[pc["signal"] >= SIGNAL_MIN]
    counts = pe["outcome"].value_counts()
    true_, false_ = counts.get("trafny", 0), counts.get("fałszywy", 0)
    sens = pc["detected"].mean()
    prec = true_ / (true_ + false_) if true_ + false_ else 0.0
    f1 = 2 * sens * prec / (sens + prec) if sens + prec else 0.0
    leads = pc["lead_weeks"].dropna()
    return {
        "akcje wykryte": f"{pc['detected'].sum()}/{len(pc)} ({100 * sens:.0f}%)",
        "z sygnałem": f"{strong['detected'].sum()}/{len(strong)}",
        "wyprzedzenie (mediana tyg.)": leads.median() if len(leads) else None,
        "alarmy trafne/fałszywe": f"{true_}/{false_}",
        "trafność %": round(100 * prec),
        "F1": round(f1, 3),
    }


def label(v):
    okno = "1 tydzień" if v["window"] == 1 else f"suma {v['window']} tyg."
    return f"{okno}, potw. {v['confirm']}, min {v['min']}"


if __name__ == "__main__":
    complaints, campaigns, vehicles = load()
    signal = signal_strength(complaints, vehicles)
    series = all_weekly_series(complaints)

    # Krok 1: strojenie na 2021-2023
    tune_rows = []
    for v in VARIANTS:
        pc, pe = evaluate(series, vehicles, v, TUNE)
        tune_rows.append({"ustawienie": label(v), **summarize(pc, pe, signal)})
    tune = pd.DataFrame(tune_rows)
    print("STROJENIE (akcje i alarmy 2021-2023):")
    print(tune.to_string(index=False))

    best = VARIANTS[int(tune["F1"].idxmax())]
    print(f"\nWybrane ustawienie (najwyższe F1 na 2021-2023): {label(best)}")

    # Krok 2: jeden test na 2024-2026 wybranym ustawieniem
    pc, pe = evaluate(series, vehicles, best, TEST)
    print("\nTEST (akcje i alarmy 2024-2026, nieużywane przy strojeniu):")
    print(pd.DataFrame([{"ustawienie": label(best), **summarize(pc, pe, signal)}]).to_string(index=False))

    pc["signal"] = pc["campaign"].map(signal).fillna(0)
    pc.to_csv("data/processed/validation_campaigns.csv", index=False)
    pe.to_csv("data/processed/validation_alarms.csv", index=False)
