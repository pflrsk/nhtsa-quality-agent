"""
Analiza skarg: liczenie tygodniowe i wykrywanie wzrostów (alarmów).

Metoda alarmu to karta kontrolna typu c (Shewharta), standard w kontroli jakości:
  norma          = średnia liczba skarg z poprzednich BASELINE_WEEKS tygodni
  górna granica  = norma + 3 * pierwiastek(norma)
  alarm          = tydzień, w którym skarg jest więcej niż granica i co najmniej MIN_COMPLAINTS
  potwierdzony   = alarm w CONFIRM_WEEKS kolejnych tygodniach (pojedynczy skok to często przypadek)

Wariant z oknem (window > 1): zamiast skarg z jednego tygodnia patrzymy na sumę z ostatnich
`window` tygodni. Wtedy oczekiwana suma = window * norma, a granica = window * norma + 3 * pierwiastek
(window * norma). Dzięki temu łapiemy wzrost rozłożony nierówno na kilka tygodni.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/processed")

BASELINE_WEEKS = 26   # z ilu poprzednich tygodni liczymy normę
MIN_COMPLAINTS = 5    # alarm dopiero od tylu skarg w tygodniu (chroni przed alarmami przy normie ~0)
CONFIRM_WEEKS = 2     # ile tygodni z rzędu musi być alarm, żeby uznać go za potwierdzony
LEAD_WINDOW_WEEKS = 26  # w walidacji liczymy tylko alarmy z tylu tygodni przed akcją serwisową


def load_complaints():
    return pd.read_csv(DATA / "complaints.csv", parse_dates=["fail_date", "received_date"])


def weekly_counts(complaints, make, model, component_group, model_years=None):
    """Liczba skarg w każdym tygodniu dla danego modelu i komponentu.

    Liczymy unikalne numery skarg (odino), bo jedna skarga może mieć kilka wierszy.
    Datą jest dzień otrzymania skargi przez NHTSA: tylko to producent mógł wtedy wiedzieć.
    Tygodnie bez skarg dostają 0, żeby nie zniknęły z wykresu i z liczenia normy.
    """
    s = complaints[
        (complaints["make"] == make)
        & (complaints["model"] == model)
        & (complaints["component_group"] == component_group)
    ]
    if model_years is not None:
        s = s[s["model_year"].isin(model_years)]
    s = s.drop_duplicates("odino")
    return s.set_index("received_date")["odino"].resample("W-MON").count()


def detect_spikes(weekly, confirm_weeks=CONFIRM_WEEKS, window=1, min_complaints=MIN_COMPLAINTS):
    """Dla każdego tygodnia: liczba skarg (w oknie), norma, górna granica i czy był alarm.

    window=1 to zwykła karta tygodniowa, window=4 to suma z ostatnich 4 tygodni.
    Norma liczona jest z 26 tygodni PRZED oknem (shift(window)), żeby wzrost
    nie podnosił sam sobie normy.
    """
    counted = weekly.rolling(window).sum()
    baseline = weekly.shift(window).rolling(BASELINE_WEEKS, min_periods=BASELINE_WEEKS).mean() * window
    upper = baseline + 3 * np.sqrt(baseline)
    alarm = (counted > upper) & (counted >= min_complaints)
    confirmed = alarm.astype(int).rolling(confirm_weeks).sum() == confirm_weeks
    return pd.DataFrame({
        "complaints": weekly, "counted": counted, "baseline": baseline, "upper_limit": upper,
        "alarm": alarm, "confirmed": confirmed,
    })


def first_alarm_before(result, event_date, window_weeks=LEAD_WINDOW_WEEKS):
    """Pierwszy potwierdzony alarm w oknie `window_weeks` tygodni przed zdarzeniem (np. akcją serwisową).

    Okno jest potrzebne, bo alarm sprzed kilku lat dotyczy innego problemu, a nie tej akcji.
    Zwraca datę alarmu albo None, jeśli w oknie nie było potwierdzonego alarmu.
    """
    start = event_date - pd.Timedelta(weeks=window_weeks)
    hits = result[result["confirmed"] & (result.index >= start) & (result.index < event_date)]
    return hits.index.min() if len(hits) else None


def plot_spikes(result, title, event_date=None, path=None, ylabel="skargi w tygodniu"):
    """Wykres: słupki skarg, linia normy, linia granicy, czerwone słupki = alarm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = np.where(result["confirmed"], "#c0392b", np.where(result["alarm"], "#e8a29a", "#4a6fa5"))
    ax.bar(result.index, result["counted"], width=5, color=colors)
    ax.plot(result.index, result["baseline"], color="#333333", lw=1.5, label="norma")
    ax.plot(result.index, result["upper_limit"], color="#333333", lw=1, ls="--", label="górna granica")
    if event_date is not None:
        ax.axvline(event_date, color="#e67e22", lw=2, label="akcja serwisowa")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    # Przykład: Jeep Cherokee 2017-2019, układ napędowy, akcja serwisowa 25V011000 z 2025-04-29.
    complaints = load_complaints()
    weekly = weekly_counts(complaints, "JEEP", "CHEROKEE", "POWER TRAIN", model_years=[2017, 2018, 2019])
    result = detect_spikes(weekly)

    event = pd.Timestamp("2025-04-29")
    print("Wszystkie alarmy (jasne) i potwierdzone (2 tygodnie z rzędu):")
    print(result[result["alarm"]].round(1))
    first = first_alarm_before(result, event)
    print("\nPierwszy potwierdzony alarm w ciągu 26 tygodni przed akcją:", first.date())
    print("Wyprzedzenie:", (event - first).days // 7, "tygodni")

    Path("figures").mkdir(exist_ok=True)
    plot_spikes(
        result.loc["2023-06":],
        "Jeep Cherokee 2017-2019, POWER TRAIN: alarmy karty kontrolnej",
        event_date=event,
        path="figures/alarmy_cherokee.png",
    )
