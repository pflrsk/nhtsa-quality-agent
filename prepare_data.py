"""
Przygotowanie danych: z surowych plików NHTSA (data/raw) do czystych tabel (data/processed).

Uruchomienie:  python prepare_data.py

Wynik:
  data/processed/complaints.csv  - skargi kierowców na pojazdy Stellantis (FCA US), 2020-2026
  data/processed/campaigns.csv          - akcje serwisowe (recalle) Stellantis od 2020 roku, jeden wiersz = jedna akcja
  data/processed/campaign_vehicles.csv  - modele, roczniki i komponenty objęte każdą akcją
  data/processed/campaign_parts.csv     - części wymieniane w ramach tych akcji

Wszystkie trzy tabele akcji łączy kolumna "campaign" (numer akcji).
"""

from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")

# Producent w danych NHTSA. Stellantis działa w USA jako FCA US LLC.
STELLANTIS = "FCA US"

# ---------------------------------------------------------------------------
# Skargi
# Pliki nie mają nagłówków. Numery kolumn i nazwy pochodzą z opisu pól NHTSA (CMPL.txt).
# Kolumny numerujemy od 0, więc pole nr 2 z dokumentacji to indeks 1.
# ---------------------------------------------------------------------------
COMPLAINT_COLS = {
    1: "odino",        # numer skargi (ten sam dla kilku komponentów jednej skargi)
    2: "manufacturer",
    3: "make",
    4: "model",
    5: "model_year",
    6: "crash",        # Y / N
    7: "fail_date",    # kiedy wystąpiła usterka
    8: "fire",         # Y / N
    9: "injured",
    10: "deaths",
    11: "component",
    16: "received_date",  # kiedy NHTSA dostała skargę
    17: "miles",
    19: "description",
    45: "product_type",   # V = pojazd, E = część/wyposażenie kupione osobno, T = opony, C = fotelik
}

COMPLAINT_FILES = [
    ("COMPLAINTS_RECEIVED_2020-2024.zip", "COMPLAINTS_RECEIVED_2020-2024.txt"),
    ("COMPLAINTS_RECEIVED_2025-2026.zip", "COMPLAINTS_RECEIVED_2025-2026.txt"),
]


def read_nhtsa(zip_name, cols):
    """Czyta plik NHTSA (tekst rozdzielony tabulatorami) bezpośrednio z zipa."""
    df = pd.read_csv(
        RAW / zip_name,
        sep="\t",
        header=None,
        usecols=list(cols.keys()),
        dtype=str,           # wszystko jako tekst, typy ustawimy sami
        encoding="latin-1",  # w opisach są znaki spoza UTF-8
        quoting=3,           # cudzysłowy w opisach to zwykłe znaki, nie separatory
    )
    return df.rename(columns=cols)


def to_date(series):
    """Tekst 20241231 -> data. Błędne wartości zamienia na puste (NaT)."""
    return pd.to_datetime(series, format="%Y%m%d", errors="coerce")


# NHTSA używa równolegle dwóch nazw dla tego samego komponentu: skargi zwykle mają
# pierwszą z nich, akcje serwisowe drugą. Sprowadzamy je do jednej nazwy, żeby dało się
# porównać skargi z akcjami.
COMPONENT_SYNONYMS = {
    "ENGINE AND ENGINE COOLING": "ENGINE",
    "FUEL/PROPULSION SYSTEM": "FUEL SYSTEM",
    "VISIBILITY/WIPER": "VISIBILITY",
}


def main_group(component):
    """'SERVICE BRAKES, HYDRAULIC' -> 'SERVICE BRAKES', 'STRUCTURE:BODY' -> 'STRUCTURE'."""
    group = component.str.split(r"[,:]", n=1).str[0].str.strip()
    return group.replace(COMPONENT_SYNONYMS)


def prepare_complaints():
    parts = [read_nhtsa(zip_name, COMPLAINT_COLS) for zip_name, _ in COMPLAINT_FILES]
    df = pd.concat(parts, ignore_index=True)

    df = df[df["manufacturer"].str.contains(STELLANTIS, na=False)]
    # Tylko skargi na pojazdy. Skargi na części kupione osobno mają w kolumnie "model"
    # nazwę części (np. "RUNNING BOARD KIT"), a nie model auta.
    df = df[df["product_type"] == "V"]
    df = df.drop_duplicates(subset=["odino", "component"])
    df["make"] = df["make"].replace({"ALFA": "ALFA ROMEO"})  # jedna literówka w danych

    df["fail_date"] = to_date(df["fail_date"])
    df["received_date"] = to_date(df["received_date"])
    df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
    df.loc[df["model_year"] == 9999, "model_year"] = pd.NA
    for col in ["injured", "deaths", "miles"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["component_group"] = main_group(df["component"])
    df["description"] = df["description"].fillna("").str.strip()

    return df.drop(columns=["manufacturer", "product_type"])


# ---------------------------------------------------------------------------
# Akcje serwisowe (recalle), plik FLAT_RCL_POST_2010
# ---------------------------------------------------------------------------
CAMPAIGN_COLS = {
    1: "campaign",          # numer kampanii, np. 25V123000
    2: "make",
    3: "model",
    4: "model_year",
    6: "component",
    10: "campaign_type",      # V = pojazdy, E = części/wyposażenie, T = opony, C = foteliki
    11: "vehicles_affected",  # liczba pojazdów objętych akcją
    14: "manufacturer",
    15: "report_date",        # kiedy producent zgłosił akcję do NHTSA
    19: "defect",
    20: "consequence",
    21: "remedy",
    24: "part_name",          # nazwa części u producenta, np. "Module, Body Control"
    26: "part_number",
}


def prepare_campaigns():
    """Zwraca trzy tabele: akcje, objęte pojazdy oraz części (łączy je numer akcji)."""
    df = read_nhtsa("FLAT_RCL_POST_2010.zip", CAMPAIGN_COLS)

    df = df[df["manufacturer"].str.contains(STELLANTIS, na=False)]
    # Tylko akcje dotyczące pojazdów. Akcje typu E dotyczą części sprzedawanych osobno
    # (np. dywaniki, lusterka Mopar): mają w kolumnie "model" nazwę części i rocznik 9999.
    df = df[df["campaign_type"] == "V"].drop(columns=["campaign_type"])
    df["report_date"] = to_date(df["report_date"])
    df = df[df["report_date"] >= "2020-01-01"]

    df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
    df.loc[df["model_year"] == 9999, "model_year"] = pd.NA
    df["vehicles_affected"] = pd.to_numeric(df["vehicles_affected"], errors="coerce")
    df["component_group"] = main_group(df["component"])

    # Surowy plik ma osobny wiersz dla każdego numeru części objętej akcją.
    # Części zapisujemy w osobnej tabeli (łączy je numer kampanii), a w tabeli akcji
    # po ich usunięciu zostałyby identyczne wiersze, więc usuwamy powtórzenia.
    parts = df[["campaign", "part_name", "part_number"]].dropna(subset=["part_name"]).drop_duplicates()

    # Informacje o całej akcji (liczba aut, data, opisy) są takie same we wszystkich
    # jej wierszach, więc trzymamy je w osobnej tabeli: jeden wiersz = jedna akcja.
    # Dzięki temu suma kolumny vehicles_affected daje poprawny wynik.
    campaign_info = ["campaign", "report_date", "vehicles_affected", "defect", "consequence", "remedy"]
    campaigns = df[campaign_info].drop_duplicates()

    # Pojazdy objęte akcją. NHTSA przypisuje komponent na kilku poziomach naraz,
    # np. "SEAT BELTS" i "SEAT BELTS:FRONT:RETRACTOR", każdy w osobnym wierszu.
    # Po grupowaniu to ten sam komponent, więc łączymy je, a szczegóły zapisujemy razem.
    key = ["campaign", "make", "model", "model_year", "component_group"]
    vehicles = (
        df[key + ["component"]]
        .drop_duplicates()
        .groupby(key, dropna=False)["component"]
        .agg(lambda x: "; ".join(sorted(set(x))))
        .reset_index()
    )
    return campaigns, vehicles, parts


def check_unique(df, key, table_name):
    """Sprawdza założenie, co oznacza jeden wiersz tabeli.

    Jeśli w kolumnach `key` są powtórzenia, to jeden wiersz NIE jest tym, czym myślimy,
    i późniejsze liczenie da zawyżone wyniki. Wtedy przerywamy zamiast zapisywać plik.
    """
    repeats = df.duplicated(subset=key).sum()
    if repeats:
        raise ValueError(f"{table_name}: {repeats} powtórzeń dla kolumn {key}")


def check_values(df, table_name):
    """Sprawdza, czy wartości w kolumnach mają sens (marka, rocznik, daty)."""
    problems = []
    makes = {"JEEP", "RAM", "DODGE", "CHRYSLER", "FIAT", "ALFA ROMEO", "PLYMOUTH", "STERLING", "EAGLE"}
    if not df["make"].isin(makes).all():
        problems.append(f"nieznane marki: {sorted(set(df['make']) - makes)}")
    years = df["model_year"].dropna()
    if not years.between(1950, 2030).all():
        problems.append(f"podejrzane roczniki: {sorted(years[~years.between(1950, 2030)].unique())}")
    if problems:
        raise ValueError(f"{table_name}: " + "; ".join(problems))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    complaints = prepare_complaints()
    check_unique(complaints, ["odino", "component"], "skargi")
    check_values(complaints, "skargi")
    complaints.to_csv(OUT / "complaints.csv", index=False)
    print(f"Skargi: {len(complaints)} wierszy, {complaints['odino'].nunique()} unikalnych skarg")

    campaigns, vehicles, parts = prepare_campaigns()
    check_unique(campaigns, ["campaign"], "akcje serwisowe")
    check_unique(vehicles, ["campaign", "make", "model", "model_year", "component_group"], "pojazdy w akcjach")
    check_values(vehicles, "pojazdy w akcjach")
    check_unique(parts, ["campaign", "part_name", "part_number"], "części")
    campaigns.to_csv(OUT / "campaigns.csv", index=False)
    vehicles.to_csv(OUT / "campaign_vehicles.csv", index=False)
    parts.to_csv(OUT / "campaign_parts.csv", index=False)
    print(f"Akcje serwisowe: {len(campaigns)} akcji, łącznie {int(campaigns['vehicles_affected'].sum()):,} pojazdów")
    print(f"Pojazdy w akcjach: {len(vehicles)} wierszy (model + rocznik + komponent)")
    print(f"Części w akcjach: {len(parts)} wierszy")
