"""
Mapowanie skarg na znane źródła wad (akcje serwisowe).

Pytanie, na które odpowiada narzędzie: "Której znanej wady dotyczy ta skarga - czy może to coś nowego?"

Jak działa:
  1. Kandydaci: akcje serwisowe, które obejmują ten model i rocznik i były ogłoszone przed skargą.
  2. Ranking: porównanie opisu skargi z opisem wady każdej akcji (TF-IDF + podobieństwo cosinusowe).
  3. (osobny krok) Weryfikacja przez model językowy: czy skarga naprawdę opisuje tę wadę.

Skąd wiemy, czy wynik jest dobry: około 3 tys. skarg zawiera numer akcji, której dotyczą
(dopisany przez NHTSA). Usuwamy z tekstu numer i zdania o akcji, zostają same objawy,
a narzędzie ma odgadnąć akcję. Strojenie: skargi sprzed 2024, wynik: skargi z lat 2024-2026.

Uruchomienie: python mapping.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

DATA = Path("data/processed")
TEST_FROM = pd.Timestamp("2024-01-01")
SAME_DEFECT_SIMILARITY = 0.9   # akcje z prawie identycznym opisem wady traktujemy jak tę samą wadę
MIN_SYMPTOM_CHARS = 150        # skargi z krótszym opisem objawów pomijamy w teście

CAMPAIGN_NUMBER = r"\b(\d{2}V\d{3})(?:000)?\b"


def load():
    complaints = pd.read_csv(DATA / "complaints.csv", parse_dates=["received_date"]).drop_duplicates("odino")
    campaigns = pd.read_csv(DATA / "campaigns.csv", parse_dates=["report_date"])
    vehicles = pd.read_csv(DATA / "campaign_vehicles.csv")
    parts = pd.read_csv(DATA / "campaign_parts.csv")
    return complaints, campaigns, vehicles, parts


def symptoms_only(text):
    """Zostawia tylko zdania o objawach: usuwa numery akcji i zdania o akcji, powiadomieniu, częściach."""
    text = re.sub(CAMPAIGN_NUMBER, " ", str(text))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pattern = r"recall|campaign|notification|remedy|parts? (?:was|were) (?:not|un)available"
    return " ".join(s for s in sentences if not re.search(pattern, s, re.I)).strip()


def defect_core(defect):
    """Opis wady bez wstępu 'Chrysler ... is recalling certain 2018-2021 Jeep ... vehicles.'"""
    return re.sub(r"^.*?vehicles?(?: equipped with[^.]*)?\.\s*", "", str(defect))


def same_defect_groups(campaigns):
    """Numer grupy dla każdej akcji: akcje z prawie identycznym opisem wady (np. rozszerzenia
    na kolejne roczniki) dostają ten sam numer."""
    core = campaigns["defect"].map(defect_core)
    X = TfidfVectorizer(stop_words="english").fit_transform(core)
    sim = (X @ X.T).toarray()
    group = list(range(len(campaigns)))
    def find(i):
        while group[i] != i:
            i = group[i]
        return i
    for i, j in zip(*np.where(np.triu(sim, 1) >= SAME_DEFECT_SIMILARITY)):
        group[find(j)] = find(i)
    return dict(zip(campaigns["campaign"], [find(i) for i in range(len(campaigns))]))


class CampaignIndex:
    """Wyszukiwarka akcji serwisowych dla skargi."""

    def __init__(self, campaigns, vehicles, parts, extra_texts=()):
        self.campaigns = campaigns.set_index("campaign")
        part_names = parts.groupby("campaign")["part_name"].apply(lambda s: " ".join(s.astype(str)))
        components = vehicles.groupby("campaign")["component"].apply(lambda s: " ".join(set(s.astype(str))))
        docs = (components.reindex(self.campaigns.index).fillna("") + ". "
                + self.campaigns["defect"].fillna("") + " " + self.campaigns["consequence"].fillna("") + " "
                + part_names.reindex(self.campaigns.index).fillna(""))
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
        self.vectorizer.fit(pd.concat([docs, pd.Series(list(extra_texts), dtype=str)]))
        self.doc_vectors = self.vectorizer.transform(docs)
        self.row = {c: i for i, c in enumerate(self.campaigns.index)}
        cover = vehicles[["campaign", "make", "model", "model_year"]].drop_duplicates()
        self.coverage = cover.groupby(["make", "model", "model_year"])["campaign"].apply(list).to_dict()

    def candidates(self, make, model, model_year, date):
        """Akcje obejmujące ten model i rocznik, ogłoszone najpóźniej tydzień po dacie skargi."""
        found = self.coverage.get((make, model, model_year), [])
        limit = date + pd.Timedelta(days=7)
        return [c for c in found if self.campaigns.at[c, "report_date"] <= limit]

    def rank(self, text, candidates):
        """Kandydaci posortowani od najbardziej podobnego opisu wady. Zwraca listę (akcja, podobieństwo)."""
        if not candidates:
            return []
        q = self.vectorizer.transform([text])
        scores = (self.doc_vectors[[self.row[c] for c in candidates]] @ q.T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")
        return [(candidates[i], float(scores[i])) for i in order]


def build_benchmark(complaints, campaigns, index):
    """Skargi z dokładnie jednym numerem akcji w tekście, z samymi objawami i listą kandydatów."""
    numbers = complaints["description"].fillna("").str.findall(CAMPAIGN_NUMBER)
    b = complaints[numbers.str.len() == 1].copy()
    b["campaign"] = numbers[numbers.str.len() == 1].str[0] + "000"
    b = b[b["campaign"].isin(campaigns["campaign"])]
    b["symptoms"] = b["description"].map(symptoms_only)
    no_failure = b["symptoms"].str.contains(r"not experienced a failure|no failure", case=False)
    b = b[~no_failure & (b["symptoms"].str.len() >= MIN_SYMPTOM_CHARS)]
    b["candidates"] = [index.candidates(r.make, r.model, r.model_year, r.received_date) for r in b.itertuples()]
    covered = pd.Series([c in cs for c, cs in zip(b["campaign"], b["candidates"])], index=b.index)
    return b[covered & (b["candidates"].str.len() >= 2)]


def evaluate(bench, index, groups):
    """Wyniki na zbiorze testowym: trafienia za pierwszym razem i odróżnianie wady znanej od nowej."""
    newest = lambda cs: max(cs, key=lambda c: index.campaigns.at[c, "report_date"])
    rows, known_scores, new_scores = [], [], []
    for r in bench.itertuples():
        ranked = index.rank(r.symptoms, r.candidates)
        top = ranked[0][0]
        rows.append({
            "tfidf": top == r.campaign,
            "tfidf_same_defect": groups[top] == groups[r.campaign],
            "newest": groups[newest(r.candidates)] == groups[r.campaign],
            "random": np.mean([groups[c] == groups[r.campaign] for c in r.candidates]),
        })
        # "nowy problem": usuwamy prawdziwą akcję i wszystkie akcje z tą samą wadą
        others = [c for c in r.candidates if groups[c] != groups[r.campaign]]
        if others:
            known_scores.append(ranked[0][1])
            new_scores.append(index.rank(r.symptoms, others)[0][1])
    res = pd.DataFrame(rows)
    y = np.r_[np.ones(len(known_scores)), np.zeros(len(new_scores))]
    auc = roc_auc_score(y, np.r_[known_scores, new_scores])
    return res.mean(), auc


if __name__ == "__main__":
    complaints, campaigns, vehicles, parts = load()
    train_texts = complaints.loc[complaints["received_date"] < TEST_FROM, "description"].fillna("")
    index = CampaignIndex(campaigns, vehicles, parts, extra_texts=train_texts.sample(20000, random_state=0))
    groups = same_defect_groups(campaigns)
    bench = build_benchmark(complaints, campaigns, index)
    test = bench[bench["received_date"] >= TEST_FROM]
    print(f"Skargi testowe (2024-2026): {len(test)}, akcji: {test['campaign'].nunique()}, "
          f"mediana kandydatów: {test['candidates'].str.len().median():.0f}")
    m, auc = evaluate(test, index, groups)
    print(f"Losowy wybór:                 {100 * m['random']:.0f}%")
    print(f"Zawsze najnowsza akcja:       {100 * m['newest']:.0f}%")
    print(f"TF-IDF, dokładnie ta akcja:   {100 * m['tfidf']:.0f}%")
    print(f"TF-IDF, ta sama wada:         {100 * m['tfidf_same_defect']:.0f}%")
    print(f"Wada znana czy nowa (AUC):    {auc:.2f}")
