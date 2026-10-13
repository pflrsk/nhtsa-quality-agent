"""
Test: czy Gemini rozpozna, której znanej akcji serwisowej dotyczy skarga, albo że nie pasuje do żadnej.

Uruchomienie (w folderze quality-agent):  python exp\\llm_test.py
Klucz: zmienna GEMINI_API_KEY, a jeśli jej nie ma, skrypt poprosi o wklejenie (nigdzie go nie zapisuje).
Można przerwać i uruchomić ponownie: skrypt pomija już zrobione przypadki.
"""
import getpass, json, os, re, sys, time
from pathlib import Path
from google import genai
from google.genai import types

HERE = Path(__file__).parent
SAMPLE, RESULTS = HERE / "llm_sample.jsonl", HERE / "llm_results.jsonl"
PAUSE = 6.5  # sekundy między zapytaniami (limit darmowy ~10 na minutę)

SYSTEM = (
    "You are a vehicle quality engineer. You get a consumer complaint about a vehicle and a list of known "
    "safety recall defects that apply to this vehicle. Decide which listed defect the complaint describes, "
    "judging by the symptoms and the failing part. If the complaint describes a different problem than every "
    "listed defect, answer NONE. Reply with JSON: {\"answer\": \"<letter or NONE>\", \"reason\": \"<max 20 words>\"}"
)

def prompt(case):
    opts = "\n".join(f"{k}: {v['defect']}" for k, v in case["options"].items())
    return f"Vehicle: {case['vehicle']}\n\nComplaint:\n{case['complaint']}\n\nKnown recall defects:\n{opts}"

def pick_model(client):
    """Wybiera najnowszy dostępny model 'flash' (stare wersje Google wyłącza, więc nie wpisujemy nazwy na sztywno)."""
    if os.environ.get("GEMINI_MODEL"):
        return [os.environ["GEMINI_MODEL"]]
    names = []
    for m in client.models.list():
        name = m.name.replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" not in actions or "flash" not in name:
            continue
        if any(w in name for w in ["lite", "image", "tts", "audio", "live", "thinking", "8b"]):
            continue
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        stable = not any(w in name for w in ["preview", "exp"])
        names.append((stable, float(v.group(1)) if v else 0, name))
    if not names:
        sys.exit("Nie znalazlem zadnego modelu 'flash'. Dostepne modele: " + ", ".join(m.name for m in client.models.list()))
    names.sort(reverse=True)
    return [n for _, _, n in names]


def call(client, model, case, thinking_off):
    cfg = dict(system_instruction=SYSTEM, temperature=0, response_mime_type="application/json")
    if thinking_off:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return client.models.generate_content(model=model, contents=prompt(case), config=types.GenerateContentConfig(**cfg))


def main():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or getpass.getpass("Wklej klucz Gemini (nie bedzie widoczny): ")
    client = genai.Client(api_key=key.strip())
    models = pick_model(client)
    MODEL = models[0]
    print("Dostepne modele flash (w tej kolejnosci):", ", ".join(models))
    thinking_off = True
    cases = [json.loads(l) for l in SAMPLE.open(encoding="utf-8")]
    done = {json.loads(l)["id"] for l in RESULTS.open(encoding="utf-8")} if RESULTS.exists() else set()
    todo = [c for c in cases if c["id"] not in done]
    print(f"Model: {MODEL}. Zrobione: {len(done)}, zostalo: {len(todo)}")
    for i, case in enumerate(todo, 1):
        for attempt in range(6):
            try:
                resp = call(client, MODEL, case, thinking_off)
                out = json.loads(resp.text)
                break
            except Exception as e:
                msg = str(e)
                if thinking_off and ("thinking" in msg.lower() or "INVALID_ARGUMENT" in msg):
                    thinking_off = False  # nowszy model nie przyjmuje tego ustawienia - próbujemy bez niego
                    continue
                busy = ("503" in msg or "UNAVAILABLE" in msg) and attempt >= 1
                gone = "NOT_FOUND" in msg or "404" in msg or "PerDay" in msg or "per day" in msg.lower()
                if (busy or gone) and models.index(MODEL) + 1 < len(models):
                    MODEL = models[models.index(MODEL) + 1]
                    print(f"  model przeciazony, przechodze na {MODEL}")
                    continue
                if "NOT_FOUND" in msg or "404" in msg:
                    sys.exit(f"Model {MODEL} jest niedostepny: {msg[:200]}")
                if "PerDay" in msg or "per day" in msg.lower():
                    print("Dzienny limit wyczerpany. Uruchom ponownie jutro - skrypt dokonczy od tego miejsca.")
                    sys.exit(1)
                wait = 20 * (attempt + 1)
                print(f"  blad ({msg[:80]}...), czekam {wait} s")
                time.sleep(wait)
        else:
            print("Za duzo bledow, przerywam."); sys.exit(1)
        letter = str(out.get("answer", "")).strip().upper()
        pred = case["options"].get(letter, {}).get("campaign", "NONE") if letter != "NONE" else "NONE"
        rec = {"id": case["id"], "model": MODEL, "trial": case["trial"], "truth": case["truth"], "pred": pred,
               "reason": out.get("reason", ""), "tfidf_pred": case["tfidf_pred"], "tfidf_max": case["tfidf_max"]}
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{len(done) + i}/{len(cases)}] {case['trial']:5s} prawda={case['truth']} model={pred} {'OK' if pred == case['truth'] else 'X'}")
        time.sleep(PAUSE)
    res = [json.loads(l) for l in RESULTS.open(encoding="utf-8")]
    known = [r for r in res if r["trial"] == "znana"]; new = [r for r in res if r["trial"] == "nowa"]
    print("\nWYNIK")
    print(f"Znana wada -> wskazal wlasciwa akcje: {sum(r['pred'] == r['truth'] for r in known)}/{len(known)}")
    print(f"Nowy problem -> odpowiedzial NONE:    {sum(r['pred'] == 'NONE' for r in new)}/{len(new)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrzerwano. Uruchom ponownie, zeby dokonczyc.")
