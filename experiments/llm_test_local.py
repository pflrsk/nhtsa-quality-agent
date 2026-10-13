"""
Ten sam test co llm_test.py, ale z modelem lokalnym przez Ollama (bez limitów, dane nie wychodzą z komputera).

Wymaga: zainstalowanej Ollamy i pobranego modelu, np.:  ollama pull qwen3:8b
Uruchomienie (w folderze quality-agent):  python exp\\llm_test_local.py            (domyślnie qwen3:8b)
                                          python exp\\llm_test_local.py gemma3:4b  (inny model)
                                          python exp\\llm_test_local.py --think    (model najpierw "myśli", wolniej)
Wyniki: exp\\llm_results_<model>.jsonl, można przerwać i uruchomić ponownie.
"""
import json, re, sys, time, urllib.error, urllib.request
from pathlib import Path

from llm_test import SAMPLE, SYSTEM, prompt

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
THINK = "--think" in sys.argv
MODEL = ARGS[0] if ARGS else "qwen3:8b"
RESULTS = Path(__file__).parent / f"llm_results_{MODEL.replace(':', '-')}{'-think' if THINK else ''}.jsonl"
URL = "http://localhost:11434/api/chat"


def ask(case, think_flag=True):
    body = {"model": MODEL, "stream": False, "format": "json", "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 3000},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(case)}]}
    if think_flag:
        body["think"] = THINK  # czy model ma najpierw "pomyśleć"; starsze modele/wersje Ollamy tego nie znają
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        content = json.loads(r.read())["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        found = re.search(r"\{.*\}", content, re.S)  # czasem JSON jest otoczony innym tekstem
        return json.loads(found.group(0)) if found else {"answer": "", "reason": "brak JSON"}


def main():
    cases = [json.loads(l) for l in SAMPLE.open(encoding="utf-8")]
    done = {json.loads(l)["id"] for l in RESULTS.open(encoding="utf-8")} if RESULTS.exists() else set()
    todo = [c for c in cases if c["id"] not in done]
    print(f"Model lokalny: {MODEL}{' (z myśleniem)' if THINK else ''}. Zrobione: {len(done)}, zostalo: {len(todo)}")
    think_flag = True
    for i, case in enumerate(todo, 1):
        t0 = time.time()
        try:
            out = ask(case, think_flag)
        except urllib.error.HTTPError as e:
            if e.code == 400 and think_flag:
                think_flag = False
                out = ask(case, think_flag)
            elif e.code == 404:
                sys.exit(f"Brak modelu {MODEL}. Pobierz go: ollama pull {MODEL}")
            else:
                raise
        except urllib.error.URLError:
            sys.exit("Nie moge polaczyc sie z Ollama. Czy jest uruchomiona (ikonka przy zegarze)?")
        except json.JSONDecodeError:
            out = {"answer": "", "reason": "niepoprawny JSON"}
        letter = str(out.get("answer", "")).strip().upper()
        if letter == "NONE":
            pred = "NONE"
        elif letter in case["options"]:
            pred = case["options"][letter]["campaign"]
        else:
            pred = "BLAD"  # model nie dał poprawnej odpowiedzi (np. za długo "myślał") - liczymy jako błąd
        rec = {"id": case["id"], "model": MODEL, "trial": case["trial"], "truth": case["truth"], "pred": pred,
               "reason": out.get("reason", ""), "tfidf_pred": case["tfidf_pred"], "tfidf_max": case["tfidf_max"]}
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{len(done) + i}/{len(cases)}] {case['trial']:5s} prawda={case['truth']} model={pred} "
              f"{'OK' if pred == case['truth'] else 'X'}  ({time.time() - t0:.1f} s)")
    print(f"\nGotowe. Wyniki w {RESULTS.name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrzerwano. Uruchom ponownie, zeby dokonczyc.")
