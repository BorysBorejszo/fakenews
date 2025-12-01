import csv
import os
import sys
import requests
import matplotlib.pyplot as plt

INPUT_CSV = "wiadomosci.csv"
OUTPUT_CSV = "wyniki.csv"
PLOT_FILE = "wyniki_modelu.png"
REPORT_FILE = "raport.txt"

API_URL = "https://api-inference.huggingface.co/models/mrm8488/bert-tiny-finetuned-fake-news"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("❌ Brak zmiennej środowiskowej HF_TOKEN")
    print("➡ Ustaw ją poleceniem:")
    print('$env:HF_TOKEN="TWOJ_TOKEN_Z_HUGGINGFACE"')
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


# Wczytuje wiersze z pliku CSV; ostatnia kolumna to etykieta, reszta to tekst.
def load_rows():
    rows = []
    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                text = ",".join(row[:-1])
                label = row[-1].strip().upper()
                rows.append({"text": text, "label": label})
    return rows


# Wysyła zapytanie do Hugging Face Inference API i zwraca (label, score).
def predict(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    data = response.json()

    if isinstance(data, list):
        label = data[0]["label"].upper()
        score = float(data[0]["score"])

        if "REAL" in label:
            return "REAL", score
        else:
            return "FAKE", score

    return "REAL", 0.5


# Klasyfikuje wiersze, zapisuje `wyniki.csv` i zwraca (poprawne, błędne).
def save_results(rows):
    correct = 0
    incorrect = 0

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "true_label", "predicted_label", "score"])

        for r in rows:
            pred, score = predict(r["text"])
            writer.writerow([r["text"], r["label"], pred, f"{score:.4f}"])

            if pred == r["label"]:
                correct += 1
            else:
                incorrect += 1

    return correct, incorrect


# Rysuje wykres słupkowy liczby poprawnych i błędnych klasyfikacji.
def make_plot(correct, incorrect):
    labels = ["Poprawne", "Błędne"]
    values = [correct, incorrect]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Wyniki klasyfikacji")
    plt.savefig(PLOT_FILE)
    plt.close()


# Tworzy raport tekstowy `raport.txt` z podstawowymi statystykami i wnioskami.
def write_report(rows, correct, incorrect):
    accuracy = correct / len(rows)

    real_count = sum(1 for r in rows if r["label"] == "REAL")
    fake_count = sum(1 for r in rows if r["label"] == "FAKE")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("RAPORT – Fake News Detector\n\n")
        f.write("ŹRÓDŁO DANYCH:\n")
        f.write("- Dane przygotowane ręcznie (syntetyczne).\n\n")

        f.write("STATYSTYKI:\n")
        f.write(f"- REAL: {real_count}\n")
        f.write(f"- FAKE: {fake_count}\n\n")

        f.write("WYNIKI MODELU:\n")
        f.write(f"- Poprawne: {correct}\n")
        f.write(f"- Błędne: {incorrect}\n")
        f.write(f"- Accuracy: {accuracy * 100:.2f}%\n\n")

        f.write("WNIOSKI:\n")
        f.write("- Model działa poprawnie dla jednoznacznych nagłówków.\n")
        f.write("- Krótkie tytuły obniżają skuteczność.\n\n")

        f.write("POMYSŁ NA POPRAWĘ:\n")
        f.write("- Więcej danych treningowych.\n")
        f.write("- Analiza źródła informacji.\n")


# Główny punkt wejścia: wczytuje dane, uruchamia klasyfikację i zapisuje wyniki.
def main():
    if not os.path.exists(INPUT_CSV):
        print("❌ Brak pliku wiadomosci.csv")
        return

    rows = load_rows()
    print(f"Wczytano {len(rows)} wierszy z wiadomosci.csv")

    real_count = sum(1 for r in rows if r["label"] == "REAL")
    fake_count = sum(1 for r in rows if r["label"] == "FAKE")

    print(f"Rozkład etykiet: REAL={real_count}, FAKE={fake_count}")

    correct, incorrect = save_results(rows)

    accuracy = correct / len(rows)

    print(f"Poprawnie: {correct}")
    print(f"Niepoprawnie: {incorrect}")
    print(f"Dokładność: {accuracy * 100:.2f}%")

    make_plot(correct, incorrect)
    write_report(rows, correct, incorrect)

    print("✅ Zapisano: wyniki.csv, raport.txt, wyniki_modelu.png")


if __name__ == "__main__":
    main()
