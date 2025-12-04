# Fake News Detector

Prosty skrypt do klasyfikacji nagłówków jako REAL/FAKE przy użyciu Hugging Face Inference API.

Pliki w repo:

- `fake_news_detector.py` — główny skrypt (wczytuje `wiadomosci.csv`, wywołuje API, zapisuje wyniki)
- `wiadomosci.csv` — wejściowe dane (ostatnia kolumna powinna zawierać etykietę REAL lub FAKE)
- `wyniki.csv` — wygenerowane wyniki (tekst, etykieta prawdziwa, etykieta przewidziana, score)
- `raport.txt` — prosty raport statystyczny
- `wyniki_modelu.png` — wykres słupkowy poprawnych/błędnych klasyfikacji

Wymagania (pip):

- `requests`
- `matplotlib`

(Uwaga: zgodnie z prośbą nie dołączono `requirements.txt`; zamiast tego instrukcje instalacji znajdują się poniżej.)

Szybkie instrukcje (macOS / zsh):

1) (opcjonalnie) utwórz i aktywuj virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Zaktualizuj pip i zainstaluj zależności:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install requests matplotlib
```

3) Ustaw token Hugging Face (zmienna środowiskowa `HF_TOKEN`). Przykład dla `zsh` (macOS):

```bash
export HF_TOKEN="TWOJ_TOKEN_Z_HUGGINGFACE"
```

(Użytkownicy PowerShell mogą użyć: `$env:HF_TOKEN="TWOJ_TOKEN"` — skrypt wyświetli pomoc jeśli zmienna nie jest ustawiona.)

4) Przygotuj `wiadomosci.csv` — każdy wiersz: tekst,...,ETYKIETA (ostatnia kolumna to etykieta `REAL` lub `FAKE`).

5) Uruchom skrypt:

```bash
python3 fake_news_detector.py
```

Po uruchomieniu w katalogu pojawią się:

- `wyniki.csv` — szczegóły predykcji
- `raport.txt` — podsumowanie i statystyki
- `wyniki_modelu.png` — wykres słupkowy

Uwagi i wskazówki:

- Skrypt używa bezpośrednio HTTP (biblioteka `requests`) do wywołania Hugging Face Inference API — nie ma potrzeby instalowania `transformers`, jeśli używasz wyłącznie API.
- Jeśli planujesz uruchamiać modele lokalnie lub trenować je, rozważ dodanie `transformers` i `torch` w osobnym środowisku.
- Jeśli chcesz, mogę dodać `requirements.txt` z zamrożonymi wersjami (np. `requests==...`, `matplotlib==...`) — daj znać.

Kontakt / dalsze kroki:

- Jeśli chcesz, automatycznie wygeneruję `requirements.txt` z aktualnie zainstalowanymi wersjami w środowisku.
- Mogę też rozszerzyć README o przykładowy format `wiadomosci.csv` lub dodać skrypt testowy.

Jak nie dziala odpalic komenda 
HF_TOKEN="Twoj_token" python3 fake_news_detector.py