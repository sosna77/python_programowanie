# Symulacja Modelu Isinga

Symulacja Monte Carlo dwuwymiarowego modelu Isinga. Projekt wykorzystuje bibliotekę `numba` do akceleracji numerycznej oraz środowisko `uv` do zarządzania pakietami.

## Wymagania

* [uv](https://docs.astral.sh/uv/) (menedżer projektów i pakietów w Pythonie)
* `ffmpeg` (opcjonalnie, do zapisu animacji w formacie `.mp4`. Do zapisu `.gif` wystarcza wbudowane Pillow).

## Instalacja i Uruchomienie

Nie musisz ręcznie tworzyć środowiska wirtualnego (venv). Narzędzie `uv` zrobi to automatycznie.

Uruchomienie z parametrami domyślnymi:

```bash
uv run run_ising.py

```

Uruchomienie z własnymi parametrami, podglądem animacji i zapisem magnetyzacji:

```bash
uv run run_ising.py --N 150 --M 2000 --beta 0.44 --show-animation --magnetization-file mag_data.csv

```

Wszystkie pliki wyjściowe (dane CSV, animacje) zapisywane są automatycznie w katalogu `output/`.

## Parametry z linii poleceń

| Argument | Krótki | Domyślnie | Opis |
| --- | --- | --- | --- |
| `--size` | `-N` | 100 | Rozmiar siatki $N \times N$ |
| `--steps` | `-M` | 1000 | Liczba makrokroków symulacji |
| `--beta` | `-b` | 0.4406 | Odwrotność temperatury ($\beta$) |
| `--exchange` | `-J` | 1.0 | Całka wymiany $J$ |
| `--field` | `-B` | 0.0 | Zewnętrzne pole magnetyczne $B$ |

**Opcje wyjścia i wizualizacji:**

* `--magnetization-file <plik>`: Zapisuje wartość magnetyzacji w funkcji czasu do pliku `.csv`.
* `--show-animation`: Wyświetla okno z animacją po zakończeniu obliczeń.
* `--animation-file <plik>`: Zapisuje animację na dysk (np. `video.mp4` lub `anim.gif`).

## Struktura projektu

```text
project/
├── ising/                  # Pakiet z logiką
│   ├── __init__.py
│   ├── simulation.py       # Obliczenia fizyczne (Numba)
│   ├── visualization.py    # Generowanie wykresów i klatek animacji
│   └── io_utils.py         # Zapis plików na dysk
├── output/                 # Katalog generowany podczas działania
├── pyproject.toml          # Konfiguracja środowiska uv
<!-- ├── uv.lock                 # Zablokowane wersje zależności -->
└── run_ising.py            # Skrypt uruchomieniowy (CLI)

```