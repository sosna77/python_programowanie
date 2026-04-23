# Symulacja Modelu Isinga 2D
Projekt to symulacja modelu Isinga na sieci 2D, przy użyciu algorytmu **Metropolis-Hastings**. Kod został napisany zarówno w wersji podstawowej jak i zoptymalizowany za pomocą `@njit` z pakietu **Numba**.

## Użycie
Skrypty znajdują się w katalogu `src/`:

```bash
cd /lab_002/src
```

### 1. Uruchomienie domyślne 
Uruchamia siatkę 100x100 na 100 kroków symulacji z parametrami domyślnymi:
```bash
python ising.py         # wersja podstawowa
python ising_nb         # wersja zoptymalizowana
```

### 2. Uruchomienie z własnymi parametrami
Parametry wywołania można dostosować za pomocą flag:

```bash
python ising_nb.py -N 50 -b 0.44 -M 500
```

**Dostępne parametry:**

| Flaga | Pełna nazwa | Opis                                 | Domyślnie |
|-------|-------------|--------------------------------------|-----------|
| `-N`    | `--size`      | Rozmiar siatki (N x N)               |        10 |
| `-J`    | `--exchange`  | Całka wymiany (oddziaływanie spinów) |         1 |
| `-b`    | `--beta`      | Odwrotność temperatury (1/kT)        |       0.2 |
| `-B`    | `--field`     | Zewnętrzne pole magnetyczne          |         1 |
| `-M`    | `--steps`     | Liczba kroków czasowych (klatek)     |       100 |


### Pomoc
Aby wyświetlić listę opcji:

```bash

python ising_nb.py --help
```