# Framework do uruchamiania symulacji

## Framework

Wewnątrz skryptu `base.py` znajduje się cały framework zawierający w sobie zarówno `@dataclass` przechowujące istotne dane, jak i klasy abstrakcyjne zawierające `@abstractmethod` oraz jedną zwyczajną klasę `Simulation`, która zawiera metodę `run()`, wywołującą symulację.

## Implementacja

W celu implementacji frameworku do konkretnego projektu, można w prosty sposób zaimportować dane z pliku `base.py`:
```python
from base import *
```

Należy również zaimplementować wszystkie klasy `@dataclass` jak i klasy abstrakcyjne z uwzględnieniem ich abstrakcyjnych metod. Nie należy implementować jedynie klasy `Simulation`.

Do wywołania nie jest konieczna zmiana parametrów wywołania metody `run()`, wystarczy ustawić porządane dane konfiguracyjne w klasie dziedziczącej po `SimulationConfiguration` oraz wywołać skrypt przez IDE lub bezpośrednio w terminalu: 

```bash
python implementacja.py
```

## Przykład: Oscylator harmoniczny z tłumieniem

W skrypcie `oscillator.py` zaimplementowany został framework z `base.py` do przypadku oscylatora harmonicznego z tłumieniem danego wzorem:
$$
m \frac{d^2x}{dt^2}=-kx-c\frac{dx}{dt}
$$

Symulacja może być wykonana zarówno przy użyciu metody Eulera, jak i Verlet w zależności od podanej konfiguracji. 

Zapisuje on dane związane ze stanem oscylatora w każdym kroku czasowym tj. `krok`,`t`, `x`, `v`, oraz statystyki z każdego kroku tj. `E_kinetic`, `E_potential`, `E_total` w pliku `.csv` w automatycznie tworzonym katalogu `.data/`.

Ponadto tworzona są wizualizacje, zapisywane w automatycznie tworzonym katalogu `./plots`.

Wywołanie implementacji z oscylatorem przez IDE lub:
```bash
python oscillator.py
```

## Przykład: model SIR na siatce kwadratowej