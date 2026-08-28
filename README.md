# CC3084 - Proyecto 2: Análisis Exploratorio

**Reto 11, Negocios:** [Repeat Buyers Prediction — Challenge the Baseline](https://tianchi.aliyun.com/competition/entrance/231576/information)
(Tianchi / Alibaba, dataset IJCAI-15)


| Integrante | Carné |
|---|---|
| Mia Alejandra Fuentes Mérida | 23775 |
| Roberto José Barreda Siekavizza | 23354 |
| Javier Eduardo España Pacheco | 23361 |
| Ángel Esteban Esquit Hernández | 23221 |

## Setup

```bash
pip install -r requirements.txt
```

```python
from src import load
log = load.load_log(nrows=1_000_000)   # nrows para no cargar todo
train = load.load_train()
```