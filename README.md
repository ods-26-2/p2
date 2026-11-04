# p2

Runtime de aceleração P2: biblioteca Python `p2` e ferramentas para carregar e executar modelos TensorRT na Jetson Orin Nano, com backend de CPU (ONNX Runtime) para notebooks.

- API (6c): `load`, `infer`, `info`, `release` — TensorRT na Jetson (FP16 padrão), CPU (ONNX Runtime) no notebook
- Ferramenta disponível: `p2-build` (`--cpu-only` no notebook; compilação FP16/FP32 na Jetson)

```python
import numpy as np
import p2

with p2.load("yolo11n-1.0.0") as h:
    out, stats = p2.infer(h, np.zeros((1, 3, 640, 640), dtype=np.float32))
```
- Contrato da API (P2 1.1, lib 0.2.x; a 0.1.0 implementa a 1.0): [docs/contrato.md](docs/contrato.md); guia de integração do I1: [docs/guia-integracao.md](docs/guia-integracao.md)
- Em desenvolvimento na branch `dev-v0.2.0`: forma dinâmica, lote, dtypes por tensor e modelos com várias etapas ([spec 009](specs/ativas/009-v020-forma-dinamica-etapas.md))
- Exemplo completo (pré e pós-processamento YOLO): [examples/yolo_demo.py](examples/yolo_demo.py)
- Plano completo: [plano-p2.md](plano-p2.md); decisões: [docs/decisoes.md](docs/decisoes.md)

## Requisitos

- Python 3.10.12
- Notebook: qualquer Linux x86_64
- Jetson: JetPack 6.2.1 (L4T 36.4.7), TensorRT 10.3.0, CUDA 12.6

## Instalação

### Notebook (CPU)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[cpu]"
```

### Jetson (TensorRT)

O `tensorrt` vem do JetPack e não do pip, então a venv precisa enxergar os pacotes do sistema:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e ".[trt]"
python scripts/check_jetson.py
```

O `check_jetson.py` confere Python, TensorRT, numpy, `cuda-python` e uma compilação de teste; sai com código 0 quando a placa está pronta.

## `p2-build`

Lê um pacote MODEL (pasta com `metadata.json` e o `.onnx`), valida hash, forma de entrada e número de classes e grava `<raiz>/<engine_ref>/` com `manifest.json`.

```bash
p2-build <pasta-do-pacote> [--cpu-only | --precision FP16 FP32] [--force] [--engine-dir <raiz>]
```

- Raiz dos engines: `--engine-dir`, senão `$P2_ENGINE_DIR`, senão `/opt/ods/engines`
- Pasta existente é recusada; `--force` substitui (a anterior só é trocada se tudo der certo)
- Códigos de saída: 0 sucesso, 1 erro de validação ou gravação, 2 erro de uso
- Sem `--cpu-only` (na Jetson): compila `fp16.engine` e `fp32.engine`; `--precision FP16` ou `FP32` escolhe uma

Exemplo com o pacote de teste:

```bash
mkdir -p /tmp/yolo11n-pkg
cp tests/fixtures/yolo11n-1.0.0/metadata.json models/yolo11n.onnx /tmp/yolo11n-pkg/
p2-build /tmp/yolo11n-pkg --cpu-only --engine-dir /tmp/p2-engines
```

## `p2-bench` e `p2-list`

```bash
p2-bench app-epi-1.0.0 --precision FP16 --duration 30m --warmup 50   # grava reports/bench/<ref>-<prec>-<data>.json
p2-list [--engine-dir <raiz>]                                         # engines, variantes e compatibilidade com o TensorRT
```

Método do orçamento de memória: [docs/orcamento.md](docs/orcamento.md).

## Desenvolvimento

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
mypy
ruff check
ruff format --check src tests scripts examples
python -m build --wheel
```

- Testes de TensorRT ficam em `tests/jetson/` (marca `jetson`) e são pulados sem `tensorrt`. Na placa: `pip install -e ".[trt,cpu]" -r requirements-dev.txt` e `pytest -m jetson`.
- Equivalência de outro modelo: `P2_ENGINE_DIR=<raiz> P2_EQUIVALENCE_ENGINE=app-epi-1.0.0 P2_EQUIVALENCE_IMAGES=<pasta> P2_EQUIVALENCE_REPORT=reports/equivalencia/app-epi-1.0.0.md pytest -m jetson tests/jetson/test_equivalence.py`
- Os testes do yolo11n precisam de `models/yolo11n.onnx` (fora do git); o pacote é montado numa pasta temporária com o `metadata.json` de `tests/fixtures/yolo11n-1.0.0/`; sem o arquivo, são pulados.
- Specs de desenvolvimento em [specs/INDEX.md](specs/INDEX.md).
