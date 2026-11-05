# Guia rápido do P2

Como instalar a lib `p2`, compilar um modelo e chamá-lo em Python. Para um exemplo completo, com pré e pós-processamento YOLO, leia [examples/example.py](../examples/example.py). Tipos, erros e formatos estão em [contrato.md](contrato.md).

## 1. Instalar

**Jetson** (JetPack 6.2.1). O TensorRT vem do JetPack, então a venv precisa enxergar os pacotes do sistema:

```bash
/usr/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install "p2[trt] @ https://github.com/ods-26-2/p2/releases/download/v0.1.0/p2-0.1.0-py3-none-any.whl"
```

**Notebook** (CPU, Python 3.10):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install "p2[cpu] @ https://github.com/ods-26-2/p2/releases/download/v0.1.0/p2-0.1.0-py3-none-any.whl"
```

Para conferir a Jetson, rode `python3 scripts/check_jetson.py` a partir de um clone do repositório. Ele precisa terminar com `0 de 5 verificações falharam`.

## 2. Compilar o ONNX

O `p2-build` recebe a pasta do pacote MODEL, que contém o `metadata.json` e o `.onnx`. Ele grava `<raiz>/<artifact_id>-<artifact_version>/`, por exemplo `app-plates-1.0.0/`.

```bash
export P2_ENGINE_DIR=~/p2-engines        # raiz dos engines; padrão /opt/ods/engines

p2-build pacote-app-plates/              # Jetson: compila FP16 e FP32 no TensorRT
p2-build pacote-app-plates/ --cpu-only   # notebook: prepara a pasta para rodar na CPU
p2-list                                  # lista os engines e a compatibilidade
```

- Opções:
  - `--precision FP16` compila só uma precisão.
  - `--force` substitui uma pasta existente.
  - `--engine-dir <raiz>` troca a raiz dos engines.
- Na Jetson, a compilação leva alguns minutos e termina com um `infer` de teste em cada variante.
- Hash, forma ou classes divergentes param o build com uma mensagem em português, sem gravar nada.

## 3. Usar em Python

```python
import numpy as np
import p2

with p2.load("app-plates-1.0.0") as h:      # FP16 na Jetson, FP32 na CPU
    info = p2.info(h)
    # {"engine_version": "app-plates-1.0.0-fp16", "input_shape": {"images": [1, 3, 640, 640]},
    #  "output_shape": {"output0": [1, 8, 8400]}, "precision": "FP16", "max_batch": 1}

    x = np.zeros((1, 3, 640, 640), dtype=np.float32)   # entrada já pré-processada (RGB, /255, CHW)
    out, stats = p2.infer(h, x)                         # ou p2.infer(h, {"images": x})
    out["output0"].shape                                # (1, 8, 8400)
    stats                                               # {"latency_ms": ..., "gpu_memory_mb": ...}
```

- `load(engine_ref, {"precision": "FP32"})` escolhe a precisão.
- Sem `with`, libere a memória com `p2.release(h)`.
- O backend é escolhido sozinho: TensorRT na Jetson, ONNX Runtime no notebook. `P2_BACKEND=trt` ou `P2_BACKEND=ort` força um deles.
- Publique `info["engine_version"]` como `artifact_version` dos eventos.

**Erros:** todos saem como `p2.P2Error`, com `e.code` igual a um dos códigos abaixo.

```python
try:
    h = p2.load("app-plates-1.0.0")
except p2.P2Error as e:
    print(e.code, e.detail)
```

| Código | Causa |
| --- | --- |
| `ENGINE_NOT_FOUND` | Modelo não compilado |
| `INCOMPATIBLE_VERSION` | TensorRT mudou; recompile com `p2-build --force` |
| `OUT_OF_MEMORY` | Memória insuficiente |
| `INVALID_INPUT_SHAPE` | Entrada com nome, forma ou dtype errado |
| `RUNTIME_ERROR` | Falha na execução |

## 4. Rodar os testes

Os testes ficam no repositório, não no wheel:

```bash
git clone https://github.com/ods-26-2/p2.git && cd p2
pip install -r requirements-dev.txt          # Jetson: pip install -e ".[trt,cpu]" -r requirements-dev.txt
pytest                                       # testes de CPU (qualquer máquina)
pytest -m jetson                             # testes de TensorRT (só na Jetson)
```

Os testes do yolo11n precisam de `models/yolo11n.onnx`. Sem esse arquivo, são pulados.
