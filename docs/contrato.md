# Contrato P2 1.0

Runtime de aceleração P2 · biblioteca `p2` 0.1.x · 30/10/2026

Este documento detalha os tipos e erros da API fechada do P2 na Arquitetura 6c (item 8 da seção 10, "SIS-1 detalha tipos e erros"). A assinatura das quatro funções não muda; o que está aqui é o comportamento que I1 pode assumir e o que o MOD-1 precisa entregar no pacote MODEL.

- Público: I1 (consome a biblioteca), MOD-1 (entrega o pacote MODEL), B1 (registro de contratos)
- Mudanças depois do envio geram uma nova versão deste documento
- Decisões de origem: [decisoes.md](decisoes.md)

## 1. Instalação

| Quando | Comando |
| --- | --- |
| Release 0.1.0 | `pip install https://github.com/ods-26-2/p2/releases/download/v0.1.0/p2-0.1.0-py3-none-any.whl` |
| Antes da 0.1.0, no notebook | `pip install "p2[cpu] @ git+https://github.com/ods-26-2/p2.git"` |
| Na Jetson | wheel com o extra `trt`; o `tensorrt` vem do JetPack 6.2.1 (TensorRT 10.3.0), não do pip |

- Python 3.10; dependência obrigatória `numpy>=1.21.6,<2`
- `import p2`

## 2. API

```python
import p2

def load(engine_ref: str, options: dict[str, str] | None = None) -> p2.Handle: ...
def infer(
    handle: p2.Handle,
    input: np.ndarray | dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], p2.InferStats]: ...
def info(handle: p2.Handle) -> p2.EngineInfo: ...
def release(handle: p2.Handle) -> None: ...
```

| Função | Entrada | Retorno | Erros (`P2Error.code`) |
| --- | --- | --- | --- |
| `load` | `engine_ref`; `options` com `precision` | `Handle` | `ENGINE_NOT_FOUND`, `INCOMPATIBLE_VERSION`, `OUT_OF_MEMORY` |
| `infer` | `Handle`; array solto ou `{nome: array}` | `(saídas, stats)` | `INVALID_INPUT_SHAPE`, `RUNTIME_ERROR` |
| `info` | `Handle` | `EngineInfo` | — |
| `release` | `Handle` | `None` | — |

### 2.1 `load`

- `engine_ref`: `"<artifact_id>-<artifact_version>"`, por exemplo `"app-epi-1.0.0"`
- `options`: `None` ou um dicionário só com a chave `"precision"`, valor `"FP16"` ou `"FP32"`
  - Sem `precision`: FP16 na Jetson; FP32 na CPU (o `info` informa a precisão usada)
  - `"FP16"` explícito na CPU gera `ENGINE_NOT_FOUND`
- O engine é procurado em `$P2_ENGINE_DIR/<engine_ref>/`; padrão `/opt/ods/engines/<engine_ref>/`
- Backend escolhido automaticamente (TensorRT quando disponível, senão CPU com ONNX Runtime); `P2_BACKEND=trt` ou `P2_BACKEND=ort` força um deles

### 2.2 `infer`

**Entrada**

- Array solto (o modelo tem uma entrada) ou `{nome: array}` com os nomes de `info()["input_shape"]`
- `dtype` `float32` e forma exatamente igual à do `info`, incluindo o lote 1
- Nome, forma ou tipo diferente gera `INVALID_INPUT_SHAPE`; não há conversão automática de tipo
- Array não contíguo é aceito e copiado internamente

**Saída**

- `saídas`: `{nome: np.ndarray}` com os nomes e formas de `info()["output_shape"]`; os arrays são cópias e podem ser guardados ou alterados
- `stats`:

| Campo | Tipo | Significado |
| --- | --- | --- |
| `latency_ms` | `float` | Tempo total da chamada, incluindo as cópias entre CPU e GPU |
| `gpu_memory_mb` | `float` | Memória do handle (pesos, ativações e buffers), calculada no `load` e constante por handle; `0.0` no backend de CPU |

A 6c diz que `gpu_memory_mb` é "medido na chamada": o valor devolvido é a memória do handle no momento da chamada, que não muda depois do `load`. Ele não inclui o contexto CUDA nem as bibliotecas do processo; o orçamento do container vem da medição da E6 (`docs/orcamento.md`).

### 2.3 `info`

```python
class EngineInfo(TypedDict):
    engine_version: str                      # "<engine_ref>-fp16" ou "<engine_ref>-fp32"
    input_shape: dict[str, list[int]]
    output_shape: dict[str, list[int]]
    precision: Literal["FP16", "FP32"]
    max_batch: int                           # sempre 1 nesta versão
```

- `engine_version` é o valor a publicar como `artifact_version` nos eventos de I1

### 2.4 `release` e ciclo de vida

- `release` libera a memória do handle; chamar de novo não faz nada
- `infer` ou `info` depois do `release` gera `ValueError`
- O handle também funciona como gerenciador de contexto:

```python
with p2.load("app-epi-1.0.0") as h:
    out, stats = p2.infer(h, x)
```

### 2.5 Concorrência

- Cada handle tem uma trava: chamadas simultâneas ao mesmo handle rodam em sequência, sem erro
- Handles diferentes são independentes
- Chamada síncrona: `infer` só retorna com as saídas prontas

## 3. Erros

```python
class P2Error(Exception):
    code: p2.ErrorCode      # compara igual à string: e.code == "ENGINE_NOT_FOUND"
    detail: str             # explicação em português
# str(e) == f"{e.code}: {e.detail}", por exemplo "ENGINE_NOT_FOUND: pasta /opt/ods/engines/app-epi-1.0.0 não existe"
```

| Função | `code` | Quando ocorre |
| --- | --- | --- |
| `load` | `ENGINE_NOT_FOUND` | Pasta do `engine_ref` inexistente, manifesto ausente ou variante de precisão não compilada |
| `load` | `INCOMPATIBLE_VERSION` | Versão do TensorRT no manifesto diferente da instalada |
| `load` | `OUT_OF_MEMORY` | Falha de alocação ao desserializar o engine ou criar os buffers |
| `infer` | `INVALID_INPUT_SHAPE` | Nome, forma ou tipo da entrada diferente do declarado em `info` |
| `infer` | `RUNTIME_ERROR` | Falha na cópia ou na execução |

Erros de uso, fora desses cinco casos, saem como `ValueError`: opção desconhecida em `options`, precisão inválida, handle usado depois do `release`.

## 4. Exemplos com o `app-epi-1.0.0`

```python
import numpy as np
import p2

h = p2.load("app-epi-1.0.0")

p2.info(h)
# {
#     "engine_version": "app-epi-1.0.0-fp16",
#     "input_shape": {"images": [1, 3, 640, 640]},
#     "output_shape": {"output0": [1, 7, 8400]},
#     "precision": "FP16",
#     "max_batch": 1,
# }

x = np.zeros((1, 3, 640, 640), dtype=np.float32)
out, stats = p2.infer(h, x)
out["output0"].shape        # (1, 7, 8400)
stats                       # {"latency_ms": 8.3, "gpu_memory_mb": 46.2}

p2.release(h)
```

Os números de `stats` são ilustrativos; os reais saem da medição na placa.

A saída `[1, 7, 8400]` tem 4 linhas de caixa (`cx, cy, w, h`) e 3 de confiança, uma por classe do `metadata.json` (`pessoa`, `capacete`, `colete`). Pré-processamento (BGR→RGB, letterbox, normalização) e pós-processamento (limiar, NMS, nomes de classes) ficam com I1; o P2 só executa o modelo.

**Tratamento de erro:**

```python
import p2

try:
    h = p2.load("app-epi-1.0.0", {"precision": "FP16"})
except p2.P2Error as e:
    if e.code == "ENGINE_NOT_FOUND":
        ...  # engine não instalado na placa: rodar p2-build
    elif e.code == "INCOMPATIBLE_VERSION":
        ...  # TensorRT mudou: recompilar com p2-build --force
    elif e.code == "OUT_OF_MEMORY":
        ...  # memória insuficiente: liberar outros handles
    raise

try:
    out, stats = p2.infer(h, x)
except p2.P2Error as e:
    if e.code == "INVALID_INPUT_SHAPE":
        ...  # erro do chamador: conferir p2.info(h)["input_shape"] e dtype float32
    else:
        ...  # RUNTIME_ERROR: falha na execução
finally:
    p2.release(h)
```

## 5. Pacote MODEL (entregue pelo MOD-1)

O `p2-build` lê uma pasta com o `metadata.json` e o `.onnx`. O `descriptor.json` não é lido; `metrics` e `dataset_version` ficam com B5 e I1.

| Campo | Tipo | Exemplo (`app-epi`) | Uso e validação |
| --- | --- | --- | --- |
| `artifact_id` | texto | `"app-epi"` | Forma o `engine_ref` |
| `artifact_version` | texto | `"1.0.0"` | Forma o `engine_ref` |
| `onnx_file` | texto | `"app-epi.onnx"` | Arquivo a compilar, relativo à pasta do pacote |
| `hash_sha256` | texto | `"3da7aa2a…63122"` | Deve bater com o sha256 do `onnx_file` |
| `input_shape` | lista de inteiros positivos | `[1, 3, 640, 640]` | Deve ser igual à forma da entrada do ONNX |
| `classes` | lista de nomes | `["pessoa", "capacete", "colete"]` | A saída do ONNX deve ter `4 + len(classes)` linhas (eixo 1) |
| `precision` | texto | `"FP32"` | Precisão do ONNX; deve ser `"FP32"` |

**Requisitos do ONNX:**

- Uma entrada e uma saída
- Formas estáticas (exportado sem `dynamic=True`), lote 1
- ONNX IR versão 10 ou menor (o export do Ultralytics gera IR 7, opset 12); IR mais novo não carrega no ONNX Runtime 1.23 nem no parser do TensorRT 10.3
- Entrada e saída `float32` (exigido pela API; o `p2-build` registra o tipo no manifesto sem validá-lo)

Qualquer divergência nos campos acima, no hash, na forma ou no número de classes faz o `p2-build` parar com código de saída 1 e uma mensagem em português, sem gravar nada. Exemplos: `hash diferente: esperado 3da7…, obtido 91c2…`; `saída deve ter 7 linhas (4 + 3 classes), obtido forma [1, 84, 8400]`.

## 6. Pasta do engine e `manifest.json`

O `p2-build` grava uma pasta por `engine_ref`, que o `load` usa para achar e validar o engine:

```
/opt/ods/engines/app-epi-1.0.0/
├─ app-epi.onnx
├─ metadata.json
├─ fp16.engine
├─ fp32.engine
└─ manifest.json
```

```json
{
  "schema_version": 1,
  "engine_ref": "app-epi-1.0.0",
  "source": {
    "onnx_file": "app-epi.onnx",
    "sha256": "3da7aa2a6b50c1aa27333990aa4450a3ef34f2e9a3d05f9367bf1b9f6d363122",
    "metadata_file": "metadata.json"
  },
  "inputs":  { "images":  { "shape": [1, 3, 640, 640], "dtype": "float32" } },
  "outputs": { "output0": { "shape": [1, 7, 8400],     "dtype": "float32" } },
  "max_batch": 1,
  "variants": {
    "FP16": { "file": "fp16.engine", "engine_version": "app-epi-1.0.0-fp16", "sha256": "…" },
    "FP32": { "file": "fp32.engine", "engine_version": "app-epi-1.0.0-fp32", "sha256": "…" }
  },
  "platform": { "jetpack": "6.2.1", "l4t": "36.4.7", "tensorrt": "10.3.0", "cuda": "12.6" },
  "build": { "tool": "p2-build 0.1.0", "built_at": "…", "tf32": false, "workspace_mb": 1024 }
}
```

- Com `p2-build --cpu-only` (notebook), a pasta tem só o `.onnx`, o `metadata.json` e o `manifest.json`, com `variants` e `platform` vazios (`{}`); o backend de CPU executa o `onnx_file`
- A pasta é gravada inteira ou não é gravada; uma pasta existente só é substituída com `--force`
- O container de I1 monta `/opt/ods/engines` como somente leitura

## 7. Fora desta versão

INT8, forma dinâmica, lote maior que 1 e modelos com várias etapas ficam para depois ([depois.md](depois.md)). Nenhum deles muda as assinaturas acima: todos acrescentam variantes ao manifesto ou etapas novas.
