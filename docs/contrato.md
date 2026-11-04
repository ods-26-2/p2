# Contrato P2 1.1

Runtime de aceleração P2 · biblioteca `p2` 0.2.x · 04/11/2026

Este documento detalha os tipos e erros da API fechada do P2 na Arquitetura 6c (item 8 da seção 10, "SIS-1 detalha tipos e erros"). A assinatura das quatro funções não muda. O que está aqui é o comportamento que os componentes consumidores podem assumir e o que o MOD-1 e o B5 precisam entregar no pacote MODEL.

- Público: componentes consumidores (I1, I2, I4, I5, I7, I8), MOD-1 e B5 (entregam o pacote MODEL), B1 (registro de contratos)
- Mudanças depois do envio geram uma nova versão deste documento
- A versão 1.0 (lib 0.1.x) está na tag `v0.1.0`; as diferenças estão na seção 8
- Decisões de origem: [decisoes.md](decisoes.md)

> **Situação:** contrato da lib 0.2.0, em desenvolvimento. A pré-release `0.2.0a1` já traz os tipos desta versão (`p2.TensorArray`), mas o comportamento ainda é o da 0.1.0: modelos estáticos, `float32`, um engine por modelo. Até a 0.2.0, programe contra este documento e teste com modelos estáticos.

## 1. Instalação

| Quando | Comando |
| --- | --- |
| Release 0.1.0 (contrato 1.0) | `pip install https://github.com/ods-26-2/p2/releases/download/v0.1.0/p2-0.1.0-py3-none-any.whl` |
| Pré-release 0.2.0a1 (tipos da 1.1, comportamento da 1.0) | `pip install https://github.com/ods-26-2/p2/releases/download/v0.2.0a1/p2-0.2.0a1-py3-none-any.whl` |
| Release 0.2.0 (este contrato) | `pip install https://github.com/ods-26-2/p2/releases/download/v0.2.0/p2-0.2.0-py3-none-any.whl` |
| Na Jetson | wheel com o extra `trt`; o `tensorrt` vem do JetPack 6.2.1 (TensorRT 10.3.0), não do pip |
| No notebook | wheel com o extra `cpu` (ONNX Runtime) |

- Python 3.10; dependência obrigatória `numpy>=1.21.6,<2`
- `import p2`

## 2. API

```python
import p2

def load(engine_ref: str, options: dict[str, str] | None = None) -> p2.Handle: ...
def infer(
    handle: p2.Handle,
    input: np.ndarray | dict[str, np.ndarray],
) -> tuple[dict[str, p2.TensorArray], p2.InferStats]: ...
def info(handle: p2.Handle) -> p2.EngineInfo: ...
def release(handle: p2.Handle) -> None: ...
```

`p2.TensorArray` é `numpy.typing.NDArray[np.generic]`, porque a saída pode ser `float32`, `int32`, `int64` ou `bool`. Quem usa um type checker converte para o tipo esperado sem copiar: `np.asarray(out["output0"], dtype=np.float32)`.

| Função | Entrada | Retorno | Erros (`P2Error.code`) |
| --- | --- | --- | --- |
| `load` | `engine_ref`; `options` com `precision` | `Handle` | `ENGINE_NOT_FOUND`, `INCOMPATIBLE_VERSION`, `OUT_OF_MEMORY` |
| `infer` | `Handle`; array solto ou `{nome: array}` | `(saídas, stats)` | `INVALID_INPUT_SHAPE`, `RUNTIME_ERROR` |
| `info` | `Handle` | `EngineInfo` | — |
| `release` | `Handle` | `None` | — |

### 2.1 `load`

- `engine_ref`: `"<artifact_id>-<artifact_version>"`, por exemplo `"app-epi-1.0.0"` ou `"whisper-small-encoder-1.0.0"`
- `options`: `None` ou um dicionário só com a chave `"precision"`, valor `"FP16"` ou `"FP32"`
  - Sem `precision`: FP16 na Jetson; FP32 na CPU (o `info` informa a precisão usada)
  - `"FP16"` explícito na CPU gera `ENGINE_NOT_FOUND`
- O engine é procurado em `$P2_ENGINE_DIR/<engine_ref>/`; padrão `/opt/ods/engines/<engine_ref>/`
- Backend escolhido automaticamente: TensorRT quando disponível, senão CPU com ONNX Runtime. `P2_BACKEND=trt` ou `P2_BACKEND=ort` força um deles
- Em modelo com forma dinâmica, a memória da GPU é reservada no `load` para a forma máxima do perfil (seção 5.3)

### 2.2 `infer`

**Entrada**

- Array solto (modelo com uma entrada) ou `{nome: array}` com os nomes de `info()["input_shape"]`. Com mais de uma entrada, o dicionário é obrigatório
- `dtype` igual ao declarado no manifesto da entrada (`float32`, `int32`, `int64` ou `bool`). Não há conversão automática
- Forma:
  - Eixo fixo: exatamente o valor do `info`, incluindo o lote
  - Eixo dinâmico (`-1` no `info`): qualquer valor entre `min` e `max` do perfil
- Nome, forma ou tipo fora disso gera `INVALID_INPUT_SHAPE`
- Array não contíguo é aceito e copiado internamente

**Saída**

- `saídas`: `{nome: np.ndarray}` com os nomes de `info()["output_shape"]`
  - Cada array vem no `dtype` do tensor de saída
  - Em modelo dinâmico, a forma é calculada a cada chamada. Exemplo: lote 4 na entrada → lote 4 na saída
  - Os arrays são cópias e podem ser guardados ou alterados
- `stats`:

| Campo | Tipo | Significado |
| --- | --- | --- |
| `latency_ms` | `float` | Tempo total da chamada, incluindo as cópias entre CPU e GPU |
| `gpu_memory_mb` | `float` | Memória do handle (pesos, ativações e buffers), calculada no `load` e constante por handle; em modelo dinâmico, reflete a forma máxima; `0.0` no backend de CPU |

A 6c diz que `gpu_memory_mb` é "medido na chamada". O valor devolvido é a memória do handle no momento da chamada, que não muda depois do `load`. Ele não inclui o contexto CUDA nem as bibliotecas do processo: o orçamento do container vem da medição com `p2-bench` (`docs/orcamento.md`).

### 2.3 `info`

```python
class EngineInfo(TypedDict):
    engine_version: str                      # "<engine_ref>-fp16" ou "<engine_ref>-fp32"
    input_shape: dict[str, list[int]]        # -1 nos eixos dinâmicos
    output_shape: dict[str, list[int]]       # -1 nos eixos que dependem da entrada
    precision: Literal["FP16", "FP32"]
    max_batch: int                           # primeira dimensão de max; 1 nos modelos de lote fixo 1
```

- `engine_version` é o valor a publicar como `artifact_version` nos eventos do componente
- Continuam os cinco campos da 6c. Os limites `min`/`opt`/`max` e o `dtype` de cada tensor ficam no `manifest.json` (seção 7) e no `metadata.json` que o próprio grupo entregou
- `max_batch` é a maior primeira dimensão entre as entradas: `max[0]` do perfil, ou a forma fixa. Tensores escalares ficam de fora. Sem nenhuma entrada com eixo, vale 1

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
- Handles diferentes são independentes, inclusive as etapas de um mesmo modelo
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
| `load` | `ENGINE_NOT_FOUND` | Pasta do `engine_ref` inexistente, manifesto ausente ou inválido, variante de precisão não compilada, saída com forma dependente dos dados |
| `load` | `INCOMPATIBLE_VERSION` | Versão do TensorRT no manifesto diferente da instalada |
| `load` | `OUT_OF_MEMORY` | Falha de alocação ao desserializar o engine ou criar os buffers (na forma máxima) |
| `infer` | `INVALID_INPUT_SHAPE` | Nome, `dtype` ou forma da entrada fora do declarado (forma fixa ou intervalo do perfil) |
| `infer` | `RUNTIME_ERROR` | Falha na cópia ou na execução |

Erros de uso, fora desses cinco casos, saem como `ValueError`: opção desconhecida em `options`, precisão inválida, handle usado depois do `release`.

## 4. Modelos por grupo

Cada etapa de um modelo é um `engine_ref` próprio, com pacote MODEL próprio. O laço entre as etapas fica no componente consumidor; o P2 só executa cada engine.

| Grupo | Modelo | Etapas (`engine_ref`) | O que usa da 1.1 |
| --- | --- | --- | --- |
| I1 | YOLO de EPI | `app-epi-1.0.0` | Nada novo: igual à 1.0 |
| I2 | ReID | `<reid>-<versão>` | Lote dinâmico: N recortes de pessoa numa chamada |
| I4 | MediaPipe (pose, mãos, rosto) | um por submodelo, depois da conversão para ONNX | `task` sem classes; vários `engine_ref` |
| I5 | Whisper | `whisper-<tam>-encoder-<versão>`, `whisper-<tam>-decoder-<versão>` | Duas etapas; tokens `int64` com comprimento dinâmico; decodificação token a token no I5 |
| I5 | VAD | `<vad>-<versão>` | Entrada escalar `int64` (taxa de amostragem); áudio com comprimento dinâmico |
| I7 | OCR em dois estágios | `<detector>-<versão>`, `<reconhecedor>-<versão>` | Duas etapas; recorte da placa no I7; lote dinâmico de recortes |
| I8 | Modelos sem classes | `<modelo>-<versão>` | `task` diferente de `detect`, `classes` opcional |

### 4.1 Exemplo: decoder do Whisper

```python
import numpy as np
import p2

encoder = p2.load("whisper-tiny-encoder-1.0.0")
decoder = p2.load("whisper-tiny-decoder-1.0.0")

p2.info(decoder)["input_shape"]
# {"tokens": [1, -1], "audio_features": [1, 1500, 384]}

features = p2.infer(encoder, {"mel": mel})[0]["audio_features"]
tokens = np.array([[50258, 50259, 50359]], dtype=np.int64)   # forma [1, 3], dentro de [1, 1]..[1, 448]
for _ in range(MAX_TOKENS):
    out, _ = p2.infer(decoder, {"tokens": tokens, "audio_features": features})
    logits = np.asarray(out["logits"], dtype=np.float32)      # forma [1, tokens.shape[1], 51865]
    next_token = int(logits[0, -1].argmax())
    if next_token == END_OF_TEXT:
        break
    tokens = np.concatenate([tokens, [[next_token]]], axis=1)

p2.release(decoder)
p2.release(encoder)
```

Os nomes dos tensores e as formas são ilustrativos; os reais vêm do export do B5 (seção 9). Com cache KV, o decoder tem mais entradas e saídas, e o laço passa o cache de uma chamada para a próxima.

### 4.2 Exemplo: lote de recortes (ReID)

```python
h = p2.load("reid-1.0.0")
p2.info(h)["input_shape"]      # {"images": [-1, 3, 256, 128]}
p2.info(h)["max_batch"]        # 16

crops = np.stack(person_crops).astype(np.float32)   # forma [4, 3, 256, 128]
out, _ = p2.infer(h, crops)
out["embeddings"].shape        # (4, 512)
```

## 5. Exemplos com o `app-epi-1.0.0`

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
        ...  # erro do chamador: conferir p2.info(h)["input_shape"], o perfil e o dtype
    else:
        ...  # RUNTIME_ERROR: falha na execução
finally:
    p2.release(h)
```

## 6. Pacote MODEL (entregue pelo MOD-1 e pelo B5)

O `p2-build` lê uma pasta com o `metadata.json` e o `.onnx`. O `descriptor.json` não é lido; `metrics` e `dataset_version` ficam com B5 e com o consumidor.

| Campo | Tipo | Obrigatório | Exemplo | Uso e validação |
| --- | --- | --- | --- | --- |
| `artifact_id` | texto | sim | `"app-epi"` | Forma o `engine_ref` |
| `artifact_version` | texto | sim | `"1.0.0"` | Forma o `engine_ref` |
| `onnx_file` | texto | sim | `"app-epi.onnx"` | Arquivo a compilar, relativo à pasta do pacote |
| `hash_sha256` | texto | sim | `"3da7aa2a…63122"` | Deve bater com o sha256 do `onnx_file` |
| `precision` | texto | sim | `"FP32"` | Precisão do ONNX; deve ser `"FP32"` |
| `task` | texto | não (padrão `"detect"`) | `"speech-decoder"` | Um de `detect`, `classify`, `segment`, `ocr`, `reid`, `speech-encoder`, `speech-decoder`, `vad`, `other`; define as validações abaixo |
| `input_shape` | lista ou `{nome: lista}` | sim | `[1, 3, 640, 640]` ou `{"tokens": [1, -1], "audio_features": [1, 1500, 384]}` | Igual às entradas do ONNX; `-1` nos eixos dinâmicos; `[]` para escalar. Lista só com uma entrada |
| `dynamic_shapes` | `{nome: {min, opt, max}}` | só com eixo dinâmico | `{"tokens": {"min": [1, 1], "opt": [1, 64], "max": [1, 448]}}` | Limites do perfil de otimização (seção 6.1) |
| `classes` | lista de nomes | só em `detect` | `["pessoa", "capacete", "colete"]` | Em `detect`, a saída deve ter `4 + len(classes)` linhas (eixo 1) |

### 6.1 `dynamic_shapes`

- Uma entrada por tensor que tem pelo menos um eixo dinâmico no ONNX. Todo eixo dinâmico precisa de limite
- `min`, `opt` e `max` têm o mesmo número de eixos da entrada
- Nos eixos fixos, repetem o valor fixo
- Nos eixos dinâmicos: `0 < min <= opt <= max`
- `opt` é a forma em que o TensorRT otimiza e a que o `p2-bench` mede; use a forma mais frequente em produção
- `max` define a memória reservada no `load`: escolha o menor valor que atende o caso de uso
- Perfil em entrada sem eixo dinâmico, ou entrada inexistente, é recusado

### 6.2 Validações por `task`

| `task` | Entradas e saídas | `classes` |
| --- | --- | --- |
| `detect` | 1 entrada e 1 saída; saída com `4 + len(classes)` linhas | obrigatório |
| demais | quantas o modelo tiver | opcional (não validado) |

### 6.3 Requisitos do ONNX

- ONNX IR versão 10 ou menor (o export do Ultralytics gera IR 7, opset 12); IR mais novo não carrega no ONNX Runtime 1.23 nem no parser do TensorRT 10.3
- Entradas e saídas com `dtype` `float32`, `int32`, `int64` ou `bool`; qualquer outro (por exemplo `float16` ou `uint8` na I/O) é recusado
- Formas declaradas em todas as entradas e saídas. Eixos simbólicos (`dim_param`) contam como dinâmicos
- Formas de saída que dependem dos dados (por exemplo `NonZero`) não são suportadas
- Um perfil de otimização por engine

Qualquer divergência nos campos acima, no hash, na forma, no perfil ou no número de classes faz o `p2-build` parar com código de saída 1 e uma mensagem em português, sem gravar nada. Exemplos:

- `hash diferente: esperado 3da7…, obtido 91c2…`
- `saída deve ter 7 linhas (4 + 3 classes), obtido forma [1, 84, 8400]`
- `entrada 'tokens' tem eixo dinâmico [1, -1]; declare min/opt/max em dynamic_shapes`

### 6.4 Exemplo: pacote do decoder do Whisper

```json
{
  "artifact_id": "whisper-tiny-decoder",
  "artifact_version": "1.0.0",
  "onnx_file": "decoder.onnx",
  "hash_sha256": "…",
  "precision": "FP32",
  "task": "speech-decoder",
  "input_shape": { "tokens": [1, -1], "audio_features": [1, 1500, 384] },
  "dynamic_shapes": { "tokens": { "min": [1, 1], "opt": [1, 64], "max": [1, 448] } }
}
```

## 7. Pasta do engine e `manifest.json`

O `p2-build` grava uma pasta por `engine_ref`, que o `load` usa para achar e validar o engine:

```
/opt/ods/engines/app-epi-1.0.0/
├─ app-epi.onnx
├─ metadata.json
├─ fp16.engine
├─ fp32.engine
└─ manifest.json
```

**Modelo estático só com `float32`: `schema_version: 1`**, idêntico ao da 1.0. A lib 0.1.0 continua lendo essas pastas.

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

**Modelo com eixo dinâmico ou tensor que não é `float32`: `schema_version: 2`.**

- Eixo dinâmico aparece como `-1` em `shape`
- Toda entrada dinâmica tem o bloco `profile`, copiado do `dynamic_shapes`
- Escalares têm `shape: []`

```json
{
  "schema_version": 2,
  "engine_ref": "whisper-tiny-decoder-1.0.0",
  "inputs": {
    "tokens": {
      "shape": [1, -1], "dtype": "int64",
      "profile": { "min": [1, 1], "opt": [1, 64], "max": [1, 448] }
    },
    "audio_features": { "shape": [1, 1500, 384], "dtype": "float32" }
  },
  "outputs": { "logits": { "shape": [1, -1, 51865], "dtype": "float32" } },
  "max_batch": 1,
  "…": "source, variants, platform e build como na versão 1"
}
```

- Com `p2-build --cpu-only` (notebook), a pasta tem só o `.onnx`, o `metadata.json` e o `manifest.json`, com `variants` e `platform` vazios (`{}`); o backend de CPU executa o `onnx_file`
- A pasta é gravada inteira ou não é gravada; uma pasta existente só é substituída com `--force`
- O container do consumidor monta `/opt/ods/engines` como somente leitura

## 8. Mudanças desde a 1.0

| Tema | 1.0 (lib 0.1.x) | 1.1 (lib 0.2.x) |
| --- | --- | --- |
| Assinaturas | `load`, `infer`, `info`, `release` | Iguais. O retorno de `infer` é anotado como `dict[str, p2.TensorArray]` |
| Tipo dos tensores | Só `float32` | `float32`, `int32`, `int64`, `bool`, por tensor |
| Forma | Estática | Estática ou dinâmica, com perfil min/opt/max |
| Lote | 1 | Fixo ou dinâmico (`max_batch` = `max[0]`) |
| Modelos com várias etapas | — | Um `engine_ref` por etapa |
| `metadata.json` | `classes` obrigatório; `input_shape` em lista | `task`; `classes` só em `detect`; `input_shape` em lista ou dicionário; `dynamic_shapes` |
| `manifest.json` | `schema_version: 1` | v1 para modelo estático `float32`; v2 com `-1`, `profile` e dtypes |
| Compatibilidade | — | Manifesto v1 carrega igual; pacote sem `task` = `detect`; YOLO estático sem nenhuma diferença |

## 9. Pré-requisitos fora do P2

| Com quem | O quê |
| --- | --- |
| B5 e VER-4 | Exportar encoder e decoder do Whisper em ONNX separados e decidir se o decoder usa cache KV: com cache, mais entradas e mais velocidade |
| B5 e INV-2 | Converter cada submodelo do MediaPipe (pose, mãos, rosto) para ONNX; cada um vira uma etapa |
| MOD-1 e B5 | Combinar os campos novos do pacote MODEL (`task`, `dynamic_shapes`, `input_shape` em dicionário) |
| Cada consumidor | Medir cada etapa com `p2-bench`: com dois ou mais engines no mesmo processo, o orçamento é a soma das etapas mais o custo fixo do processo |

## 10. Fora desta versão

INT8, vários perfis de otimização por engine e tensores que ficam na GPU entre chamadas (por exemplo o cache KV do decoder) ficam para depois ([depois.md](depois.md)). Nenhum deles muda as assinaturas acima.
