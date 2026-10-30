# Guia de integração do P2

Passo a passo para as equipes de inferência (I1 e as próximas) usarem o P2 na Jetson Orin Nano. Os tipos e erros estão no [contrato](contrato.md); este guia mostra como montar o container e o código.

## 1. O que o P2 faz e o que fica com você

| P2 faz | Você faz |
| --- | --- |
| Carregar o engine certo para o `engine_ref` e a precisão | Ler a câmera ou a imagem |
| Executar o modelo na GPU (ou na CPU no notebook) | **BGR → RGB**, letterbox, normalização (pré-processamento) |
| Validar nome, forma e tipo da entrada | Decodificar a saída, limiar, NMS, nomes das classes (pós-processamento) |
| Informar `engine_version`, latência e memória do handle | Publicar eventos com `artifact_version` |

O [`examples/yolo_demo.py`](../examples/yolo_demo.py) tem o pré e o pós-processamento completos de um YOLO de detecção; copie dele.

## 2. Na placa: preparar os engines

Feito uma vez por modelo e por versão do TensorRT, fora do container de I1:

```bash
pip install p2-0.1.0-py3-none-any.whl            # no Python do sistema da Jetson, com o TensorRT do JetPack
sudo p2-build <pasta-do-pacote-MODEL>            # grava /opt/ods/engines/<engine_ref>/
p2-list                                          # confere variantes e compatibilidade
```

Se o JetPack ou o TensorRT mudar, o `p2-list` mostra "incompatível" e o `load` gera `INCOMPATIBLE_VERSION`: rode `p2-build --force` de novo.

## 3. No container de I1

**Dockerfile:**

```dockerfile
# FROM: imagem base com TensorRT 10.3 e CUDA 12.6 — pendente com P3
FROM <imagem-base-P3>

RUN pip install "p2[trt] @ https://github.com/ods-26-2/p2/releases/download/v0.1.0/p2-0.1.0-py3-none-any.whl"
```

- O `tensorrt` vem da imagem base (JetPack), não do pip.
- `numpy` precisa ficar em `>=1.21.6,<2`; o wheel já declara isso.

**Montagem dos engines** (somente leitura):

```bash
docker run --runtime nvidia -v /opt/ods/engines:/opt/ods/engines:ro <imagem-de-I1>
```

- Se montar em outro caminho, defina `P2_ENGINE_DIR` com esse caminho.
- Limite de memória do container: o orçamento em [orcamento.md](orcamento.md) (provisório 1.024 MB até a medição).

## 4. Código

```python
import cv2
import numpy as np
import p2

ENGINE_REF = "app-epi-1.0.0"

handle = p2.load(ENGINE_REF)                       # FP16 na Jetson
info = p2.info(handle)
artifact_version = info["engine_version"]          # "app-epi-1.0.0-fp16": publique este valor
size = info["input_shape"]["images"][2]            # 640

frame_bgr = cv2.imread("quadro.jpg")               # OpenCV entrega BGR
tensor = preprocess(frame_bgr, size)               # BGR→RGB, letterbox 114, /255, CHW, lote 1, float32
outputs, stats = p2.infer(handle, tensor)
detections = postprocess(outputs["output0"], ...)  # [1, 7, 8400] → caixas, classes, confianças

p2.release(handle)
```

- `preprocess` e `postprocess`: use `letterbox` e `postprocess` do `examples/yolo_demo.py`.
- Um handle por modelo, carregado uma vez no início; não faça `load` a cada quadro.
- O mesmo handle pode ser usado por várias threads: as chamadas rodam em sequência.
- Prefira `with p2.load(...) as handle:` quando o handle tiver vida curta.

> **BGR → RGB.** O OpenCV lê em BGR e o modelo foi treinado em RGB. Esquecer a conversão não gera erro nenhum: só derruba a acurácia em silêncio. O `letterbox` do exemplo já faz a conversão.

## 5. Erros

```python
try:
    handle = p2.load(ENGINE_REF)
except p2.P2Error as e:
    match e.code:
        case "ENGINE_NOT_FOUND":       # engine não instalado: rodar p2-build na placa
            ...
        case "INCOMPATIBLE_VERSION":   # TensorRT mudou: p2-build --force
            ...
        case "OUT_OF_MEMORY":          # outro processo usando a GPU, ou orçamento estourado
            ...
    raise
```

| `code` | Ação sugerida em I1 |
| --- | --- |
| `ENGINE_NOT_FOUND` | Falha de implantação: alertar e não iniciar |
| `INCOMPATIBLE_VERSION` | Falha de implantação: recompilar os engines |
| `OUT_OF_MEMORY` | Tentar de novo após liberar memória; se persistir, rever o orçamento |
| `INVALID_INPUT_SHAPE` | Erro de programação: conferir `p2.info(handle)["input_shape"]` e `float32` |
| `RUNTIME_ERROR` | Falha na GPU: registrar, descartar o quadro e seguir; se repetir, recarregar o handle |

`ValueError` indica uso errado da API (opção desconhecida, handle usado depois do `release`).

## 6. Testar sem a placa

No notebook, o mesmo código roda na CPU:

```bash
pip install "p2[cpu] @ https://github.com/ods-26-2/p2/releases/download/v0.1.0/p2-0.1.0-py3-none-any.whl"
p2-build <pasta-do-pacote-MODEL> --cpu-only --engine-dir ~/engines
P2_ENGINE_DIR=~/engines python examples/yolo_demo.py app-epi-1.0.0 quadro.jpg --out anotada.jpg
```

Na CPU só existe FP32: `engine_version` termina em `-fp32` e `gpu_memory_mb` é `0.0`.

## 7. Checklist da primeira build

- [ ] `p2-list` na placa mostra o `engine_ref` como compatível
- [ ] Container sobe com `/opt/ods/engines` montado e `p2.load` funciona
- [ ] `info()["engine_version"]` termina em `-fp16`
- [ ] Imagem de teste dá as mesmas detecções do `yolo_demo.py`
- [ ] Eventos publicados com `artifact_version` igual ao `engine_version`
- [ ] Memória do container dentro do orçamento
