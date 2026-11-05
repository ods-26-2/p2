# 006 — E5: backend TensorRT e equivalência

- Estado: ativa
- Criada: 2026-10-30
- Fechada: —
- Base: 002 (compilador, T8), 003 (stubs), 004 (contrato), 005 (API, Protocols); plano-p2.md (E5, "Testes e verificação")
- Substituída por: —
- Tags: E5, tensorrt, backend, gpu, cuda, cudart, equivalencia, yolo, nms, jetson

## Objetivo
- `p2.load` usa TensorRT na Jetson com a mesma API da CPU; detecções equivalentes entre CPU, TRT FP32 e TRT FP16

## Fora de escopo
- `p2-bench`, `p2-list`, orçamento (E6); `examples/yolo_demo.py` (E7)

## Requisitos
- R1: backend automático = TRT se `tensorrt` e `cuda-python` importáveis, senão CPU; `P2_BACKEND=trt` sem as libs → `ENGINE_NOT_FOUND` com dica
- R2: TRT: precisão padrão FP16; variante ausente (ou pasta `--cpu-only`) → `ENGINE_NOT_FOUND`; `platform.tensorrt` ≠ instalado (3 componentes) → `INCOMPATIBLE_VERSION`
- R3: engine ausente ou sha256 ≠ manifesto → `ENGINE_NOT_FOUND`; desserialização, contexto, `cudaMalloc` ou stream falham → `OUT_OF_MEMORY`
- R4: um buffer de GPU por tensor de I/O, `set_tensor_address`, stream próprio; saídas no host pré-alocadas por handle
- R5: `memory_mb` = bytes do engine + `device_memory_size_v2` + soma dos buffers, em MB; constante por handle
- R6: `run`: `cudaMemcpyAsync` H→D, `execute_async_v3`, D→H, `cudaStreamSynchronize`; qualquer falha → `RUNTIME_ERROR`
- R7: `close`: `cudaFree` dos buffers, destrói stream, contexto e engine; idempotente
- R8: `p2-build` faz um `infer` com entrada zerada em cada variante antes de publicar a pasta (002/R3, 002/T8)
- R9: `tests/helpers/yolo.py`: letterbox 114 + BGR→RGB, decodificação `[1, 4+nc, N]`, NMS por classe, IoU
- R10: equivalência com a CPU FP32 como referência: TRT FP32 IoU ≥ 0,98 e Δconf ≤ 0,01; TRT FP16 IoU ≥ 0,95 e Δconf ≤ 0,03; detecções a ±0,03 do limiar ignoradas
- R11: relatório Markdown da equivalência quando `P2_EQUIVALENCE_REPORT` aponta um arquivo
- R12: caso 4 do yolo11n na CPU: `bus.jpg` tem `person` com confiança ≥ 0,5

## Regras
- Valem 001/RG1–RG3, 003/RG1, 005/RG1 e docs/contrato.md
- RG1: `p2.backends.trt` só é importado no `load` (005/RG1)

## Decisões
- D1 [assumida] seleção de variante e versão em `backends/variants.py` puro (sem `tensorrt`) — testável no notebook
- D2 [assumida] sha256 do engine conferido no `load` → `ENGINE_NOT_FOUND`; depois disso, falha de desserialização = `OUT_OF_MEMORY` — separa arquivo corrompido de falta de memória
- D3 [assumida] automático não cai para a CPU quando a pasta não tem engines (`--cpu-only`): `ENGINE_NOT_FOUND`; usar `P2_BACKEND=ort` — evita rodar na CPU sem perceber
- D4 [assumida] smoke do `p2-build` via `EngineCompiler.smoke_test(directory, manifest, precision)` usando o `TrtBackend` na pasta temporária — reusa o backend
- D5 [assumida] equivalência configurável por `P2_EQUIVALENCE_ENGINE` e `P2_EQUIVALENCE_IMAGES`; padrão: yolo11n compilado na hora + `assets/*.jp*g` — imagens de EPI não estão no repositório
- D6 [assumida] `opencv-python>=4.7,<4.12` em requirements-dev (só testes; `importorskip`) — decodificar JPEG
- D7 [assumida] limiar de confiança 0,25 e IoU do NMS 0,45 nos testes — padrões do Ultralytics
- D8 [assumida] stream passado ao TensorRT como `int(stream)` — handle numérico aceito por `execute_async_v3`
- D9 [assumida] `CollectingLogger` movido de `build/compiler.py` (002) para `backends/trt.py` — evita import circular com o smoke
- D10 [assumida] `set_tensor_address` falso → `ENGINE_NOT_FOUND` (engine sem o tensor do manifesto)
- D11 [assumida] `tests/jetson/` vira pacote (`__init__.py`) por causa de dois `conftest` no mypy; engines compilados uma vez por sessão

## Tarefas
- [x] T1: stubs: `cudart` (streams, MemcpyAsync, erro de alocação), TRT (contexto, execução, device memory)
- [x] T2: `backends/variants.py` + testes (R2, R3, R5, D1, D2)
- [x] T3: `backends/trt.py` (R2–R7, D8)
- [x] T4: `api.select_backend` automático/trt (R1, D3)
- [x] T5: smoke no `p2-build` (R8, D4) + testes com compilador falso
- [x] T6: `tests/helpers/yolo.py` + testes; caso 4 do yolo11n (R9, R12, D6, D7)
- [x] T7: `tests/jetson/test_trt.py`: ciclo na GPU, FP16/FP32, 5 códigos, `gpu_memory_mb > 0`
- [x] T8: `tests/jetson/test_equivalence.py` + relatório (R10, R11, D5)
- [ ] T9 (usuário, Jetson): rodar `pytest -m jetson`; gerar `reports/equivalencia/app-epi-1.0.0.md`

## Mapa de código

### typings/cuda/cudart.pyi, typings/tensorrt/__init__.pyi (de 003)
- + cudaStream_t, cudaStreamCreate/Synchronize/Destroy, cudaMemcpyAsync, cudaErrorMemoryAllocation; + IExecutionContext, create_execution_context, device_memory_size_v2 [T1]

### src/p2/backends/variants.py
- def select_variant(manifest, precision, installed_tensorrt: str) -> VariantInfo: sem engines / variante ausente → ENGINE_NOT_FOUND; versão ≠ → INCOMPATIBLE_VERSION [R2, D1]
- def read_engine(directory: Path, variant: VariantInfo) -> bytes: ausente ou sha256 ≠ → ENGINE_NOT_FOUND [R3, D2]
- def memory_mb(*, engine_bytes: int, device_memory: int, buffer_bytes: int) -> float [R5]

### src/p2/backends/trt.py
- class CollectingLogger(trt.ILogger): erros do TensorRT para mensagens [D9]
- class EngineParts (frozen, slots): runtime/engine/contexto/logger vivos juntos [R4] (→ 009: DeviceTensor → DeviceInput, DeviceOutput)
- class TrtSession: run (MemcpyAsync, execute_async_v3, sync) → RUNTIME_ERROR; close idempotente [R6, R7, D8]
- class TrtBackend: padrão FP16; open: variante, sha, desserializa, aloca, stream, bind; falha → OUT_OF_MEMORY e libera [R2–R5, D10]

### src/p2/api.py (de 005)
- select_backend: "" → TRT se importável, senão CPU; "trt" sem libs → ENGINE_NOT_FOUND [R1, D3]

### src/p2/build/output.py, src/p2/build/compiler.py (de 002)
- EngineCompiler.smoke_test(directory, manifest, precision); write_compiled chama após o manifesto; TrtCompiler.smoke_test abre TrtBackend e roda zeros [R8, D4]

### tests/
- helpers/yolo.py: letterbox, decode, iou, nms, detect, compare (Comparison.within) [R9, R10, D7]
- test_yolo_helpers.py: letterbox/RGB, decode, IoU, NMS por classe, compare (margem, faltas, extras) [R9, R10]
- test_variants.py: seleção, versões, sha, memória [R2, R3, R5]
- test_api_cpu.py (de 005): trt sem libs, automático cai na CPU [R1]
- test_build_compiled.py (de 002): smoke por variante, falha não publica [R8]
- test_yolo11n.py: caso 4 (bus.jpg, person ≥ 0,5) [R12]
- jetson/conftest.py: sintético e yolo11n compilados por sessão; `trt_env` [D11]
- jetson/test_trt.py: FP16 padrão, FP32, memória constante, cópias, threads, release, 5 códigos, pasta --cpu-only, automático prefere TRT
- jetson/test_equivalence.py: CPU × TRT FP32 × FP16, tolerâncias, relatório [R10, R11, D5]

## Validação
- notebook: `pytest`, `mypy`, `ruff check`, `ruff format --check`
- Jetson: `pytest -m jetson`; relatório do app-epi

## Log
- 2026-10-30: criada (modo auto)
- 2026-10-30: T1–T8: notebook pytest 184 passed, 3 skipped (jetson); mypy strict 38 arquivos, ruff ok; caso 4 na CPU: bus.jpg → bus 0,94, person 0,90/0,85/0,83; nada executado com TensorRT real
