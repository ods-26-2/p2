# 005 — E2: núcleo da lib e backend de CPU

- Estado: fechada
- Criada: 2026-10-30
- Fechada: 2026-10-30
- Base: 001 (errors, manifest), 003 (stubs, RG1), 004 (contrato, D1–D5); plano-p2.md (E2)
- Substituída por: —
- Tags: E2, api, load, infer, info, release, handle, backend, onnxruntime, cpu, erros, manifesto

## Objetivo
- `import p2` com `load`/`infer`/`info`/`release` no notebook (ONNX Runtime), seguindo docs/contrato.md

## Fora de escopo
- Backend TensorRT (E5), `p2-bench`/`p2-list` (E6)
- Caso 4 do `test_yolo11n.py` (imagem + NMS; depende de `tests/helpers/yolo.py`, E5)

## Requisitos
- R1: `ErrorCode` (5 códigos, `str` + `Enum`) e `P2Error(code, detail)` com `str(e) == "CODE: detalhe"` [004/D1, D3]
- R2: `read_manifest(dir)`: pasta ou manifesto ausente, JSON ou campos inválidos → `ENGINE_NOT_FOUND`
- R3: `load`: `engine_ref` texto sem `/`; `options` só `precision` FP16/FP32, senão `ValueError`; manifesto com o mesmo `engine_ref`
- R4: backend: `P2_BACKEND` `ort`/`trt`/vazio; automático = CPU até a E5
- R5: CPU: só FP32; sem `precision` usa FP32; FP16 explícito → `ENGINE_NOT_FOUND`; `gpu_memory_mb = 0.0`; ONNX ausente ou inválido → `ENGINE_NOT_FOUND`; `MemoryError` → `OUT_OF_MEMORY`
- R6: `infer`: array solto (1 entrada) ou `{nome: array}` com os nomes exatos; float32; forma exata; senão `INVALID_INPUT_SHAPE`; não contíguo é copiado; saídas copiadas; falha na execução → `RUNTIME_ERROR`
- R7: `stats = {"latency_ms", "gpu_memory_mb"}`; `latency_ms` mede a chamada inteira (`perf_counter`)
- R8: `info` com os 5 campos; `engine_version = <engine_ref>-fp16|fp32`
- R9: trava por handle; `release` idempotente; `infer`/`info` depois do `release` → `ValueError`; `with` libera ao sair
- R10: `p2` exporta `load`, `infer`, `info`, `release`, `P2Error`, `ErrorCode`, `Handle`, `EngineInfo`, `InferStats`, `__version__`

## Regras
- Valem 001/RG1–RG3, 003/RG1 e docs/contrato.md
- RG1: `import p2` não importa `onnxruntime` nem `tensorrt`; o backend é importado só no `load`

## Decisões
- D1 [assumida] alterar `errors.py` e `manifest.py` de 001 (acrescentar, sem mudar o existente) — são os módulos previstos no plano
- D2 [assumida] stubs mínimos de `onnxruntime` em `typings/` (003/RG1), com as exceções do pybind listadas para captura específica
- D3 [assumida] import do backend dentro de `select_backend` (`# noqa: PLC0415`) — RG1; extras opcionais
- D4 [assumida] `P2_BACKEND=trt` antes da E5 → `ValueError` "backend TensorRT ainda não disponível"; valor desconhecido → `ValueError`
- D5 [assumida] backend sem a lib instalada (onnxruntime) → `P2Error(ENGINE_NOT_FOUND)` com dica `pip install p2[cpu]` — I1 só trata `P2Error`
- D6 [assumida] manifesto com `engine_ref` diferente da pasta → `ENGINE_NOT_FOUND`
- D7 [assumida] `engine_version` calculado por `<engine_ref>-<precisão minúscula>` nos dois backends — regra do plano; TRT confere com `variants`
- D8 [assumida] `Session.run` pode devolver buffers internos; o `Handle` copia sempre — contrato único para CPU e TRT
- D9 [assumida] `EngineInfo` e `InferStats` ficam em `handle.py`; `info()` devolve cópia nova a cada chamada
- D10 [assumida] `OrtBackend.open` confere nomes de entrada/saída do ONNX contra o manifesto → `ENGINE_NOT_FOUND` — falha cedo no `load`
- D11 [assumida] ONNX sintético dos testes e do `check_jetson.py` com `ir_version=10` e saída dependente da entrada (`arange + média`) — onnx 1.23 gera IR 14, que o ORT 1.23 recusa
- D12 [assumida] fixtures `engine_root`/`synth_engine` em `tests/conftest.py` (devolvem `Path`, sem tipo a importar); `build_engine` em `tests/helpers/` (de 001/D16)

## Tarefas
- [x] T1: stubs `typings/onnxruntime/` (D2)
- [x] T2: `errors.py`: `ErrorCode`, `P2Error` (R1)
- [x] T3: `manifest.py`: `engine_dir`, `read_manifest`, `as_precision` (R2)
- [x] T4: `backends/base.py` (`Session`, `Backend`) e `backends/ort.py` (R5)
- [x] T5: `handle.py` (R8, R9)
- [x] T6: `api.py` e `__init__.py` (R3, R4, R6, R7, R10, RG1)
- [x] T7: testes: `test_errors.py`, `test_api_cpu.py`, `test_manifest.py` (leitura), `test_yolo11n.py` casos 2, 3, 5

## Mapa de código

### typings/onnxruntime/
- `__init__.pyi`: InferenceSession (init, get_inputs/outputs, run), NodeArg, SessionOptions; `capi/onnxruntime_pybind11_state.pyi`: exceções do pybind [D2]

### src/p2/errors.py (de 001)
- class ErrorCode(str, Enum): 5 códigos; `__str__` devolve o valor [R1, 004/D1]
- class P2Error(P2BaseError): `code: ErrorCode`, `detail: str`; `str(e) == "CODE: detalhe"` [R1, 004/D3]

### src/p2/manifest.py (de 001)
- class ManifestFormatError(P2BaseError): erro interno de formato [R2]
- def as_precision(value: object) -> Precision | None [R3]
- def engine_dir(engine_ref: str, env: Mapping[str, str]) -> Path [R3]
- def read_manifest(directory: Path) -> Manifest: ausente/ilegível/inválido → ENGINE_NOT_FOUND [R2]
- def parse_manifest(raw: object) -> Manifest: narrowing completo, schema_version 1 [R2]

### src/p2/backends/base.py
- FloatArray: TypeAlias = NDArray[float32] (→ 009: + TensorArray)
- class Session(Protocol): memory_mb, run(inputs) -> Mapping[str, FloatArray], close() [D8]
- class Backend(Protocol): default_precision, open(manifest, directory, precision) -> Session

### src/p2/backends/ort.py
- ORT_ERRORS: tupla das exceções do pybind; CPU_PRECISION = "FP32" [D2]
- class OrtSession: run → RUNTIME_ERROR em falha; memory_mb 0.0; close solta a sessão [R5, R6]
- class OrtBackend: open: só FP32, ONNX ausente/inválido → ENGINE_NOT_FOUND, MemoryError → OUT_OF_MEMORY, confere I/O [R5, D10]

### src/p2/handle.py
- class EngineInfo(TypedDict), class InferStats(TypedDict) [R7, R8, D9]
- class Handle: trava, released, input_shapes(), info() (cópia), run() (copia saídas), release() idempotente, ensure_open(), with [R8, R9, D8] (→ 009: run → `dict[str, TensorArray]`)

### src/p2/api.py
- def load(engine_ref: str, options: Mapping[str, object] | None = None) -> Handle [R3, R5, D6]
- def infer(handle: Handle, input: InferInput) -> tuple[dict[str, FloatArray], InferStats]: valida, mede, copia [R6, R7] (→ 009: retorno `dict[str, TensorArray]`)
- def info(handle) -> EngineInfo; def release(handle) -> None [R8, R9]
- def select_backend(env: Mapping[str, str]) -> Backend: P2_BACKEND; import tardio do ORT [R4, D3, D4, D5, RG1] (→ 006: automático TRT, D4 substituída)

### src/p2/__init__.py
- exporta API, P2Error, ErrorCode, Handle, EngineInfo, InferStats, `__version__` [R10] (→ 009: + TensorArray)

### tests/
- conftest.py: fixtures `engine_root`, `synth_engine` [D12]
- helpers/model_package.py: `build_engine`, `PackageSpec.fails_at_runtime`, IR 10 [D11]
- test_api_cpu.py: ciclo, info, array/dict, não contíguo, cópias, with, release, 4 threads, options, engine_ref, P2_BACKEND, import leve, exports [R3–R10]
- test_errors.py: ErrorCode/P2Error, ENGINE_NOT_FOUND (pasta, manifesto, ref, ONNX ausente/corrompido/divergente, FP16), INVALID_INPUT_SHAPE (8 casos), RUNTIME_ERROR [R1, R2, R5, R6]
- test_manifest.py: leitura ida e volta (CPU e TRT), rejeições de formato [R2]
- test_yolo11n.py: casos 2, 3, 5 [R5, R8] (→ 008: pacote montado por `helpers/yolo11n.py`)

## Validação
- `pytest`, `mypy`, `ruff check`, `ruff format --check`
- manual: `p2-build tests/fixtures/yolo11n-1.0.0 --cpu-only` + `p2.load("yolo11n-1.0.0")` → `output0` `(1, 84, 8400)`

## Log
- 2026-10-30: criada (modo auto)
- 2026-10-30: T1–T7: pytest 142 passed; mypy strict (24 arquivos), ruff ok; smoke manual yolo11n na CPU: info ok, `output0` (1, 84, 8400) float32, FP16 → ENGINE_NOT_FOUND
- 2026-10-30: achado: onnx 1.23 gera IR 14 e o ORT 1.23 recusa (>11) → D11 e 004/D6; yolo11n real é IR 7/opset 12
- 2026-10-30: fechada (modo auto)
