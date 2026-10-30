# 002 — p2-build: compilação TensorRT

- Estado: ativa
- Criada: 2026-10-30
- Fechada: —
- Base: 001, 003 (stubs), 005 (manifesto, IR 10); plano-p2.md (E4)
- Substituída por: —
- Tags: p2-build, tensorrt, compilação, jetson, E4

## Objetivo
- `p2-build` sem `--cpu-only` compila FP16/FP32 na Jetson e grava `variants` e `platform` no manifesto

## Fora de escopo
- Backend TRT do `load`/`infer` (E5)
- Tudo o que é de 001 (pacote, validação, staging, CLI)

## Requisitos
- R1: por precisão (`--precision`, padrão FP16 FP32): FP16 com `BuilderFlag.FP16`; FP32 sem FP16 e sem TF32; workspace 1 GiB; grava `fp16.engine`/`fp32.engine` e o sha256 em `variants` com `engine_version = <engine_ref>-fp16|fp32`
- R2: `platform` com `jetpack` (`dpkg-query -W nvidia-jetpack`), `l4t` (`/etc/nv_tegra_release`), `tensorrt` (`tensorrt.__version__`), `cuda`
- R3: após compilar, cada variante é desserializada e o I/O do engine (nomes, formas, dtype) tem de ser igual ao do ONNX; o `infer` com entrada zerada entra quando houver o backend TRT (E5)
- R4: erro de parser/compilação → mensagem com o erro original do TensorRT, exit ≠ 0
- R5: sem `tensorrt` instalado, `p2-build` sem `--cpu-only` → BuildError com dica (substitui 001/D14)

## Regras
- Valem 001/RG1–RG5 e as decisões de 001
- RG1: engines são gravados dentro de `staged_engine_dir` (001/RG5)

## Decisões
- Tipagem do `tensorrt`/`cudart`: 003/D2 (stubs em `typings/`, estendidos)
- D1 [assumida] smoke `infer` (R3) adiado para a E5; agora: desserializar e conferir I/O — sem duplicar o backend TRT
- D2 [assumida] compilador injetado: `write_compiled(..., compiler: EngineCompiler)` (Protocol); `TrtCompiler` em `build/compiler.py` — orquestração testável no notebook
- D3 [assumida] leitura da plataforma em `build/platform.py` sem `tensorrt`; `cudart` importado dentro da função — testável e opcional
- D4 [assumida] `platform.tensorrt` com 3 componentes (`10.3.0.30` → `10.3.0`) — formato do plano; E5 compara do mesmo jeito
- D5 [assumida] `jetpack`, `l4t` ou `cuda` ilegíveis → omitidos com aviso; `tensorrt` sempre presente — só ele decide `INCOMPATIBLE_VERSION`
- D6 [assumida] TF32 desligado nas duas variantes (manifesto tem um único `build.tf32: false`)
- D7 [assumida] `write_compiled` com 6 parâmetros (`# noqa: PLR0913`) — espelha `write_cpu_only` + precisões e compilador
- D8 [assumida] alterar `cli/build.py`, `build/output.py` e `test_cli_build.py` de 001

## Tarefas
- [x] T1: stubs do TensorRT/cudart estendidos (Builder flags, Runtime, ICudaEngine, ILogger)
- [x] T2: `build/platform.py` + testes (R2, D3–D5)
- [x] T3: `build/output.py`: `EngineCompiler`, `write_compiled` + testes com compilador falso (R1, R3, RG1, D2)
- [x] T4: `build/compiler.py`: `TrtCompiler` (R1, R3, R4)
- [x] T5: `cli/build.py` sem `--cpu-only` (R5)
- [x] T6: `tests/jetson/`: compilação do sintético + `test_yolo11n.py` caso 6 (`@pytest.mark.jetson`)
- [ ] T7 (usuário, Jetson): rodar `pytest -m jetson` e `p2-build` do yolo11n e app-epi; registrar
- [x] T8 (E5): smoke `infer` com entrada zerada por variante (R3) — feito em 006/R8

## Mapa de código

### typings/tensorrt/__init__.pyi (de 003), typings/cuda/cudart.pyi (de 003)
- + ILogger (Severity, log), DataType, nptype, Dims, TensorIOMode, BuilderFlag, set/clear_flag, IHostMemory.__buffer__, ICudaEngine, Runtime; + cudaRuntimeGetVersion [T1]

### src/p2/build/platform.py
- def parse_l4t(text: str) -> str | None: `R36 … REVISION: 4.7` → `36.4.7` [R2]
- def parse_jetpack(text: str) -> str | None; def short_tensorrt_version(version: str) -> str; def format_cuda_version(version: int) -> str [R2, D4]
- def read_platform(tensorrt_version: str | None) -> PlatformInfo (→ 007: versão opcional, + installed_tensorrt): dpkg-query, /etc/nv_tegra_release, cudart; ilegível → omite com aviso [R2, D3, D5]

### src/p2/build/output.py (de 001)
- ENGINE_FILES: Final[Mapping[Precision, str]] [R1]
- class EngineCompiler(Protocol): compile(onnx_path, engine_path, precision) -> OnnxIO; platform() -> PlatformInfo [D2]
- def write_compiled(package, io, *, engine_root, force, stamp, precisions, compiler) -> Path: staging, compila cada precisão, confere I/O, manifesto [R1, R3, RG1, D7]

### src/p2/build/compiler.py
- (CollectingLogger movido para backends/trt.py → 006/D9)
- class TrtCompiler: platform(); compile(): parse, workspace 1 GiB, sem TF32, FP16 flag, serializa, desserializa e lê I/O; erros → BuildError [R1, R3, R4, D6]
- def engine_io(engine: trt.ICudaEngine) -> OnnxIO [R3]

### src/p2/cli/build.py (de 001)
- run(): sem --cpu-only → write_compiled com `_trt_compiler()` (import tardio; sem TRT → BuildError) [R5]

### tests/
- test_platform.py: parsers e omissão de campos [R2]
- test_build_compiled.py: compilador falso: arquivos, variants, platform, 1 precisão, I/O divergente, falha preserva pasta [R1, R3, RG1]
- test_cli_build.py (de 001): sem TensorRT → BuildError [R5]
- jetson/test_build_trt.py: sintético FP16/FP32, --precision, erro de parser, yolo11n caso 6 (`jetson`, skip sem tensorrt) [R1–R4]

## Validação
- na Jetson: `pytest -m jetson`, `mypy`, `ruff check`, `ruff format --check`
- manual: `p2-build <pacote-yolo11n>` (metadata da fixture + `models/yolo11n.onnx`, 008/R9) e `p2-build <pacote-app-epi>`

## Log
- 2026-10-30: T1–T6 (modo auto): pytest 161 passed, 1 skipped (jetson); mypy strict 29 arquivos, ruff ok; nada executado com TensorRT real
- 2026-10-30: criada a partir de 001 (001/D17): antigos 001/R12–R15 → R1–R4; 001/T9–T11 → T1–T3
