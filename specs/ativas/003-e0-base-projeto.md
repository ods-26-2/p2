# 003 — E0: base do projeto e verificação da Jetson

- Estado: ativa
- Criada: 2026-10-30
- Fechada: —
- Base: 001 (pyproject, requirements-dev); plano-p2.md (E0, "Decisões consolidadas", "Deixado para depois")
- Substituída por: —
- Tags: E0, pyproject, readme, docs, decisoes, jetson, check_jetson, stubs, tensorrt, cuda

## Objetivo
- Completar a E0: README, `docs/decisoes.md`, `docs/depois.md`, `scripts/check_jetson.py`
- Stubs locais de `tensorrt` e `cuda.cudart` para tipagem estrita (reuso em 002 e E5)

## Fora de escopo
- `pyproject.toml`/`requirements-dev.txt` já feitos em 001 (só ajustes abaixo)
- Rodar o `check_jetson.py` na placa (feito pelo usuário; resultado vai no Log)

## Requisitos
- R1: `README.md` em português: o que é o p2, instalação no notebook (`.[cpu]`) e na Jetson (`.[trt]`, venv com `--system-site-packages`), uso do `p2-build`, comandos de desenvolvimento
- R2: `docs/decisoes.md`: tabela "Decisões consolidadas" do plano + decisões de specs com alcance de projeto, com origem `NNN/D#`
- R3: `docs/depois.md`: tabela "Deixado para depois" do plano
- R4: `pyproject.toml` com `readme = "README.md"`
- R5: `scripts/check_jetson.py` confere: Python 3.10.12; `import tensorrt` com versão 10.3.0; numpy em `>=1.21.6,<2`; `cudaMalloc`/`cudaMemcpy`/`cudaFree` via `cuda-python` com ida e volta de dados; compilação de um ONNX sintético mínimo com TensorRT
- R6: cada verificação loga `OK` ou `FALHA` com detalhe em português; exit 0 só se todas passarem, senão 1; uma falha não impede as seguintes
- R7: partes sem GPU (comparação de versões, relatório, exit code) testáveis no notebook

## Regras
- Valem 001/RG1–RG3
- RG1 [global]: bibliotecas sem tipos (`tensorrt`, `cuda`) usam stubs mínimos em `typings/`, só com a API usada; sem `Any`, sem `ignore_missing_imports`

## Decisões
- D1 [assumida] entry points `p2-bench`/`p2-list` só entram no pyproject quando os módulos existirem (E6) — evita comandos quebrados no wheel (entraram em 007/R10)
- D2 [assumida] stubs locais em `typings/` (`mypy_path`) para `tensorrt` e `cuda.cudart` — sem Any; resolve a pendência de tipagem de 002
- D3 [assumida] saída do `check_jetson.py` via `logging` (001/RG3), sem `print`
- D4 [assumida] versões comparadas por tupla de inteiros própria, sem `packaging` — sem dependência nova
- D5 [assumida] TensorRT aceito se `__version__` começa com `10.3.0` (ex.: `10.3.0.30`)
- D6 [assumida] `scripts/` entra no mypy e no pytest (`pythonpath`) para testar o `check_jetson.py` sem empacotá-lo
- D7 [assumida] imports de `tensorrt`, `cuda.cudart` e `onnx` dentro das verificações (`# noqa: PLC0415`) — lib ausente vira FALHA em vez de derrubar o script

## Tarefas
- [x] T1: `README.md` + `readme` no pyproject (R1, R4)
- [x] T2: `docs/decisoes.md` e `docs/depois.md` (R2, R3)
- [x] T3: stubs `typings/tensorrt/__init__.pyi`, `typings/cuda/cudart.pyi`; config mypy/pytest (D2, D6, RG1)
- [x] T4: `scripts/check_jetson.py` (R5, R6, D3–D5)
- [x] T5: `tests/test_check_jetson.py` (R7)
- [ ] T6 (usuário, Jetson): rodar `check_jetson.py` na placa e registrar o resultado

## Mapa de código

### README.md
- instalação notebook/Jetson, uso do `p2-build`, comandos de dev [R1]

### pyproject.toml (de 001)
- `readme = "README.md"`; mypy `files` + `scripts`, `mypy_path = "typings"`; pytest `pythonpath = ["scripts"]`; ruff `src` + `scripts` [R4, D2, D6]

### docs/decisoes.md
- decisões do plano + tomadas em specs, com origem `NNN/D#` [R2]

### docs/depois.md
- itens adiados do plano [R3]

### typings/tensorrt/__init__.pyi
- stub mínimo: `__version__`, Logger, Builder, INetworkDefinition, OnnxParser, ParserError, IBuilderConfig, MemoryPoolType, IHostMemory [D2, RG1]

### typings/cuda/cudart.pyi
- stub mínimo: cudaError_t, cudaMemcpyKind, cudaMalloc, cudaMemcpy, cudaFree [D2, RG1]

### scripts/check_jetson.py
- class CheckResult (frozen, slots): name, ok, detail; Check: TypeAlias = Callable[[], CheckResult] [R6]
- def parse_version(text: str) -> tuple[int, ...]; def version_in_range(version, *, minimum, maximum_exclusive) -> bool [D4]
- def check_python_version / check_tensorrt_version / check_numpy_version(...) -> CheckResult: regras puras [R5, D5]
- def check_python / check_tensorrt / check_numpy / check_cuda_roundtrip / check_tensorrt_build() -> CheckResult: verificações reais [R5, D7]
- def synthetic_onnx() -> bytes: Relu [1, 3, 8, 8], opset 17, IR 10 [R5] (IR por 005/D11)
- def run_checks(checks: Sequence[Check]) -> int: loga OK/FALHA, todas rodam, exit 0/1 [R6, D3]
- def main() -> int

### tests/test_check_jetson.py
- cobre R7: versões, faixas, TRT 10.3.0.x, libs ausentes viram FALHA, ONNX sintético válido, run_checks continua após falha e exit codes

## Validação
- `pytest`, `mypy`, `ruff check`, `ruff format --check`; `python -m build --wheel` gera `p2-0.1.0-py3-none-any.whl`
- na Jetson: `python3 scripts/check_jetson.py` → exit 0

## Log
- 2026-10-30: criada (modo auto)
- 2026-10-30: T1–T5: pytest 75 passed; mypy strict (16 arquivos, stubs de typings/ usados no lugar do cuda-bindings 13 da .venv), ruff ok; wheel gerado sem scripts/; `check_jetson.py` no notebook: Python e numpy OK, 3 FALHA esperadas (sem TRT/cudart), exit 1
