# 001 — p2-build

- Estado: fechada
- Criada: 2026-10-30
- Fechada: 2026-10-30
- Base: plano-p2.md (E3; seções "Pacote MODEL e manifesto", "Teste com o yolo11n")
- Substituída por: —
- Tags: p2-build, cli, pacote-model, manifesto, onnx, E3

## Objetivo
- Comando `p2-build` que lê um pacote MODEL, valida e grava `<engine_dir>/<engine_ref>/` com `manifest.json`
- Modo `--cpu-only` no notebook (E3); compilação TensorRT (E4) em 002

## Fora de escopo
- Compilação TensorRT, `platform`, smoke `infer` (E4) → 002
- Leitura/validação do manifesto pelo `load`, backends, API (E2)
- `p2-bench`, `p2-list` (E6)
- `schema/manifest.schema.json` (formato antigo; fica como está)
- Download do pacote pelo registro de B5

## Requisitos
- R1: ler `<pasta>/metadata.json`; campos `artifact_id`, `artifact_version`, `onnx_file`, `hash_sha256`, `input_shape`, `classes`, `precision`; ausente ou tipo errado → erro
- R2: `engine_ref = f"{artifact_id}-{artifact_version}"`
- R3: sha256 do `onnx_file` igual a `hash_sha256`; senão `hash diferente: esperado …, obtido …`
- R4: `precision` do metadata deve ser `"FP32"`
- R5: ONNX com 1 entrada e formas estáticas; forma da entrada igual a `input_shape`
- R6: ONNX com 1 saída; `shape[1] == 4 + len(classes)`
- R7: raiz dos engines: `--engine-dir` > `P2_ENGINE_DIR` > `/opt/ods/engines`
- R8: pasta `<raiz>/<engine_ref>` existente → recusa; `--force` substitui (D4)
- R9: `--cpu-only`: copia `onnx_file` e `metadata.json`; manifesto com `variants: {}` e `platform: {}`
- R10: `manifest.json` no formato do plano: `schema_version: 1`, `engine_ref`, `source`, `inputs`, `outputs`, `max_batch: 1`, `variants`, `platform`, `build {tool: "p2-build <versão>", built_at ISO-8601 UTC, tf32: false, workspace_mb: 1024}`
- R11: CLI `p2-build <pasta> [--precision FP16 FP32] [--cpu-only] [--force] [--engine-dir P]`; exit 0 em sucesso, ≠ 0 em erro com mensagem em português

## Regras
- RG1 [global]: Python 3.10.12; ruff/mypy com alvo py310
- RG2 [global]: código em inglês; documentação e mensagens de erro em português
- RG3 [global]: `logging.getLogger(__name__)` sob `"p2"`; sem `print`
- RG4: validações R3 a R6 acontecem antes de criar qualquer arquivo no destino
- RG5: a pasta de destino só aparece completa (escrita em staging + rename)

## Decisões
- D1 [rejeitada] E4 em spec 002 → E3 + E4 na 001 — escolha do usuário [substituída por D17]
- D2 [aprovada] `typing_extensions` como dependência obrigatória (`override`, `Self`, `TypeIs`); `TypeVar` no lugar de PEP 695; `class X(str, Enum)` no lugar de `StrEnum` — Python 3.10
- D3 [aprovada] pacote `onnx>=1.16` nos extras `cpu` e `trt` para ler I/O — um caminho só, sem executar
- D4 [aprovada] `--force` com staging em `<raiz>/.<engine_ref>.tmp-*` e troca só no sucesso — pasta antiga intacta em falha
- D5 [aprovada] `P2BaseError(Exception)` base; `BuildError(P2BaseError)`; `P2Error` (E2) herdará da base; CLI captura `BuildError` → exit 1
- D6 [aprovada] `--precision` com `--cpu-only` → erro de uso do argparse (exit 2)
- D7 [aprovada] `pyproject.toml` mínimo da E0 (p2 0.1.0, numpy, typing_extensions, extras `cpu`/`trt`, entry point `p2-build`, config ruff/mypy py310/pytest) + `requirements-dev.txt` (pytest, mypy, ruff, onnx, build)
- D8 [aprovada] backend de build `setuptools>=68` — já na .venv, src/ layout, entry points
- D9 [aprovada] .venv com `numpy<2` (1.26.x) e opencv `<4.12` via `requirements.txt` — mesmo numpy 1.x da Jetson
- D10 [aprovada] `classes` = lista não vazia de strings; outro formato → BuildError — simples; rever se app-epi vier diferente
- D11 [aprovada] `types-protobuf` em requirements-dev para tipar `DecodeError` do `onnx.load` — sem ignore
- D12 [assumida] `onnx_file` com subpasta é copiado achatado (só o nome do arquivo) e o manifesto registra esse nome — pasta do engine plana como no plano
- D13 [assumida] troca com `--force`: renomeia a antiga para `.<ref>.old-*`, move a nova, apaga a antiga; se mover falhar, restaura — Linux não troca pasta não vazia atomicamente
- D14 [assumida] até a 002, sem `--cpu-only` → BuildError `compilação TensorRT ainda não disponível; use --cpu-only` (exit 1), após validar o pacote — comportamento provisório explícito
- D15 [assumida] logging da CLI: `basicConfig(INFO, "%(levelname)s: %(message)s")` em stderr; mensagens internas do argparse ficam em inglês — evita reimplementar o argparse
- D16 [assumida] gerador do pacote sintético em `tests/helpers/` (importável e tipado) em vez de `conftest.py`; testes da CLI em `test_cli_build.py` — fixtures de conftest não são importáveis com tipo
- D18 [assumida] classes da fixture yolo11n vêm do `names` do ONNX (`traffic light`), não de `assets/coco80.txt` (sem espaços) — o plano manda copiar do ONNX
- D17 [aprovada] dividir: E4 (antigos R12–R15, T9–T11) → 002 — spec passou de 150 linhas; 001 fecha no notebook

## Tarefas
- [x] T1: `pyproject.toml` + `requirements-dev.txt`; venv com ferramentas (D7, RG1)
- [x] T2: `errors.py`: base do pacote + `BuildError` (D5)
- [x] T3: `manifest.py`: TypedDicts do manifesto, `resolve_engine_root`, `write_manifest` (R7, R10) + `tests/test_manifest.py`
- [x] T4: `build/package.py`: `read_package`, `sha256_file`, `read_onnx_io`, `validate_package` (R1–R6, D3, D10, D11)
- [x] T5: `build/output.py`: staging, cópia, manifesto, troca atômica (R8, R9, RG4, RG5, D4)
- [x] T6: `cli/build.py`: `main(argv) -> int`, argparse, logging, exit codes (R11, D6)
- [x] T7: `tests/helpers/model_package.py` (ONNX sintético) + `tests/test_package.py` (R1–R6) + `tests/test_cli_build.py` (R7–R11)
- [x] T8: fixture `tests/fixtures/yolo11n-1.0.0/metadata.json` + `tests/test_yolo11n.py` caso 1 (skip sem `model.onnx`)

## Mapa de código

### pyproject.toml
- p2 0.1.0, setuptools, src/; deps numpy<2, typing_extensions; extras cpu/trt; script `p2-build = p2.cli.build:main`; ruff/mypy py310, pytest marca `jetson` [D2, D3, D7, D8] (→ 003)

### requirements-dev.txt
- `-e .[cpu]`, pytest, mypy, ruff, types-protobuf, build [D7, D11]

### src/p2/errors.py
- class P2BaseError(Exception): base de todas as exceções do pacote [D5] (→ 005: + ErrorCode, P2Error)
- class BuildError(P2BaseError): falha do p2-build; mensagem em português; CLI → exit 1 [D5, R11]

### src/p2/manifest.py
- Precision: TypeAlias = Literal["FP16", "FP32"]
- SCHEMA_VERSION = 1; MANIFEST_FILE = "manifest.json"; ENGINE_DIR_ENV = "P2_ENGINE_DIR"; DEFAULT_ENGINE_ROOT = /opt/ods/engines (todas Final) [R7, R10]
- TypedDicts TensorSpec, SourceInfo, VariantInfo, BuildInfo, Manifest: formato do plano [R10] (→ 009: + ShapeProfile, TensorSpec.profile, DYNAMIC_DIM, TENSOR_DTYPES)
- PlatformInfo(TypedDict, total=False): vazio no --cpu-only [R9]
- Manifest.variants: dict[Precision, VariantInfo], vazio no --cpu-only [R9]
- def resolve_engine_root(override: Path | None, env: Mapping[str, str]) -> Path: override > env não vazio > padrão [R7]
- def write_manifest(manifest: Manifest, directory: Path) -> Path: grava JSON indentado UTF-8 [R10] (→ 005: + leitura)

### src/p2/build/package.py
- METADATA_FILE = "metadata.json"; SOURCE_PRECISION = "FP32"; BOX_ROWS = 4; CLASS_AXIS = 1 (Final) (→ 009: + Task, DEFAULT_TASK)
- class ModelPackage (frozen, slots): root, metadata_path, onnx_path, engine_ref, sha256 (minúsculo), input_shape: tuple[int, ...], classes: tuple[str, ...] [R1, R2]
- class OnnxIO (frozen, slots): inputs/outputs: dict[str, TensorSpec] [R5, R6]
- def read_package(root: Path) -> ModelPackage: lê e valida metadata, exige FP32 [R1, R2, R4, D10]
- def validate_package(package: ModelPackage) -> OnnxIO: arquivo, hash, I/O; sem escrita [R3, R5, R6, RG4]
- def sha256_file(path: Path) -> str: hash em blocos de 1 MiB [R3]
- def read_onnx_io(path: Path) -> OnnxIO: `onnx.load` sem dados externos; ignora initializers; dim simbólica → BuildError [R5, D3, D11]
- _require_*: narrowing de `object` do JSON com mensagens em português [R1]

### src/p2/build/output.py
- TOOL_NAME = "p2-build"; MAX_BATCH = 1; WORKSPACE_MB = 1024 (Final) [R10]
- class BuildStamp (frozen, slots): built_at: datetime, tool_version: str [R10]
- def build_manifest(package, io, *, variants: Mapping[Precision, VariantInfo], platform: PlatformInfo, stamp: BuildStamp) -> Manifest [R10, D12]
- def write_cpu_only(package, io, *, engine_root: Path, force: bool, stamp: BuildStamp) -> Path: staging + cópia + manifesto vazio [R9] (→ 002: + write_compiled)
- def copy_sources(package: ModelPackage, directory: Path) -> None: copia onnx e metadata pelo nome [R9, D12]
- @contextmanager staged_engine_dir(engine_root: Path, engine_ref: str, *, force: bool) -> Iterator[Path]: recusa sem force antes de criar; staging `.<ref>.tmp-*`; commit no sucesso; limpa sempre; OSError → BuildError [R8, RG5, D4]
- _commit: troca com backup e restauração [D13]

### src/p2/cli/build.py
- PROG = "p2-build"; ALL_PRECISIONS: tuple[Precision, ...] = ("FP16", "FP32"); EXIT_OK = 0; EXIT_BUILD_ERROR = 1 (Final) [R11]
- class BuildArgs (frozen, slots): package, precisions, cpu_only, force, engine_dir: Path | None [R11]
- def main(argv: Sequence[str] | None = None) -> int: logging, BuildError → exit 1 [R11, D5, D15]
- def run(args: BuildArgs) -> Path: lê, valida, resolve raiz, grava cpu-only [R7, R9] (→ 002: compila sem --cpu-only; D14 substituída)
- def parse_args(argv: Sequence[str] | None) -> BuildArgs: narrowing do Namespace; --precision + --cpu-only → exit 2 [R11, D6]

### tests/helpers/model_package.py
- class PackageSpec (frozen, slots): forma da entrada, linhas da saída, I/O extra, nome do onnx, overrides/drop do metadata [D16]
- def write_package(root: Path, spec: PackageSpec | None = None) -> Path: ONNX sintético + metadata com hash real (→ 005: IR 10, saída dependente da entrada, build_engine)

### tests/fixtures/yolo11n-1.0.0/metadata.json
- pacote do yolo11n (80 classes COCO do ONNX, hash do plano); `model.onnx` fora do git [D18] (→ 008: `onnx_file` = `yolo11n.onnx`, ONNX em `models/`)

### tests/test_yolo11n.py
- caso 1 do plano: `--cpu-only` gera o manifesto esperado; skip sem `model.onnx` (→ 008: pacote montado por `helpers/yolo11n.py`)

### tests/test_manifest.py
- cobre R7 (prioridade, env vazio) e R10 (ida e volta do JSON)

### tests/test_package.py
- cobre R1 (ausência e tipos de cada campo, JSON inválido), R2, R3 (hash, maiúsculas, arquivo ausente), R4, R5 (forma, dim dinâmica, 2 entradas, ONNX corrompido), R6 (classes, 2 saídas)

### tests/test_cli_build.py
- cobre R7 (opção, env), R8/D4 (recusa, --force, falha no staging preserva), R9, R10 (conteúdo do manifesto), R11/D6 (exit 0/1/2), RG4, D12, raiz não gravável (→ 002: D14 virou 'sem TensorRT')

## Validação
- `pytest`, `mypy --strict src tests`, `ruff check`, `ruff format --check`
- manual: `p2-build tests/fixtures/yolo11n-1.0.0 --cpu-only --engine-dir <tmp>`

## Log
- 2026-10-30: criada a partir do plano-p2.md (E3, E4); D1–D11 com o usuário
- 2026-10-30: T1: `.venv/` estava somente leitura; usuário instalou; numpy 1.26.4, opencv 4.11.0, onnx/ort 1.23.2, pytest 9.2.0, mypy 2.4.1, ruff 0.17.3
- 2026-10-30: PYTHON.md ganhou seção "Ambiente virtual" a pedido do usuário; .gitignore com egg-info, caches, /build/, /dist/
- 2026-10-30: T4–T6: smoke manual com yolo11n real e `p2-build` instalado (exit 0/1/2, --force, falha preserva pasta, sem permissão) ok
- 2026-10-30: D17: R12–R15 e T9–T11 movidos para 002 (lá R1–R4, T1–T3)
- 2026-10-30: T8: pytest 51 passed (yolo11n skip sem model.onnx verificado); mypy strict, ruff check/format ok
- 2026-10-30: fechada (modo auto)
