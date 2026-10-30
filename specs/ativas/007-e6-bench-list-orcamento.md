# 007 — E6: p2-bench, p2-list e orçamento

- Estado: ativa
- Criada: 2026-10-30
- Fechada: —
- Base: 002 (platform), 005 (API), 006 (backend TRT, variants); plano-p2.md (E6)
- Substituída por: —
- Tags: E6, p2-bench, p2-list, cli, memoria, orcamento, latencia, nvpmodel, jetson

## Objetivo
- Medir latência e memória de um `engine_ref` e listar os engines instalados; base para `docs/orcamento.md`

## Fora de escopo
- Executar as medições de 30 min e o teste do limite do container (usuário, na placa)

## Requisitos
- R1: `p2-bench <engine_ref> [--precision FP16|FP32] [--duration 30m] [--warmup 50] [--out arquivo.json]`
- R2: `--duration` aceita `90`, `45s`, `30m`, `1h`; inválido → erro de uso (exit 2)
- R3: entrada aleatória float32 com a forma do `info`, gerada uma vez antes do laço; warmup fora das estatísticas
- R4: latência (do `stats["latency_ms"]`) p50, p95, máxima e média; FPS médio; `gpu_memory_mb`; iterações
- R5: memória do sistema por `MemAvailable` de `/proc/meminfo`: antes do `load`, depois do `load`, mínimo durante o laço; custo fixo = antes − depois do load; pico = antes − mínimo; limite = pico × 1,3
- R6: registra `nvpmodel -q` (modo de energia) e as versões da plataforma; ausentes → `null`
- R7: grava o JSON em `--out`, senão em `reports/bench/<engine_ref>-<precisão>-<AAAAMMDD-HHMMSS>.json`; resumo no log
- R8: `P2Error` → exit 1 com a mensagem
- R9: `p2-list [--engine-dir P]`: uma linha por pasta com manifesto: `engine_ref`, variantes, compatibilidade (compatível, incompatível com versões, só CPU, sem TensorRT instalado); manifesto inválido aparece como erro; pastas ocultas ignoradas; raiz ausente → exit 1
- R10: entry points `p2-bench` e `p2-list` no pyproject (003/D1)
- R11: `docs/orcamento.md`: método, fórmulas e tabela a preencher com as medições

## Regras
- Valem 001/RG1–RG3, 003/RG1, 005/RG1
- RG1: os comandos usam só a API pública (`p2.load/infer/info/release`) para medir

## Decisões
- D1 [assumida] `read_platform(tensorrt_version: str | None)`: sem TensorRT omite o campo (de 002) — bench na CPU também registra a placa
- D2 [assumida] memória amostrada no máximo a cada 0,5 s durante o laço — custo desprezível, pico capturado
- D3 [assumida] saída do `p2-list` por `logging` em stdout (`%(message)s`) — 001/RG3 sem `print`, e a tabela vai para stdout
- D4 [assumida] relatório JSON como `TypedDict` (`BenchReport`), com `null` para o que não existe
- D5 [assumida] semente fixa (0) para a entrada aleatória — medições repetíveis
- D6 [assumida] leitores de memória e de `nvpmodel` injetáveis em `run_bench` — testes sem a placa
- D7 [assumida] `installed_tensorrt()` em `build/platform.py` (import tardio) — usado por bench e list
- D8 [assumida] relatório não gravável → exit 1 com mensagem; `docs/orcamento.md` sem números até a medição — nada inventado

## Tarefas
- [x] T1: `read_platform` opcional (D1)
- [x] T2: `cli/bench.py` (R1–R8, D2, D4–D6)
- [x] T3: `cli/list.py` (R9, D3)
- [x] T4: entry points + reinstalação (R10)
- [x] T5: testes `test_cli_bench.py`, `test_cli_list.py`
- [x] T6: `docs/orcamento.md` com método e tabela (R11)
- [ ] T7 (usuário, Jetson): `p2-bench app-epi-1.0.0` FP16 e FP32 por 30 min, placa em repouso; preencher o orçamento; avisar o MOD-1
- [ ] T8 (usuário, Jetson): testar se o limite de memória do container conta a memória do CUDA

## Mapa de código

### src/p2/build/platform.py (de 002)
- read_platform(tensorrt_version: str | None) [D1]; def installed_tensorrt() -> str | None [D7]

### src/p2/cli/bench.py
- TypedDicts LatencySummary, MemorySummary, BenchReport [R4, R5, D4]
- class BenchConfig, Probes, Measurement (frozen, slots) [D6]
- def parse_duration(text: str) -> float; parse_meminfo(text) -> float | None; parse_power_mode(text) -> str | None [R2, R5, R6]
- def summarize_latency(latencies: Sequence[float]) -> LatencySummary; memory_summary(before, after_load, minimum) -> MemorySummary (limite = pico × 1,3) [R4, R5]
- def run_bench(config: BenchConfig, probes: Probes) -> BenchReport: load, entrada aleatória (semente 0), warmup, laço até a duração, amostra memória a cada 0,5 s, release [R3–R6, D2, D5, RG1]
- def default_report_path(engine_ref, precision, now) -> Path; write_report(report, path) -> Path [R7]
- SYSTEM_PROBES; def main(argv) -> int: P2Error/OSError → exit 1; uso → exit 2 [R1, R2, R8, D8]

### src/p2/cli/list.py
- def compatibility(manifest: Manifest, installed: str | None) -> str [R9]
- def describe(directory: Path, installed: str | None) -> str; def list_engines(root: Path, installed: str | None) -> list[str]: ignora ocultas [R9]
- def main(argv) -> int: tabela por logging em stdout; raiz ausente → exit 1 [R9, D3]

### pyproject.toml (de 003)
- + `p2-bench = p2.cli.bench:main`, `p2-list = p2.cli.list:main` [R10]

### docs/orcamento.md
- método, fórmulas, tabela FP16/FP32 a preencher, teste do limite do container [R11]

### tests/
- test_cli_bench.py: duração, meminfo, nvpmodel, percentis, orçamento, run_bench na CPU com sondas falsas, JSON, erros (engine, uso, gravação) [R1–R8]
- test_cli_list.py: compatibilidade (4 casos), listagem com inválida e oculta, main (tabela, vazia, raiz ausente) [R9]

## Validação
- `pytest`, `mypy`, `ruff check`, `ruff format --check`; `p2-bench` curto e `p2-list` no notebook

## Log
- 2026-10-30: criada (modo auto)
- 2026-10-30: T1–T6: pytest 212 passed, 3 skipped; mypy strict 42 arquivos, ruff ok; `pip install -e .` instalou p2-bench e p2-list; smoke no notebook: yolo11n CPU 3 s → 104 iterações, 34,5 FPS, p50 27,85 ms; p2-list mostrou só CPU, inválida e ignorou oculta
