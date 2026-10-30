# 008 — E7: exemplo, guia de integração e release 0.1.0

- Estado: ativa
- Criada: 2026-10-30
- Fechada: —
- Base: 004 (contrato), 005 (API), 006 (backend TRT, helpers/yolo), 007 (orçamento); plano-p2.md (E7)
- Substituída por: —
- Tags: E7, release, wheel, exemplo, yolo_demo, guia, integracao, I1, docker, letterbox, nms

## Objetivo
- I1 integra o P2 sem ajuda: exemplo completo, guia passo a passo e wheel 0.1.0 publicado

## Fora de escopo
- Ações externas sem confirmação do usuário: tag, push, release no GitHub
- Imagem base do Dockerfile (pendência com P3)

## Requisitos
- R1: `examples/yolo_demo.py`: BGR→RGB, letterbox com preenchimento 114, `infer`, decodificação, limiar, NMS por classe e caixas de volta ao quadro original
- R2: o exemplo usa só a API pública do `p2`; nomes de classes do `metadata.json` da pasta do engine; tamanho de entrada do `info`
- R3: CLI `python examples/yolo_demo.py <engine_ref> <imagem> [--conf 0.25] [--iou 0.45] [--out anotada.jpg]`; `P2Error` → exit 1
- R4: `docs/guia-integracao.md`: instalação no Dockerfile, montagem de `/opt/ods/engines` somente leitura, `artifact_version` de `info()["engine_version"]`, tratamento de erros, BGR→RGB em destaque, orçamento
- R5: `python -m build --wheel` gera `dist/p2-0.1.0-py3-none-any.whl` sem `tests/`, `scripts/`, `examples/`
- R6: wheel instalado num venv limpo do notebook com `[cpu]`: `import p2`, `p2-build --cpu-only`, `p2.load`/`infer` do yolo11n e os três comandos
- R7 (usuário): wheel testado num venv limpo na Jetson
- R8 (externo, com confirmação): tag `v0.1.0` e release com `p2-0.1.0-py3-none-any.whl` em `ods-26-2/p2`

## Regras
- Valem 001/RG1–RG3, 003/RG1, docs/contrato.md
- RG1: nenhuma ação externa (push, tag, release) sem confirmação explícita do usuário

## Decisões
- D1 [assumida] o exemplo é autocontido (não importa `tests/helpers/yolo.py`) — I1 copia o arquivo; duplicação aceita
- D2 [assumida] `examples/` entra no mypy, no ruff e no `pythonpath` do pytest; teste do exemplo com o yolo11n na CPU
- D3 [assumida] teste do wheel num venv temporário no scratchpad, fora da `.venv` — exceção à regra de venv única do PYTHON.md, só para validar o pacote limpo
- D4 [assumida] Dockerfile do guia com `FROM` marcado como pendente do P3
- D5 [assumida] `conf` e `iou` agrupados em `Thresholds` no exemplo (limite de 5 parâmetros do ruff)
- D6 [aprovada] publicar agora em `joaoVs17/p2_runtime`; URLs de plano/contrato/guia/README/decisões continuam em `ods-26-2/p2`, que será o local principal depois — escolha do usuário

## Tarefas
- [x] T1: `examples/yolo_demo.py` + config mypy/ruff/pytest (R1–R3, D1, D2)
- [x] T2: `tests/test_yolo_demo.py` (R1, R2)
- [x] T3: `docs/guia-integracao.md` (R4, D4)
- [x] T4: gerar o wheel e conferir conteúdo (R5)
- [x] T5: venv limpo no notebook (R6, D3)
- [ ] T6 (usuário): wheel na Jetson (R7)
- [ ] T7 (externo, aguardando confirmação): tag e release (R8)
- [ ] T8 (usuário): acompanhar a primeira build de I1 na Jetson

## Mapa de código

### examples/yolo_demo.py
- class Letterboxed, Thresholds, Detection (frozen, slots) [R1, D5]
- def letterbox(image_bgr: ImageArray, size: int) -> Letterboxed: BGR→RGB, 114, /255, CHW [R1]
- def box_iou(...); def nms(boxes, scores, classes, iou: float) -> list[int]: por classe [R1]
- def postprocess(output, boxed, image_size, class_names, thresholds=None) -> list[Detection]: limiar, NMS, volta ao quadro original [R1]
- def detect(handle: p2.Handle, image_bgr, class_names, thresholds=None) -> list[Detection]: tamanho do `info` [R2]
- def class_names_for(engine_ref: str) -> list[str]: `metadata.json` da pasta do engine [R2]
- def draw(image_bgr, detections) -> ImageArray; def main(argv) -> int [R3]

### pyproject.toml (de 003)
- mypy `files`, ruff `src` e pytest `pythonpath` + `examples` [D2]

### docs/guia-integracao.md
- divisão de responsabilidades, p2-build na placa, Dockerfile (`p2[trt]`), montagem ro, código, BGR→RGB, erros, notebook, checklist [R4, D4]

### tests/test_yolo_demo.py
- letterbox, volta ao quadro original, classe sem nome, limiares, demo na bus.jpg, engine ausente, imagem ilegível [R1–R3]

## Validação
- `pytest`, `mypy`, `ruff check`, `ruff format --check`; venv limpo com o wheel

## Log
- 2026-10-30: criada (modo auto); usuário pediu para aguardar as ações externas
- 2026-10-30: T1–T3: demo na bus.jpg (CPU): bus 0,94, person 0,90/0,85/0,83/0,40, caixas conferidas na imagem anotada
- 2026-10-30: T4: `dist/p2-0.1.0-py3-none-any.whl` (28 KB): 19 módulos de `p2`, 3 entry points, sem tests/scripts/examples
- 2026-10-30: T5: venv limpo (scratchpad) com `[cpu]`: numpy 1.26.4, onnx/ort 1.23.2; p2-build, p2-list, load/infer (1, 84, 8400), p2-bench ok, p2 importado do site-packages
- 2026-10-30: pytest 219 passed, 3 skipped; mypy strict 44 arquivos; ruff ok
- 2026-10-30: achado: remoto `joaoVs17/p2_runtime` ≠ `ods-26-2/p2` das URLs (D6)
- 2026-10-30: .gitignore ampliado (engines TRT, bytecode, runs/, cobertura, .env, editores); commits 4d58487 e 2b8b479 na branch v1.0.0; tag anotada v0.1.0 em 2b8b479
- 2026-10-30: push e release não feitos: sem credencial do GitHub nesta sessão (HTTPS sem helper; SSH com ~/.ssh/github recusado em modo não interativo) e sem `gh`; pendentes com o usuário
