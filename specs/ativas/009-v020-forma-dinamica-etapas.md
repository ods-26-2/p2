# 009 — v0.2.0: forma dinâmica, várias etapas e tipos por tensor

- Estado: ativa
- Criada: 2026-11-04
- Fechada: —
- Base: 001 (pacote, manifesto), 004 (contrato 1.0, RG1), 005 (API), 006 (backend TRT), 007 (bench), 008 (release 0.1.0); plano-p2.md (E8)
- Substituída por: —
- Tags: v0.2.0, E8, interface, contrato-1.1, forma-dinamica, perfil, lote, dtype, task, etapas, whisper, vad, ocr, reid, I2, I4, I5, I7, I8

## Objetivo
- Interface comum da 6c para I5 (Whisper, VAD), I7 (OCR em 2 estágios), I2 (ReID em lote), I4 e I8 (modelos sem classes)
- `load`/`infer`/`info`/`release` sem mudança de assinatura; modelos estáticos (YOLO) iguais à 0.1.0
- Nesta etapa: só a interface (contrato 1.1, formatos, tipos públicos). Implementação aguarda decisão do usuário

## Fora de escopo
- Laço entre etapas (decodificação token a token em I5, recorte da placa em I7): fica no consumidor
- Pré/pós-processamento; INT8; telemetria; vários perfis de otimização por engine
- Export dos ONNX (Whisper, MediaPipe): B5, VER-4, INV-2

## Requisitos
- R1: etapas como engines separados: cada etapa é um `engine_ref` e um pacote MODEL próprios (ex.: `whisper-small-encoder-1.0.0`, `whisper-small-decoder-1.0.0`)
- R2: `metadata.json` com `dynamic_shapes` opcional: `{entrada: {"min": [...], "opt": [...], "max": [...]}}`; eixos fixos repetem o valor; `0 < min <= opt <= max`; todo eixo dinâmico das entradas do ONNX precisa de perfil; perfil em entrada sem eixo dinâmico é recusado
- R3: `p2-build` cria um perfil de otimização do TensorRT com esses limites
- R4: manifesto `schema_version: 2`: eixo dinâmico = `-1` em `shape`; entrada dinâmica com bloco `profile` {min, opt, max}; v1 continua sendo lido sem mudança
- R5: `info`: `-1` nos eixos dinâmicos; `max_batch` = primeira dimensão de `max` (D5)
- R6: `infer` aceita qualquer forma entre `min` e `max`; fora → `INVALID_INPUT_SHAPE`; formas de saída calculadas a cada chamada
- R7: buffers da GPU alocados no `load` para a forma `max`; `gpu_memory_mb` reflete esse máximo
- R8: lote > 1: primeira dimensão dinâmica processa N recortes numa chamada (I2, I7)
- R9: dtype por tensor no manifesto: `float32`, `int32`, `int64`, `bool`; `infer` valida o declarado (antes: só float32); tipo errado → `INVALID_INPUT_SHAPE`; saídas no dtype do tensor
- R10: `metadata.json` com `task`: `detect`, `classify`, `segment`, `ocr`, `reid`, `speech-encoder`, `speech-decoder`, `vad`, `other`; ausente = `detect`
- R11: `4 + len(classes)` e 1 entrada/1 saída só em `detect`; `classes` opcional nas demais
- R12: `input_shape` em lista (1 entrada) ou `{nome: forma}`; `-1` nos eixos dinâmicos; igual ao ONNX
- R13: compatibilidade: manifesto v1 carrega igual; pacote sem `task` = `detect`; YOLO estático com o mesmo manifesto e o mesmo `info` da 0.1.0
- R14: `docs/contrato.md` → "Contrato P2 1.1" (lib 0.2.x), com o que muda para cada grupo (004/RG1)
- R15: tipos públicos da interface no código: `p2.TensorArray`; `ShapeProfile`, `TensorSpec.profile`, `DYNAMIC_DIM`, `TENSOR_DTYPES` em `manifest.py`; `Task` em `build/package.py`
- R16: testes: ONNX sintético com eixo dinâmico; entrada int64; forma fora do perfil → `INVALID_INPUT_SHAPE`; lote de 4 numa chamada; manifesto v1 carregando; Jetson: Whisper tiny (encoder + decoder) transcrevendo um áudio curto
- R17: plano com a etapa E8 (v0.2.0), pré-requisitos externos e pendências

## Regras
- Valem 001/RG1–RG3, 003/RG1, 004/RG1, 005/RG1, 007/RG1 (exceção em D14)
- RG1: modelo estático só float32 gera manifesto v1 idêntico ao da 0.1.0 (D4)
- RG2 [global]: pendências abertas ficam em `specs/INDEX.md` ("Pendências abertas") e são revisadas antes de cada pedido de mudança (D13)

## Decisões
- D1 [assumida] contrato 1.1 no mesmo `docs/contrato.md`, com "Mudanças desde a 1.0"; a 1.0 fica na tag `v0.1.0` — um documento vigente só
- D2 [assumida] `info` mantém os 5 campos da 6c; faixas min/opt/max só no `manifest.json` — 6c fechada; revisar se I5 precisar ler o `max` em tempo de execução
- D3 [assumida] `infer` devolve `dict[str, p2.TensorArray]` (`NDArray[np.generic]`) — saídas podem ser int64/bool; só a anotação muda
- D4 [assumida] `p2-build` grava v1 quando tudo é estático e float32, v2 caso contrário — pastas de modelos estáticos seguem legíveis pela lib 0.1.0 de I1
- D5 [assumida] `max_batch` = maior primeira dimensão (de `max`, ou da forma fixa) entre entradas com pelo menos 1 eixo; nenhuma → 1
- D6 [assumida] tensores escalares aceitos com `shape: []` (ex.: `sr` int64 do VAD)
- D7 [assumida] `input_shape` obrigatório em todas as tasks — conferência explícita com o ONNX
- D8 [assumida] I/O com outro dtype (float16, uint8 etc.) → `p2-build` recusa — conjunto fechado do R9
- D9 [assumida] saída com forma dependente dos dados (ex.: `NonZero`) não suportada → `ENGINE_NOT_FOUND` no `load` TRT
- D10 [assumida] versão `0.2.0.dev0` na `dev-v0.2.0` até a implementação [substituída por D16]
- D11 [assumida] tipos do R15 entram já, sem uso em runtime — exceção a "sem abstrações especulativas": pedido explícito de interface
- D12 [assumida] implementação adiada; rascunho incompleto em `git stash` "rascunho da implementação 0.2.0"
- D13 [assumida] pendências centralizadas no INDEX (sempre carregado) + regra no CLAUDE.md — pedido do usuário
- D14 [assumida] smoke do `p2-build` na forma `max`; `p2-bench` na forma `opt`, lendo o manifesto (float aleatório; int e bool zerados) — `info` não tem dtype nem perfil
- D16 [aprovada] pré-release `v0.2.0a1` com a interface (comportamento da 0.1.0); `v0.2.0` final só com o Whisper funcionando; `main` avança junto — escolha do usuário
- D15 [assumida] erros de pacote (perfil, task, dtype, input_shape) → `BuildError`, exit 1, mensagem em português, sem gravar nada (001/RG4)

## Tarefas
- [x] T1: spec, plano (E8), `docs/depois.md` (R17)
- [x] T2: `docs/contrato.md` 1.1 (R1–R14, D1–D9)
- [x] T3: tipos públicos no código; versão 0.2.0.dev0 (R15, D3, D10, D11)
- [x] T4: `docs/decisoes.md`, README; INDEX com pendências; regra no CLAUDE.md (D13, RG2)
- [ ] T5 (aguarda decisão): manifesto v2: leitura, gravação v1/v2, `max_batch` (R4, R5, R13, D4–D6)
- [ ] T6 (aguarda decisão): pacote por `task`, `input_shape` em dicionário, `dynamic_shapes`, dtypes (R2, R10–R12, D7, D8, D15)
- [ ] T7 (aguarda decisão): `infer` com dtype e perfil; ORT preserva o dtype (R6, R9)
- [ ] T8 (aguarda decisão): TRT: perfil no compilador, `set_input_shape`, buffers em `max`, saídas por chamada (R3, R7, R8, D9)
- [ ] T9 (aguarda decisão): smoke e `p2-bench` com formas e dtypes (D14)
- [ ] T10 (aguarda decisão): testes do R16 no notebook; versões Jetson dos testes
- [ ] T11 (usuário, Jetson; depende de B5/VER-4): Whisper tiny de ponta a ponta
- [x] T13: pré-release `v0.2.0a1` (D16)
- [ ] T12 (externo, com confirmação): release `v0.2.0`

## Mapa de código

### docs/contrato.md (de 004)
- Contrato P2 1.1: etapas, forma dinâmica, lote, dtype, `task`, manifesto v2, compatibilidade, o que muda por grupo [R1–R14]

### src/p2/backends/base.py (de 005), src/p2/handle.py (de 005), src/p2/api.py (de 005, 006), src/p2/__init__.py (de 005)
- TensorArray: TypeAlias = NDArray[np.generic]; `infer` e `Handle.run` devolvem `dict[str, TensorArray]`; `p2.TensorArray` exportado [R15, D3]

### src/p2/manifest.py (de 001, 005)
- DYNAMIC_DIM: Final = -1; TENSOR_DTYPES: Final[frozenset[str]]; class ShapeProfile(TypedDict): min, opt, max; TensorSpec.profile: NotRequired[ShapeProfile] [R4, R9, R15]

### src/p2/build/package.py (de 001)
- class Task(str, Enum): 9 tarefas; DEFAULT_TASK = Task.DETECT [R10, R15]

### examples/yolo_demo.py (de 008)
- `detect` converte a saída para float32 antes do pós-processamento [D3]

### plano-p2.md, docs/depois.md, docs/decisoes.md, README.md, specs/INDEX.md, CLAUDE.md
- E8 no plano; itens adiados → E8; decisões 009; pendências abertas [R17, D13]

## Validação
- `pytest`, `mypy`, `ruff check`, `ruff format --check src tests scripts examples`

## Log
- 2026-11-04: criada (modo auto) a partir do texto do usuário ("v0.1.1" → v0.2.0); escopo reduzido pelo usuário a "interface apenas"; rascunho da implementação guardado no stash (D12)
- 2026-11-04: T1–T4: contrato 1.1, plano E8, tipos públicos, decisões, pendências no INDEX; pytest 213 passed, 9 skipped (219/3 com models/yolo11n.onnx); mypy strict 45 arquivos; ruff ok; nada commitado
- 2026-11-04: T13: versão 0.2.0a1; wheel p2-0.2.0a1-py3-none-any.whl sha256 6c9935cf…bd3036d, 19 módulos, sem tests/scripts/examples/assets; venv limpo [cpu] ok; pytest 215 passed, 10 skipped; mypy e ruff ok; commit 990b3da + merge bcbbd37 na develop
