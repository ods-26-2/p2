# 009 — v0.2.0: forma dinâmica, várias etapas e tipos por tensor

- Estado: ativa
- Criada: 2026-11-04
- Fechada: —
- Base: 001 (pacote, manifesto), 004 (contrato 1.0, RG1), 005 (API), 006 (backend TRT), 007 (bench), 008 (release 0.1.0); plano-p2.md (E8)
- Substituída por: —
- Tags: v0.2.0, E8, implementacao, interface, contrato-1.1, forma-dinamica, perfil, lote, dtype, task, etapas, whisper, vad, ocr, reid, I2, I4, I5, I7, I8

## Objetivo
- Interface comum da 6c para I5 (Whisper, VAD), I7 (OCR em 2 estágios), I2 (ReID em lote), I4 e I8 (modelos sem classes)
- `load`/`infer`/`info`/`release` sem mudança de assinatura; modelos estáticos (YOLO) iguais à 0.1.0
- Whisper-small em FP16 na Jetson (encoder e decoder como engines separados)

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
- R16: testes: ONNX sintético com eixo dinâmico; entrada int64; forma fora do perfil → `INVALID_INPUT_SHAPE`; lote de 4 numa chamada; manifesto v1 carregando; Jetson: Whisper-small FP16 (encoder + decoder) transcrevendo um áudio curto
- R17: plano com a etapa E8 (v0.2.0), pré-requisitos externos e pendências
- R18: eixo com `min == max` fixado no ONNX antes do parse do TensorRT (exports com todos os eixos simbólicos; `Conv` exige canal fixo)
- R19: ONNX com pesos externos → `BuildError`
- R22: `examples/whisper_example.py`: `WhisperBridge` com a forma do `p2_runtime` do I5 (`load`, `info`, `infer`, `release`; áudio entra, `{text, confidence, token_probs}` sai); ignora `options` exceto `precision`; áudio > 30 s em trechos
- R21: `examples/whisper_setup.py`: baixa (hashes fixos), monta os 2 pacotes e compila; seção no guia rápido
- R20: `examples/whisper_demo.py`: WAV 16 kHz mono → log-mel → encoder → decodificação gulosa → texto, só com a API pública

## Regras
- Valem 001/RG1–RG3, 003/RG1, 004/RG1, 005/RG1, 007/RG1 (exceção em D14)
- RG1: modelo estático só float32 gera manifesto v1 idêntico ao da 0.1.0 (D4)
- RG3 [global]: o que funciona na 0.1.0 continua funcionando sem mudança para quem usa modelo estático: API, métodos públicos do `Handle`, manifesto, saídas e erros (pedido do usuário)
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
- D12 [assumida] implementação adiada; rascunho incompleto em `git stash` "rascunho da implementação 0.2.0" [substituída: implementação retomada a pedido do usuário]
- D13 [assumida] pendências centralizadas no INDEX (sempre carregado) + regra no CLAUDE.md — pedido do usuário
- D14 [assumida] smoke do `p2-build` na forma `max`; `p2-bench` na forma `opt`, lendo o manifesto (float aleatório; int e bool zerados) — `info` não tem dtype nem perfil
- D16 [aprovada] pré-release `v0.2.0a1` com a interface (comportamento da 0.1.0); `v0.2.0` final só com o Whisper funcionando; `main` avança junto — escolha do usuário
- D17 [assumida] `infer` do Whisper recebe log-mel `float32 [1, 80, 3000]`, não áudio; áudio → mel fica no I5 — pré-processamento fora do P2; STFT no grafo não é suportado pelo parser do TensorRT 10.3; aguarda confirmação do usuário
- D18 [assumida] `whisper_demo.py` com mel em numpy e BPE próprio, sem dependência nova — referência copiável para o I5
- D19 [rejeitada] aviso de NaN/infinito no smoke do `p2-build` → removido — o P2 não faz mais do que o planejado (pedido do usuário)
- D20 [aprovada] ponte áudio → texto como exemplo no P2, para o I5 copiar; fora da lib — escolha do usuário (opção B)
- D15 [assumida] erros de pacote (perfil, task, dtype, input_shape) → `BuildError`, exit 1, mensagem em português, sem gravar nada (001/RG4)

## Tarefas
- [x] T1: spec, plano (E8), `docs/depois.md` (R17)
- [x] T2: `docs/contrato.md` 1.1 (R1–R14, D1–D9)
- [x] T3: tipos públicos no código; versão 0.2.0.dev0 (R15, D3, D10, D11)
- [x] T4: `docs/decisoes.md`, README; INDEX com pendências; regra no CLAUDE.md (D13, RG2)
- [x] T5: manifesto v2: leitura, gravação v1/v2, `max_batch` (R4, R5, R13, D4–D6)
- [x] T6: pacote por `task`, `input_shape` em dicionário, `dynamic_shapes`, dtypes (R2, R10–R12, D7, D8, D15)
- [x] T7: `infer` com dtype e perfil; ORT preserva o dtype (R6, R9)
- [x] T8: TRT: perfil no compilador, `set_input_shape`, buffers em `max`, saídas por chamada, eixos fixos (R3, R7, R8, R18, D9); não executado com TensorRT real
- [x] T9: smoke e `p2-bench` com formas e dtypes (D14)
- [x] T10: testes do R16 no notebook; versões Jetson dos testes
- [x] T15: `examples/whisper_demo.py`, pesos externos (R19, R20, D17, D18)
- [x] T19: pré-release `v0.2.0a3` com a implementação (D16)
- [x] T18: `examples/whisper_example.py`, testes e guia rápido (R22, D20)
- [x] T17: `examples/whisper_setup.py`, testes e guia rápido (R21)
- [x] T16: análise de compatibilidade com a 0.1.0; `Handle.input_shapes()` e `FloatArray` restaurados (R13, RG3)
- [ ] T11 (usuário, Jetson): `pytest -m jetson`, incluindo Whisper-small FP16 de ponta a ponta
- [x] T13: pré-release `v0.2.0a1` (D16)
- [x] T14: pré-release `v0.2.0a2`: example.py do app-plates e guia rápido (D16; 008/T12–T13)
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

### src/p2/manifest.py, api.py, handle.py, backends/ort.py (implementação)
- SCHEMA_VERSION = 2, STATIC_SCHEMA_VERSION = 1, SUPPORTED_SCHEMAS, STATIC_DTYPE, PROFILE_KEYS (Final) [R4, D4]
- def manifest_schema(inputs, outputs) -> int; def max_batch(inputs) -> int; def profile_problem(shape, profile) -> str | None; def is_dynamic(shape) -> bool [R2, R4, R5, D4, D5]
- api.shape_fits(shape, spec) -> bool; _check_array valida dtype e perfil; Handle.input_specs() [R6, R9]
- OrtSession.run e Handle.run preservam o dtype [R9]

### src/p2/build/package.py, output.py, compiler.py (de 001, 002, 006)
- ModelPackage + task, dynamic_shapes; input_shape: DeclaredShapes; read_onnx_io: eixo simbólico → -1, pesos externos → BuildError [R2, R10–R12, R19, D8]
- def pin_fixed_axes(model: bytes, profiles) -> bytes [R18]
- output: with_profiles, io_matches (eixo -1 aceita qualquer valor); EngineCompiler.compile(..., profiles) [R3, R18]
- TrtCompiler: perfil de otimização; smoke em `max` com dtype [R3, D14]

### src/p2/backends/trt.py (de 006)
- class DeviceInput, DeviceOutput (substituem DeviceTensor); max_shape(spec) -> list[int]; TrtSession.run: set_input_shape, formas de saída do contexto, cópia do tamanho real [R6–R9, D9]

### src/p2/cli/bench.py (de 007)
- def bench_inputs(specs) -> dict[str, TensorArray]: forma `opt`, float aleatório, int/bool zerados [D14]

### examples/whisper_demo.py
- Tokenizer(prompt, end_of_text, decode); read_tokenizer, read_wav (WAV 16 kHz mono), mel_filters, log_mel(samples) -> [1, 80, 3000] [R20, D17, D18]
- def transcribe(encoder, decoder, samples, *, tokenizer, language) -> str; main [R20]

### examples/whisper_example.py
- class WhisperBridge: load(engine_ref, options) -> BridgeHandle, info, infer(handle, audio) -> (Transcript, Metrics), release [R22, D20]
- whisper_demo.generate(...) -> (tokens, probabilidades) [R22]

### examples/whisper_setup.py, docs/guia-rapido.md (de 008)
- class RemoteFile, Stage; ENCODER, DECODER, FILES (Final); download, convert_sample, write_package(directory, stage, onnx_path) -> Path, main [R21]

### tests/
- helpers/dynamic_package.py: ONNX com eixos dinâmicos, int64, bool, escalar [R16]
- helpers/whisper.py: pacotes de `models/whisper-*` via `whisper_setup` [R16]; test_whisper_setup.py: pacote, hash, download [R21]; test_whisper_example.py: formato do I5, options, trechos, áudio vazio, release [R22]
- test_dynamic.py: manifesto v1/v2, info, lote, dtypes, perfil, tasks, validações, eixos fixos, bench [R2–R13, R18, R19]
- test_whisper_demo.py: wav, log-mel, tokenizer, transcrição na CPU (skip sem modelo) [R16, R20]
- jetson/test_trt.py, jetson/test_whisper.py: dinâmico na GPU; Whisper-small FP16 [R16]

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
- 2026-11-04: pré-release `v0.2.0a1` publicada via `gh` (--prerelease) com o wheel; download pela URL com sha256 6c9935cf…bd3036d; `v0.1.0` segue como Latest; develop, main e dev-v0.2.0 no remoto
- 2026-11-04: T14: versão 0.2.0a2; wheel sha256 6514602f…dd29f4cc, 19 módulos; venv limpo [cpu] ok; pytest 217 passed, 11 skipped; mypy e ruff ok
- 2026-11-05: pré-release `v0.2.0a2` publicada via `gh` (--prerelease); download pela URL com sha256 6514602f…dd29f4cc; `v0.1.0` segue como Latest; develop e main em 3b394ed
- 2026-11-05: T5–T10, T15 (branch dev-v0.2.0, modo auto): rascunho do stash concluído; achados: `np.ascontiguousarray` virava escalar em 1-d; encoder real tem canal simbólico (R18); decoder real tem 49 saídas
- 2026-11-05: Whisper-small real (onnx-community, IR 7, opset 14) na CPU: "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."; pytest com modelos 294 passed, 4 skipped; sem modelos 285 passed, 13 skipped; mypy strict 53 arquivos; ruff ok; nada executado com TensorRT real
- 2026-11-05: T16: testes da tag v0.1.0 no código novo: 208 de 222 passam; 11 falhas só em interno ou intencional (exportações/versão, compilador falso, schema 2, mensagem de eixo dinâmico). Lado a lado com o wheel 0.1.0 e yolo11n: manifesto idêntico, saídas bit a bit iguais (lib × pasta), erros e mensagens iguais. TensorRT/CUDA simulados: mesmas 77 chamadas na mesma ordem. Regressão achada e corrigida: `Handle.input_shapes()`
- 2026-11-05: T17: whisper_setup.py na CPU com download real do tokenizer e do áudio: transcrição ok; pasta existente e falta de TensorRT dão erro claro; pytest 291 passed, 13 skipped; mypy strict 55 arquivos; ruff ok
- 2026-11-05: T18: `TranscritorWhisper` do I5 sem alteração (cópia) com `WhisperBridge` na CPU: 4 áudios em português e 1 de 40 s; envelope com texto, confiança e `artifact_version`; pytest com modelos 310 passed, 4 skipped; sem modelos 300 passed, 14 skipped; mypy strict 57 arquivos; ruff ok
- 2026-11-05: auditoria de escopo de `src/p2`: sem lógica de Whisper, sem dependência nova, API igual + `TensorArray`; removido warn_non_finite (D19); extras mantidos: R18 (necessário para compilar) e R19 (recusa); engine do Whisper ainda não compilado
- 2026-11-05: T19: versão 0.2.0a3; wheel sha256 9bb07c2a…5a4c45, 19 módulos; venv limpo [cpu] ok; com modelos reais pytest 310 passed, 4 skipped; mypy e ruff ok
