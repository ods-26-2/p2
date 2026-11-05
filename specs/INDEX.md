# Índice de specs

Próximo ID: 010

## Specs

| ID | Título | Estado | Escopo | Tags | Resumo |
|----|--------|--------|--------|------|--------|
| 009 | v0.2.0: forma dinâmica, várias etapas e tipos por tensor | ativa | `docs/contrato.md`, `plano-p2.md`, `docs/depois.md`, `docs/decisoes.md`, `src/p2/manifest.py`, `src/p2/build/package.py`, `src/p2/api.py`, `src/p2/handle.py`, `src/p2/backends/**`, `src/p2/build/**`, `src/p2/cli/bench.py`, `pyproject.toml`, `examples/whisper_demo.py`, `examples/whisper_example.py`, `examples/whisper_setup.py`, `tests/test_whisper_example.py`, `tests/test_whisper_setup.py`, `tests/test_dynamic.py`, `tests/test_whisper_demo.py`, `tests/helpers/dynamic_package.py`, `tests/helpers/whisper.py`, `tests/jetson/test_whisper.py` | v0.2.0, E8, implementacao, interface, contrato-1.1, forma-dinamica, perfil, lote, dtype, task, etapas, whisper, vad, ocr, reid | Implementação na pré-release 0.2.0a3; Whisper-small validado na CPU; falta TensorRT na placa |
| 008 | E7: exemplo, guia de integração e release 0.1.0 | ativa | `examples/**`, `docs/guia-integracao.md`, `pyproject.toml`, `tests/test_yolo_demo.py`, `tests/helpers/yolo11n.py`, `tests/fixtures/yolo11n-1.0.0/**`, `tests/test_yolo11n.py`, `tests/jetson/conftest.py`, `tests/jetson/test_build_trt.py`, `dist/` | E7, release, wheel, exemplo, yolo_demo, guia, integracao, I1, docker, letterbox, nms, yolo11n, fixture, models | Exemplo YOLO completo, guia para I1, release 0.1.0 publicada; faltam Jetson e I1 |
| 007 | E6: p2-bench, p2-list e orçamento | ativa | `src/p2/cli/bench.py`, `src/p2/cli/list.py`, `src/p2/build/platform.py`, `pyproject.toml`, `docs/orcamento.md`, `tests/test_cli_bench.py`, `tests/test_cli_list.py` | E6, p2-bench, p2-list, cli, memoria, orcamento, latencia, nvpmodel, jetson | Mede latência/memória, lista engines e compatibilidade, método do orçamento |
| 006 | E5: backend TensorRT e equivalência | ativa | `src/p2/backends/trt.py`, `src/p2/backends/variants.py`, `src/p2/api.py`, `src/p2/build/compiler.py`, `src/p2/build/output.py`, `typings/**`, `tests/helpers/yolo.py`, `tests/jetson/**`, `tests/test_variants.py`, `tests/test_yolo_helpers.py`, `tests/test_yolo11n.py` | E5, tensorrt, backend, gpu, cuda, cudart, equivalencia, yolo, nms, jetson | Backend TRT do load/infer, smoke do p2-build, equivalência CPU×FP32×FP16 |
| 004 | E1: contrato da API | ativa | `docs/contrato.md`, `README.md`, `docs/decisoes.md` | E1, contrato, api, 6c, erros, manifesto, pacote-model, MOD-1, B1, I1 | Contrato P2 1.0 para I1 e MOD-1: tipos, erros, exemplos, pacote |
| 003 | E0: base do projeto e verificação da Jetson | ativa | `README.md`, `docs/decisoes.md`, `docs/depois.md`, `scripts/**`, `typings/**`, `pyproject.toml`, `tests/test_check_jetson.py` | E0, readme, docs, decisoes, jetson, check_jetson, stubs, tensorrt, cuda | README, docs de decisões, stubs TRT/CUDA e verificação do ambiente da placa |
| 002 | p2-build: compilação TensorRT | ativa | `src/p2/build/compiler.py`, `src/p2/build/platform.py`, `src/p2/build/output.py`, `src/p2/cli/build.py`, `typings/tensorrt/**`, `tests/jetson/**`, `tests/test_platform.py`, `tests/test_build_compiled.py`, `tests/test_cli_build.py` | p2-build, tensorrt, compilação, jetson, E4 | Compila FP16/FP32 na Jetson, preenche variants/platform, smoke infer |
| 005 | E2: núcleo da lib e backend de CPU | fechada | `src/p2/api.py`, `src/p2/handle.py`, `src/p2/backends/**`, `src/p2/errors.py`, `src/p2/manifest.py`, `src/p2/__init__.py`, `typings/onnxruntime/**`, `tests/conftest.py`, `tests/test_api_cpu.py`, `tests/test_errors.py`, `tests/test_manifest.py`, `tests/test_yolo11n.py`, `tests/helpers/**` | E2, api, load, infer, info, release, handle, backend, onnxruntime, cpu, erros, manifesto | API 6c na CPU com ONNX Runtime, erros, trava, ciclo de vida |
| 001 | p2-build | fechada | `src/p2/build/package.py`, `src/p2/build/output.py`, `src/p2/cli/build.py`, `src/p2/manifest.py`, `src/p2/errors.py`, `pyproject.toml`, `requirements-dev.txt`, `tests/test_manifest.py`, `tests/test_package.py`, `tests/test_cli_build.py`, `tests/helpers/**`, `tests/test_yolo11n.py`, `tests/fixtures/yolo11n-1.0.0/**` | p2-build, cli, pacote-model, manifesto, onnx, E3 | Lê pacote MODEL, valida, grava pasta do engine_ref com manifesto (--cpu-only) |

## Regras globais

- 001/RG1: Python 3.10.12; ruff e mypy com alvo py310
- 001/RG2: nomes de código em inglês; documentação e mensagens de erro em português
- 001/RG3: logging via `getLogger(__name__)` sob a hierarquia `"p2"`; sem `print`
- 003/RG1: libs sem tipos (`tensorrt`, `cuda`) usam stubs mínimos em `typings/`; sem `Any`, sem `ignore_missing_imports`
- 009/RG2: pendências abertas ficam na seção abaixo; revisá-las antes de cada pedido de mudança e mantê-las atualizadas
- 009/RG3: o que funciona na 0.1.0 continua funcionando sem mudança para modelos estáticos (API, `Handle`, manifesto, saídas, erros)

## Pendências abertas

Atualizado em 2026-11-05 (pré-release v0.2.0a3 publicada). Ler antes de qualquer alteração; avisar o usuário quando o pedido tocar ou destravar uma pendência.

### v0.2.0 (009, branch `dev-v0.2.0`)
- Implementação 009/T5–T10 e T15–T18 na `v0.2.0a3`, validada na CPU com Whisper-small real; caminho TensorRT dinâmico nunca executado; engine do Whisper ainda não compilado. `git stash` "rascunho da implementação 0.2.0" pode ser descartado
- Confirmar 009/D17: `infer` do Whisper recebe log-mel `float32 [1, 80, 3000]`, áudio → mel no I5
- Decisões assumidas a revisar: 009/D1–D15, sobretudo D2 (`info` sem perfis), D3 (retorno `TensorArray`), D4 (manifesto v1 para estático); 009/D17, D18
- Enviar o Contrato P2 1.1 a I2, I4, I5, I7, I8, MOD-1, B5 e B1 (substitui o envio da 1.0 em 004/T3)
- Externos: B5 + VER-4 exportam o Whisper (encoder/decoder; opset ≥ 17 por causa do LayerNorm em FP16; cache KV? sem cache, exportar só `logits`); B5 + INV-2 convertem o MediaPipe; MOD-1 + B5 aceitam `task`/`dynamic_shapes`
- I5 (repo `../i5`, analisado em 2026-11-05): `src/p2_runtime.py` é substituto local com contrato próprio (áudio entra, `{"text", "confidence"}` sai; 1 `engine_ref`; `options` extras; `P2RuntimeError`); não troca direto pelo P2 real. `examples/whisper_example.py` (`WhisperBridge`, 009/D20) funciona via `p2_module` sem mudar o I5 (CPU, áudios em português). Falta: I5 copiar `whisper_example.py` e `whisper_demo.py`, reescrever `test_contrato_p2_runtime_*` e usar Python 3.10 e `numpy<2` na placa
- 009/T11 (Jetson): `pytest -m jetson` com Whisper-small FP16 de ponta a ponta; 009/T12: release `v0.2.0` final (`v0.2.0a3` é a pré-release atual)

### Jetson (v0.1.0)
- 003/T6: `check_jetson.py` com exit 0
- 002/T7: `pytest -m jetson`; `p2-build` do yolo11n e do app-epi
- 006/T9: `pytest -m jetson` completo; `reports/equivalencia/app-epi-1.0.0.md`; ajustar as tolerâncias após a 1ª execução
- 007/T7: `p2-bench app-epi-1.0.0` FP16 e FP32 por 30 min; preencher `docs/orcamento.md`; avisar o MOD-1
- 007/T8: limite de memória do container conta a memória do CUDA?
- 008/T6: wheel da URL num venv limpo na placa; 008/T8: primeira build do I1 publicando `artifact_version: "app-epi-1.0.0-fp16"`

### Modelos
- `app-plates-1.0.0` em uso na placa (pacote fora do repositório): detector YOLO de 4 classes `placa_{1L,2L}_{BR,ME}`, estático FP32, cabe no contrato 1.0; não está no plano: definir se substitui ou acompanha o `app-epi-1.0.0` e qual consumidor o usa

### Dados de teste
- `models/yolo11n.onnx` no clone (fora do git; usuário coloca); arquivos do Whisper em `models/` vêm de `examples/whisper_setup.py`
- ~20 imagens de pessoas com EPI em `tests/fixtures/imagens/` (plano); hoje só `assets/bus.jpg` e `assets/dogi.jpeg`
- Pacote MODEL do `app-epi-1.0.0` (depende do canal de B5/MOD-1)

### Outras equipes (plano)
- P3: imagem base com TensorRT 10.3 para o `FROM`; `/opt/ods/engines` somente leitura no container
- B5/MOD-1: canal de entrega do pacote MODEL; mAP oficial (0,936 ou 0,9517); mAP com o engine FP16
- MOD-1: correção da saída do app-epi para `[1, 7, 8400]` (vai junto com o contrato)
- I1/B1: o exemplo do evento `ods.i1.detections` usa `artifact_version: "app-epi-1.0.0"`; contrato e plano (E7) pedem o `engine_version` (`"app-epi-1.0.0-fp16"`): alinhar qual vale; confirmar o schema do evento com o `app-plates`

### Fechamento de specs
- 003, 002, 006, 007: após registrar as execuções na Jetson; 004: após o envio do contrato; 008: após a build do I1; 009: após T11–T12

### Adiados (docs/depois.md)
- CI, INT8, vários perfis por engine, tensores residentes na GPU, carregamento sob demanda, telemetria, orçamento das outras builds, busca no registro de B5
