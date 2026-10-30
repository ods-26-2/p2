# Índice de specs

Próximo ID: 009

## Specs

| ID | Título | Estado | Escopo | Tags | Resumo |
|----|--------|--------|--------|------|--------|
| 008 | E7: exemplo, guia de integração e release 0.1.0 | ativa | `examples/**`, `docs/guia-integracao.md`, `pyproject.toml`, `tests/test_yolo_demo.py`, `tests/helpers/yolo11n.py`, `tests/fixtures/yolo11n-1.0.0/**`, `tests/test_yolo11n.py`, `tests/jetson/conftest.py`, `tests/jetson/test_build_trt.py`, `dist/` | E7, release, wheel, exemplo, yolo_demo, guia, integracao, I1, docker, letterbox, nms, yolo11n, fixture, models | Exemplo YOLO completo, guia para I1, wheel 0.1.0 testado; release aguarda confirmação |
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
