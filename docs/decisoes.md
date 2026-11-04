# Decisões do P2

Decisões fechadas do projeto. Qualquer dúvida nova durante a implementação vira uma linha a mais aqui. A coluna "Origem" indica de onde veio: `plano` (seção "Decisões consolidadas" do [plano](../plano-p2.md)) ou a spec e a decisão (`NNN/D#`, em [specs/](../specs/INDEX.md)).

## Consolidadas no plano

| Tema | Decisão | Origem |
| --- | --- | --- |
| Linguagem | Python 3.10.12 | plano |
| Código | Nomes em inglês, sem comentários | plano |
| Documentação e mensagens de erro | Em português | plano |
| Distribuição | Wheel `p2-<versão>-py3-none-any.whl` anexado a uma release de `ods-26-2/p2`; `import p2` | plano |
| URL da 0.1.0 | `https://github.com/ods-26-2/p2/releases/download/v0.1.0/p2-0.1.0-py3-none-any.whl` | plano |
| Dependências obrigatórias | `numpy>=1.21.6,<2` e `typing_extensions>=4.10` | plano, 001/D2 |
| Extra `cpu` | `onnxruntime>=1.19`, `onnx>=1.16` | plano, 001/D3 |
| Extra `trt` | `cuda-python==12.6.*` (`cudart`), `onnx>=1.16`; o `tensorrt` vem do JetPack, não do pip | plano, 001/D3 |
| Ambiente da placa | JetPack 6.2.1, L4T 36.4.7, TensorRT 10.3.0, CUDA 12.6 | plano |
| Erros da 6c | `p2.P2Error(code, detail)` com os 5 códigos | plano |
| Erros de uso | `ValueError` (opção desconhecida, handle já liberado) | plano |
| `options` | Só `precision`: `"FP16"` (padrão) ou `"FP32"` | plano |
| Entrada do `infer` | Array solto ou `{nome: array}`; float32; forma exata do `info`; tipo errado gera `INVALID_INPUT_SHAPE`; array não contíguo é copiado | plano |
| Saída do `infer` | Tupla `({nome: ndarray}, {"latency_ms", "gpu_memory_mb"})`, com arrays copiados | plano |
| `latency_ms` | Tempo total da chamada, incluindo as cópias entre CPU e GPU | plano |
| `gpu_memory_mb` | Memória do handle (pesos, ativações e buffers), calculada no `load` e constante por handle | plano |
| `engine_version` | `<engine_ref>-fp16` ou `<engine_ref>-fp32` | plano |
| Concorrência | Trava por handle; chamadas simultâneas rodam em sequência | plano |
| Ciclo de vida do handle | `release` idempotente; suporte a `with p2.load(...) as h:` | plano |
| Backends | TensorRT ou CPU, escolhido automaticamente; forçável com `P2_BACKEND=trt\|ort` | plano |
| Precisão na CPU | Só FP32. Sem `precision`, a CPU usa FP32 e o `info` informa; FP16 explícito gera `ENGINE_NOT_FOUND` | plano |
| Pasta dos engines | `P2_ENGINE_DIR`, padrão `/opt/ods/engines/<engine_ref>/` | plano |
| `engine_ref` | `artifact_id-artifact_version` do `metadata.json`, por exemplo `app-epi-1.0.0` | plano |
| Ferramenta de compilação | `p2-build`, instalada com o wheel; roda no sistema da Jetson | plano |
| Modo sem GPU | `p2-build --cpu-only` monta a pasta só com ONNX, metadados e manifesto | plano |
| Precisões compiladas | FP16 e FP32 por padrão; `--precision` escolhe | plano |
| TF32 | Desligado no engine FP32 | plano |
| Memória de trabalho do compilador | 1 GB | plano |
| Validações antes de compilar | Hash, `input_shape` e `4 + len(classes)`: qualquer divergência faz a compilação falhar | plano |
| Engine já existente | Recusa sobrescrever; `--force` permite | plano |
| Teste após compilar | Um `infer` com entrada zerada em cada variante | plano |
| Logs | Módulo `logging`, logger `"p2"`, sem `print` | plano |
| Testes | `pytest`; testes de TensorRT marcados para rodar só na placa | plano |
| Orçamento | 1.024 MB por processo de I1, provisório; uma aplicação por vez na placa | plano |

## Tomadas durante a implementação

| Tema | Decisão | Origem |
| --- | --- | --- |
| Recursos de tipagem no 3.10 | `typing_extensions` para `override`, `Self`, `TypeIs`; `TypeVar` no lugar de PEP 695; `class X(str, Enum)` no lugar de `StrEnum` | 001/D2 |
| Leitura de I/O do ONNX | Pacote `onnx`, só o grafo, sem executar | 001/D3 |
| Exceções | `P2BaseError` é a base do pacote; `BuildError` (p2-build) e `P2Error` (API) herdam dela | 001/D5 |
| `--force` | Monta em `<raiz>/.<engine_ref>.tmp-*` e troca só no sucesso; em falha a pasta anterior fica intacta | 001/D4, 001/D13 |
| `--precision` com `--cpu-only` | Erro de uso (código 2) | 001/D6 |
| Backend de build | `setuptools>=68`, layout `src/` | 001/D8 |
| numpy no notebook | `<2` (1.26.x), igual ao da Jetson; opencv `<4.12` | 001/D9 |
| Campo `classes` do `metadata.json` | Lista não vazia de nomes | 001/D10 |
| `onnx_file` em subpasta | Copiado só com o nome do arquivo; o manifesto registra esse nome | 001/D12 |
| Saída do `p2-build` | `logging` em stderr (`NÍVEL: mensagem`); exit 0 sucesso, 1 erro, 2 uso | 001/D15 |
| `p2-bench` e `p2-list` no wheel | Entry points instalados com o wheel desde a E6 | 003/D1, 007/R10 |
| Memória no `p2-bench` | `MemAvailable` antes do `load`, depois do `load` e mínimo durante a execução (amostra a cada 0,5 s); limite = pico × 1,3 | 007/R5, 007/D2 |
| Entrada do `p2-bench` | Aleatória, float32, semente 0, gerada uma vez antes do laço | 007/R3, 007/D5 |
| Saída do `p2-list` | Tabela em stdout via `logging` | 007/D3 |
| Bibliotecas sem tipos | Stubs mínimos em `typings/` para `tensorrt` e `cuda.cudart` | 003/D2 |
| `P2Error.code` | `p2.ErrorCode` (`str` + `Enum`), compara igual à string do código | 004/D1 |
| Tipos públicos | `p2.Handle`, `p2.EngineInfo` e `p2.InferStats` exportados para anotação | 004/D2 |
| `P2Error.detail` | Texto em português; `str(e)` é `"CODE: detalhe"` | 004/D3 |
| Handle liberado | `infer`/`info` depois do `release` geram `ValueError`; `release` repetido não faz nada | 004/D4 |
| Versão do contrato | "Contrato P2 1.0", válido para a lib 0.1.x; "Contrato P2 1.1" para a 0.2.x, no mesmo arquivo (a 1.0 fica na tag `v0.1.0`) | 004/D5, 009/D1 |
| IR do ONNX | Pacote MODEL com ONNX IR ≤ 10; ONNX gerado nos testes e no `check_jetson.py` fixa `ir_version=10` | 004/D6 |
| `import p2` | Não importa `onnxruntime` nem `tensorrt`; o backend é importado só no `load` | 005/RG1 |
| Backend automático | TensorRT se `tensorrt` e `cuda-python` importáveis, senão CPU; `P2_BACKEND=trt` sem as libs gera `ENGINE_NOT_FOUND` | 006/R1 |
| Pasta `--cpu-only` no backend TRT | `ENGINE_NOT_FOUND` (não cai para a CPU sozinho); use `P2_BACKEND=ort` | 006/D3 |
| Engine corrompido | sha256 do `.engine` conferido no `load`; diferente gera `ENGINE_NOT_FOUND` | 006/D2 |
| Backend sem a lib instalada | `P2Error(ENGINE_NOT_FOUND)` com a dica `pip install 'p2[cpu]'` | 005/D5 |
| Manifesto de outro `engine_ref` | `ENGINE_NOT_FOUND` | 005/D6 |
| Teste após compilar | Desserializa cada engine, confere I/O com o ONNX e roda um `infer` com entrada zerada antes de publicar a pasta | 002/D1, 006/R8 |
| `platform.tensorrt` | 3 componentes (`10.3.0.30` → `10.3.0`) | 002/D4 |
| `platform` incompleto | `jetpack`, `l4t` ou `cuda` ilegíveis são omitidos com aviso; `tensorrt` sempre presente | 002/D5 |
| ONNX diferente do manifesto | Entradas ou saídas com nomes diferentes geram `ENGINE_NOT_FOUND` no `load` | 005/D10 |
| Etapas de um modelo | Um `engine_ref` e um pacote MODEL por etapa; o laço entre etapas fica no consumidor | 009/R1 |
| `info` na 0.2.x | Mantém os 5 campos da 6c; `-1` nos eixos dinâmicos; perfis e dtypes só no `manifest.json` | 009/D2 |
| Tipo de retorno do `infer` | `dict[str, p2.TensorArray]` (`NDArray[np.generic]`); o consumidor converte com `np.asarray(x, dtype=...)` | 009/D3 |
| Versão do manifesto | v1 para modelo estático só com `float32` (legível pela lib 0.1.0); v2 com `-1`, `profile` e dtypes | 009/D4 |
| `max_batch` | Maior primeira dimensão entre as entradas (`max[0]` do perfil ou forma fixa); sem entrada com eixo, 1 | 009/D5 |
| Tensores escalares | `shape: []` (por exemplo a taxa de amostragem do VAD) | 009/D6 |
| dtypes aceitos | `float32`, `int32`, `int64`, `bool`; outros na I/O são recusados pelo `p2-build` | 009/R9, 009/D8 |
| Saída com forma dependente dos dados | Não suportada; `ENGINE_NOT_FOUND` no `load` | 009/D9 |
