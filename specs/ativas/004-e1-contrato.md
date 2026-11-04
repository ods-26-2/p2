# 004 — E1: contrato da API

- Estado: ativa
- Criada: 2026-10-30
- Fechada: —
- Base: 001 (pacote MODEL, manifesto, p2-build), 003 (decisoes.md); plano-p2.md (E1, "Contrato da API", "Pacote MODEL e manifesto")
- Substituída por: —
- Tags: E1, contrato, api, 6c, erros, manifesto, pacote-model, MOD-1, B1, I1

## Objetivo
- `docs/contrato.md`: o que I1 e MOD-1 precisam para programar contra o P2 sem ler o código

## Fora de escopo
- Implementação da API (E2)
- Envio ao MOD-1 e ao B1 (usuário)

## Requisitos
- R1: assinaturas de `load`, `infer`, `info`, `release`, com tipos, retornos e erros por função
- R2: `P2Error` (atributos, `str`, os 5 códigos e quando ocorre cada um) e `ValueError` de uso
- R3: semântica: entrada/saída do `infer`, `latency_ms`, `gpu_memory_mb`, `engine_version`, precisão na CPU, concorrência, ciclo de vida e `with`, variáveis `P2_ENGINE_DIR` e `P2_BACKEND`
- R4: exemplos com `app-epi-1.0.0`: `info`, `infer` e tratamento de erro com `try`
- R5: pacote MODEL que o MOD-1 entrega (campos, validações do `p2-build`), pasta gerada e `manifest.json`
- R6: instalação para I1 (URL da 0.1.0 e branch principal até lá)

## Regras
- Valem 001/RG2 (português) e as decisões de `docs/decisoes.md`
- RG1: mudança no contrato depois de enviado gera nova versão do documento e nova spec

## Decisões
- D1 [assumida] `P2Error.code` é `p2.ErrorCode` (`str` + `Enum`): compara igual à string (`e.code == "ENGINE_NOT_FOUND"`) — conjunto fechado (PYTHON.md) sem obrigar I1 a importar o enum
- D2 [assumida] tipos públicos exportados para anotação: `p2.Handle`, `p2.EngineInfo`, `p2.InferStats` (TypedDict) — I1 tipa sem `dict` genérico
- D3 [assumida] `P2Error.detail: str` em português; `str(e) == f"{code}: {detail}"` — formato do plano/E2
- D4 [assumida] `info`/`infer` em handle liberado → `ValueError`; `release` repetido não faz nada — plano: "handle usado depois do release"
- D6 [assumida] pacote MODEL exige ONNX IR ≤ 10 — onnx 1.23 gera IR 14 por padrão, ORT 1.23 aceita até 11, TRT 10.3 vem com onnx 1.16 (IR 10); achado na 005
- D5 [assumida] contrato versionado como "Contrato P2 1.0", válido para a lib 0.1.x — referência estável para B1

## Tarefas
- [x] T1: `docs/contrato.md` (R1–R6)
- [x] T2: link no README e linhas novas em `docs/decisoes.md`
- [ ] T3 (usuário): enviar ao MOD-1 e ao B1; registrar no Log

## Mapa de código

### docs/contrato.md
- Contrato P2 1.0: instalação, API, erros, exemplos app-epi, pacote MODEL, manifesto [R1–R6, D1–D5] (→ 009: Contrato P2 1.1; a 1.0 fica na tag v0.1.0)

### README.md (de 003), docs/decisoes.md (de 003)
- link para o contrato; linhas 004/D1–D5 [T2]

## Validação
- conferência manual contra plano-p2.md e docs/decisoes.md

## Log
- 2026-10-30: criada (modo auto)
- 2026-10-30: T1–T2: contrato conferido contra plano-p2.md, decisoes.md e o p2-build atual (dtype só registrado, não validado)
- 2026-10-30: requisito de IR ≤ 10 acrescentado ao contrato (D6), antes do envio
- 2026-11-04: contrato passou para a 1.1 na 009 (RG1); o envio da T3 deve usar a 1.1
