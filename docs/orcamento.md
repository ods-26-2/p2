# Orçamento de memória do P2

Limite de memória por processo de I1 na Jetson Orin Nano, medido com `p2-bench`. O valor em vigor é o **provisório de 1.024 MB** até a medição abaixo ser feita; uma aplicação por vez na placa.

> **Situação:** medições pendentes (E6). Os campos marcados com "—" são preenchidos com os relatórios de `reports/bench/`.

## Método

1. Placa em repouso: só o sistema rodando, sem outros containers nem processos de inferência.
2. Registrar o modo de energia (`nvpmodel -q`); o `p2-bench` grava no relatório.
3. Rodar 30 minutos por precisão:

   ```bash
   p2-bench app-epi-1.0.0 --precision FP16 --duration 30m --warmup 50
   p2-bench app-epi-1.0.0 --precision FP32 --duration 30m --warmup 50
   ```

4. Cada execução grava `reports/bench/<engine_ref>-<precisão>-<AAAAMMDD-HHMMSS>.json`.

O `p2-bench` mede a memória do sistema por `MemAvailable` (`/proc/meminfo`) em três momentos e calcula:

| Medida | Fórmula | Campo no JSON |
| --- | --- | --- |
| Custo fixo | disponível antes do `load` − disponível depois do `load` | `memory.fixed_cost_mb` |
| Pico | disponível antes do `load` − menor disponível durante a execução | `memory.peak_mb` |
| Limite do componente | pico × 1,3 | `memory.limit_mb` |

- O `gpu_memory_mb` do `infer` mede só o handle (engine, ativações e buffers) e **não** substitui essa medição: o contexto CUDA e as bibliotecas do processo custam centenas de MB a mais.
- Na Jetson, CPU e GPU dividem a mesma memória; por isso `MemAvailable` cobre as duas.

## Resultados · `app-epi-1.0.0`

| Precisão | Modo de energia | FPS médio | Latência p50 / p95 / máx (ms) | `gpu_memory_mb` | Custo fixo (MB) | Pico (MB) | Limite (MB) | Relatório |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | — | — | — | — | — | — | — | — |
| FP32 | — | — | — | — | — | — | — | — |

**Limite adotado para I1:** — MB (maior limite entre as precisões em uso; hoje FP16). Até lá, vale o provisório de 1.024 MB.

## Limite do container

O P3 aplica o limite como limite de memória do container de I1. É preciso confirmar na placa se esse limite conta a memória alocada pelo CUDA:

| Teste | Resultado |
| --- | --- |
| Container com limite abaixo do pico medido roda `p2-bench` | — (esperado: falha por falta de memória se o limite contar o CUDA) |
| Container com o limite adotado roda 30 min sem erro | — |

Se o limite não contar a memória do CUDA, o orçamento vale só como acordo entre as equipes, e o controle precisa ser feito por outra via (por exemplo, uma aplicação por vez na placa).

## Outros modelos

Cada modelo novo de B5 é medido do mesmo jeito e ganha uma seção como a do `app-epi-1.0.0`.
