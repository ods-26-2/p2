# Deixado para depois

Itens fora do MVP do P2. Nenhum deles muda a API da E1: todos acrescentam variantes ao manifesto ou etapas novas.

- **[Adiado a pedido]**: item que o responsável pediu explicitamente para deixar para depois.
- **[Sugerido para depois]**: item que o plano propõe adiar.

| Item | Origem | O que envolve |
| --- | --- | --- |
| CI no GitHub Actions | **[Adiado a pedido]** | Rodar os testes de CPU a cada push e gerar e anexar o wheel automaticamente a cada tag |
| INT8 | **[Adiado a pedido]** | Dataset de calibração de B5, variante `INT8` no manifesto e medição da perda de acurácia; avaliar só se o FP16 ficar lento |
| Forma dinâmica | **[Adiado a pedido]** | Perfis de otimização no `p2-build` e formas com faixa no manifesto e no `info` |
| Lote maior que 1 | **[Adiado a pedido]** | `max_batch > 1`, perfis no `p2-build` e validação da primeira dimensão no `infer` |
| Modelos com várias etapas | **[Adiado a pedido]** | Whisper de I5 (encoder e decoder) e OCR em dois estágios de I7: um `engine_ref` por etapa ou vários engines por `engine_ref` |
| Carregamento sob demanda | [Sugerido para depois] | Política de quais modelos ficam residentes, a partir dos números de E6 |
| Telemetria do dispositivo | [Sugerido para depois] | Expor temperatura, clock e memória da placa, que a 6c permite a P2 sem virar pipeline gráfico |
| Orçamento das outras builds | [Sugerido para depois] | Medir cada modelo novo de B5 com `p2-bench` e acrescentar linhas em `docs/orcamento.md` |
| Busca no registro de B5 | [Sugerido para depois] | `p2-build` baixar o pacote pela URL do registro, em vez de receber um caminho local |
