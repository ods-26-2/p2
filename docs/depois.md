# Deixado para depois

Itens fora do MVP do P2. Nenhum deles muda as assinaturas da API: todos acrescentam variantes ao manifesto ou etapas novas. Forma dinâmica, lote maior que 1 e várias etapas entraram na v0.2.0 (E8, spec 009).

- **[Adiado a pedido]**: item que o responsável pediu explicitamente para deixar para depois.
- **[Sugerido para depois]**: item que o plano propõe adiar.

| Item | Origem | O que envolve |
| --- | --- | --- |
| CI no GitHub Actions | **[Adiado a pedido]** | Rodar os testes de CPU a cada push e gerar e anexar o wheel automaticamente a cada tag |
| INT8 | **[Adiado a pedido]** | Dataset de calibração de B5, variante `INT8` no manifesto e medição da perda de acurácia; avaliar só se o FP16 ficar lento |
| Vários perfis por engine | [Sugerido para depois] | Mais de um perfil de otimização (por exemplo lote 1 e lote 16) e escolha do perfil no `infer` |
| Tensores residentes na GPU | [Sugerido para depois] | Manter tensores na GPU entre chamadas (cache KV do decoder) sem copiar para a CPU; exige função nova, fora da 6c |
| Carregamento sob demanda | [Sugerido para depois] | Política de quais modelos ficam residentes, a partir dos números de E6 |
| Telemetria do dispositivo | [Sugerido para depois] | Expor temperatura, clock e memória da placa, que a 6c permite a P2 sem virar pipeline gráfico |
| Orçamento das outras builds | [Sugerido para depois] | Medir cada modelo novo de B5 com `p2-bench` e acrescentar linhas em `docs/orcamento.md` |
| Busca no registro de B5 | [Sugerido para depois] | `p2-build` baixar o pacote pela URL do registro, em vez de receber um caminho local |
