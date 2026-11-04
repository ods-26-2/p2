# Regras do projeto

Este arquivo é carregado antes de qualquer pedido. Suas regras têm prioridade sobre hábitos padrão.

@specs/INDEX.md
@PYTHON.md

## 1. Princípios

- Todo código nasce de uma spec. Nenhuma feature, correção não trivial ou refatoração começa sem spec.
- Specs são a memória do projeto e o meio mais barato de recuperar contexto: **ler a spec antes de ler o código**.
- Toda decisão é apresentada ao usuário para concordar ou discordar (seção 6), exceto no modo auto.
- Specs são escritas para o modelo ler: telegráficas, em listas, com assinaturas tipadas, sem prosa.

## 2. Estrutura

```
specs/
  INDEX.md            registro de todas as specs (sempre carregado)
  _template.md        modelo de spec (ler apenas ao criar uma)
  ativas/NNN-slug.md
  fechadas/NNN-slug.md
```

- `NNN`: 3 dígitos sequenciais, nunca reutilizados. O próximo ID fica no topo do INDEX.
- `slug`: kebab-case, até 5 palavras.

## 3. Estados

| Estado  | Pasta       | Significado |
|---------|-------------|-------------|
| ativa   | `ativas/`   | Em desenvolvimento ou com tarefas pendentes. Editável livremente. |
| fechada | `fechadas/` | Entregue e validada. Imutável, exceto o **Mapa de código** (para continuar fiel ao código atual) e o campo `Substituída por`. |

- Fechar exige: todas as tarefas `[x]`, mapa de código atualizado, validação executada e aprovação do usuário (no modo auto, fechar ao concluir e informar).
- Fechar = mover o arquivo com `git mv` para `fechadas/`, preencher `Fechada:` e atualizar o INDEX.
- Spec fechada não é reaberta. Nova mudança gera nova spec que a referencia em `Base`.

## 4. Fluxo para qualquer pedido

1. **Classificar o pedido**
   - Pergunta ou explicação → responder, sem spec.
   - Mudança trivial (typo, formatação, renomear variável local, ajuste de mensagem/log) → executar sem spec; se tocar símbolo listado em algum mapa de código, atualizar a linha.
   - Qualquer outra mudança de código → seguir os passos abaixo.
2. **Localizar contexto pelo INDEX**: cruzar arquivos, áreas e termos do pedido com as colunas `Escopo` e `Tags`.
   - Revisar "Pendências abertas" do INDEX: avisar o usuário quando o pedido tocar, depender ou destravar uma pendência.
   - Ler toda spec **ativa** relacionada.
   - Ler spec **fechada** apenas se seu escopo cobre arquivos que serão alterados ou se suas regras se aplicam.
   - Não ler specs sem relação.
3. **Recuperar contexto pela spec**: usar o mapa de código para saber quais arquivos e símbolos abrir. Abrir só o necessário. Ler arquivos inteiros apenas se a spec não bastar ou estiver desatualizada.
4. **Escolher onde registrar**
   - O pedido continua uma spec ativa → atualizar essa spec (novos R#, T#).
   - Caso contrário → criar nova spec em `ativas/` a partir de `_template.md`, com `Base` apontando para as specs relacionadas.
5. **Apresentar o rascunho** (objetivo, requisitos, tarefas, decisões pendentes) de forma resumida e aguardar aprovação (seção 6). Em plan mode, o plano apresentado é o rascunho da spec; o arquivo é criado após aprovação.
6. **Implementar tarefa por tarefa**. Ao concluir cada uma: marcar `[x]` e atualizar o mapa de código.
7. **Validar**: rodar testes, type checker e linter existentes. Registrar o resultado no Log.
8. **Atualizar o INDEX**.
9. **Propor o fechamento** quando todas as tarefas estiverem concluídas.

## 5. Mapa de código

Registro resumido do código gerado ou alterado pela spec, para reconhecê-lo sem abrir arquivos.

```
### src/pacote/modulo.py
- class Nome(Base): papel em até 12 palavras [R1]
  - metodo(self, a: int) -> str: o que faz [R2]
- def funcao(x: Item, *, limite: int = 10) -> list[Item]: o que faz; aplica RG1 [R3]
- LIMITE: Final[int] = papel

### tests/pacote/test_modulo.py
- cobre R1, R2, R3; casos de borda de limite
```

- Uma linha por símbolo público ou relevante, com assinatura tipada. Nunca copiar código-fonte.
- Referenciar requisitos (`R#`), regras (`RG#`) e decisões (`D#`).
- Símbolos privados só quando têm lógica não óbvia.
- Testes: uma linha por arquivo, dizendo o que cobre.
- Alteração em símbolo de spec fechada: atualizar a linha na spec original com `(→ NNN)` e listar o símbolo na nova spec com `(de NNN)`.
- Símbolo removido: apagar a linha do mapa e registrar no Log da spec atual (`removido X de NNN`).
- Divergência entre código e spec: o código é a verdade. Corrigir a spec e avisar o usuário.

## 6. Decisões

**É decisão**: arquitetura ou padrão, nova dependência, modelo de dados ou esquema, interface pública (assinatura, endpoint, formato de dados), comportamento não especificado ou ambíguo, trade-off (desempenho × simplicidade etc.), desvio de regra (deste arquivo, do PYTHON.md ou de uma spec), remoção de código ou arquivos, alteração em código de spec fechada, fechamento de spec.

**Não é decisão**: detalhe interno que já segue regras definidas.

### Modo normal

- Nunca decidir sozinho: apresentar e aguardar resposta.
- Agrupar as decisões pendentes em uma única mensagem; usar a ferramenta de perguntas (AskUserQuestion) quando disponível.
- Formato:
  ```
  **D3 — título curto**
  Contexto: 1 linha
  A) opção — prós/contras
  B) opção — prós/contras
  Recomendo: A, porque ...
  ```
- Se uma decisão surgir no meio da implementação: parar nesse ponto, perguntar e não avançar no que depende dela.
- Registrar na spec: `D3 [aprovada] escolha — motivo`. Se o usuário discordar: `D3 [rejeitada] proposta → escolha do usuário — motivo`.

### Modo auto

Vale quando o Claude Code está em modo auto ou o usuário pede explicitamente para decidir sozinho.

- Seguir a recomendação, registrar `D3 [assumida] escolha — motivo` e continuar.
- Ao final da resposta, listar todas as decisões assumidas para revisão.

## 7. Regras de spec (RG)

- Convenções, restrições e invariantes definidas durante uma spec vão na seção `Regras` com ID `RG#`.
- Regra com alcance além da spec: marcar `[global]` e registrá-la em "Regras globais" do INDEX.
- Regras de specs fechadas continuam valendo até serem substituídas: `RG2 [substituída por 007/RG1]`.
- Antes de implementar, conferir as regras das specs lidas e as regras globais. Conflito entre regras é decisão.

## 8. INDEX.md

- Atualizar sempre que uma spec for criada, mudar de escopo ou for fechada.
- Uma linha por spec: `ID | Título | Estado | Escopo (paths/globs) | Tags | Resumo (≤ 15 palavras)`.
- Ativas primeiro, depois fechadas em ordem decrescente de ID.
- Incrementar `Próximo ID` ao criar uma spec.
- Manter "Pendências abertas" atualizada: acrescentar o que surgir, remover o que for concluído, atualizar a data.

## 9. Economia de contexto

- Spec com até ~150 linhas; se crescer além disso, dividir em specs ligadas por `Base`.
- Não duplicar informação entre specs: referenciar como `NNN/R2`, `NNN/RG1`.
- Log com uma linha por entrada, datada.
- Não reler specs já lidas na mesma sessão, salvo se foram alteradas.