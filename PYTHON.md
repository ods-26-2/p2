# Regras de Python

## Versão e ferramentas

- Python ≥ 3.12, salvo se o projeto definir outra versão.
- Type checker em modo estrito (`mypy --strict` ou pyright `strict`). Código só está pronto quando passa sem erros.
- Lint e formatação: `ruff check` e `ruff format`.
- Testes: pytest.

## Ambiente virtual

- Sempre usar a venv local `.venv/` na raiz do projeto; nunca instalar pacotes no Python do sistema.
- Antes de rodar `python`, `pip`, testes, type checker ou linter, conferir que a `.venv` está ativa (`sys.prefix` aponta para `.venv`) ou chamar os binários de `.venv/bin/` diretamente.
- Se `.venv/` não existir, criá-la com `python -m venv .venv` (versão do Python do projeto) antes de instalar qualquer dependência.

## Tipos (obrigatório)

- Anotar todos os parâmetros e retornos de funções e métodos, incluindo `-> None` e `__init__`.
- Anotar atributos de classe, atributos de instância e variáveis de módulo. Variáveis locais apenas quando o tipo não é inferível (ex.: `itens: list[Item] = []`).
- Sintaxe moderna: `list[int]`, `dict[str, int]`, `X | None`. Nunca `List`, `Dict`, `Optional`, `Union`.
- Parâmetros com tipos abstratos de `collections.abc` (`Sequence`, `Mapping`, `Iterable`, `Callable`); retornos com tipos concretos.
- Genéricos com sintaxe PEP 695: `def primeiro[T](xs: Sequence[T]) -> T`, `class Repo[T]: ...`, `type UserId = int`.
- Dados estruturados:
  - `@dataclass(frozen=True, slots=True)` por padrão;
  - `TypedDict` para dicionários que cruzam fronteiras (JSON, APIs);
  - pydantic quando o projeto já usa validação.
  - Proibido `dict[str, Any]` como modelo de dados.
- Conjuntos fechados de valores: `Enum`/`StrEnum` ou `Literal`.
- Interfaces por forma: `Protocol` em vez de herança.
- `Final` para constantes, `ClassVar` para atributos de classe, `@override` em métodos sobrescritos, `Self` em retornos do próprio tipo.
- `NewType` para identificadores e valores que não devem se misturar.
- `Any` proibido; usar `object` com narrowing. Exceção somente via decisão aprovada.
- Narrowing explícito com `isinstance`, `match`, `TypeIs`/`TypeGuard`. Exaustividade com `assert_never`.
- `cast` e `# type: ignore` só como último recurso, sempre com código específico: `# type: ignore[arg-type]`.
- Testes também são totalmente tipados.

## Sem comentários

- **Não escrever comentários nem docstrings, exceto quando o usuário pedir explicitamente.**
- A clareza vem de nomes precisos, tipos e funções pequenas.
- Permitido apenas o que ferramentas exigem: `# type: ignore[...]`, `# noqa: CODIGO`, `# pragma: no cover`, shebang e cabeçalho de licença obrigatório.
- Nunca comentários de histórico ou pendência (`# alterado`, `# novo`, `# TODO`): histórico vai no Log da spec, pendências vão nas Tarefas.
- Nunca deixar código comentado.
- Ao editar código que já tem comentários, não removê-los nem alterá-los sem pedido.

## Estrutura e estilo

- Nomes PEP 8: `snake_case` para funções e variáveis, `PascalCase` para classes, `UPPER_CASE` para constantes.
- Funções curtas com uma responsabilidade; early return; no máximo 3 níveis de aninhamento.
- Funções puras sempre que possível; I/O nas bordas.
- Sem estado global mutável; dependências passadas por parâmetro.
- Parâmetros keyword-only (`*`) em funções com mais de 2 parâmetros ou com flags booleanas.
- Sem argumentos default mutáveis.
- Valores mágicos viram constantes `Final` ou enums.
- Imports absolutos, sem `*`. Import dentro de função só para quebrar ciclo (é decisão).
- `pathlib.Path` para caminhos, f-strings para formatação, `with` para recursos.
- `match` para despacho sobre tipos e estruturas.
- Sem abstrações especulativas: só generalizar quando houver o segundo uso real.

## Erros

- Exceções específicas; exceções de domínio herdam de uma exceção base do pacote.
- Nunca `except:` nem `except Exception: pass`.
- Reempacotar com `raise NovoErro(...) from err`.
- Validar entradas nas fronteiras; o interior confia nos tipos.

## Logs e saída

- `logging` com logger por módulo (`logger = logging.getLogger(__name__)`); `print` apenas em saída de CLI.

## Async

- Nunca chamar código bloqueante dentro de `async def`.
- Concorrência estruturada com `asyncio.TaskGroup`.

## Dependências

- Nova dependência é decisão. Declarar em `pyproject.toml`.

## Testes

- Funções `test_*` com `-> None`, fixtures tipadas, `pytest.mark.parametrize` para variações.
- Todo requisito `R#` testável tem pelo menos um teste.

## Configuração de referência

Usar quando o projeto ainda não tiver configuração própria (criá-la é decisão).

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "ANN", "SIM", "RUF", "PL", "ERA"]

[tool.mypy]
python_version = "3.12"
strict = true
```