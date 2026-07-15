# modules/ KNOWLEDGE BASE

**Generated:** 2026-07-15
**Project:** app_hr — Core business logic

## OVERVIEW
8 modules implementing the AI data analysis pipeline: Excel ingest → SQLite → schema profile → multi-turn LLM agent → read-only SQL → result explanation.

## WHERE TO LOOK
| Module | File | Role | LOC |
|--------|------|------|-----|
| SQL agent | `sql_agent.py` | Multi-turn LLM explorer (max 8 turns) | 449 |
| SQLite manager | `sqlite_manager.py` | `TrainingDatabase` ORM; build/supplement/read | 329 |
| LLM client | `llm_client.py` | OpenAI-compatible wrapper (deepseek/openai/glm/ollama/custom) | 111 |
| ER diagram | `er_diagram.py` | Matplotlib ER diagram of `persons` + `training_records` | 100 |
| Safe executor | `safe_executor.py` | AST whitelist sandbox for AI-generated code | 92 |
| Prompt builder | `prompt_builder.py` | Assembles system + user prompts from `prompts/*.txt` | 89 |
| Data profile | `data_profile.py` | `build_profile()` — schema/stats/sample rows | 80 |
| Excel loader | `excel_loader.py` | `ExcelLoader.load()` for .xlsx/.csv ingest | 50 |

## CONVENTIONS
- All modules import `utils/logger.py` and `config/settings.py`.
- Lazy imports in `app_hr.py`: `modules.er_diagram` (L211) and `modules.sql_agent` (L272) are imported inside UI handlers, not at top level.
- Prompt templates in `prompts/` are read at runtime by `prompt_builder.py` and `sql_agent.py`.

## ANTI-PATTERNS
1. **Never import dangerous modules in AI-generated code** — `safe_executor.py` enforces `FORBIDDEN_IMPORTS` at AST level.
2. **Never call dangerous builtins** — `safe_executor.py` blocks `eval`, `exec`, `compile`, `__import__`, `open` via `FORBIDDEN_CALLS`.
3. **SQL write forbidden** — `sql_agent.py` rejects any query containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `ATTACH`, `DETACH`, `REPLACE`.
4. **No business logic in Python** — `sqlite_manager.py` must not encode HR concepts (班子成员, 中层干部). All reasoning lives in `prompts/agent_system.txt`.
5. **No print/input/network/file I/O in generated code** — enforced by both prompt rules (`prompts/code_gen.txt`) and AST check.
