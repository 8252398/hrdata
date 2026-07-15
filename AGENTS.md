# PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-15
**Project:** app_hr — AI Excel data analysis platform
**Stack:** Python 3.11/3.12, Streamlit, Pandas, SQLite, OpenAI-compatible LLM

## OVERVIEW
Local-only, Chinese-language AI data analysis platform. Users upload Excel files; LLM explores a SQLite database via multi-turn agent, then generates read-only SQL. Streamlit renders results. No external data leaves the machine.

## STRUCTURE
```
hrdata/
├── app_hr.py              # Streamlit entry point (284 LOC)
├── modules/               # Core business logic (8 modules, 1300 LOC)
├── config/settings.py     # LLM providers, DB path, safe-execution policy
├── utils/logger.py        # Cross-cutting logger (imported by 9 files)
├── prompts/               # LLM prompt templates (3 .txt files)
├── data/                  # Sample Excel + generated training.db
├── fonts/                 # NotoSansCJK-Regular.ttc for matplotlib
├── tests/                 # EMPTY — placeholder only
└── *.md (×6)              # README + ARCHITECTURE + CODING_RULES + SQL_AGENT + PROMPT_RULES + USERNEEDS
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Start app | `app_hr.py` | `streamlit run app_hr.py` |
| Architecture rules | `ARCHITECTURE.md` | Three-way separation: LLM thinks / Python computes / Streamlit renders |
| Coding conventions | `CODING_RULES.md` | PEP 8, type hints, no bare except, logging only |
| Agent design | `SQL_AGENT.md` | Multi-turn schema discovery → read-only SQL |
| Prompt design | `PROMPT_RULES.md` | Short prompts, no full DataFrame, no hardcoded in code |
| Business requirements | `USERNEEDS.md` | DB schema driven by HR training data |
| LLM client | `modules/llm_client.py` | Unified OpenAI-compatible wrapper |
| Safe execution | `modules/safe_executor.py` | AST whitelist sandbox |
| SQL agent | `modules/sql_agent.py` | Multi-turn explorer (max 8 turns) |
| Data layer | `modules/sqlite_manager.py` | TrainingDatabase ORM |
| Logger | `utils/logger.py` | `get_logger()` used everywhere |
| Config | `config/settings.py` | Paths, LLMConfig, FORBIDDEN_IMPORTS/FORBIDDEN_CALLS |

## CODE MAP
| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| app_hr.py | entry | root | — | Streamlit UI, imports all modules |
| `get_logger` | function | `utils/logger.py` | 9 | Logger factory |
| `LLMConfig` | class | `config/settings.py` | 7 | LLM provider list + paths |
| `FORBIDDEN_IMPORTS` | set | `config/settings.py` | 3 | Safe-execution blacklist |
| `FORBIDDEN_CALLS` | set | `config/settings.py` | 3 | Safe-execution blacklist |
| `TrainingDatabase` | class | `modules/sqlite_manager.py` | 3 | SQLite data layer |
| `SQLAgent` | class | `modules/sql_agent.py` | 1 | Multi-turn LLM agent |
| `LLMClient` | class | `modules/llm_client.py` | 1 | OpenAI-compatible client |
| `SafeExecutor` | class | `modules/safe_executor.py` | 1 | AST sandbox |
| `PromptBuilder` | class | `modules/prompt_builder.py` | 2 | Assembles prompts from `prompts/` |
| `ExcelLoader` | class | `modules/excel_loader.py` | 1 | xlsx → DataFrame |
| `build_profile` | function | `modules/data_profile.py` | 1 | Schema/stats builder |
| `draw_er_diagram` | function | `modules/er_diagram.py` | 1 | Matplotlib ER diagram |

## CONVENTIONS
- **No lint/format config** — conventions live in markdown docs, not in `pyproject.toml`/`.flake8`.
- **Python 3.12+** per `CODING_RULES.md`, but `Dockerfile` and venv use 3.11.
- **PEP 8** + mandatory type hints + docstrings on public functions.
- **Never bare `except: pass`** — use `except Exception as e: logger.exception(e); raise`.
- **Logging only** — stdlib `logging`; never `print()` to Streamlit page.
- **Prompts in `prompts/` only** — no long prompts inlined in business code.
- **Config in `config/` only** — no hardcoded API keys, base URLs, model names, or paths.
- **Module layout**: business logic → `modules/`, utilities → `utils/`.
- **Git commits**: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:` — one logical goal per commit.

## ANTI-PATTERNS (THIS PROJECT)
1. **Bare except swallow** — forbidden by `CODING_RULES.md` §五.
2. **Python containing business logic** — all reasoning lives in prompts; Python is neutral (`SQL_AGENT.md` §Python职责).
3. **Direct Text2SQL** — must use multi-turn agent with schema discovery first (`SQL_AGENT.md` §Agent设计).
4. **Hardcoded secrets/paths** — forbidden; read from `config/settings.py` or `.env`.
5. **Inlined long prompts** — must live in `prompts/*.txt` (`CODING_RULES.md` §九).
6. **AI-generated dangerous imports** — `os`, `subprocess`, `socket`, `requests`, `shutil`, `pathlib`, `sys`, `importlib`, `builtins` are blocked by `FORBIDDEN_IMPORTS` in `config/settings.py`.
7. **AI-generated dangerous calls** — `eval`, `exec`, `compile`, `__import__`, `open` are blocked by `FORBIDDEN_CALLS`.
8. **SQL writes** — `INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/ATTACH/DETACH/REPLACE` are blocked by `sql_agent.py` `_FORBIDDEN_KEYWORDS`.
9. **Schema/DataFrame oversharing** — never send full DataFrame to LLM; send profile + 5-10 sample rows only.

## UNIQUE STYLES
- **Agent-first SQL**: LLM does not generate SQL immediately. It explores the DB with `PRAGMA` and `SELECT DISTINCT`, reasons, then queries. Max 8 turns (`MAX_AGENT_TURNS = 8`).
- **AST sandbox**: All AI-generated Python code is parsed and validated against `ALLOWED_IMPORTS`/`FORBIDDEN_IMPORTS`/`FORBIDDEN_CALLS` before `exec` in a scrubbed namespace.
- **Chinese prompts**: System prompts and business concepts are in Chinese. LLM must not hallucinate domain values (e.g., "党委书记") if absent from the DB.
- **Single SQLite source of truth**: After Excel ingest, all analysis goes through `data/training.db`. No DataFrame analysis on raw uploads.

## COMMANDS
```bash
# Local dev
streamlit run app_hr.py

# Docker
docker compose up --build

# Install deps
pip install -r requirements.txt
```

## NOTES
- `tests/` is empty scaffolding despite being at v2.1.0. No test runner installed.
- No CI/CD configured (no `.github/workflows/`).
- `ARCHITECTURE.md` §八 lists modules `chart_generator.py`, `markdown_formatter.py`, `export_manager.py` that do **not exist** in `modules/`. Actual modules include `sql_agent.py`, `sqlite_manager.py`, `er_diagram.py` not listed in the doc. **Treat code as source of truth.**
- `__pycache__/` and `app_hr.log` are tracked in git despite `.gitignore` listing them. Cleanup needed.
- `.omo/` directory (agent runtime state) is also tracked. Add to `.gitignore`.
- `plotly` is present in venv but missing from `requirements.txt`.
