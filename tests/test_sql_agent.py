"""Tests for modules/sql_agent.py — TDD: RED first, then GREEN.

Uses FakeLLMClient (scripted responses) + temp TrainingDatabase.
Run: pytest tests/test_sql_agent.py -q
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from modules.sql_agent import SQLAgent, MAX_AGENT_TURNS

# A FINAL SQL that passes _check_final_columns gate (>=8 cols, all required fields)
_VALID_FINAL_SQL = (
    "SELECT p.employee_code, p.name, p.unit, p.department, p.cadre_flag, "
    "t.course_name, t.hours, t.training_type, t.organizer, t.institution, "
    "t.start_date, t.end_date "
    "FROM persons p JOIN training_records t ON p.employee_code = t.employee_code "
    "LIMIT 1"
)


def _skip_to_review(gen, event):
    """Skip status and sql events to reach review/ask/result."""
    while event["type"] in ("status", "sql"):
        event = next(gen)
    return event


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════

class FakeLLMClient:
    """Queue-based fake LLM that returns scripted content strings."""

    def __init__(self, responses: list[str]):
        self._queue = list(responses)
        self._idx = 0
        self.chat_messages_calls: list[dict] = []

    def chat_messages(self, messages, temperature=0.1, max_tokens=4096, timeout=180):
        if self._idx >= len(self._queue):
            raise RuntimeError("FakeLLMClient queue exhausted")
        resp = self._queue[self._idx]
        self._idx += 1
        self.chat_messages_calls.append({"messages": messages, "response": resp})
        return resp

    def chat(self, user_message, system_message=""):
        return "这是总结说明。"


@pytest.fixture
def temp_db():
    """Create a minimal TrainingDatabase-like object with persons + training_records."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_training.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE persons (
                employee_code TEXT PRIMARY KEY,
                name TEXT,
                unit TEXT,
                department TEXT,
                cadre_flag TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE training_records (
                id INTEGER PRIMARY KEY,
                employee_code TEXT,
                course_name TEXT,
                hours INTEGER,
                study_type TEXT,
                training_type TEXT,
                training_method TEXT,
                organizer TEXT,
                institution TEXT,
                start_date TEXT,
                end_date TEXT
            )
        """)
        # Insert sample data
        conn.executemany(
            "INSERT INTO persons VALUES (?,?,?,?,?)",
            [
                ("E001", "张三", "总部", "人事部", "集团领导"),
                ("E002", "李四", "分公司A", "技术部", "工程师"),
                ("E003", "王五", "分公司B", "财务部", "干部|总部"),
            ],
        )
        conn.executemany(
            "INSERT INTO training_records VALUES (NULL,?,?,?,?,?,?,?,?,?,?)",
            [
                ("E001", "领导力培训", 40, "面授", "管理", "集中", "集团党校", "北京", "2024-01-01", "2024-01-05"),
                ("E002", "技术培训", 20, "在线", "技术", "自学", "外部机构", "上海", "2024-02-01", "2024-02-03"),
            ],
        )
        conn.commit()
        conn.close()

        # Minimal TrainingDatabase-like wrapper
        class MiniDB:
            def __init__(self, path):
                self.db_path = path
                self._conn = None

            def connect(self):
                if self._conn is None:
                    self._conn = sqlite3.connect(str(self.db_path))
                    self._conn.row_factory = sqlite3.Row
                return self._conn

            def close(self):
                if self._conn:
                    self._conn.close()
                    self._conn = None

            def query_to_df(self, sql, params=()):
                import pandas as pd
                return pd.read_sql_query(sql, self.connect(), params=params)

            def get_stats(self):
                c = self.connect()
                persons = c.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
                records = c.execute("SELECT COUNT(*) FROM training_records").fetchone()[0]
                return {"人员数": persons, "培训记录数": records}

        db = MiniDB(db_path)
        yield db
        db.close()


# ═══════════════════════════════════════════════════════════
# 1. Review event fields
# ═══════════════════════════════════════════════════════════

def test_review_event_fields(temp_db):
    """Agent yields review event with all expected fields after SQL execution."""
    client = FakeLLMClient(["[推理]探索人员表结构\n[SQL]\nSELECT * FROM persons LIMIT 1"])
    agent = SQLAgent()
    gen = agent.run_iter(
        question="测试",
        llm_client=client,
        db=temp_db,
    )

    event = _skip_to_review(gen, next(gen))
    assert event["type"] == "review"
    assert "sql" in event
    assert "result" in event
    assert "result_summary" in event
    assert "turn" in event
    assert "reasoning" in event
    assert "is_final" in event
    assert "exploratory" in event
    assert "remaining_turns" in event
    assert "rows" in event
    assert "cols" in event
    assert event["turn"] == 1
    assert event["reasoning"] == "探索人员表结构"
    assert event["is_final"] is False
    assert event["exploratory"] is True
    assert event["remaining_turns"] == MAX_AGENT_TURNS - 1


# ═══════════════════════════════════════════════════════════
# 2. Approve → result (final)
# ═══════════════════════════════════════════════════════════

def test_approve_final_yields_result(temp_db):
    """Approve a FINAL SQL → agent calls LLM for summary → yields result."""
    client = FakeLLMClient([f"[推理]已获取足够信息，生成最终查询\n[FINAL]\n{_VALID_FINAL_SQL}"])
    agent = SQLAgent()
    gen = agent.run_iter(question="测试", llm_client=client, db=temp_db)

    event = _skip_to_review(gen, next(gen))
    assert event["is_final"] is True
    assert event["reasoning"] == "已获取足够信息，生成最终查询"
    result_event = gen.send({"action": "approve", "feedback": "", "turn": 1})
    assert result_event["type"] == "result"
    assert "sql" in result_event
    assert "result" in result_event
    assert "explanation" in result_event       # Q4=A: summary folded into result
    assert "turns" in result_event
    assert result_event["turns"] == 1


# ═══════════════════════════════════════════════════════════
# 3. Reject → correction message in messages (Q2=A)
# ═══════════════════════════════════════════════════════════

def test_reject_appends_correction_message(temp_db):
    """Reject feedback is appended to messages with original SQL + summary."""
    client = FakeLLMClient([
        "[推理]探索人员表结构\n[SQL]\nSELECT * FROM persons LIMIT 1",
        "[推理]修正查询，返回2条记录\n[SQL]\nSELECT * FROM persons LIMIT 2",
    ])
    agent = SQLAgent()
    gen = agent.run_iter(question="测试", llm_client=client, db=temp_db)

    event = _skip_to_review(gen, next(gen))

    # Reject with feedback
    event2 = gen.send({"action": "reject", "feedback": "应该查 2 条", "turn": 1})
    event2 = _skip_to_review(gen, event2)

    assert event2["turn"] == 2
    # Inspect the messages snapshot via get_state_snapshot
    snap = agent.get_state_snapshot()
    msgs = snap["messages"]
    # The last user message should be the reject correction
    last_user = [m for m in msgs if m["role"] == "user"][-1]
    assert "用户拒绝了你的输出" in last_user["content"]
    assert "应该查 2 条" in last_user["content"]
    assert "SELECT * FROM persons LIMIT 1" in last_user["content"]  # rejected SQL included


# ═══════════════════════════════════════════════════════════
# 4. Append → instruction message
# ═══════════════════════════════════════════════════════════

def test_append_appends_instruction_message(temp_db):
    """Append feedback is appended as instruction message."""
    client = FakeLLMClient([
        "[推理]探索人员表结构\n[SQL]\nSELECT * FROM persons LIMIT 1",
        "[推理]追加条件，返回3条记录\n[SQL]\nSELECT * FROM persons LIMIT 3",
    ])
    agent = SQLAgent()
    gen = agent.run_iter(question="测试", llm_client=client, db=temp_db)

    event = _skip_to_review(gen, next(gen))

    event2 = gen.send({"action": "append", "feedback": "再加一个条件", "turn": 1})
    event2 = _skip_to_review(gen, event2)

    snap = agent.get_state_snapshot()
    msgs = snap["messages"]
    last_user = [m for m in msgs if m["role"] == "user"][-1]
    assert "用户追加指令" in last_user["content"]
    assert "再加一个条件" in last_user["content"]


# ═══════════════════════════════════════════════════════════
# 5. Extend → max_turns bump
# ═══════════════════════════════════════════════════════════

def test_extend_bumps_max_turns(temp_db):
    """Extend action increases max_turns by MAX_AGENT_TURNS_EXTEND."""
    from config.settings import MAX_AGENT_TURNS_EXTEND
    # Need enough responses for 9 turns (8 original + 1 extended)
    responses = ["[推理]继续探索\n[SQL]\nSELECT * FROM persons LIMIT 1"] * (MAX_AGENT_TURNS + 2)
    client = FakeLLMClient(responses)
    agent = SQLAgent()
    gen = agent.run_iter(question="测试", llm_client=client, db=temp_db)

    # Burn through all turns with reject
    for _ in range(MAX_AGENT_TURNS):
        event = _skip_to_review(gen, next(gen))
        # At last turn, reject should trigger exhausted
        if event["remaining_turns"] == 0:
            exhausted = gen.send({"action": "reject", "feedback": "x", "turn": event["turn"]})
            assert exhausted["type"] == "exhausted"
            assert exhausted["turn"] == MAX_AGENT_TURNS
            # Extend
            review_after_extend = gen.send({"action": "extend"})
            # After extend, the generator continues the loop; next yield is review for turn 9
            # but there might be a status event first if the loop re-enters
            review_after_extend = _skip_to_review(gen, review_after_extend)
            assert review_after_extend["type"] == "review"
            assert review_after_extend["turn"] == MAX_AGENT_TURNS + 1
            snap = agent.get_state_snapshot()
            assert snap["max_turns"] == MAX_AGENT_TURNS + MAX_AGENT_TURNS_EXTEND
            break
        else:
            gen.send({"action": "reject", "feedback": "x", "turn": event["turn"]})


# ═══════════════════════════════════════════════════════════
# 6. Resume from messages skips prompt build
# ═══════════════════════════════════════════════════════════

def test_resume_skips_prompt_build(temp_db):
    """run_iter with resume_messages does NOT call PromptBuilder (no schema query)."""
    resume_msgs = [
        {"role": "user", "content": "[System Instructions]\nschema\n\nquestion"},
        {"role": "assistant", "content": "SELECT * FROM persons LIMIT 1"},
        {"role": "user", "content": "继续"},
    ]
    client = FakeLLMClient(["SELECT * FROM persons LIMIT 2"])
    agent = SQLAgent()
    gen = agent.run_iter(
        question="测试",
        llm_client=client,
        db=temp_db,
        resume_messages=resume_msgs,
        start_turn=2,
        prior_history=[{"turn": 1, "sql": "SELECT * FROM persons LIMIT 1", "rows": 1, "exploratory": True}],
    )

    event = next(gen)
    while event["type"] != "review":
        event = next(gen)

    assert event["turn"] == 2
    # PromptBuilder should NOT have been called (no schema PRAGMA in call log)
    # The fake client only received one call (the turn-2 LLM call)
    assert len(client.chat_messages_calls) == 1
    # That call's messages should be the resumed messages
    assert client.chat_messages_calls[0]["messages"] == resume_msgs


# ═══════════════════════════════════════════════════════════
# 7. run() wrapper auto-approves reviews
# ═══════════════════════════════════════════════════════════

def test_run_auto_approves_reviews(temp_db):
    """Blocking run() auto-approves every review and returns result."""
    client = FakeLLMClient([
        "[推理]探索人员表结构\n[SQL]\nSELECT * FROM persons LIMIT 1",
        f"[推理]已获取足够信息，生成最终查询\n[FINAL]\n{_VALID_FINAL_SQL}",
    ])
    agent = SQLAgent()
    result = agent.run(question="测试", llm_client=client, db=temp_db)
    assert result["sql"] == _VALID_FINAL_SQL
    assert isinstance(result["result"], pd.DataFrame)
    assert result["turns"] == 2
    assert len(result["history"]) == 2


# ═══════════════════════════════════════════════════════════
# 8. get_state_snapshot exposes internals
# ═══════════════════════════════════════════════════════════

def test_get_state_snapshot(temp_db):
    """get_state_snapshot returns messages, history, turn, max_turns."""
    client = FakeLLMClient(["SELECT * FROM persons LIMIT 1"])
    agent = SQLAgent()
    gen = agent.run_iter(question="测试", llm_client=client, db=temp_db)

    event = _skip_to_review(gen, next(gen))

    snap = agent.get_state_snapshot()
    assert "messages" in snap
    assert "history" in snap
    assert "turn" in snap
    assert "max_turns" in snap
    assert snap["turn"] == 1
    assert snap["max_turns"] == MAX_AGENT_TURNS
    assert len(snap["history"]) == 1


# ═══════════════════════════════════════════════════════════
# 9. ASK coexists with review
# ═══════════════════════════════════════════════════════════

def test_ask_coexists_with_review(temp_db):
    """LLM outputs ASK: → yields ask event, not review."""
    client = FakeLLMClient([
        "[推理]无法确定'班子成员'的具体范围\n[ASK]\n您说的'班子成员'具体指哪些？",
        "[推理]探索人员表结构\n[SQL]\nSELECT * FROM persons LIMIT 1",
    ])
    agent = SQLAgent()
    gen = agent.run_iter(question="测试", llm_client=client, db=temp_db)

    event = next(gen)
    while event["type"] == "status":
        event = next(gen)

    assert event["type"] == "ask"
    assert event["text"] == "您说的'班子成员'具体指哪些？"

    # Answer the ask
    event2 = gen.send("集团领导, 总部部门正职")
    event2 = _skip_to_review(gen, event2)

    assert event2["type"] == "review"
    assert event2["turn"] == 2
