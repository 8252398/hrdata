"""Integration test — full human-in-the-loop flow without Streamlit UI.

Exercises: SQLAgent.run_iter → review events → approve/reject/append/extend.
Run: pytest tests/test_integration.py -q
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from modules.sql_agent import SQLAgent, MAX_AGENT_TURNS


class FakeLLMClient:
    """Queue-based fake LLM."""

    def __init__(self, responses: list[str]):
        self._queue = list(responses)
        self._idx = 0

    def chat_messages(self, messages, temperature=0.1, max_tokens=4096, timeout=180):
        if self._idx >= len(self._queue):
            raise RuntimeError("FakeLLMClient queue exhausted")
        resp = self._queue[self._idx]
        self._idx += 1
        return resp

    def chat(self, user_message, system_message=""):
        return "总结说明。"


@pytest.fixture
def temp_db():
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
        conn.executemany(
            "INSERT INTO persons VALUES (?,?,?,?,?)",
            [
                ("E001", "张三", "总部", "人事部", "集团领导"),
                ("E002", "李四", "分公司A", "技术部", "工程师"),
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


def _skip_to_review(gen, event):
    while event["type"] in ("status", "sql"):
        event = next(gen)
    return event


_VALID_FINAL_SQL = (
    "SELECT p.employee_code, p.name, p.unit, p.department, p.cadre_flag, "
    "t.course_name, t.hours, t.training_type, t.organizer, t.institution, "
    "t.start_date, t.end_date "
    "FROM persons p JOIN training_records t ON p.employee_code = t.employee_code "
    "LIMIT 1"
)


# ═══════════════════════════════════════════════════════════
# Integration: full approve → reject → append → extend → final
# ═══════════════════════════════════════════════════════════

def test_full_human_in_the_loop_flow(temp_db):
    """Simulate a complete session with approve, reject, append, extend, final."""
    client = FakeLLMClient([
        "[推理]探索人员表结构\n[SQL]\nSELECT * FROM persons LIMIT 1",
        "[推理]查看单位列表\n[SQL]\nSELECT DISTINCT unit FROM persons",
        "[推理]继续查看单位列表\n[SQL]\nSELECT DISTINCT unit FROM persons",
        "[推理]再次查看单位\n[SQL]\nSELECT DISTINCT unit FROM persons",
        f"[推理]已获取足够信息，生成最终SQL\n[FINAL]\n{_VALID_FINAL_SQL}",
    ])
    agent = SQLAgent()
    db = temp_db
    gen = agent.run_iter(
        question="测试查询",
        llm_client=client,
        db=db,
        max_turns_override=3,
    )

    # Turn 1: approve
    ev1 = _skip_to_review(gen, next(gen))
    assert ev1["type"] == "review"
    assert ev1["turn"] == 1
    assert ev1["is_final"] is False
    gen.send({"action": "approve", "feedback": "", "turn": 1})

    # Turn 2: reject
    ev2 = _skip_to_review(gen, next(gen))
    assert ev2["type"] == "review"
    assert ev2["turn"] == 2
    assert ev2["is_final"] is False
    gen.send({"action": "reject", "feedback": "需要更多字段", "turn": 2})

    # Turn 3: append (triggers exhausted, then extend)
    ev3 = _skip_to_review(gen, next(gen))
    assert ev3["type"] == "review"
    assert ev3["turn"] == 3
    assert ev3["remaining_turns"] == 0  # max_turns=3, turn=3
    assert ev3["is_final"] is False
    exhausted = gen.send({"action": "append", "feedback": "加上开始时间", "turn": 3})
    assert exhausted["type"] == "exhausted"
    assert exhausted["turn"] == 3

    # Extend
    ev4 = gen.send({"action": "extend"})
    ev4 = _skip_to_review(gen, ev4)
    assert ev4["type"] == "review"
    assert ev4["turn"] == 4
    snap4 = agent.get_state_snapshot()
    assert snap4["max_turns"] == 3 + 8  # MAX_AGENT_TURNS_EXTEND
    gen.send({"action": "approve", "feedback": "", "turn": 4})

    # Turn 5: final → approve → result
    ev5 = _skip_to_review(gen, next(gen))
    assert ev5["type"] == "review"
    assert ev5["is_final"] is True
    result = gen.send({"action": "approve", "feedback": "", "turn": 5})
    assert result["type"] == "result"
    assert result["turns"] == 5
    assert result["explanation"] == "总结说明。"

    # Verify history has 5 turns
    snap = agent.get_state_snapshot()
    assert len(snap["history"]) == 5


# ═══════════════════════════════════════════════════════════
# Integration: ASK coexistence with review
# ═══════════════════════════════════════════════════════════

def test_ask_then_review(temp_db):
    """LLM asks a question, human answers, then SQL + review."""
    client = FakeLLMClient([
        "[推理]无法确定'班子成员'的具体范围\n[ASK]\n您说的'班子成员'指哪些？",
        "[推理]探索人员表结构\n[SQL]\nSELECT * FROM persons LIMIT 1",
    ])
    agent = SQLAgent()
    gen = agent.run_iter(question="测试", llm_client=client, db=temp_db)

    # Turn 1: ASK
    ev = next(gen)
    while ev["type"] == "status":
        ev = next(gen)
    assert ev["type"] == "ask"
    gen.send("集团领导, 总部部门正职")

    # Turn 2: review
    ev2 = next(gen)
    while ev2["type"] in ("status", "sql"):
        ev2 = next(gen)
    assert ev2["type"] == "review"
    assert ev2["turn"] == 2
