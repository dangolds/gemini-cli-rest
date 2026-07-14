"""
Unit tests for transcript_repair (recovering agy-truncated answers) and its
wire-in through AgySession._repair_responses.

agy's transcript.jsonl truncates any field over 4096 chars to
2048-head + "\\n<truncated N bytes>\\n" + 2048-tail, and the bridge reads the
answer from that log — so a >4KB answer reaches the caller with a hole in the
MIDDLE. The full text survives in agy's per-conversation SQLite DB, where each
step's `step_payload` is a schema-less protobuf blob. These pin the module's
contract: recovery is VERIFIED (byte-exact reconstruction) or the original is
left untouched — a repair can never make an answer worse or splice in a
stranger's text.

No server / agy / Docker for the module tests: a real temp sqlite DB is built
with hand-encoded protobuf blobs (a tiny wire-format encoder lives below). The
wire-in tests drive AgySession._repair_responses against the same temp DB.
Run:  ./.venv/bin/python -m pytest test_transcript_repair.py -v
"""

import sqlite3

import pytest

import transcript_repair
import server


# --- tiny protobuf wire-format encoder (enough to hand-build step payloads) --

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _string_field(field_num: int, s: str) -> bytes:
    """Length-delimited (wire type 2) string field."""
    data = s.encode("utf-8")
    return bytes([(field_num << 3) | 2]) + _varint(len(data)) + data


def _varint_field(field_num: int, val: int) -> bytes:
    """Varint (wire type 0) field."""
    return bytes([(field_num << 3) | 0]) + _varint(val)


def _submessage_field(field_num: int, inner: bytes) -> bytes:
    """Length-delimited (wire type 2) nested message field."""
    return bytes([(field_num << 3) | 2]) + _varint(len(inner)) + inner


# --- temp conversation DB ---------------------------------------------------

CID = "1111aaaa-2222-3333-4444-555566667777"


def _make_db(state_dir, payloads):
    """Write conversations/<CID>.db with a `steps` table holding *payloads*.

    Rows are inserted oldest-first with ascending idx; _db_texts reads them
    idx DESC (newest first), matching how it hunts the latest answer.
    """
    conv = state_dir / "conversations"
    conv.mkdir(parents=True, exist_ok=True)
    db = conv / f"{CID}.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE steps (idx INTEGER, step_payload BLOB)")
    conn.executemany(
        "INSERT INTO steps (idx, step_payload) VALUES (?, ?)",
        list(enumerate(payloads)),
    )
    conn.commit()
    conn.close()
    return db


def _truncated(head: str, tail: str, elided_bytes: int) -> str:
    """The holed content agy would leave: head + marker + tail."""
    return f"{head}\n<truncated {elided_bytes} bytes>\n{tail}"


# --- module: repair() -------------------------------------------------------

def test_no_marker_is_untouched(tmp_path):
    content = "a perfectly ordinary short answer with no truncation at all"
    rep = transcript_repair.repair(content, CID, tmp_path)
    assert rep.text == content
    assert rep.elided == 0
    assert rep.repaired is False
    assert rep.holed is False


def test_verified_repair_reconstructs_exactly(tmp_path):
    head, tail = "A" * 2048, "Z" * 2048
    middle = "M" * 5000                       # all ASCII: bytes == chars
    full = head + middle + tail
    _make_db(tmp_path, [_string_field(1, full)])
    content = _truncated(head, tail, len(middle.encode("utf-8")))

    rep = transcript_repair.repair(content, CID, tmp_path)
    assert rep.repaired is True
    assert rep.holed is False
    assert rep.elided == 5000
    assert rep.text == full                   # byte-exact reconstruction
    assert "<truncated" not in rep.text


def test_matching_head_wrong_tail_is_refused(tmp_path):
    head, tail = "A" * 2048, "Z" * 2048
    middle = "M" * 5000
    # Same head, same total byte count, but the tail differs: verification
    # must still refuse it and hand back the untouched (holed) content.
    decoy = head + middle + ("Y" * 2048)
    _make_db(tmp_path, [_string_field(1, decoy)])
    content = _truncated(head, tail, len(middle.encode("utf-8")))

    rep = transcript_repair.repair(content, CID, tmp_path)
    assert rep.repaired is False
    assert rep.holed is True
    assert rep.text == content


def test_wrong_byte_count_is_refused(tmp_path):
    head, tail = "A" * 2048, "Z" * 2048
    middle = "M" * 5000
    # Right head AND tail but the wrong number of bytes between them.
    decoy = head + ("M" * (5000 + 17)) + tail
    _make_db(tmp_path, [_string_field(1, decoy)])
    content = _truncated(head, tail, len(middle.encode("utf-8")))

    rep = transcript_repair.repair(content, CID, tmp_path)
    assert rep.repaired is False
    assert rep.holed is True
    assert rep.text == content


def test_non_ascii_byte_math_verifies(tmp_path):
    head = "é" * 1024                          # 1024 chars, 2048 bytes
    tail = "ü" * 1024                          # 1024 chars, 2048 bytes
    middle = "ñ" * 500                         # 500 chars, 1000 bytes
    full = head + middle + tail
    _make_db(tmp_path, [_string_field(1, full)])
    # agy counts the elision in BYTES, not chars.
    content = _truncated(head, tail, len(middle.encode("utf-8")))
    assert len(middle) != len(middle.encode("utf-8"))   # guard: the point

    rep = transcript_repair.repair(content, CID, tmp_path)
    assert rep.repaired is True
    assert rep.text == full
    assert rep.elided == 1000                 # bytes, not the 500 chars


def test_two_markers_refuses_to_guess(tmp_path):
    head, tail = "A" * 2048, "Z" * 2048
    full = head + ("M" * 5000) + tail
    _make_db(tmp_path, [_string_field(1, full)])
    # Two markers => the head/tail split would be a guess. Refuse.
    content = (
        f"{head}\n<truncated 100 bytes>\nMIDDLE"
        f"\n<truncated 200 bytes>\n{tail}"
    )
    rep = transcript_repair.repair(content, CID, tmp_path)
    assert rep.repaired is False
    assert rep.text == content
    assert rep.holed is True                  # elided from the first marker


def test_missing_db_returns_original(tmp_path):
    # No conversations/<CID>.db at all.
    head, tail = "A" * 2048, "Z" * 2048
    content = _truncated(head, tail, 5000)
    rep = transcript_repair.repair(content, CID, tmp_path)   # must not raise
    assert rep.repaired is False
    assert rep.holed is True
    assert rep.text == content


def test_full_text_buried_in_nested_submessage(tmp_path):
    head, tail = "A" * 2048, "Z" * 2048
    middle = "M" * 5000
    full = head + middle + tail
    # Bury the answer one level deep: a nested message whose leading varint
    # field (0xC8 0x01) makes the outer chunk fail UTF-8 decode, so _texts
    # descends instead of yielding the raw submessage bytes as "text".
    inner = _varint_field(2, 200) + _string_field(1, full)
    _make_db(tmp_path, [_submessage_field(3, inner)])
    content = _truncated(head, tail, len(middle.encode("utf-8")))

    rep = transcript_repair.repair(content, CID, tmp_path)
    assert rep.repaired is True
    assert rep.text == full


# --- wire-in: AgySession._repair_responses ----------------------------------

def _session(cid=CID):
    sess = server.AgySession(name="unit")
    sess._conversation_id = cid
    return sess


def test_repair_responses_repairs_per_element(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "AGY_STATE_DIR", tmp_path)
    monkeypatch.setattr(server, "AGY_REPAIR", True)
    head, tail = "A" * 2048, "Z" * 2048
    middle = "M" * 5000
    full = head + middle + tail
    _make_db(tmp_path, [_string_field(1, full)])
    truncated = _truncated(head, tail, len(middle.encode("utf-8")))

    # A healthy element and a truncated one in the SAME turn: each is handled
    # individually, and the healthy one is returned byte-identical.
    healthy = "a short intermediate note"
    out = _session()._repair_responses([healthy, truncated])
    assert out[0] == healthy                  # untouched, no sqlite opened
    assert out[1] == full                     # repaired from the DB
    assert "<truncated" not in out[1]


def test_repair_responses_respects_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "AGY_STATE_DIR", tmp_path)
    monkeypatch.setattr(server, "AGY_REPAIR", False)   # disabled
    head, tail = "A" * 2048, "Z" * 2048
    middle = "M" * 5000
    full = head + middle + tail
    _make_db(tmp_path, [_string_field(1, full)])
    truncated = _truncated(head, tail, len(middle.encode("utf-8")))

    out = _session()._repair_responses([truncated])
    assert out == [truncated]                 # left exactly as-is


def test_repair_responses_no_conversation_id_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "AGY_STATE_DIR", tmp_path)
    monkeypatch.setattr(server, "AGY_REPAIR", True)
    truncated = _truncated("A" * 2048, "Z" * 2048, 5000)
    out = _session(cid=None)._repair_responses([truncated])
    assert out == [truncated]
