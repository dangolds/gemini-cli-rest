"""Repair agy-truncated responses from agy's own conversation database.

agy's transcript.jsonl is a LOG, not the model's output. It truncates any field
longer than 4096 chars, keeping the first 2048 and the last 2048 and splicing

    \\n<truncated N bytes>\\n

between them (the step is flagged with truncated_fields: ["content"]). The
bridge recovers the model's answer from that log, so before this module every
agy answer over ~4KB reached the caller with a hole in the MIDDLE — head and
tail intact, which is exactly why it went unnoticed for so long: the response
length looked plausible and a head preview looked perfect.

The untruncated text survives in agy's per-conversation SQLite database
(conversations/<conversation-id>.db), where each step's `step_payload` column
is a protobuf blob. We do not have agy's .proto, so rather than assume field
numbers we walk the wire format generically and collect every length-delimited
field that decodes as plausible UTF-8 text — one of them is the whole answer.

Recovery is VERIFIED, never guessed. A candidate is accepted only when it
provably reconstructs the truncated text: same head, same tail, and exactly the
elided byte count in between. Anything else leaves the response untouched. So a
repair can never make an answer worse, an unrelated blob can never be spliced
in, and a model answer that merely quotes a "<truncated N bytes>" marker cannot
be corrupted by us.

codex needs none of this — its rollout JSONL stores whole messages.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path as FsPath

logger = logging.getLogger("transcript-repair")

# Exactly how agy marks an elision in transcript.jsonl.
TRUNCATION_MARKER = re.compile(r"\n<truncated (\d+) bytes>\n")

# Guards against a pathological blob: how deep we descend into nested protobuf
# messages, and the shortest string worth considering as a candidate answer.
_MAX_DEPTH = 8
_MIN_TEXT = 32


@dataclass(frozen=True)
class Repair:
    """Outcome of one repair attempt.

    `text` is always the best available answer: the repaired one when recovery
    verified, otherwise the original (holed) text — callers can use it blindly.
    """

    text: str
    elided: int = 0            # bytes agy cut out (0 => nothing was truncated)
    repaired: bool = False     # were those bytes recovered?
    original_len: int = 0

    @property
    def holed(self) -> bool:
        """Bytes are missing and we could NOT get them back."""
        return self.elided > 0 and not self.repaired


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _texts(buf: bytes, depth: int = 0):
    """Yield every length-delimited protobuf field that looks like UTF-8 text.

    Protobuf's wire format is self-describing enough to walk without a schema:
    each field is a (tag, wire-type) varint followed by a payload whose size the
    wire type determines. A length-delimited field (wire type 2) is either a
    string or a nested message, and the format does not say which — so we try to
    decode it as text and, failing that, descend into it as a message.
    """
    if depth > _MAX_DEPTH:
        return
    i = 0
    end = len(buf)
    while i < end:
        try:
            tag, i = _varint(buf, i)
            wire = tag & 7
            if wire == 0:      # varint
                _, i = _varint(buf, i)
            elif wire == 1:    # 64-bit
                i += 8
            elif wire == 5:    # 32-bit
                i += 4
            elif wire == 2:    # length-delimited: string, bytes, or submessage
                size, i = _varint(buf, i)
                chunk = buf[i:i + size]
                i += size
                if len(chunk) != size:
                    return  # truncated/misparsed blob — stop, keep what we have
                try:
                    text = chunk.decode("utf-8")
                except UnicodeDecodeError:
                    yield from _texts(chunk, depth + 1)
                    continue
                printable = sum(c.isprintable() or c in "\n\r\t" for c in text)
                if len(text) >= _MIN_TEXT and printable / len(text) > 0.95:
                    yield text
                else:
                    yield from _texts(chunk, depth + 1)
            else:
                return  # groups (3/4) or garbage: not worth guessing past
        except (IndexError, ValueError):
            return


def _db_texts(conversation_id: str, state_dir: FsPath):
    """Yield every text-ish string in the conversation's stored protobuf steps."""
    db = state_dir / "conversations" / f"{conversation_id}.db"
    if not db.is_file():
        logger.warning("No conversation DB at %s — cannot repair", db)
        return
    try:
        # Read-only: agy owns this file and may be mid-write on the next turn.
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error as exc:
        logger.warning("Cannot open conversation DB %s: %s", db, exc)
        return
    try:
        # Newest steps first: the answer we are repairing is the latest one.
        rows = conn.execute(
            "SELECT step_payload FROM steps ORDER BY idx DESC"
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("Cannot read steps from %s: %s", db, exc)
        return
    finally:
        conn.close()
    for (payload,) in rows:
        if payload:
            yield from _texts(payload)


def repair(content: str, conversation_id: str, state_dir: FsPath) -> Repair:
    """Restore the bytes agy elided from *content*, from the conversation DB.

    Returns the content unchanged when it was never truncated, or when no
    candidate in the DB provably reconstructs it.
    """
    markers = TRUNCATION_MARKER.findall(content)
    if not markers:
        return Repair(content, original_len=len(content))

    match = TRUNCATION_MARKER.search(content)
    elided = int(match.group(1))
    result = Repair(content, elided=elided, original_len=len(content))

    if len(markers) > 1:
        # agy elides a field exactly once (one head, one tail), so more than one
        # marker means our head/tail split would be a guess. Refuse to guess.
        logger.error(
            "Response has %d truncation markers — refusing to guess the split",
            len(markers),
        )
        return result

    head, tail = content[:match.start()], content[match.end():]
    # agy counts the elision in BYTES, so verify the length in bytes: an answer
    # with any non-ASCII in it has more bytes than chars.
    want = len(head.encode("utf-8")) + elided + len(tail.encode("utf-8"))

    for candidate in _db_texts(conversation_id, state_dir):
        if (
            len(candidate.encode("utf-8")) == want
            and candidate.startswith(head)
            and candidate.endswith(tail)
        ):
            return Repair(
                candidate, elided=elided, repaired=True, original_len=len(content)
            )
    return result
