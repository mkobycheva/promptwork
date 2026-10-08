"""Deterministic sentence and list evidence segmentation.

Abbreviation handling is heuristic: ambiguous abbreviations may still hide a
sentence boundary. This is not a language-aware grammatical parser.
"""
import re

_CLOSERS = '"\'»”’」』)]}'
_OPENERS = '"\'«“‘「『([{'
_CLOSERS_RE = re.escape(_CLOSERS)
_MARKER = re.compile(r'^\s*(?:[-*•]|\d{1,2}[.)])(?:\s+|$)')
_ENDS_SENTENCE = re.compile(r'(?P<term>[.!?…:;。！？]+)[' + _CLOSERS_RE + r']*$')
_TERMINATOR = re.compile(
    r'(?P<cjk>[。！？]+[' + _CLOSERS_RE + r']*)'
    r'|(?P<lat>[.!?…]+)[' + _CLOSERS_RE + r']*(?=\s+(?P<next>\S))'
)
_NEVER_END = {
    'ст', 'стор', 'п', 'пп', 'ч', 'т', 'тт', 'абз', 'розд', 'гл', 'ім', 'див',
    'напр', 'пор', 'проф', 'акад', 'вул', 'просп', 'обл',
    'no', 'nos', 'art', 'sec', 'fig', 'dr', 'mr', 'mrs', 'ms', 'prof', 'vs',
    'cf', 'e.g', 'i.e',
}
_UNITS = {'грн', 'тис', 'млн', 'млрд', 'коп', 'руб', 'р', 'рр', 'м',
          'т.д', 'т.п', 'etc', 'inc', 'ltd'}


def _is_boundary(before: str, terminator: str, nxt: str) -> bool:
    if terminator[-1] in '!?':
        return True
    if terminator != '.':
        return nxt.isupper() or nxt in _OPENERS
    if before.strip().isdigit():
        return False
    words = before.split()
    raw = words[-1].lstrip(_OPENERS) if words else ''
    word = raw.lower()
    if word in _NEVER_END:
        return False
    if len(raw) == 1 and raw.isalpha() and raw.isupper():
        return False
    if word in _UNITS:
        return nxt.isupper() or nxt in _OPENERS
    return nxt.isalpha() or nxt in _OPENERS


def _split_sentences(line: str) -> list[str]:
    parts, start = [], 0
    for match in _TERMINATOR.finditer(line):
        if match.group('lat') is not None and not _is_boundary(
                line[start:match.start()], match.group('lat'), match.group('next')):
            continue
        parts.append(line[start:match.end()])
        start = match.end()
    parts.append(line[start:])
    return parts


def _ends_sentence(current: str, following: str) -> bool:
    """Apply the same abbreviation rules across a newline as within prose."""
    match = _ENDS_SENTENCE.search(current)
    if match is None:
        return False
    term = match.group('term')
    if term[-1] in ':;。！？':
        return True
    # Find the last sentence fragment so a bare year after an earlier sentence
    # is handled consistently with _split_sentences.
    fragment = _split_sentences(current)[-1].strip()
    ending = _ENDS_SENTENCE.search(fragment)
    if ending is None:
        return True  # CJK punctuation already consumed the entire fragment.
    return _is_boundary(fragment[:ending.start()], ending.group('term'), following[0])


def _logical_lines(text: str) -> list[str]:
    """Join wrapped prose, preserving items, paragraphs, and sentence endings."""
    lines, current = [], ''
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                lines.append(current)
                current = ''
            continue
        is_item = bool(_MARKER.match(line))
        line = _MARKER.sub('', line).strip()
        if not line:
            # A marker-only line must not cause following[0] to fail or get
            # attached to the previous item.
            if current:
                lines.append(current)
                current = ''
            continue
        if current and (is_item or _ends_sentence(current, line)):
            lines.append(current)
            current = ''
        current = f'{current} {line}' if current else line
    if current:
        lines.append(current)
    return lines


def split_evidence(text: str, min_chunk_chars: int = 12,
                   keep_short_turns: bool = True,
                   join_wrapped_lines: bool = True) -> list[str]:
    """Split evidence, optionally joining ordinary hard-wrapped prose.

    With joining disabled every newline is an evidence boundary. With joining
    enabled, unmarked continuation lines also join an unfinished list item.
    """
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    if not isinstance(min_chunk_chars, int) or min_chunk_chars < 0:
        raise ValueError('min_chunk_chars must be a nonnegative integer')
    lines = (_logical_lines(text) if join_wrapped_lines else
             [_MARKER.sub('', line).strip() for line in text.splitlines()])
    chunks = []
    for line in lines:
        for part in _split_sentences(line):
            part = part.strip()
            if part and len(part) >= min_chunk_chars:
                chunks.append(part)
    if not chunks and keep_short_turns and text.strip():
        return [text.strip()]
    return chunks
