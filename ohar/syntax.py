"""Dependency-free, line-oriented syntax tokenization for terminal rendering."""

from __future__ import annotations

import re

__all__ = ["TOKEN_KINDS", "normalize_language", "highlight_line"]

TOKEN_KINDS = {
    "plain",
    "keyword",
    "string",
    "comment",
    "number",
    "function",
    "builtin",
    "operator",
    "punctuation",
    "property",
    "tag",
    "attribute",
    "type",
    "variable",
}

_ALIASES = {
    "python": "python",
    "py": "python",
    "python3": "python",
    "javascript": "javascript",
    "js": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "node": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "tsx": "typescript",
    "json": "json",
    "jsonc": "jsonc",
    "bash": "bash",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "html": "html",
    "htm": "html",
    "xml": "html",
    "svg": "html",
    "css": "css",
    "scss": "css",
    "toml": "toml",
    "yaml": "yaml",
    "yml": "yaml",
    "sql": "sql",
    "rust": "rust",
    "rs": "rust",
    "go": "go",
    "golang": "go",
    "java": "java",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "cc": "cpp",
    "hpp": "cpp",
    "csharp": "csharp",
    "cs": "csharp",
    "diff": "diff",
    "patch": "diff",
}

_QUOTED_KEYS = frozenset({"json", "jsonc", "yaml"})
_PROPERTY_KEYS = frozenset({"json", "jsonc", "yaml", "toml", "css", "javascript", "typescript"})
_MAX_HIGHLIGHT_CHARS = 4_000
_NUMBER_RE = re.compile(
    r"(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+"
    r"|\d[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?\d+)?)"
)
_IDENT_RE = re.compile(r"[A-Za-z_$][\w$]*", re.UNICODE)
_HTML_TAG_RE = re.compile(r"[A-Za-z][\w:.-]*")
_HTML_ATTR_RE = re.compile(r"[@:*A-Za-z_][\w:.-]*", re.UNICODE)

_KEYWORDS = {
    "python": frozenset(
        "False None True and as assert async await break class continue def del elif else "
        "except finally for from global if import in is lambda nonlocal not or pass raise "
        "return try while with yield match case".split()
    ),
    "javascript": frozenset(
        "as async await break case catch class const continue debugger default delete do "
        "else export extends finally for from function get if import in instanceof let new "
        "of return set static super switch this throw try typeof var void while with yield".split()
    ),
    "typescript": frozenset(
        "as async await break case catch class const continue debugger declare default delete do "
        "else enum export extends finally for from function get if implements import in infer "
        "instanceof interface keyof let namespace new of private protected public readonly return "
        "set static super switch this throw try type typeof var void while yield satisfies".split()
    ),
    "rust": frozenset(
        "as async await break const continue crate dyn else enum extern false fn for if impl in "
        "let loop match mod move mut pub ref return self Self static struct super trait true type "
        "unsafe use where while".split()
    ),
    "go": frozenset(
        "break case chan const continue default defer else fallthrough false for func go goto if "
        "import interface map nil package range return select struct switch true type var".split()
    ),
    "java": frozenset(
        "abstract assert boolean break byte case catch char class const continue default do double "
        "else enum extends final finally float for goto if implements import instanceof int interface "
        "long native new package private protected public return short static strictfp super switch "
        "synchronized this throw throws transient try void volatile while true false null".split()
    ),
    "c": frozenset(
        "auto break case char const continue default do double else enum extern float for goto if "
        "inline int long register restrict return short signed sizeof static struct switch typedef "
        "union unsigned void volatile while".split()
    ),
    "cpp": frozenset(
        "alignas alignof and and_eq asm auto bitand bitor bool break case catch char class compl "
        "const constexpr continue decltype default delete do double else enum explicit export extern "
        "false float for friend goto if inline int long mutable namespace new noexcept not nullptr "
        "operator or private protected public register reinterpret_cast return short signed sizeof "
        "static static_cast struct switch template this throw true try typedef typename union unsigned "
        "using virtual void volatile while".split()
    ),
    "csharp": frozenset(
        "abstract as async await base bool break byte case catch char class const continue decimal "
        "default delegate do double else enum event explicit extern false finally fixed float for "
        "foreach get goto if implicit in int interface internal is lock long namespace new null object "
        "operator out override params private protected public readonly record ref return sbyte sealed "
        "set short sizeof stackalloc static string struct switch this throw true try typeof uint ulong "
        "unchecked unsafe ushort using var virtual void volatile while".split()
    ),
    "bash": frozenset(
        "case do done elif else esac fi for function if in select then time until while true false".split()
    ),
    "sql": frozenset(
        "add all alter and as asc between by case cast check column commit constraint create cross "
        "database default delete desc distinct drop else end exists foreign from full grant group "
        "having in index inner insert into is join key left like limit not null offset on or order "
        "outer primary references right rollback select set table then union unique update using "
        "values view when where with".split()
    ),
    "json": frozenset({"true", "false", "null"}),
    "jsonc": frozenset({"true", "false", "null"}),
    "toml": frozenset({"true", "false"}),
    "yaml": frozenset(
        "true false null yes no on off ~ inf nan".split()
    ),
    "css": frozenset(
        "important inherit initial unset revert auto none block flex grid inline relative absolute "
        "fixed sticky".split()
    ),
}

_BUILTINS = {
    "python": frozenset(
        "abs all any ascii bin bool breakpoint bytearray bytes callable chr classmethod compile "
        "complex delattr dict dir divmod enumerate eval exec filter float format frozenset getattr "
        "globals hasattr hash help hex id input int isinstance issubclass iter len list locals map "
        "max memoryview min next object oct open ord pow print property range repr reversed round "
        "set setattr slice sorted staticmethod str sum super tuple type vars zip".split()
    ),
    "javascript": frozenset(
        "Array Boolean Date Error JSON Map Math Number Object Promise Proxy RegExp Set String Symbol "
        "console document globalThis window process require fetch".split()
    ),
    "typescript": frozenset(
        "Array Boolean Date Error JSON Map Math Number Object Promise Proxy Record RegExp Set String "
        "Symbol Partial Readonly Pick Omit console document window process".split()
    ),
    "bash": frozenset(
        "alias bg break cd command declare dirs disown echo enable eval exec exit export false fg "
        "getopts hash help history jobs kill let local logout mapfile printf pwd read readonly return "
        "set shift shopt source test times trap true type typeset ulimit umask unalias unset wait"
        .split()
    ),
    "sql": frozenset(
        "avg count coalesce concat current_date current_timestamp date extract greatest least length "
        "lower max min nullif now replace round sum trim upper".split()
    ),
    "go": frozenset("append cap close complex copy delete imag len make new panic print println real recover".split()),
}

_TYPES = {
    "python": frozenset("bool bytes dict float frozenset int list object set str tuple type".split()),
    "javascript": frozenset("boolean number string object bigint symbol undefined any unknown never void".split()),
    "typescript": frozenset("boolean number string object bigint symbol undefined any unknown never void".split()),
    "rust": frozenset(
        "bool char f32 f64 i8 i16 i32 i64 i128 isize str u8 u16 u32 u64 u128 usize String Vec Option Result".split()
    ),
    "go": frozenset(
        "bool byte complex64 complex128 error float32 float64 int int8 int16 int32 int64 rune string uint uint8 uint16 uint32 uint64 uintptr".split()
    ),
    "java": frozenset("boolean byte char double float int long short void String Integer List Map".split()),
    "c": frozenset(
        "bool char double float int int8_t int16_t int32_t int64_t long short size_t uint8_t uint16_t uint32_t uint64_t unsigned void".split()
    ),
    "cpp": frozenset(
        "bool char double float int long short size_t string uint8_t uint16_t uint32_t uint64_t unsigned void".split()
    ),
    "csharp": frozenset(
        "bool byte char decimal double dynamic float int long object sbyte short string uint ulong ushort void".split()
    ),
    "sql": frozenset(
        "bigint binary bit blob boolean char date decimal double float int integer numeric real text time timestamp varchar".split()
    ),
}

_BLOCK_COMMENT_LANGS = frozenset(
    {"javascript", "typescript", "rust", "go", "java", "c", "cpp", "csharp", "css", "sql", "jsonc"}
)
_SLASH_COMMENT_LANGS = frozenset(
    {"javascript", "typescript", "rust", "go", "java", "c", "cpp", "csharp", "jsonc"}
)
_DASH_COMMENT_LANGS = frozenset({"sql"})
_HASH_COMMENT_LANGS = frozenset({"python", "bash", "toml", "yaml"})
_QUOTE_CHARS = {
    "python": ("'", '"'),
    "javascript": ("'", '"', "`"),
    "typescript": ("'", '"', "`"),
    "json": ('"',),
    "jsonc": ('"',),
    "rust": ('"', "'"),
    "go": ('"', "'", "`"),
    "java": ('"', "'"),
    "c": ('"', "'"),
    "cpp": ('"', "'"),
    "csharp": ('"', "'"),
    "bash": ("'", '"'),
    "sql": ("'", '"'),
    "css": ("'", '"'),
    "toml": ("'", '"'),
    "yaml": ("'", '"'),
}
_CASE_INSENSITIVE = frozenset({"sql"})


def normalize_language(language: str | None) -> str | None:
    if not isinstance(language, str):
        return None
    key = language.strip().lower().lstrip(".")
    return _ALIASES.get(key)


def _is_ident_start(char: str) -> bool:
    return char == "_" or char == "$" or char.isalpha()


def _is_ident_char(char: str) -> bool:
    return char == "_" or char == "$" or char.isalnum()


def _string_end(line: str, start: int, quote: str, *, triple: bool = False) -> tuple[int, bool]:
    if triple and line.startswith(quote * 3, start):
        closing = line.find(quote * 3, start + 3)
        return (len(line), True) if closing < 0 else (closing + 3, True)
    i = start + 1
    while i < len(line):
        char = line[i]
        if char == "\\" and i + 1 < len(line):
            i += 2
            continue
        if char == quote:
            return i + 1, False
        i += 1
    return len(line), False


def _classify_ident(
    language: str,
    ident: str,
    line: str,
    end: int,
) -> str:
    key = ident.lower() if language in _CASE_INSENSITIVE else ident
    keywords = _KEYWORDS.get(language, frozenset())
    if key in keywords:
        return "keyword"
    builtins = _BUILTINS.get(language, frozenset())
    if key in builtins:
        return "builtin"
    types = _TYPES.get(language, frozenset())
    if key in types:
        return "type"
    look = end
    while look < len(line) and line[look] in " \t":
        look += 1
    if look < len(line) and line[look] == "(":
        return "function"
    if language in _PROPERTY_KEYS and look < len(line) and line[look] in ":=":
        return "property"
    if language in {"javascript", "typescript"} and ident in {"this", "self"}:
        return "variable"
    return "plain"


def _generic_tokens(line: str, language: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    size = len(line)
    quotes = _QUOTE_CHARS.get(language, ())
    while i < size:
        char = line[i]

        if char.isspace():
            start = i
            while i < size and line[i].isspace():
                i += 1
            tokens.append((line[start:i], "plain"))
            continue

        if language in _BLOCK_COMMENT_LANGS and line.startswith("/*", i):
            close = line.find("*/", i + 2)
            end = size if close < 0 else close + 2
            tokens.append((line[i:end], "comment"))
            i = end
            continue

        if language in _SLASH_COMMENT_LANGS and line.startswith("//", i):
            tokens.append((line[i:], "comment"))
            break

        if language in _DASH_COMMENT_LANGS and line.startswith("--", i):
            tokens.append((line[i:], "comment"))
            break

        if language in _HASH_COMMENT_LANGS and char == "#":
            if language == "python" or i == 0 or line[i - 1].isspace() or line[i - 1] in ";([{":
                tokens.append((line[i:], "comment"))
                break

        if language == "python" and char in "fFrRbBuU" and i + 1 < size and line[i + 1] in quotes:
            end, _ = _string_end(line, i + 1, line[i + 1], triple=line.startswith(line[i + 1] * 3, i + 1))
            tokens.append((line[i:end], "string"))
            i = end
            continue

        if char in quotes:
            triple = language == "python" and line.startswith(char * 3, i)
            end, _ = _string_end(line, i, char, triple=triple)
            kind = "property" if language in _QUOTED_KEYS and _next_nonspace(line, end) == ":" else "string"
            tokens.append((line[i:end], kind))
            i = end
            continue

        if language == "bash" and char == "$":
            end = i + 1
            if end < size and line[end] == "{":
                close = line.find("}", end + 1)
                end = size if close < 0 else close + 1
            else:
                while end < size and (line[end].isalnum() or line[end] == "_"):
                    end += 1
                if end == i + 1:
                    end = min(size, i + 2)
            tokens.append((line[i:end], "variable"))
            i = end
            continue

        if char.isdigit():
            match = _NUMBER_RE.match(line, i)
            if match:
                tokens.append((match.group(0), "number"))
                i = match.end()
                continue

        if _is_ident_start(char):
            match = _IDENT_RE.match(line, i)
            if match:
                ident = match.group(0)
                tokens.append((ident, _classify_ident(language, ident, line, match.end())))
                i = match.end()
                continue

        if char in "{}[](),;":
            tokens.append((char, "punctuation"))
            i += 1
            continue

        if char in "+-*/%=<>!&|^~?:.@\\":
            start = i
            i += 1
            while i < size and line[i] in "+-*/%=<>!&|^~?:.@\\":
                i += 1
            tokens.append((line[start:i], "operator"))
            continue

        tokens.append((char, "plain"))
        i += 1
    return tokens or [("", "plain")]


def _next_nonspace(line: str, start: int) -> str:
    index = start
    while index < len(line) and line[index] in " \t":
        index += 1
    return line[index] if index < len(line) else ""


def _html_tokens(line: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    size = len(line)
    in_tag = False
    while i < size:
        if line.startswith("<!--", i):
            close = line.find("-->", i + 4)
            end = size if close < 0 else close + 3
            tokens.append((line[i:end], "comment"))
            i = end
            continue
        char = line[i]
        if char == "<":
            end = min(size, i + 1)
            if end < size and line[end] in "/!?":
                end += 1
            tokens.append((line[i:end], "punctuation"))
            match = _HTML_TAG_RE.match(line, end)
            if match:
                tokens.append((match.group(0), "tag"))
                end = match.end()
            in_tag = True
            i = end
            continue
        if char == ">":
            tokens.append((char, "punctuation"))
            i += 1
            in_tag = False
            continue
        if in_tag and char in "\"'":
            end, _ = _string_end(line, i, char)
            tokens.append((line[i:end], "string"))
            i = end
            continue
        if in_tag and char == "=":
            tokens.append((char, "operator"))
            i += 1
            continue
        if in_tag and _is_ident_start(char):
            match = _HTML_ATTR_RE.match(line, i)
            if match:
                tokens.append((match.group(0), "attribute"))
                i = match.end()
                continue
        if in_tag and char in "/:":
            tokens.append((char, "punctuation"))
            i += 1
            continue
        if not in_tag:
            start = i
            next_tag = line.find("<", i + 1)
            i = size if next_tag < 0 else next_tag
            tokens.append((line[start:i], "plain"))
            continue
        tokens.append((char, "plain"))
        i += 1
    return tokens or [("", "plain")]


def _diff_tokens(line: str) -> list[tuple[str, str]]:
    if not line:
        return [("", "plain")]
    if line.startswith(("@@", "+++", "---", "diff ", "index ")):
        return [(line, "keyword")]
    if line.startswith("+"):
        return [(line, "string")]
    if line.startswith("-"):
        return [(line, "comment")]
    if line.startswith("\\"):
        return [(line, "comment")]
    return [(line, "plain")]


def highlight_line(line: str, language: str | None) -> list[tuple[str, str]]:
    if not isinstance(line, str):
        raise TypeError("line must be a string.")
    canonical = normalize_language(language)
    if not line:
        return [("", "plain")]
    if canonical is None:
        return [(line, "plain")]
    if len(line) > _MAX_HIGHLIGHT_CHARS:
        return [(line, "plain")]
    if canonical == "html":
        tokens = _html_tokens(line)
    elif canonical == "diff":
        tokens = _diff_tokens(line)
    else:
        tokens = _generic_tokens(line, canonical)
    if "".join(text for text, _ in tokens) != line:
        return [(line, "plain")]
    return tokens
