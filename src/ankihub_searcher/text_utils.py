"""Text processing and Markdown parsing utilities for Smart Search.

Provides:
1. clean_html_and_cloze: HTML, Cloze, and media cleaner for Anki note fields.
2. clean_tags: Smart tag cleaner stripping AnKing / Mastering Radiology boilerplate.
3. parse_markdown_units: Markdown batch parser with empty-header handling and breadcrumbs.
4. normalize_inline_latex: Inline LaTeX cleaner converting math and Greek letters to Unicode.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class LectureUnit:
    """Represents an atomic lecture unit extracted from Markdown or plain text."""

    unit_title: str
    deck_name: str
    body_text: str
    breadcrumb: str = ""
    keywords: List[str] = field(default_factory=list)


# Prefixes or segments that invalidate subsequent segments in the same tag (e.g., #AK_Other::AnKing_Image::Marcos)
_NOISE_CONTAINER_SEGMENTS = {
    "anking_image",
    "amboss",
    "contributors",
    "drradiology",
    "ankore",
    "ak_other",
    "expn",
    "uworld",
    "bootcamp",
    "physeo",
    "pixorize",
    "picmonic",
    "sketchy",
}

# Boilerplate patterns in AnKing / Mastering Radiology
_BOILERPLATE_TAG_SEGMENTS = {
    "ak_step1_v12",
    "ak_step2_v12",
    "ak_step3_v12",
    "ak_other",
    "ak_original_decks",
    "anking_image",
    "only_step_1",
    "only_step_2",
    "expn",
    "amboss",
    "low/highyield",
    "highyield",
    "lowyield",
    "contributors",
    "drradiology",
    "ankore",
    "physeo_image_update",
    "subjects",
    "systems",
    "textbooks",
    "firstaid",
    "first aid",
    "bootcamp",
    "physeo",
    "pixorize",
    "sketchy",
    "sketchy_2",
    "sketchy_extra",
    "picmonic",
    "uworld",
    "bnb",
    "pathoma",
    "mastering_radiology",
    "mastering radiology",
}

# Regexes for cleaning
_RE_STYLE_SCRIPT = re.compile(r"<(style|script)[^>]*>[\s\S]*?</\1>", re.IGNORECASE)
_RE_MEDIA_SOUND = re.compile(r"\[sound:[^\]]+\]|<(img|audio|video|source)[^>]*>", re.IGNORECASE)
_RE_PROMO_LINKS = re.compile(
    r"<a\b[^>]*>(?:Watch associated (?:Bootcamp|Physeo|Sketchy|Pixorize|Picmonic)[^<]*|Watch [^<]*video[^<]*)</a>|"
    r"Watch associated (?:Bootcamp|Physeo|Sketchy|Pixorize|Picmonic)[^\n<]*",
    re.IGNORECASE,
)
_RE_BLOCK_TAGS = re.compile(r"<(?:br\s*/?|/p|/div|/li|/tr|/td|/h[1-6])>", re.IGNORECASE)
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_CLOZE = re.compile(r"\{\{c\d+::(.*?)(?:::([^}]*))?\}\}", re.DOTALL)
_RE_NUMERIC_PREFIX = re.compile(r"^\d{1,3}_+")
_RE_WHITESPACE = re.compile(r"[ \t]+")
_RE_MULTILINE = re.compile(r"\n{3,}")
_RE_HEADER = re.compile(r"^(#{1,6})\s+(.+)$")
_RE_HR = re.compile(r"^[-=_*]{3,}$")
_RE_RANDOM_CODE = re.compile(r"^[a-zA-Z0-9]{5,10}$")
_RE_KEYWORD_BULLET = re.compile(r"^(?:[-*+•]|\d+[\.\)])\s*")


def clean_html_and_cloze(raw_html: Optional[str]) -> str:
    """Strips HTML tags, audio/media tokens, unwraps Cloze deletions, and normalizes text."""
    if not raw_html:
        return ""

    text = raw_html

    # 1. Strip styles and scripts
    text = _RE_STYLE_SCRIPT.sub(" ", text)

    # 2. Strip sounds, images, videos
    text = _RE_MEDIA_SOUND.sub(" ", text)

    # 3. Strip external promo/video links
    text = _RE_PROMO_LINKS.sub(" ", text)

    # 4. Replace block elements with space or newline
    text = _RE_BLOCK_TAGS.sub(" \n ", text)

    # 5. Unwrap cloze deletions {{c1::answer::hint}} -> "answer (hint)" or "answer"
    def _cloze_replacer(match: re.Match) -> str:
        answer = match.group(1).strip()
        hint = match.group(2)
        if hint is not None and hint.strip():
            h = hint.strip()
            # If hint is just another cloze index or trivial symbol, omit
            if h.lower() not in {"...", "-", "?"}:
                return f"{answer} ({h})"
        return answer

    # Handle potentially nested or sequential clozes
    for _ in range(3):
        if "{{" in text:
            text = _RE_CLOZE.sub(_cloze_replacer, text)
        else:
            break

    # 6. Strip any remaining HTML tags
    text = _RE_HTML_TAGS.sub(" ", text)

    # 7. Unescape HTML entities
    text = html.unescape(text)

    # 8. Normalize whitespace
    text = _RE_WHITESPACE.sub(" ", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = _RE_MULTILINE.sub("\n\n", text)

    return text.strip()


def clean_tags(tags: Optional[List[str]]) -> str:
    """Strips AnKing / Mastering Radiology boilerplate and returns a deduplicated taxonomy string.

    Example:
    Input: ["#AK_Step1_v12::#FirstAid::04_Pathology::02_Inflammation::04_Acute_inflammation::*Hageman_Factor",
            "#AK_Other::AnKing_Image::Marcos"]
    Output: "Pathology, Inflammation, Acute inflammation, Hageman Factor"
    """
    if not tags:
        return ""

    clean_tokens: List[str] = []
    seen = set()

    for tag in tags:
        if not tag:
            continue
        segments = tag.split("::")
        skip_remaining = False

        for seg in segments:
            if skip_remaining:
                break

            seg = seg.strip()
            if not seg:
                continue

            # Strip leading symbols #, *, ^, !
            seg = seg.lstrip("#*^!").strip()
            if not seg:
                continue

            seg_lower = seg.lower()

            # If segment is a known noise container, skip this and all subsequent segments
            if seg_lower in _NOISE_CONTAINER_SEGMENTS:
                skip_remaining = True
                continue

            # Filter boilerplate
            if seg_lower in _BOILERPLATE_TAG_SEGMENTS:
                continue

            # Filter purely numeric segments (e.g. "04")
            if seg.isdigit():
                continue

            # Filter known noise prefixes
            if any(
                seg_lower.startswith(prefix)
                for prefix in (
                    "ak_step",
                    "ak_other",
                    "ak_original",
                    "only_step",
                    "anking_image",
                    "amboss",
                    "expn",
                )
            ):
                continue

            # Filter alphanumeric hash codes (e.g. mixed letters and digits like "9OaNuk")
            if any(c.isdigit() for c in seg) and any(c.isalpha() for c in seg) and not ("_" in seg or " " in seg):
                continue

            # Strip leading numeric prefix like 01_, 04_
            seg_cleaned = _RE_NUMERIC_PREFIX.sub("", seg).strip()

            # Replace underscores with spaces
            seg_cleaned = seg_cleaned.replace("_", " ").strip()

            if not seg_cleaned or len(seg_cleaned) < 2:
                continue

            # Deduplicate case-insensitively
            key = seg_cleaned.lower()
            if key not in seen:
                seen.add(key)
                clean_tokens.append(seg_cleaned)

    return ", ".join(clean_tokens)


def parse_markdown_units(
    raw_text: str, base_deck_name: str = "", naming_style: str = "dash"
) -> List[LectureUnit]:
    """Splits Markdown text into LectureUnit instances based on ATX headers.

    - Empty headers (followed immediately by subheaders without substantive text) are
      skipped as units, but preserved in breadcrumb context.
    - If no headers are present, returns a single unit with the entire text.
    - If the entire text is empty or contains only empty headers, returns [].
    """
    if not raw_text or not raw_text.strip():
        return []

    lines = raw_text.splitlines()

    # Detect if any valid ATX headers exist outside code blocks
    headers_found = False
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block and _RE_HEADER.match(stripped):
            headers_found = True
            break

    # If no headers found: return single unit
    if not headers_found:
        body = raw_text.strip()
        if not body:
            return []
        if base_deck_name:
            deck_name = base_deck_name.strip()
            unit_title = base_deck_name.strip()
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            deck_name = f"Smart Search {timestamp}"
            unit_title = "Smart Search"
        return [LectureUnit(unit_title=unit_title, deck_name=deck_name, body_text=body, breadcrumb="")]

    # Parse headers and sections
    raw_sections: List[Tuple[int, str, List[str]]] = []
    current_level = 0
    current_header = ""
    current_body: List[str] = []
    prelude: List[str] = []

    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            if current_header:
                current_body.append(line)
            else:
                prelude.append(line)
            continue

        if not in_code_block:
            m = _RE_HEADER.match(stripped)
            if m:
                # Save previous section
                if current_header:
                    raw_sections.append((current_level, current_header, current_body))
                elif any(p.strip() for p in prelude):
                    # Substantive prelude text before first header
                    raw_sections.append((1, "Intro", prelude))

                level = len(m.group(1))
                header_text = m.group(2).strip().rstrip("#").strip()
                current_level = level
                current_header = header_text
                current_body = []
                continue

        if current_header:
            current_body.append(line)
        else:
            prelude.append(line)

    if current_header:
        raw_sections.append((current_level, current_header, current_body))

    # Second pass: compute breadcrumbs and filter empty headers
    units: List[LectureUnit] = []
    hierarchy_stack: List[Tuple[int, str]] = []

    base_deck = base_deck_name.strip()

    for level, title, body_lines in raw_sections:
        # Maintain hierarchy stack for breadcrumbs
        while hierarchy_stack and hierarchy_stack[-1][0] >= level:
            hierarchy_stack.pop()
        hierarchy_stack.append((level, title))

        breadcrumb = " > ".join(t for _, t in hierarchy_stack)

        # Check if body has substantive text (not just empty lines or horizontal rules)
        filtered_body_lines = []
        for bline in body_lines:
            bstripped = bline.strip()
            if not bstripped or _RE_HR.match(bstripped):
                continue
            filtered_body_lines.append(bline)

        body_text = "\n".join(filtered_body_lines).strip()
        if not body_text:
            # Empty header: do not emit unit! (it stays on stack for breadcrumb context)
            continue

        # Build deck name based on naming style
        if naming_style == "subdeck":
            if base_deck:
                deck_name = f"{base_deck}::{title}"
            else:
                deck_name = title
        else:  # "dash" default
            if base_deck:
                deck_name = f"{base_deck} - {title}"
            else:
                deck_name = title

        units.append(
            LectureUnit(
                unit_title=title,
                deck_name=deck_name,
                body_text=body_text,
                breadcrumb=breadcrumb,
            )
        )

    return units


# Greek letter mappings for LaTeX normalization
_LATEX_GREEK = {
    # Lowercase
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "omicron": "ο",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    # Uppercase
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
}

# Common mathematical symbols in medical notes
_LATEX_MATH_SYMBOLS = {
    "pm": "±",
    "mp": "∓",
    "times": "×",
    "div": "÷",
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "neq": "≠",
    "ne": "≠",
    "approx": "≈",
    "sim": "∼",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "leftrightarrow": "↔",
    "Rightarrow": "⇒",
    "Leftarrow": "⇐",
    "Leftrightarrow": "⇔",
    "uparrow": "↑",
    "downarrow": "↓",
    "infty": "∞",
    "partial": "∂",
    "nabla": "∇",
    "sum": "∑",
    "prod": "∏",
    "int": "∫",
    "circ": "°",
    "degree": "°",
}

_ALL_LATEX_SYMBOLS = {**_LATEX_GREEK, **_LATEX_MATH_SYMBOLS}
_RE_LATEX_WRAPPERS = re.compile(
    r"\\(?:mathrm|mathbf|mathit|text|textbf|textit|textnormal|operatorname|bm|underline)\{([^{}]*)\}"
)
_RE_LATEX_FRAC = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_RE_LATEX_SUB_SUP = re.compile(r"([_^])\{([^{}]+)\}")
_RE_MATH_DELIMITERS = re.compile(r"(?<!\\)\${1,2}")
_RE_PAREN_OPEN = re.compile(r"\\(?:[\(\[])\s*")
_RE_PAREN_CLOSE = re.compile(r"\s*\\(?:[\)\]])")


def normalize_inline_latex(raw_text: Optional[str]) -> str:
    """Normalizes inline LaTeX in lecture text to clean Unicode representations.

    Converts Greek letters (e.g., $\\beta$ -> β, $\\kappa$ -> κ, $\\alpha$ -> α),
    unwraps formatting commands (e.g., \\mathrm{TGF}-\\beta -> TGF-β), removes
    math delimiters ($...$, $$...$$, \\(...\\)), and converts common symbols.
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Iteratively unwrap LaTeX command wrappers (e.g. \mathrm{...}, \textbf{...})
    for _ in range(5):
        if not _RE_LATEX_WRAPPERS.search(text):
            break
        text = _RE_LATEX_WRAPPERS.sub(r"\1", text)

    # 2. Convert fractions \frac{a}{b} -> a/b
    for _ in range(5):
        if not _RE_LATEX_FRAC.search(text):
            break
        text = _RE_LATEX_FRAC.sub(r"\1/\2", text)

    # 3. Replace known Greek letters and math symbols
    def _sym_repl(match: re.Match) -> str:
        cmd = match.group(1)
        trailer = match.group(2) or ""
        sym = _ALL_LATEX_SYMBOLS.get(cmd)
        if sym is not None:
            if trailer == "{}":
                return sym
            return sym + trailer
        return match.group(0)

    text = re.sub(r"\\([a-zA-Z]+)(\{\})?", _sym_repl, text)

    # 4. Unwrap braces in sub/superscripts: _{2} -> _2, ^{2} -> ^2
    text = _RE_LATEX_SUB_SUP.sub(r"\1\2", text)

    # 5. Remove math delimiters ($, $$, \(, \), \[, \])
    text = _RE_MATH_DELIMITERS.sub("", text)
    text = _RE_PAREN_OPEN.sub("", text)
    text = _RE_PAREN_CLOSE.sub("", text)

    return text


def parse_keyword_lines(text: Optional[str]) -> List[str]:
    """Parses handwritten keywords from text, one per line.

    - Strips leading bullet markers (-, *, +, •, 1., 1)).
    - Strips whitespace and trailing commas or semicolons.
    - Skips empty lines and Markdown header lines.
    - Normalizes inline LaTeX (e.g. $\\beta$ -> β).
    - Deduplicates case-insensitively while preserving original order and first casing.
    """
    if not text:
        return []

    keywords: List[str] = []
    seen = set()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        kw = _RE_KEYWORD_BULLET.sub("", stripped).strip()
        kw = kw.rstrip(" \t,;").strip()
        if not kw:
            continue

        kw = normalize_inline_latex(kw).strip()
        if not kw:
            continue

        key = kw.lower()
        if key not in seen:
            seen.add(key)
            keywords.append(kw)

    return keywords


def parse_keyword_units(
    raw_text: str, base_deck_name: str = "", naming_style: str = "dash"
) -> List[LectureUnit]:
    """Splits Markdown/plain text into LectureUnits populated with parsed keywords.

    Uses parse_markdown_units to determine units, deck names, and breadcrumbs,
    extracts keywords from each unit's body text via parse_keyword_lines, and
    discards any units that have no valid keywords.
    """
    units = parse_markdown_units(raw_text, base_deck_name=base_deck_name, naming_style=naming_style)
    result: List[LectureUnit] = []
    for unit in units:
        kws = parse_keyword_lines(unit.body_text)
        if kws:
            unit.keywords = kws
            result.append(unit)
    return result


# =====================================================================
# Structural / Syntactic Similarity Utilities
# =====================================================================

_STRUCT_POS_MAP: Dict[str, str] = {}
for _w in (
    "il", "lo", "la", "i", "gli", "le", "l", "un", "uno", "una",
    "the", "a", "an",
):
    _STRUCT_POS_MAP[_w] = "DET"
for _w in (
    "di", "del", "dello", "della", "dei", "degli", "delle",
    "a", "ad", "al", "allo", "alla", "ai", "agli", "alle",
    "da", "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "in", "nel", "nello", "nella", "nei", "negli", "nelle",
    "con", "su", "sul", "sullo", "sulla", "sui", "sugli", "sulle",
    "per", "tra", "fra",
    "of", "in", "to", "for", "with", "on", "at", "from", "by",
    "into", "through", "during", "between", "among", "within",
    "without", "under", "over",
):
    _STRUCT_POS_MAP[_w] = "ADP"
for _w in (
    "è", "e", "sono", "era", "erano", "sia", "siano", "essere",
    "ha", "hanno", "aveva", "può", "possono", "deve", "devono",
    "viene", "vengono", "consiste",
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "can", "may", "must", "does", "do", "did",
):
    _STRUCT_POS_MAP[_w] = "AUX"
for _w in (
    "che", "cui", "quale", "quali", "chi", "cosa", "dove", "quando",
    "come", "perché", "perche", "quanto", "quanta", "quanti", "quante",
    "which", "what", "who", "whom", "where", "when", "how", "why", "that",
):
    _STRUCT_POS_MAP[_w] = "WH"
for _w in (
    "ed", "o", "od", "ma", "però", "pero", "mentre", "quindi",
    "perciò", "percio", "poiché", "poiche", "perchè", "se", "anche",
    "oppure", "ovvero",
    "and", "or", "but", "while", "whereas", "if", "because", "since",
    "although", "though", "therefore", "thus", "however",
):
    _STRUCT_POS_MAP[_w] = "CONJ"
for _w in (
    "non", "no", "né", "ne", "mai", "senza",
    "not", "no", "never", "without", "neither", "nor",
):
    _STRUCT_POS_MAP[_w] = "NEG"
for _w in (
    "più", "piu", "meno", "maggiore", "minore", "alto", "alta",
    "basso", "bassa", "aumenta", "diminuisce",
    "more", "less", "greater", "lower", "higher", "increased",
    "decreased", "increases", "decreases",
):
    _STRUCT_POS_MAP[_w] = "CMP"
for _w in (
    "questo", "questa", "questi", "queste", "quello", "quella", "si", "ci",
    "this", "these", "that", "those", "it", "its", "they", "their",
):
    _STRUCT_POS_MAP[_w] = "PRON"

_RE_STRUCT_MATH = re.compile(
    r"\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[^$]+\$"
)
_RE_STRUCT_NUM = re.compile(r"^\d+(?:[.,:/-]\d+)*$")
_RE_STRUCT_UUID = re.compile(r"^[a-f0-9-]{8,}$|^[A-Za-z0-9]{10,32}$")


def extract_primary_field_raw(flds_str: Optional[str], fallback_clean_text: str = "") -> str:
    """Extracts the primary question/prompt field from Anki's \\x1f-joined flds string.

    Skips empty fields, pure media-only fields, Image Occlusion coordinate masks,
    and hidden ID/UUID fields so structural comparison operates on the actual sentence.
    """
    if flds_str:
        parts = flds_str.split("\x1f")
        for raw in parts:
            stripped = raw.strip()
            if not stripped:
                continue
            if "image-occlusion:rect:" in stripped:
                continue
            # Check plain visible text without HTML/media
            visible = _RE_MEDIA_SOUND.sub(" ", stripped)
            visible = _RE_HTML_TAGS.sub(" ", visible)
            visible = html.unescape(visible).strip()
            if not visible:
                continue
            if _RE_STRUCT_UUID.match(visible) and " " not in visible and len(parts) > 1:
                continue
            return stripped

    if fallback_clean_text:
        for line in fallback_clean_text.splitlines():
            s = line.strip()
            if s:
                return s
    return ""


def build_structural_representation(raw_text: Optional[str]) -> Dict[str, Any]:
    """Builds a lightweight, language-agnostic structural representation of a flashcard field.

    Captures:
    - lexical_tokens: ordered word tokens after HTML/cloze unwrapping
    - skeleton_tokens: ordered tokens with clozes (<CLOZE>), math (<MATH>), and numbers (<NUM>) masked
    - pos_tokens: coarse IT/EN closed-class POS skeleton with collapsed content runs
    - hints: normalized cloze hints
    - shingles: set of lexical/skeleton 2-grams for fast inverted-index candidate pre-filtering
    """
    if not raw_text:
        return {
            "lexical_tokens": [],
            "skeleton_tokens": [],
            "pos_tokens": [],
            "hints": set(),
            "lex_2grams": set(),
            "skel_2grams": set(),
            "pos_2grams": set(),
            "shingles": set(),
        }

    text = _RE_STYLE_SCRIPT.sub(" ", raw_text)
    text = _RE_MEDIA_SOUND.sub(" ", text)
    text = _RE_PROMO_LINKS.sub(" ", text)
    text = _RE_STRUCT_MATH.sub(" __MATH__ ", text)

    hints: Set[str] = set()

    def _lex_cloze_repl(match: re.Match) -> str:
        ans = match.group(1).strip()
        hint = match.group(2)
        if hint is not None:
            h = hint.strip().lower()
            if h and h not in {"...", "-", "?"}:
                hints.add(h)
        return ans

    lex_t = text
    skel_t = text
    for _ in range(3):
        if "{{" in lex_t:
            lex_t = _RE_CLOZE.sub(_lex_cloze_repl, lex_t)
        if "{{" in skel_t:
            skel_t = _RE_CLOZE.sub(" __CLOZE__ ", skel_t)

    def _tokenize_raw(s: str) -> List[str]:
        s = re.sub(r"<(?:br\s*/?|/p|/div|/li|/tr|/td|/h[1-6])>", " , ", s, flags=re.IGNORECASE)
        s = _RE_HTML_TAGS.sub(" ", s)
        s = html.unescape(s)
        s = s.replace("’", "'").replace("‘", "'").replace("→", " , ")
        return re.findall(r"__cloze__|__math__|[a-zA-ZÀ-ÿ0-9]+|[?,;:]", s.lower())

    lex_raw = _tokenize_raw(lex_t)
    skel_raw = _tokenize_raw(skel_t)

    lexical_tokens = [w for w in lex_raw if w not in {"?", ",", ";", ":"}][:80]
    skeleton_tokens = [
        ("<NUM>" if _RE_STRUCT_NUM.match(w) else ("<CLOZE>" if w == "__cloze__" else ("<MATH>" if w == "__math__" else w)))
        for w in skel_raw
        if w not in {"?", ",", ";", ":"}
    ][:80]

    pos_seq: List[str] = []
    for w in lex_raw:
        if w in {"?", ",", ";", ":"}:
            pos_seq.append(w)
        elif w == "__cloze__":
            pos_seq.append("<CLOZE>")
        elif w == "__math__":
            pos_seq.append("<MATH>")
        elif _RE_STRUCT_NUM.match(w):
            pos_seq.append("<NUM>")
        elif w in _STRUCT_POS_MAP:
            pos_seq.append(_STRUCT_POS_MAP[w])
        else:
            pos_seq.append("W")

    pos_tokens: List[str] = []
    for p in pos_seq:
        if p == "W" and pos_tokens and pos_tokens[-1] == "W":
            continue
        pos_tokens.append(p)
    pos_tokens = pos_tokens[:60]

    def _bigrams(seq: List[str]) -> Set[Tuple[str, str]]:
        if len(seq) < 2:
            return set()
        return {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}

    lex_2grams = _bigrams(lexical_tokens)
    skel_2grams = _bigrams(skeleton_tokens)
    pos_2grams = _bigrams(pos_tokens)

    shingles: Set[str] = set()
    for a, b in lex_2grams:
        shingles.add(f"L:{a}|{b}")
    for a, b in skel_2grams:
        shingles.add(f"S:{a}|{b}")
    for tok in lexical_tokens:
        if len(tok) >= 4 and tok not in _STRUCT_POS_MAP and tok not in {"__math__", "__cloze__"}:
            shingles.add(f"T:{tok}")

    return {
        "lexical_tokens": lexical_tokens,
        "skeleton_tokens": skeleton_tokens,
        "pos_tokens": pos_tokens,
        "hints": hints,
        "lex_2grams": lex_2grams,
        "skel_2grams": skel_2grams,
        "pos_2grams": pos_2grams,
        "shingles": shingles,
    }


def _lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    """Computes Longest Common Subsequence length in O(min(m, n)) space."""
    if not seq_a or not seq_b:
        return 0
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    prev = [0] * (len(seq_b) + 1)
    for tok_a in seq_a:
        curr = [0] * (len(seq_b) + 1)
        for j, tok_b in enumerate(seq_b, 1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[-1]


def _lcs_f1(seq_a: List[str], seq_b: List[str]) -> float:
    if not seq_a or not seq_b:
        return 0.0
    l = _lcs_length(seq_a, seq_b)
    return (2.0 * l) / (len(seq_a) + len(seq_b))


def _set_dice(set_a: Set[Any], set_b: Set[Any]) -> float:
    if not set_a or not set_b:
        return 0.0
    return (2.0 * len(set_a & set_b)) / (len(set_a) + len(set_b))


def compute_structural_similarity(rep_a: Dict[str, Any], rep_b: Dict[str, Any]) -> float:
    """Computes normalized structural/template similarity in [0.0, 1.0] between two representations.

    Combines:
    1. Word-level & Skeleton-level ROUGE-L (LCS F1)
    2. Coarse syntactic POS-skeleton alignment gated by template anchor overlap
    3. Word/Skeleton bigram frame overlap
    4. Opening prefix & cloze-hint similarity
    """
    lex_a = rep_a.get("lexical_tokens", [])
    lex_b = rep_b.get("lexical_tokens", [])
    skel_a = rep_a.get("skeleton_tokens", [])
    skel_b = rep_b.get("skeleton_tokens", [])
    if len(lex_a) < 2 or len(lex_b) < 2:
        return 0.0

    s_lex_lcs = _lcs_f1(lex_a, lex_b)
    s_skel_lcs = _lcs_f1(skel_a, skel_b)
    s_lcs = max(s_lex_lcs, s_skel_lcs)
    if s_lcs <= 0.0:
        return 0.0

    pos_a = rep_a.get("pos_tokens", [])
    pos_b = rep_b.get("pos_tokens", [])
    s_pos_lcs = _lcs_f1(pos_a, pos_b)
    s_pos_2g = _set_dice(rep_a.get("pos_2grams", set()), rep_b.get("pos_2grams", set()))
    s_pos = 0.60 * s_pos_lcs + 0.40 * s_pos_2g

    s_lex_2g = _set_dice(rep_a.get("lex_2grams", set()), rep_b.get("lex_2grams", set()))
    s_skel_2g = _set_dice(rep_a.get("skel_2grams", set()), rep_b.get("skel_2grams", set()))
    s_ngram = max(s_lex_2g, s_skel_2g)

    pref_lex = 0
    for x, y in zip(lex_a, lex_b):
        if x == y:
            pref_lex += 1
        else:
            break
    pref_skel = 0
    for x, y in zip(skel_a, skel_b):
        if x == y:
            pref_skel += 1
        else:
            break
    pref = max(pref_lex, pref_skel)
    s_pref = min(1.0, pref / max(1, min(5, len(lex_a), len(lex_b))))

    pos_gated = s_pos * (0.40 + 0.60 * math.sqrt(s_lcs))
    score = (
        0.35 * math.sqrt(s_lcs)
        + 0.25 * pos_gated
        + 0.25 * math.sqrt(s_ngram)
        + 0.15 * math.sqrt(s_pref)
    )

    hints_a = rep_a.get("hints", set())
    hints_b = rep_b.get("hints", set())
    if hints_a and hints_b:
        s_hint = _set_dice(hints_a, hints_b)
        if s_hint > 0.0:
            score += 0.05 * s_hint

    return round(min(1.0, max(0.0, score)), 4)
