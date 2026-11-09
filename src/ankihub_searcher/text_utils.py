"""Text processing and Markdown parsing utilities for Smart Search.

Provides:
1. clean_html_and_cloze: HTML, Cloze, and media cleaner for Anki note fields.
2. clean_tags: Smart tag cleaner stripping AnKing / Mastering Radiology boilerplate.
3. parse_markdown_units: Markdown batch parser with empty-header handling and breadcrumbs.
4. normalize_inline_latex: Inline LaTeX cleaner converting math and Greek letters to Unicode.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple


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
