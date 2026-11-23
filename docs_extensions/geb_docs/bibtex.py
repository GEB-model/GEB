"""Pandoc-style BibTeX citations for Python-Markdown, without pandoc.

Citations use the pandoc syntax. ``[@key]``, ``[@key1; @key2]`` and
``[see @key, p. 12]`` give parenthetical citations: "(Author, 2021)". A bare
``@key`` gives a narrative citation for use in a sentence: "Author (2021)".
Write ``\@key`` for a literal "@key". Text is never added or removed around a
citation, so write the spacing and punctuation in the Markdown as it should
appear. Each citation is rendered in the
configured CSL style and links to its entry in a reference list. The list is
placed where a paragraph contains only ``/// bibliography``, or appended to
the end of the page otherwise. Unknown keys fail the build unless ``strict``
is set to false.

The BibTeX file is parsed with pybtex (which also converts LaTeX markup to
Unicode) and formatted with citeproc-py, a pure-Python CSL processor.

Enable it in ``zensical.toml`` with:

    [project.markdown_extensions."geb_docs.bibtex"]
    bib_file = "docs/references.bib"
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as etree
from functools import lru_cache
from pathlib import Path
from typing import Any

from citeproc import (
    Citation,
    CitationItem,
    CitationStylesBibliography,
    CitationStylesStyle,
    formatter,
)
from citeproc.source.json import CiteProcJSON
from markdown import Markdown
from markdown.extensions import Extension
from markdown.inlinepatterns import InlineProcessor
from markdown.treeprocessors import Treeprocessor
from pybtex.backends.plaintext import Backend as PlainTextBackend
from pybtex.database import Entry, Person, parse_file
from pybtex.richtext import Text

logger = logging.getLogger("geb_docs.bibtex")

# Created once: `Text.render_as("text")` looks the backend up through package
# entry points on every call, which is very slow on network file systems.
PLAIN_TEXT: PlainTextBackend = PlainTextBackend()

# Opening bracket, then citation items separated by ";", each holding one
# @key (keys may contain letters, digits and _:.#$%&+?<>~/-).
CITATION_PATTERN: str = r"\[((?:[^\[\]@;]*@[\w:.#$%&+?<>~/-]+[^\[\];]*;?\s*)+)\]"
KEY_PATTERN: re.Pattern[str] = re.compile(r"@([\w:.#$%&+?<>~/-]+)")
# A bare @key not preceded by a word character (so e-mail addresses and
# similar text are not matched). Trailing punctuation is not part of the key.
NARRATIVE_PATTERN: str = r"(?<![\w@\\])@([\w:#$%&+?<>~/-]+(?:[.][\w:#$%&+?<>~/-]+)*)"
BIBLIOGRAPHY_MARKERS: frozenset[str] = frozenset({"/// bibliography", "\\bibliography"})

BIBTEX_TO_CSL_TYPE: dict[str, str] = {
    "article": "article-journal",
    "book": "book",
    "booklet": "pamphlet",
    "inbook": "chapter",
    "incollection": "chapter",
    "inproceedings": "paper-conference",
    "conference": "paper-conference",
    "manual": "report",
    "mastersthesis": "thesis",
    "phdthesis": "thesis",
    "techreport": "report",
    "report": "report",
    "dataset": "dataset",
    "software": "software",
    "online": "webpage",
    "misc": "document",
    "unpublished": "manuscript",
}

# BibTeX field -> CSL-JSON variable for plain text fields.
BIBTEX_TO_CSL_FIELD: dict[str, str] = {
    "title": "title",
    "journal": "container-title",
    "journaltitle": "container-title",
    "booktitle": "container-title",
    "series": "collection-title",
    "volume": "volume",
    "number": "issue",
    "pages": "page",
    "edition": "edition",
    "publisher": "publisher",
    "institution": "publisher",
    "school": "publisher",
    "organization": "publisher",
    "address": "publisher-place",
    "location": "publisher-place",
    "doi": "DOI",
    "url": "URL",
    "isbn": "ISBN",
    "issn": "ISSN",
    "note": "note",
    "version": "version",
}

MONTHS: dict[str, int] = {
    name: number
    for number, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}


def latex_to_text(value: str) -> str:
    """Convert a BibTeX/LaTeX string (accents, braces) to plain Unicode text.

    Args:
        value: The raw field value from the BibTeX file.

    Returns:
        The value as plain text.
    """
    try:
        return Text.from_latex(value).render(PLAIN_TEXT).strip()
    except Exception:  # noqa: BLE001 - malformed LaTeX should not break the build
        return value.replace("{", "").replace("}", "").strip()


def person_to_csl(person: Person) -> dict[str, str]:
    """Convert a pybtex person to a CSL-JSON name.

    Names written in braces in BibTeX (e.g. ``{OECD}``) are corporate names
    and become CSL ``literal`` names.

    Args:
        person: The pybtex person.

    Returns:
        The CSL-JSON name object.
    """
    last_names: list[str] = list(person.last_names or [])
    family: str = latex_to_text(" ".join(last_names))
    given: str = latex_to_text(
        " ".join([*(person.first_names or []), *(person.middle_names or [])])
    )
    particle: str = latex_to_text(" ".join(person.prelast_names or []))
    suffix: str = latex_to_text(" ".join(person.lineage_names or []))
    if not given and not particle and len(last_names) == 1:
        raw: str = last_names[0]
        if raw.startswith("{") and raw.endswith("}"):
            return {"literal": family}
    name: dict[str, str] = {"family": family}
    if given:
        name["given"] = given
    if particle:
        name["non-dropping-particle"] = particle
    if suffix:
        name["suffix"] = suffix
    return name


def entry_to_csl(key: str, entry: Entry) -> dict[str, Any]:
    """Convert a pybtex entry to a CSL-JSON item.

    Args:
        key: The citation key.
        entry: The pybtex entry.

    Returns:
        The CSL-JSON item.
    """
    fields: dict[str, str] = {
        name.lower(): value for name, value in (entry.fields or {}).items()
    }
    persons = entry.persons or {}
    entry_type: str = (entry.type or "misc").lower()
    item: dict[str, Any] = {
        "id": key,
        "type": BIBTEX_TO_CSL_TYPE.get(entry_type, "document"),
    }
    if entry_type == "phdthesis":
        item["genre"] = "PhD thesis"
    elif entry_type == "mastersthesis":
        item["genre"] = "Master's thesis"

    for bibtex_field, csl_field in BIBTEX_TO_CSL_FIELD.items():
        if bibtex_field in fields and csl_field not in item:
            item[csl_field] = latex_to_text(fields[bibtex_field])
    if "page" in item:
        item["page"] = item["page"].replace("--", "-")
    if "DOI" in item:
        item["DOI"] = re.sub(r"^https?://(dx\.)?doi\.org/", "", item["DOI"])
        # The DOI is rendered as a link, so a separate URL is redundant.
        item.pop("URL", None)

    date_parts: list[int] = []
    year_match = re.search(r"\d{4}", fields.get("year", fields.get("date", "")))
    if year_match:
        date_parts.append(int(year_match.group()))
        month: str = fields.get("month", "").strip().lower()
        month_number: int | None = (
            int(month) if month.isdigit() else MONTHS.get(month[:3] if month else "")
        )
        if month_number:
            date_parts.append(month_number)
        item["issued"] = {"date-parts": [date_parts]}

    for role in ("author", "editor"):
        if role in persons:
            item[role] = [person_to_csl(person) for person in persons[role]]
    return item


@lru_cache(maxsize=4)
def load_bibliography(bib_file: str, mtime: float) -> CiteProcJSON:
    """Parse a BibTeX file into a citeproc source (cached per file version).

    Args:
        bib_file: Path to the BibTeX file.
        mtime: Modification time of the file, used to invalidate the cache.

    Returns:
        The citeproc source holding all entries.
    """
    database = parse_file(bib_file, bib_format="bibtex")
    return CiteProcJSON(
        [entry_to_csl(key, entry) for key, entry in database.entries.items()]
    )


@lru_cache(maxsize=4)
def load_style(csl_file: str) -> CitationStylesStyle:
    """Load a CSL style by path or by the name of a style bundled with citeproc-py.

    Args:
        csl_file: Path to a ``.csl`` file, or a bundled style name.

    Returns:
        The citation style.
    """
    return CitationStylesStyle(csl_file, validate=False)


def anchor(key: str) -> str:
    """Return the HTML id of a reference list entry.

    Args:
        key: The citation key.

    Returns:
        The id attribute value.
    """
    return "ref-" + re.sub(r"[^\w-]", "-", key)


class CitationInlineProcessor(InlineProcessor):
    """Render ``[@key; ...]`` as an author-date citation linking to the references."""

    def __init__(self, pattern: str, extension: BibtexExtension) -> None:
        """Initialize the processor.

        Args:
            pattern: The regular expression matching a citation.
            extension: The extension holding the page's bibliography state.
        """
        super().__init__(pattern, extension.md)
        self.extension = extension
        self.markdown: Markdown = extension.md

    def handleMatch(  # noqa: N802
        self, m: re.Match[str], data: str
    ) -> tuple[etree.Element | str | None, int | None, int | None]:
        """Render one citation.

        Args:
            m: The regular expression match.
            data: The text being processed.

        Returns:
            The rendered citation and the span it replaces. Returns ``None``
            for text that only looks like a citation (e.g. an e-mail address
            in brackets), so other patterns can handle it.
        """
        items: list[CitationItem] = []
        for part in m.group(1).split(";"):
            key_match = KEY_PATTERN.search(part)
            if key_match is None:
                return None, None, None
            raw_prefix: str = part[: key_match.start()]
            if raw_prefix and not raw_prefix[-1].isspace() and raw_prefix[-1] != "-":
                return None, None, None  # e.g. "[mail user@example.com]"
            prefix: str = raw_prefix.strip()
            suffix: str = part[key_match.end() :].strip()
            key: str = key_match.group(1).rstrip(".,:")
            kwargs: dict[str, str] = {}
            if prefix:
                kwargs["prefix"] = prefix + " "
            if suffix:
                kwargs["suffix"] = suffix if suffix.startswith(",") else ", " + suffix
            items.append(CitationItem(key, **kwargs))

        html: str = self.extension.cite(items)
        return self.markdown.htmlStash.store(html), m.start(0), m.end(0)


class NarrativeCitationInlineProcessor(InlineProcessor):
    """Render a bare ``@key`` as a narrative citation: "Author (2021)"."""

    def __init__(self, pattern: str, extension: BibtexExtension) -> None:
        """Initialize the processor.

        Args:
            pattern: The regular expression matching a narrative citation.
            extension: The extension holding the page's bibliography state.
        """
        super().__init__(pattern, extension.md)
        self.extension = extension
        self.markdown: Markdown = extension.md

    def handleMatch(  # noqa: N802
        self, m: re.Match[str], data: str
    ) -> tuple[etree.Element | str | None, int | None, int | None]:
        """Render one narrative citation.

        Args:
            m: The regular expression match.
            data: The text being processed.

        Returns:
            The rendered citation and the span it replaces. Keys that are not
            in the BibTeX file are left as plain text (e.g. "@username").
        """
        html: str | None = self.extension.cite_narrative(m.group(1))
        if html is None:
            return None, None, None
        return self.markdown.htmlStash.store(html), m.start(0), m.end(0)


class BibliographyTreeprocessor(Treeprocessor):
    """Insert the reference list for all citations on the page."""

    def __init__(self, extension: BibtexExtension) -> None:
        """Initialize the processor.

        Args:
            extension: The extension holding the page's bibliography state.
        """
        super().__init__(extension.md)
        self.extension = extension
        self.markdown: Markdown = extension.md

    def run(self, root: etree.Element) -> None:
        """Place the reference list at the marker or at the end of the page.

        Args:
            root: The root element of the page.
        """
        marker_parent: etree.Element | None = None
        marker_index: int = -1
        for parent in root.iter():
            for index, child in enumerate(parent):
                if (
                    child.tag == "p"
                    and len(child) == 0
                    and (child.text or "").strip() in BIBLIOGRAPHY_MARKERS
                ):
                    marker_parent, marker_index = parent, index
                    break
            if marker_parent is not None:
                break

        references: list[tuple[str, str]] = (
            [] if self.extension.bibliography is None else self.extension.references()
        )
        if not references:
            if marker_parent is not None:
                marker_parent.remove(marker_parent[marker_index])
            return

        container = etree.Element("div", {"class": "bibliography"})
        if marker_parent is None:
            heading = etree.SubElement(container, "h2", {"id": "references"})
            heading.text = self.extension.getConfig("bibliography_title")
        reference_list = etree.SubElement(container, "ul", {"class": "references"})
        for key, html in references:
            item = etree.SubElement(reference_list, "li", {"id": anchor(key)})
            item.text = self.markdown.htmlStash.store(html)

        if marker_parent is None:
            root.append(container)
        else:
            marker_parent[marker_index] = container


class BibtexExtension(Extension):
    """Python-Markdown extension for BibTeX citations and reference lists."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the extension.

        Args:
            **kwargs: Extension configuration, see ``self.config``.
        """
        self.config = {
            "bib_file": ["docs/references.bib", "Path to the BibTeX file."],
            "csl_file": [
                "harvard-cite-them-right",
                "Path to a CSL style file, or the name of a style bundled "
                "with citeproc-py.",
            ],
            "bibliography_title": ["References", "Heading of the reference list."],
            "strict": [
                True,
                "Fail the build on citation keys missing from the BibTeX file. "
                "When false, they are logged and shown as '[@key?]'.",
            ],
        }
        super().__init__(**kwargs)
        self.bibliography: CitationStylesBibliography | None = None

    def extendMarkdown(self, md: Markdown) -> None:  # noqa: N802
        """Register the citation processors.

        The inline processor runs before links (priority 160) so that
        ``[@key]`` is not parsed as a reference link, and after escapes
        (priority 180) so that ``\\[@key\\]`` stays literal text.

        Args:
            md: The Markdown instance.
        """
        self.md = md
        md.registerExtension(self)
        # Allow "\@key" for a literal @key, as in pandoc.
        if "@" not in md.ESCAPED_CHARS:
            md.ESCAPED_CHARS.append("@")
        md.inlinePatterns.register(
            CitationInlineProcessor(CITATION_PATTERN, self), "geb_citation", 175
        )
        md.inlinePatterns.register(
            NarrativeCitationInlineProcessor(NARRATIVE_PATTERN, self),
            "geb_narrative_citation",
            174,
        )
        md.treeprocessors.register(
            BibliographyTreeprocessor(self), "geb_bibliography", 5
        )

    def reset(self) -> None:
        """Forget the citations of the previous page."""
        self.bibliography = None

    def cite(self, items: list[CitationItem]) -> str:
        """Register and render a citation.

        Args:
            items: The cited items.

        Returns:
            The citation as HTML, each work linked to its reference entry.

        Raises:
            KeyError: If a key is missing from the BibTeX file and ``strict``
                is enabled.
        """
        bibliography: CitationStylesBibliography = self.get_bibliography()
        known: list[CitationItem] = []
        for item in items:
            if item.key in bibliography.source:
                known.append(item)
            elif self.getConfig("strict"):
                raise KeyError(
                    f"Citation key '{item.key}' not found in "
                    f"{self.getConfig('bib_file')}"
                )
            else:
                logger.warning("Citation key not found in bibliography: %s", item.key)
        if not known:
            missing: str = "; ".join("@" + item.key for item in items)
            return f'<span class="citation citation-missing">[{missing}?]</span>'

        # Render each work separately so that each links to its own entry.
        parts: list[tuple[str, str]] = []
        for item in known:
            citation = Citation([item])
            bibliography.register(citation)
            parts.append((item.key, bibliography.cite(citation, lambda _: None)))

        # Author-date styles wrap a citation in parentheses: merge the
        # single-work citations into one, e.g. "(A, 2020; B, 2021)".
        if all(text.startswith("(") and text.endswith(")") for _, text in parts):
            links: str = "; ".join(
                f'<a href="#{anchor(key)}">{text[1:-1]}</a>' for key, text in parts
            )
            return f'<span class="citation">({links})</span>'
        return (
            '<span class="citation">'
            + " ".join(f'<a href="#{anchor(key)}">{text}</a>' for key, text in parts)
            + "</span>"
        )

    def get_bibliography(self) -> CitationStylesBibliography:
        """Return the page's bibliography, creating it on first use.

        Returns:
            The citeproc bibliography that collects the page's citations.
        """
        if self.bibliography is None:
            bib_file: str = self.getConfig("bib_file")
            source = load_bibliography(bib_file, Path(bib_file).stat().st_mtime)
            self.bibliography = CitationStylesBibliography(
                load_style(self.getConfig("csl_file")), source, formatter.html
            )
        return self.bibliography

    def cite_narrative(self, key: str) -> str | None:
        """Register and render a narrative citation, e.g. "Author (2021)".

        Args:
            key: The citation key.

        Returns:
            The citation as HTML linked to its reference entry, or None if the
            key is not in the BibTeX file.
        """
        bibliography: CitationStylesBibliography = self.get_bibliography()
        if key.lower() not in bibliography.source:
            return None
        citation = Citation([CitationItem(key)])
        bibliography.register(citation)
        text: str = bibliography.cite(citation, lambda _: None)
        # Author-date styles give "(Author, 2021)": rewrite to "Author (2021)".
        if text.startswith("(") and text.endswith(")") and ", " in text:
            author, year = text[1:-1].rsplit(", ", 1)
            text = f"{author} ({year})"
        return f'<span class="citation"><a href="#{anchor(key)}">{text}</a></span>'

    def references(self) -> list[tuple[str, str]]:
        """Return the formatted reference list of the current page.

        Returns:
            Pairs of citation key and formatted HTML entry, sorted by the style.
        """
        assert self.bibliography is not None
        self.bibliography.sort()
        return [
            (key, str(entry))
            for key, entry in zip(
                self.bibliography.keys, self.bibliography.bibliography(), strict=True
            )
        ]


def makeExtension(**kwargs: Any) -> BibtexExtension:  # noqa: N802
    """Create the extension (entry point used by Python-Markdown).

    Args:
        **kwargs: Extension configuration.

    Returns:
        The extension instance.
    """
    return BibtexExtension(**kwargs)
