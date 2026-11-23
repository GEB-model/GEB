"""Tests for the Markdown extensions used to build the documentation."""

from pathlib import Path

import markdown
import pytest

BIB_FILE: str = str(Path(__file__).parent.parent / "docs" / "references.bib")


def render(text: str, strict: bool = True) -> str:
    """Render Markdown with the citation extension.

    Args:
        text: The Markdown source.
        strict: Whether unknown citation keys raise an error.

    Returns:
        The rendered HTML.
    """
    return markdown.markdown(
        text,
        extensions=["geb_docs.bibtex"],
        extension_configs={"geb_docs.bibtex": {"bib_file": BIB_FILE, "strict": strict}},
    )


def test_citations_and_reference_list() -> None:
    """Citations link to their entries in an appended reference list."""
    html: str = render(
        "One [@tol2008climate], two [@tierolf2023coupled; @de2022usa], "
        "prefix and suffix [see @tol2008climate, p. 3]."
    )
    assert '<a href="#ref-tol2008climate">Tol, 2008</a>' in html
    assert '<a href="#ref-de2022usa">de Ruig <i>et al.</i>, 2022</a>' in html
    assert "see Tol, 2008, p. 3" in html
    assert '<h2 id="references">References</h2>' in html
    # Each work appears once in the reference list.
    assert html.count('<li id="ref-tol2008climate">') == 1
    assert html.count("<li ") == 3


def test_bibliography_marker() -> None:
    """The reference list replaces a `/// bibliography` paragraph."""
    html: str = render(
        "Text [@tol2008climate].\n\n## Sources\n\n/// bibliography\n\nEnd."
    )
    assert "/// bibliography" not in html
    assert '<h2 id="references">' not in html
    assert (
        html.index("Sources")
        < html.index('id="ref-tol2008climate"')
        < html.index("End.")
    )


def test_text_that_is_not_a_citation() -> None:
    """Escaped brackets, e-mail addresses and code are left alone."""
    html: str = render(
        r"Escaped \[@tol2008climate\], e-mail [x@example.com], code `[@tol2008climate]`."
    )
    assert "[@tol2008climate]" in html
    assert "[x@example.com]" in html
    assert "<code>[@tol2008climate]</code>" in html
    assert "references" not in html


def test_unknown_key() -> None:
    """Unknown keys fail in strict mode and are marked otherwise."""
    with pytest.raises(KeyError, match="does_not_exist"):
        render("[@does_not_exist]")
    assert "[@does_not_exist?]" in render("[@does_not_exist]", strict=False)


def test_latex_in_bibtex_entries() -> None:
    """LaTeX accents and braces in the BibTeX file become plain Unicode."""
    from geb_docs.bibtex import latex_to_text

    assert latex_to_text(r"Schr{\"o}dinger and {GEB}") == "Schrödinger and GEB"
