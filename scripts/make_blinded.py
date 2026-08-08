#!/usr/bin/env python3
"""Generate a double-blind manuscript from paper/manuscript.tex.

Food Quality and Preference uses double-blind review, so the Manuscript File item
must carry no author-identifying information. This script strips the author block
and corresponding-author email, removes the presentation venue, and withholds the
repository URL, writing paper/submission/manuscript_blinded.tex (built from
paper/submission/ so it still finds ../figures/).

Regenerate after any manuscript change, then rebuild the blinded PDF:
    python3 scripts/make_blinded.py
    (cd paper/submission && tectonic manuscript_blinded.tex)
"""
from pathlib import Path

SRC = Path("paper/manuscript.tex")
OUT = Path("paper/submission/manuscript_blinded.tex")

AUTHOR_BLOCK = r"""\author{
John M. Ennis\thanks{Corresponding author: john.m.ennis@aigora.com}\\
Aigora, Richmond, Virginia, USA
\and
Thierry Worch\\
FrieslandCampina, Wageningen, The Netherlands
\and
Benjamin Mahieu\\
Oniris, Nantes, France
}"""

REPLACEMENTS = [
    # 1. remove all author identifiers (names, affiliations, corresponding email)
    (AUTHOR_BLOCK, r"\author{}"),
    # 2. fix the figure path now that the file lives in paper/submission/
    (r"\graphicspath{{figures/}}", r"\graphicspath{{../figures/}}"),
    # 3. remove the presentation venue
    ("two days before this work was presented at\nSensometrics 2026.",
     "two days before this work was first presented publicly."),
    # 4. withhold the repository URL (it names the authors' organization)
    ("are available in the project repository at\n"
     r"\url{https://github.com/aigorahub/model-cognition-fit-supporting-materials}.",
     "are available in a public repository. The repository link is withheld here\n"
     "to preserve double-blind review and will be provided on acceptance."),
]


def main() -> int:
    text = SRC.read_text()
    for old, new in REPLACEMENTS:
        if old not in text:
            raise SystemExit(f"blinding target not found (manuscript changed?):\n  {old[:70]!r}")
        text = text.replace(old, new)
    text = text.replace(
        r"\begin{document}",
        "% AUTO-GENERATED blinded manuscript via scripts/make_blinded.py. Do not edit by hand.\n"
        r"\begin{document}",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
