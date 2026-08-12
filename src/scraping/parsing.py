"""Build an auditable Plato corpus directly from Perseus TEI.

The XML, rather than a previously flattened table, is the authority.  In
particular, editorial notes are removed while walking the tree and their tail
text is retained.  Every generated row points back to its source element.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

CORPUS_DIR = Path(".cache/sources/canonical-greekLit/data/tlg0059")
OUTPUT_DIR = Path("data/corpus")
NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
SOURCE_PATTERNS = ("*.perseus-grc*.xml", "*.perseus-eng*.xml")
REMOVED_FROM_SPEECH = {"note"}

# Explicit identities only.  Unknown labels get a stable, reviewable local
# identity rather than a guessed cross-language equivalence.
ALIASES = {
    "socrates": "socrates", "soc": "socrates", "σω": "socrates",
    "σωκρατης": "socrates", "σωκράτης": "socrates", "σocrates": "socrates",
    "younger_socrates": "younger_socrates", "y_soc": "younger_socrates",
    "νεωτερος σωκρατης": "younger_socrates", "νεώτερος_σωκράτης": "younger_socrates",
    "crito": "crito", "κριτων": "crito", "κρίτων": "crito", "cr": "crito", "κρ": "crito",
    "plato": "plato", "πλατων": "plato",
}

UTTERANCE_COLUMNS = [
    "utterance_id", "passage_id", "work_id", "version_id", "language",
    "sequence", "segment_type", "speaker_id", "speaker_local_id",
    "speaker_label_raw", "text_raw", "text_clean", "section_start",
    "section_end", "stephanus_markers", "source_path", "source_xpath",
    "source_fragment_ids", "note_count",
]


def local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    return nfc(" ".join(text.split()))


def element_text(element: ET.Element, *, clean: bool) -> str:
    """Extract text structurally; excluded nodes keep their following tail."""
    parts: list[str] = []

    def visit(node: ET.Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            name = local_name(child)
            if name != "label" and not (clean and name in REMOVED_FROM_SPEECH):
                visit(child)
            if child.tail:  # tail belongs to the parent, never to removed child
                parts.append(child.tail)

    visit(element)
    return normalize_whitespace("".join(parts))


def slug(value: str) -> str:
    value = nfc(value).casefold().replace("ς", "σ")
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE).strip("_")
    return value or "unknown"


def canonical_speaker(local_id: str | None, label: str | None) -> tuple[str | None, str, float]:
    candidates = [x for x in (local_id, label) if x]
    for candidate in candidates:
        key = slug(candidate.removeprefix("#").rstrip("."))
        if key in ALIASES:
            return ALIASES[key], "explicit_alias", 1.0
    if local_id:
        # Exact identifier mapping is safe within a language, but deliberately
        # not advertised as a cross-language identification.
        return f"local:{slug(local_id)}", "local_identifier", 0.7
    return None, "unresolved", 0.0


def markers(element: ET.Element) -> list[str]:
    return [m.get("n") for m in element.iter() if local_name(m) == "milestone"
            and m.get("unit") == "section" and m.get("n")]


def xpath_for(element: ET.Element, parents: dict[ET.Element, ET.Element]) -> str:
    parts: list[str] = []
    node = element
    while node in parents:
        parent = parents[node]
        same = [c for c in parent if local_name(c) == local_name(node)]
        parts.append(f"{local_name(node)}[{same.index(node) + 1}]")
        node = parent
    parts.append(local_name(node))
    return "/" + "/".join(reversed(parts))


def source_id(path: str, xpath: str) -> str:
    return "frag-" + hashlib.sha256(f"{path}#{xpath}".encode()).hexdigest()[:16]


def first_text(root: ET.Element, path: str) -> str | None:
    node = root.find(path, NS)
    return normalize_whitespace("".join(node.itertext())) if node is not None else None


def parse_version(path: Path, corpus_dir: Path) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any],
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]],
]:
    root = ET.parse(path).getroot()
    version = next((d for d in root.findall(".//tei:text/tei:body/tei:div", NS)
                    if d.get("type") in {"edition", "translation"}), None)
    if version is None or not version.get("n"):
        raise ValueError(f"No CTS edition/translation in {path}")
    version_id = version.get("n")
    assert version_id
    work_id = version_id.rsplit(".", 1)[0]
    language = version.get(XML_LANG) or root.find(".//tei:text", NS).get(XML_LANG)  # type: ignore[union-attr]
    rel = path.relative_to(corpus_dir).as_posix()
    parents = {child: parent for parent in root.iter() for child in parent}
    title = first_text(root, ".//tei:titleStmt/tei:title") or work_id

    version_row = {
        "version_id": version_id, "work_id": work_id, "language": language,
        "version_type": version.get("type"),
        "translator_or_editor": "; ".join(dict.fromkeys(
            normalize_whitespace("".join(e.itertext())) for e in root.findall(".//tei:titleStmt/tei:editor", NS)
        )) or None,
        "publication_date": first_text(root, ".//tei:publicationStmt/tei:date"),
        "publisher": first_text(root, ".//tei:publicationStmt/tei:publisher"),
        "source_urn": version_id, "source_path": rel,
        "rights": first_text(root, ".//tei:publicationStmt/tei:availability"),
    }
    work_row = {"work_id": work_id, "work_slug": slug(title), "canonical_title": title,
                "traditional_order": None, "authenticity_status": "unreviewed"}

    notes: list[dict[str, Any]] = []
    body = root.find(".//tei:text/tei:body", NS)
    assert body is not None
    for note in body.iter():
        if local_name(note) != "note":
            continue
        xp = xpath_for(note, parents)
        notes.append({
            "note_id": source_id(rel, xp), "work_id": work_id, "version_id": version_id,
            "language": language, "note_text_raw": element_text(note, clean=False),
            "note_text_clean": element_text(note, clean=False), "note_type": note.get("type") or "editorial_note",
            "responsibility": note.get("resp"), "source_path": rel, "source_xpath": xp,
        })

    candidates: list[tuple[ET.Element, str]] = []
    said_nodes = list(version.iterfind(".//tei:said", NS))
    nested_said = {x for s in said_nodes for x in s.iterfind(".//tei:said", NS)}
    candidates.extend((s, "speech") for s in said_nodes if s not in nested_said)
    # Preserve meaningful non-speech blocks; do not inherit nearby speakers.
    for node in version.iter():
        name = local_name(node)
        if name == "head":
            candidates.append((node, "heading"))
        elif name in {"stage"}:
            candidates.append((node, "stage_direction"))
        elif name in {"p", "l", "ab"} and all(
            local_name(x) != "said" for x in node.iter()
        ):
            candidates.append((node, "narration"))
    order = {node: i for i, node in enumerate(root.iter())}
    candidates.sort(key=lambda pair: order[pair[0]])

    rows: list[dict[str, Any]] = []
    speaker_rows: list[dict[str, Any]] = []
    current_section: str | None = None
    section_for: dict[ET.Element, str | None] = {}
    for node in version.iter():
        if local_name(node) == "div" and node.get("type") == "textpart" and node.get("n"):
            current_section = node.get("n")
        section_for[node] = current_section

    for node, kind in candidates:
        raw, clean = element_text(node, clean=False), element_text(node, clean=True)
        if not raw and not clean:
            continue
        local = node.get("who", "").removeprefix("#") or None
        label_node = next((c for c in node if local_name(c) == "label"), None)
        label = normalize_whitespace("".join(label_node.itertext())) if label_node is not None else None
        sid, method, confidence = canonical_speaker(local, label)
        xp = xpath_for(node, parents)
        fragment = source_id(rel, xp)
        ms = markers(node)
        section = section_for.get(node)
        passage_ref = ms[0] if ms else section
        passage_id = f"{work_id}:{passage_ref}" if passage_ref else None
        note_count = sum(local_name(x) == "note" for x in node.iter())
        rows.append({
            "utterance_id": "", "passage_id": passage_id, "work_id": work_id,
            "version_id": version_id, "language": language, "sequence": 0,
            "segment_type": "unattributed_speech" if kind == "speech" and not local else kind,
            "speaker_id": sid, "speaker_local_id": local, "speaker_label_raw": label,
            "text_raw": raw, "text_clean": clean, "section_start": section,
            "section_end": section, "stephanus_markers": ms, "source_path": rel,
            "source_xpath": xp, "source_fragment_ids": [fragment], "note_count": note_count,
            "_rend": node.get("rend"), "_fragment": fragment,
        })
        if local or label:
            speaker_rows.append({"work_id": work_id, "language": language,
                "speaker_local_id": local, "speaker_label_raw": label, "speaker_id": sid,
                "mapping_method": method, "mapping_confidence": confidence})

    # rend=merge is a continuation marker. Merge only exact adjacent speech
    # identity within a work/version and record both fragments; otherwise retain.
    merged: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for row in rows:
        prior = merged[-1] if merged else None
        can_merge = bool(row["_rend"] == "merge" and prior and
                         row["segment_type"] in {"speech", "unattributed_speech"} and
                         prior["segment_type"] == row["segment_type"] and
                         prior["speaker_local_id"] == row["speaker_local_id"] and
                         prior["section_end"] == row["section_start"])
        decisions.append({"version_id": version_id, "source_fragment_id": row["_fragment"],
                          "rend": row["_rend"], "decision": "merged" if can_merge else "retained",
                          "reason": "adjacent_same_speaker_same_passage" if can_merge else "merge_conditions_not_met"})
        if can_merge:
            prior["text_raw"] = normalize_whitespace(prior["text_raw"] + " " + row["text_raw"])
            prior["text_clean"] = normalize_whitespace(prior["text_clean"] + " " + row["text_clean"])
            prior["source_fragment_ids"].extend(row["source_fragment_ids"])
            prior["stephanus_markers"].extend(row["stephanus_markers"])
            prior["note_count"] += row["note_count"]
        else:
            merged.append(row)
    for sequence, row in enumerate(merged, 1):
        row["sequence"] = sequence
        row["utterance_id"] = f"{version_id}:u{sequence:06d}"
        row.pop("_rend"); row.pop("_fragment")
    return merged, notes, version_row, work_row, speaker_rows, decisions

def find_source_files(corpus_dir: Path) -> list[Path]:
    if files := sorted(
        {p for pattern in SOURCE_PATTERNS for p in corpus_dir.rglob(pattern)}
    ):
        return files
    else:
        raise FileNotFoundError(f"No supported TEI files under {corpus_dir}")
    return files


def build_corpus(corpus_dir: Path = CORPUS_DIR) -> dict[str, pd.DataFrame]:
    utterances: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    versions: list[dict[str, Any]] = []
    works: dict[str, dict[str, Any]] = {}
    speakers: list[dict[str, Any]] = []
    merge_decisions: list[dict[str, Any]] = []
    for path in find_source_files(corpus_dir):
        u, n, v, w, s, d = parse_version(path, corpus_dir)
        utterances.extend(u); notes.extend(n); versions.append(v); works[w["work_id"]] = w
        speakers.extend(s); merge_decisions.extend(d)
    speakers = list({tuple(row.items()): row for row in speakers}.values())
    alignments: list[dict[str, Any]] = []
    by_passage: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in utterances:
        if row["passage_id"]:
            by_passage[(row["work_id"], row["passage_id"])][row["language"]].append(row["utterance_id"])
    for (work_id, passage_id), langs in by_passage.items():
        for grc in langs.get("grc", []):
            alignments.extend(
                {
                    "passage_id": passage_id,
                    "work_id": work_id,
                    "greek_utterance_id": grc,
                    "english_utterance_id": eng,
                    "alignment_method": "shared_cts_or_stephanus_reference",
                    "alignment_confidence": (
                        0.9
                        if len(langs["grc"]) == len(langs["eng"]) == 1
                        else 0.7
                    ),
                }
                for eng in langs.get("eng", [])
            )
    return {
        "utterances": pd.DataFrame(utterances, columns=UTTERANCE_COLUMNS),
        "notes": pd.DataFrame(notes), "versions": pd.DataFrame(versions),
        "works": pd.DataFrame(works.values()), "speakers": pd.DataFrame(speakers),
        "alignments": pd.DataFrame(alignments), "merge_decisions": pd.DataFrame(merge_decisions),
    }


def qa_report(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    u = tables["utterances"]
    normalized = u["text_raw"].map(lambda x: unicodedata.is_normalized("NFC", x)).all() and u["text_clean"].map(lambda x: unicodedata.is_normalized("NFC", x)).all()
    counts = u.groupby(["work_id", "version_id", "language", "segment_type"], dropna=False).size()
    low = tables["speakers"] if tables["speakers"].empty else tables["speakers"].query("mapping_confidence < 0.8")
    passages = set(u["passage_id"].dropna())
    aligned = set() if tables["alignments"].empty else set(tables["alignments"]["passage_id"])
    monotonic = all(
        group["sequence"].tolist() == sorted(group["sequence"].tolist())
        for _, group in u.groupby("version_id", sort=False)
    )
    return {
        "utterance_count": len(u), "note_count": len(tables["notes"]),
        "row_counts": [{**dict(zip(counts.index.names, idx)), "count": int(value)} for idx, value in counts.items()],
        "null_speaker_counts": u[u["speaker_id"].isna()]["segment_type"].value_counts().to_dict(),
        "duplicate_utterance_ids": int(u["utterance_id"].duplicated().sum()),
        "empty_cleaned_utterances": int(u["text_clean"].eq("").sum()),
        "notes_removed_from_speech": int(u["note_count"].sum()),
        "merge_decisions": tables["merge_decisions"]["decision"].value_counts().to_dict(),
        "alignment_coverage": len(aligned) / len(passages) if passages else 0.0,
        "passage_order_monotonic": monotonic,
        "unicode_nfc": bool(normalized),
        "raw_clean_differences": int(u["text_raw"].ne(u["text_clean"]).sum()),
        "short_utterances_under_3_chars": int(u["text_clean"].str.len().lt(3).sum()),
        "long_utterances_over_5000_chars": int(u["text_clean"].str.len().gt(5000).sum()),
        "unresolved_or_low_confidence_speakers": low.to_dict("records"),
    }


def write_corpus(tables: dict[str, pd.DataFrame], output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, dataframe in tables.items():
        dataframe.to_parquet(output_dir / f"{name}.parquet", index=False)
    (output_dir / "qa-report.json").write_text(json.dumps(qa_report(tables), ensure_ascii=False, indent=2, default=str) + "\n")


def main() -> None:
    tables = build_corpus()
    write_corpus(tables)
    print(f"Wrote {len(tables['utterances']):,} utterances and {len(tables['notes']):,} notes to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
