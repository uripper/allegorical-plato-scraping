import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd

CORPUS_DIR = Path(".cache/sources/canonical-greekLit/data/tlg0059")
OUTPUT_FILE = Path("data/utterances.parquet")

SOURCE_PATTERNS = (
    "*.perseus-grc*.xml",
    "*.perseus-eng*.xml",
)

NS = {
    "tei": "http://www.tei-c.org/ns/1.0",
}

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

Language = Literal["grc", "eng"]
VersionType = Literal["edition", "translation"]


class UtteranceRecord(TypedDict):
    utterance_id: str
    work_id: str
    version_id: str
    version_type: VersionType
    language: Language
    sequence: int
    section_number: str | None
    speaker_local_id: str | None
    speaker_label: str | None
    text_normalized: str
    stephanus_markers: list[str]
    rend: str | None
    source_path: str


UTTERANCE_COLUMNS = [
    "utterance_id",
    "work_id",
    "version_id",
    "version_type",
    "language",
    "sequence",
    "section_number",
    "speaker_local_id",
    "speaker_label",
    "text_normalized",
    "stephanus_markers",
    "rend",
    "source_path",
]


def local_name(element: ET.Element) -> str:
    """Return an XML tag without its namespace."""

    return element.tag.rsplit("}", maxsplit=1)[-1]


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace into single spaces."""

    return " ".join(text.split())


def extract_speech_text(speech: ET.Element) -> str:
    """Extract speech text while excluding the printed speaker label."""

    parts: list[str] = []

    if speech.text:
        parts.append(speech.text)

    for child in speech:
        if local_name(child) != "label":
            parts.extend(child.itertext())

        if child.tail:
            parts.append(child.tail)

    return normalize_whitespace("".join(parts))


def extract_speaker_id(speech: ET.Element) -> str | None:
    """Extract the document-local speaker identifier."""

    speaker_reference = speech.get("who")

    if speaker_reference is None:
        return None

    return speaker_reference.removeprefix("#")


def extract_speaker_label(speech: ET.Element) -> str | None:
    """Extract the printed speaker label."""

    label = speech.findtext("./tei:label", namespaces=NS)

    return None if label is None else label.strip() or None


def extract_stephanus_markers(
    speech: ET.Element,
) -> list[str]:
    """Return Stephanus section markers embedded in a speech."""

    markers: list[str] = []

    for milestone in speech.findall(
        ".//tei:milestone[@unit='section']",
        NS,
    ):
        marker = milestone.get("n")

        if marker is not None:
            markers.append(marker)

    return markers


def parse_xml(file_path: Path) -> ET.Element:
    """Parse a TEI XML file and return its root element."""

    try:
        return ET.parse(file_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(
            f"Could not parse TEI XML file: {file_path}"
        ) from exc


def find_text_version(
    root: ET.Element,
    file_path: Path,
) -> tuple[ET.Element, str, VersionType]:
    """Return the edition or translation element and its CTS URN."""

    body = root.find(".//tei:text/tei:body", NS)

    if body is None:
        raise ValueError(f"No TEI body found in {file_path}")

    for element in body.findall("./tei:div", NS):
        raw_version_type = element.get("type")

        if raw_version_type == "edition":
            version_type: VersionType = "edition"
        elif raw_version_type == "translation":
            version_type = "translation"
        else:
            continue

        version_id = element.get("n")

        if version_id is None:
            raise ValueError(
                f"Text version is missing its CTS URN in {file_path}"
            )

        return element, version_id, version_type

    raise ValueError(
        f"No edition or translation found in {file_path}"
    )

def extract_language(
    root: ET.Element,
    version: ET.Element,
    file_path: Path,
) -> Language:
    """Extract and validate the text language."""

    text = root.find(".//tei:text", NS)

    language = version.get(XML_LANG)

    if language is None and text is not None:
        language = text.get(XML_LANG)

    if language == "grc":
        return "grc"

    if language == "eng":
        return "eng"

    raise ValueError(
        f"Unsupported or missing language {language!r} "
        f"in {file_path}"
    )

def extract_utterances(
    file_path: Path,
    *,
    corpus_dir: Path,
) -> Iterator[UtteranceRecord]:
    """Yield utterance records from one TEI text version."""

    root = parse_xml(file_path)

    version, version_id, version_type = find_text_version(
        root,
        file_path,
    )

    language = extract_language(
        root,
        version,
        file_path,
    )

    work_id = version_id.rsplit(".", maxsplit=1)[0]
    sequence = 0

    sections = version.findall(
        "./tei:div[@type='textpart']",
        NS,
    )

    for section in sections:
        section_number = section.get("n")

        for speech in section.findall(".//tei:said", NS):
            sequence += 1

            yield {
                "utterance_id": f"{version_id}:{sequence}",
                "work_id": work_id,
                "version_id": version_id,
                "version_type": version_type,
                "language": language,
                "sequence": sequence,
                "section_number": section_number,
                "speaker_local_id": extract_speaker_id(speech),
                "speaker_label": extract_speaker_label(speech),
                "text_normalized": extract_speech_text(speech),
                "stephanus_markers": extract_stephanus_markers(
                    speech
                ),
                "rend": speech.get("rend"),
                "source_path": file_path.relative_to(
                    corpus_dir
                ).as_posix(),
            }


def find_source_files(corpus_dir: Path) -> list[Path]:
    """Return all supported Greek and English TEI files."""

    source_files = sorted(
        {
            file_path
            for pattern in SOURCE_PATTERNS
            for file_path in corpus_dir.rglob(pattern)
        }
    )

    if not source_files:
        patterns = ", ".join(repr(pattern) for pattern in SOURCE_PATTERNS)

        raise FileNotFoundError(
            f"No files matching {patterns} "
            f"were found under {corpus_dir}"
        )

    return source_files


def build_utterance_dataframe(
    corpus_dir: Path,
) -> pd.DataFrame:
    """Extract all supported Plato utterances."""

    records = [
        utterance
        for file_path in find_source_files(corpus_dir)
        for utterance in extract_utterances(
            file_path,
            corpus_dir=corpus_dir,
        )
    ]

    return pd.DataFrame.from_records(
        records,
        columns=UTTERANCE_COLUMNS,
    ).convert_dtypes()


def write_dataframe(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """Write an utterance DataFrame to Parquet."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        output_file,
        index=False,
    )


def main() -> None:
    dataframe = build_utterance_dataframe(CORPUS_DIR)
    write_dataframe(dataframe, OUTPUT_FILE)

    language_counts = dataframe["language"].value_counts()

    print(
        f"Wrote {len(dataframe):,} utterances "
        f"to {OUTPUT_FILE}"
    )

    for language, count in language_counts.items():
        print(f"  {language}: {count:,}")


if __name__ == "__main__":
    main()