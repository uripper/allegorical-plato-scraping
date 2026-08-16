"""Auditable linguistic enrichment for analysis without altering source text."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import pandas as pd

MODEL_DIR = Path(".cache/stanza")
OUTPUT_DIR = Path("data/corpus")
GREEK_MODEL_PACKAGE = "perseus"
GREEK_PROCESSORS = "tokenize,pos,lemma"

# These inventories are intentionally topic-specific. The tokens remain in
# text_raw/text_clean and in tokens.parquet, so other analyses can retain them.
GREEK_FUNCTION_LEMMAS = frozenset(
    {
        "αν",
        "απο",
        "αρα",
        "αυ",
        "αυτοσ",
        "γαρ",
        "γε",
        "γουν",
        "δε",
        "δη",
        "δια",
        "εαν",
        "εγω",
        "ει",
        "ειμι",
        "εισ",
        "εκ",
        "εν",
        "επι",
        "η",
        "και",
        "κατα",
        "μεν",
        "μεντοι",
        "μετα",
        "μη",
        "ναι",
        "ο",
        "οτι",
        "ου",
        "ουν",
        "ουκουν",
        "ουτοσ",
        "παρα",
        "περι",
        "προσ",
        "συ",
        "συν",
        "τε",
        "τι",
        "τισ",
        "τοινυν",
        "υπερ",
        "υπο",
        "ω",
        "ωσ",
    }
)
GREEK_DIALOGUE_LEMMAS = frozenset(
    {
        "δοκεω",
        "εοικα",
        "ερω",
        "λεγω",
        "οιμαι",
        "παλιν",
        "φημι",
    }
)
EXCLUDED_UPOS = frozenset(
    {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "PUNCT", "SCONJ"}
)

TOKEN_COLUMNS = [
    "token_id",
    "utterance_id",
    "token_index",
    "surface",
    "normalized",
    "lookup_form",
    "lemma",
    "upos",
    "morphology",
    "topic_keep",
    "exclusion_reason",
    "analyzer",
    "analyzer_version",
    "model_package",
]


def fold_token(token: str) -> str:
    """Return a case-, accent-, and final-sigma-insensitive lookup form."""
    decomposed = unicodedata.normalize("NFD", token.casefold().replace("ς", "σ"))
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )


@dataclass(frozen=True)
class AnalyzedToken:
    surface: str
    lemma: str | None
    upos: str | None
    morphology: str | None = None


class GreekAnalyzer(Protocol):
    name: str
    analyzer_version: str
    model_package: str

    def analyze(self, text: str) -> Sequence[AnalyzedToken]: ...


class StanzaGreekAnalyzer:
    """Ancient Greek Perseus tokenizer, POS tagger, and lemmatizer."""

    name = "stanza"
    model_package = GREEK_MODEL_PACKAGE

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        try:
            import stanza
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Ancient Greek enrichment requires the 'nlp' extra; "
                "run `uv run --extra nlp scraping-models` first."
            ) from error
        self.analyzer_version = version("stanza")
        self.stanza = stanza
        try:
            self.pipeline = stanza.Pipeline(
                lang="grc",
                package=self.model_package,
                processors=GREEK_PROCESSORS,
                model_dir=str(model_dir),
                download_method=None,
                use_gpu=False,
                verbose=False,
            )
        except (
            Exception
        ) as error:  # Stanza uses several model-specific exception types.
            raise RuntimeError(
                f"Could not load the Ancient Greek {self.model_package!r} model from {model_dir}. "
                "Run `uv run --extra nlp scraping-models`."
            ) from error

    def analyze(self, text: str) -> list[AnalyzedToken]:
        return self.analyze_many([text])[0]

    def analyze_many(
        self, texts: Sequence[str], *, batch_size: int = 128
    ) -> list[list[AnalyzedToken]]:
        """Analyze documents in batches while retaining utterance boundaries."""
        results: list[list[AnalyzedToken]] = []
        for start in range(0, len(texts), batch_size):
            documents = [
                self.stanza.Document([], text=text)
                for text in texts[start : start + batch_size]
            ]
            for document in self.pipeline(documents):
                results.append(self._document_tokens(document))
        return results

    @staticmethod
    def _document_tokens(document: object) -> list[AnalyzedToken]:
        return [
            AnalyzedToken(
                surface=word.text,
                lemma=word.lemma,
                upos=word.upos,
                morphology=word.feats,
            )
            for sentence in document.sentences  # type: ignore[attr-defined]
            for word in sentence.words
        ]


def exclusion_reason(token: AnalyzedToken) -> str | None:
    """Explain why a Greek token should not enter semantic topic modeling."""
    has_lemma = bool(token.lemma and token.lemma != "_")
    lemma = fold_token(token.lemma if has_lemma else token.surface)
    if token.upos == "PUNCT" or not any(
        character.isalpha() for character in token.surface
    ):
        return "punctuation"
    if token.upos == "PROPN":
        return "proper_name"
    if lemma in GREEK_DIALOGUE_LEMMAS:
        return "dialogue_formula"
    if lemma in GREEK_FUNCTION_LEMMAS or token.upos in EXCLUDED_UPOS:
        return "function_word"
    if not has_lemma:
        return "missing_lemma"
    return None


def enrich_utterances(
    utterances: pd.DataFrame,
    analyzer: GreekAnalyzer,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Add text_topic and return a token audit table for Ancient Greek rows."""
    required = {"utterance_id", "language", "text_clean"}
    if missing := required - set(utterances.columns):
        raise ValueError(
            f"Utterances are missing columns: {', '.join(sorted(missing))}"
        )

    token_rows: list[dict[str, object]] = []
    topic_text: dict[str, str] = {}
    greek_rows = utterances[utterances["language"] == "grc"]
    row_values = list(greek_rows.itertuples(index=False))
    texts = [str(row.text_clean) for row in row_values]
    analyze_many = getattr(analyzer, "analyze_many", None)
    analyses = (
        analyze_many(texts)
        if callable(analyze_many)
        else [analyzer.analyze(text) for text in texts]
    )
    for row, analyzed_tokens in zip(row_values, analyses, strict=True):
        utterance_id = str(row.utterance_id)
        kept: list[str] = []
        for token_index, token in enumerate(analyzed_tokens, start=1):
            reason = exclusion_reason(token)
            lemma = (
                unicodedata.normalize("NFC", token.lemma.casefold())
                if token.lemma and token.lemma != "_"
                else None
            )
            if reason is None:
                assert lemma is not None
                kept.append(lemma)
            token_rows.append(
                {
                    "token_id": f"{utterance_id}:t{token_index:06d}",
                    "utterance_id": utterance_id,
                    "token_index": token_index,
                    "surface": token.surface,
                    "normalized": unicodedata.normalize(
                        "NFC", token.surface.casefold()
                    ),
                    "lookup_form": fold_token(token.surface),
                    "lemma": lemma,
                    "upos": token.upos,
                    "morphology": token.morphology,
                    "topic_keep": reason is None,
                    "exclusion_reason": reason,
                    "analyzer": analyzer.name,
                    "analyzer_version": analyzer.analyzer_version,
                    "model_package": analyzer.model_package,
                }
            )
        topic_text[utterance_id] = " ".join(kept)

    enriched = utterances.copy()
    enriched["text_topic"] = [
        topic_text.get(str(row.utterance_id), str(row.text_clean))
        for row in enriched.itertuples(index=False)
    ]
    tokens = pd.DataFrame(token_rows, columns=TOKEN_COLUMNS)
    kept_count = int(tokens["topic_keep"].sum()) if not tokens.empty else 0
    run = {
        "analyzer": analyzer.name,
        "analyzer_version": analyzer.analyzer_version,
        "model_package": analyzer.model_package,
        "languages_enriched": ["grc"],
        "greek_utterances": len(greek_rows),
        "token_count": len(tokens),
        "topic_token_count": kept_count,
        "topic_token_rate": kept_count / len(tokens) if len(tokens) else 0.0,
    }
    return enriched, tokens, run


def model_fingerprint(model_dir: Path = MODEL_DIR) -> str:
    """Hash model file paths and contents for reproducible run metadata."""
    digest = hashlib.sha256()
    root = model_dir / "grc"
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(path.relative_to(model_dir).as_posix().encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def download_models() -> None:
    """Download the explicitly selected Ancient Greek Stanza model."""
    try:
        import stanza
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Run this command with `uv run --extra nlp scraping-models`."
        ) from error
    stanza.download(
        "grc",
        package=GREEK_MODEL_PACKAGE,
        processors=GREEK_PROCESSORS,
        model_dir=str(MODEL_DIR),
    )


def enrich_corpus(output_dir: Path = OUTPUT_DIR) -> None:
    """Enrich an already-built corpus and write auditable derived artifacts."""
    utterance_path = output_dir / "utterances.parquet"
    if not utterance_path.is_file():
        raise FileNotFoundError(f"Build the corpus before enrichment: {utterance_path}")
    analyzer = StanzaGreekAnalyzer()
    utterances = pd.read_parquet(utterance_path)
    enriched, tokens, run = enrich_utterances(utterances, analyzer)
    run["model_sha256"] = model_fingerprint()
    enriched.to_parquet(utterance_path, index=False)
    tokens.to_parquet(output_dir / "tokens.parquet", index=False)
    (output_dir / "linguistic-run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n"
    )
    from scraping.parsing import qa_report

    table_names = (
        "utterances",
        "notes",
        "versions",
        "works",
        "speakers",
        "alignments",
        "merge_decisions",
        "tokens",
    )
    tables = {
        name: pd.read_parquet(output_dir / f"{name}.parquet")
        for name in table_names
        if (output_dir / f"{name}.parquet").is_file()
    }
    (output_dir / "qa-report.json").write_text(
        json.dumps(qa_report(tables), ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(
        f"Enriched {run['greek_utterances']:,} Greek utterances; "
        f"retained {run['topic_token_count']:,}/{run['token_count']:,} topic tokens"
    )


def download_main() -> None:
    download_models()


def enrich_main() -> None:
    enrich_corpus()
