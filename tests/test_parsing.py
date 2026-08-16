import unicodedata
import xml.etree.ElementTree as ET

import pandas as pd

from scraping.linguistics import AnalyzedToken, enrich_utterances, fold_token
from scraping.parsing import canonical_speaker, element_text


def test_note_removed_but_tail_retained():
    node = ET.fromstring("<said>Hello <note>Cf. Plat. 1a</note> world.</said>")
    assert element_text(node, clean=False) == "Hello Cf. Plat. 1a world."
    assert element_text(node, clean=True) == "Hello world."


def test_spoken_quote_is_not_removed():
    node = ET.fromstring("<said>He said <q>know yourself</q>.</said>")
    assert element_text(node, clean=True) == "He said know yourself."


def test_heading_mention_is_not_speaker_mapping():
    assert canonical_speaker(None, "The Speech of Socrates") == (
        None,
        "unresolved",
        0.0,
    )


def test_socrates_aliases_and_younger_socrates_are_distinct():
    assert canonical_speaker("Σωκράτης", "ΣΩ.")[0] == "socrates"
    assert canonical_speaker("Younger Socrates", "Y. Soc.")[0] == "younger_socrates"


def test_polytonic_greek_is_nfc_and_preserved():
    text = element_text(ET.fromstring("<said>ἄνθρωπος ὑπʼ ἐμοῦ</said>"), clean=True)
    assert unicodedata.is_normalized("NFC", text)
    assert text == "ἄνθρωπος ὑπʼ ἐμοῦ"


class FakeGreekAnalyzer:
    name = "fake"
    analyzer_version = "1"
    model_package = "test"

    def analyze(self, text: str) -> list[AnalyzedToken]:
        return [
            AnalyzedToken("τοίνυν", "τοίνυν", "PART"),
            AnalyzedToken("εἶπον", "λέγω", "VERB", "Mood=Ind"),
            AnalyzedToken("σώματος", "σῶμα", "NOUN", "Case=Gen"),
        ]


def test_greek_enrichment_is_lemma_based_auditable_and_lossless():
    utterances = pd.DataFrame(
        {
            "utterance_id": ["grc:1", "eng:1"],
            "language": ["grc", "eng"],
            "text_clean": ["τοίνυν εἶπον σώματος", "body and soul"],
        }
    )

    enriched, tokens, report = enrich_utterances(utterances, FakeGreekAnalyzer())

    assert enriched["text_clean"].tolist() == ["τοίνυν εἶπον σώματος", "body and soul"]
    assert enriched["text_topic"].tolist() == ["σῶμα", "body and soul"]
    assert tokens["exclusion_reason"].tolist()[:2] == [
        "function_word",
        "dialogue_formula",
    ]
    assert pd.isna(tokens["exclusion_reason"].iloc[2])
    assert tokens["topic_keep"].tolist() == [False, False, True]
    assert report["topic_token_count"] == 1


def test_fold_token_matches_polytonic_stoplist_forms():
    assert fold_token("πάλιν") == "παλιν"
    assert fold_token("οὐκοῦν") == "ουκουν"
