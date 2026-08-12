import unicodedata
import xml.etree.ElementTree as ET

from scraping.parsing import canonical_speaker, element_text


def test_note_removed_but_tail_retained():
    node = ET.fromstring("<said>Hello <note>Cf. Plat. 1a</note> world.</said>")
    assert element_text(node, clean=False) == "Hello Cf. Plat. 1a world."
    assert element_text(node, clean=True) == "Hello world."


def test_spoken_quote_is_not_removed():
    node = ET.fromstring("<said>He said <q>know yourself</q>.</said>")
    assert element_text(node, clean=True) == "He said know yourself."


def test_heading_mention_is_not_speaker_mapping():
    assert canonical_speaker(None, "The Speech of Socrates") == (None, "unresolved", 0.0)


def test_socrates_aliases_and_younger_socrates_are_distinct():
    assert canonical_speaker("Σωκράτης", "ΣΩ.")[0] == "socrates"
    assert canonical_speaker("Younger Socrates", "Y. Soc.")[0] == "younger_socrates"


def test_polytonic_greek_is_nfc_and_preserved():
    text = element_text(ET.fromstring("<said>ἄνθρωπος ὑπʼ ἐμοῦ</said>"), clean=True)
    assert unicodedata.is_normalized("NFC", text)
    assert text == "ἄνθρωπος ὑπʼ ἐμοῦ"
