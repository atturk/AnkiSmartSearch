"""Unit tests for text_utils.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ankihub_searcher.text_utils import (
    clean_html_and_cloze,
    clean_tags,
    parse_markdown_units,
    LectureUnit,
)


class TestTextUtils(unittest.TestCase):
    def test_clean_html_and_cloze(self):
        # Test basic html removal and linebreaks
        raw = "<div>Hello<br>World</div><p>Paragraph</p>"
        cleaned = clean_html_and_cloze(raw)
        self.assertIn("Hello\nWorld", cleaned)
        self.assertIn("Paragraph", cleaned)

        # Test cloze unwrapping with hint
        cloze_raw = "{{c1::Factor XII::Hageman factor}} is a {{c2::coagulation factor}}."
        cleaned = clean_html_and_cloze(cloze_raw)
        self.assertEqual(cleaned, "Factor XII (Hageman factor) is a coagulation factor.")

        # Test cloze unwrapping without hint
        cloze_simple = "The activation of {{c1::NLRP3}} inflammasome."
        self.assertEqual(clean_html_and_cloze(cloze_simple), "The activation of NLRP3 inflammasome.")

        # Test sound, img, script, and style removal
        complex_html = (
            "<style>.hide{display:none;}</style>"
            "Text with <img src='test.jpg'> and [sound:rec.mp3]"
            "<script>alert(1);</script>"
        )
        self.assertEqual(clean_html_and_cloze(complex_html), "Text with and")

        # Test promotional bootcamp/physeo/sketchy links removal
        promo_html = (
            "<div>Clinical presentation of gout.</div>"
            "<a href='https://bootcamp.com'>Watch associated Bootcamp video on Gout</a>"
            "<div>More notes.</div>"
        )
        cleaned = clean_html_and_cloze(promo_html)
        self.assertNotIn("Bootcamp", cleaned)
        self.assertIn("Clinical presentation of gout.", cleaned)
        self.assertIn("More notes.", cleaned)

        # Test empty and None
        self.assertEqual(clean_html_and_cloze(""), "")
        self.assertEqual(clean_html_and_cloze(None), "")

    def test_clean_tags(self):
        tags = [
            "#AK_Step1_v12::#FirstAid::04_Pathology::02_Inflammation::04_Acute_inflammation::*Hageman_Factor",
            "#AK_Other::AnKing_Image::Marcos",
            "#AK_Step1_v12::#AMBOSS::9OaNuk",
            "Mastering_Radiology::01_Chest::Pneumothorax",
        ]
        cleaned = clean_tags(tags)
        # Should contain meaningful medical terms without AK boilerplate, numbers, or contributor noise
        self.assertIn("Pathology", cleaned)
        self.assertIn("Inflammation", cleaned)
        self.assertIn("Acute inflammation", cleaned)
        self.assertIn("Hageman Factor", cleaned)
        self.assertIn("Chest", cleaned)
        self.assertIn("Pneumothorax", cleaned)

        # Should NOT contain noise
        self.assertNotIn("AK_Step1_v12", cleaned)
        self.assertNotIn("AnKing_Image", cleaned)
        self.assertNotIn("Marcos", cleaned)
        self.assertNotIn("9OaNuk", cleaned)
        self.assertNotIn("04_", cleaned)

    def test_parse_markdown_units_single(self):
        text = "This is a single unit text with no markdown headers."
        units = parse_markdown_units(text, base_deck_name="Patologia")
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].deck_name, "Patologia")
        self.assertEqual(units[0].unit_title, "Patologia")
        self.assertEqual(units[0].body_text, text)
        self.assertEqual(units[0].breadcrumb, "")

    def test_parse_markdown_units_with_empty_headers(self):
        markdown_text = """
# Capitolo 2 - Infiammazione
---
## 2.1 Segnali Esogeni
I patogeni rilasciano PAMPs come LPS.

## 2.2 Segnali Endogeni ed Inflammasoma
L'inflammasoma NLRP3 risponde ad acido urico e cristalli di pirofosfato di calcio.

# Capitolo 3 - Riparazione Tissutale
### 3.1 Citochine
TGF-beta e PDGF stimolano la sintesi di collagene da parte dei fibroblasti.
"""
        # Testing default dash naming style
        units = parse_markdown_units(markdown_text, base_deck_name="Patologia Generale", naming_style="dash")
        # Capitolo 2 has no body text before 2.1, Capitolo 3 has no body text before 3.1
        # So only 2.1, 2.2, and 3.1 should be emitted as units!
        self.assertEqual(len(units), 3)

        u1, u2, u3 = units
        self.assertEqual(u1.unit_title, "2.1 Segnali Esogeni")
        self.assertEqual(u1.deck_name, "Patologia Generale - 2.1 Segnali Esogeni")
        self.assertEqual(u1.breadcrumb, "Capitolo 2 - Infiammazione > 2.1 Segnali Esogeni")
        self.assertIn("PAMPs", u1.body_text)

        self.assertEqual(u2.unit_title, "2.2 Segnali Endogeni ed Inflammasoma")
        self.assertEqual(u2.deck_name, "Patologia Generale - 2.2 Segnali Endogeni ed Inflammasoma")
        self.assertEqual(u2.breadcrumb, "Capitolo 2 - Infiammazione > 2.2 Segnali Endogeni ed Inflammasoma")
        self.assertIn("NLRP3", u2.body_text)

        self.assertEqual(u3.unit_title, "3.1 Citochine")
        self.assertEqual(u3.deck_name, "Patologia Generale - 3.1 Citochine")
        self.assertEqual(u3.breadcrumb, "Capitolo 3 - Riparazione Tissutale > 3.1 Citochine")
        self.assertIn("TGF-beta", u3.body_text)

    def test_parse_markdown_units_subdeck_naming(self):
        markdown_text = """
## 1.1 First Section
Content here.
"""
        units = parse_markdown_units(markdown_text, base_deck_name="Biochimica", naming_style="subdeck")
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].deck_name, "Biochimica::1.1 First Section")

    def test_parse_markdown_units_empty_input(self):
        self.assertEqual(parse_markdown_units(""), [])
        self.assertEqual(parse_markdown_units("   \n\n  "), [])
        # Only headers with no body
        self.assertEqual(parse_markdown_units("# Header 1\n## Header 2\n---"), [])


if __name__ == "__main__":
    unittest.main()
