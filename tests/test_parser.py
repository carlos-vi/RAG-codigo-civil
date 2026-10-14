"""Tests del parser del BOE (src/ingest.py). No necesitan red ni API key.

Cada caso reproduce una particularidad del XML del BOE que ya provoco un bug
en el proyecto. Ejecutar desde la raiz:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import ingest  # noqa: E402

V = 'id_norma="BOE-A-X" fecha_publicacion="18890725" fecha_vigencia="{d}"'

# XML minimo con las particularidades reales del BOE.
SAMPLE = f"""<?xml version="1.0" encoding="utf-8"?>
<response><data><texto>
  <bloque id="l1" tipo="encabezado"><version {V.format(d='1')}>
    <p class="libro_num">LIBRO PRIMERO</p><p class="libro_tit">Texto antiguo</p></version>
  <version {V.format(d='2')}>
    <p class="libro_num">LIBRO PRIMERO</p><p class="libro_tit">De las personas</p></version></bloque>
  <bloque id="t1" tipo="encabezado"><version {V.format(d='1')}>
    <p class="titulo_num">TÍTULO I</p><p class="titulo_tit">Del nacimiento</p></version></bloque>
  <bloque id="c1" tipo="encabezado"><version {V.format(d='1')}>
    <p class="capitulo_num">CAPÍTULO I</p><p class="capitulo_tit">Disposiciones</p>
    <p class="nota_pie">Se anade por la Ley X.</p></version></bloque>

  <bloque id="a1" tipo="precepto"><version {V.format(d='1')}>
    <p class="articulo">Artículo 1.</p><p class="parrafo">Texto ANTIGUO del articulo uno que ya no rige.</p></version>
  <version {V.format(d='2')}>
    <p class="articulo">Artículo 1.</p><p class="parrafo">Texto VIGENTE del articulo uno, version actual.</p>
    <p class="nota_pie">Se modifica por la Ley Y.</p></version></bloque>

  <bloque id="a94bis" tipo="precepto"><version {V.format(d='1')}>
    <p class="articulo">Artículo 94 bis.</p>
    <p class="parrafo">La autoridad judicial confiara los animales de compania a uno de los conyuges.</p></version></bloque>

  <bloque id="a315" tipo="precepto"><version {V.format(d='1')}>
    <p class="articulo">Artículo 315.</p><p class="parrafo">(Derogado)</p></version></bloque>

  <bloque id="sprimera" tipo="precepto"><version {V.format(d='1')}>
    <p class="seccion">Sección primera. De las clases de matrimonio</p>
    <p class="nota_pie">Se modifica la rubrica.</p></version></bloque>
  <bloque id="a2" tipo="precepto"><version {V.format(d='1')}>
    <p class="articulo">Artículo 2.</p>
    <p class="parrafo">Articulo posterior a una seccion publicada como precepto sin numero.</p></version></bloque>

  <bloque id="dt" tipo="encabezado"><version {V.format(d='1')}>
    <p class="capitulo">DISPOSICIONES TRANSITORIAS</p></version></bloque>
  <bloque id="primera" tipo="precepto"><version {V.format(d='1')}>
    <p class="articulo">Primera.</p>
    <p class="parrafo">Se regiran por la legislacion anterior los derechos nacidos antes del Codigo.</p></version></bloque>
</texto></data></response>
"""


class ParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "sample.xml"
            path.write_text(SAMPLE, encoding="utf-8")
            cls.articles = ingest.parse_articles(path)
        cls.by_label = {a["etiqueta"]: a for a in cls.articles}

    def test_uses_last_version_of_each_article(self):
        text = self.by_label["Artículo 1"]["text"]
        self.assertIn("VIGENTE", text)
        self.assertNotIn("ANTIGUO", text)

    def test_footnotes_are_not_part_of_the_text(self):
        self.assertNotIn("Se modifica por la Ley Y", self.by_label["Artículo 1"]["text"])

    def test_derogated_articles_are_skipped(self):
        self.assertNotIn("Artículo 315", self.by_label)

    def test_suffixed_articles_keep_full_number(self):
        article = self.by_label["Artículo 94 bis"]
        self.assertEqual(article["articulo"], "94 bis")
        self.assertEqual(ingest.build_chunks([article])[0][0]["text"].splitlines()[0], "Artículo 94 bis.")

    def test_section_headings_published_as_articles_are_not_articles(self):
        labels = list(self.by_label)
        self.assertFalse([x for x in labels if "Sección" in x or x.strip() in ("", "Artículo")])
        self.assertEqual(self.by_label["Artículo 2"]["seccion"], "Sección primera. De las clases de matrimonio")

    def test_hierarchy_uses_only_current_heading_version(self):
        hierarchy = ingest._hierarchy(self.by_label["Artículo 1"])
        self.assertIn("LIBRO PRIMERO - De las personas", hierarchy)
        self.assertNotIn("Texto antiguo", hierarchy)
        self.assertEqual(hierarchy.count("LIBRO PRIMERO"), 1)

    def test_disposition_groups_label_their_entries(self):
        self.assertIn("Disposiciones transitorias Primera", self.by_label)
        # No debe heredar la jerarquia del articulo anterior.
        self.assertEqual(self.by_label["Disposiciones transitorias Primera"]["titulo"], "")

    def test_generic_disposiciones_section_does_not_reset_book(self):
        xml = SAMPLE.replace(
            '<bloque id="a2"',
            f'<bloque id="sg" tipo="encabezado"><version {V.format(d="1")}>'
            '<p class="seccion">Disposiciones generales</p></version></bloque><bloque id="a2"',
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "s.xml"
            path.write_text(xml, encoding="utf-8")
            articles = {a["etiqueta"]: a for a in ingest.parse_articles(path)}
        self.assertIn("LIBRO PRIMERO", articles["Artículo 2"]["libro"])

    def test_every_article_has_a_label(self):
        for a in self.articles:
            self.assertTrue(a["etiqueta"].strip())
            self.assertTrue(a["articulo"].strip())


@unittest.skipUnless(config.XML_PATH.exists(), "XML real no descargado (python src/ingest.py)")
class RealCodeTest(unittest.TestCase):
    """Comprobaciones sobre el texto real del Codigo Civil."""

    @classmethod
    def setUpClass(cls):
        cls.articles = ingest.parse_articles()
        cls.by_label = {a["etiqueta"]: a for a in cls.articles}

    def test_no_malformed_or_duplicated_labels(self):
        labels = [a["etiqueta"] for a in self.articles]
        self.assertEqual(len(labels), len(set(labels)), "etiquetas duplicadas")
        odd = [x for x in labels if not x.startswith(("Artículo ", "Disposiciones ", "Disposición "))]
        self.assertEqual(odd, [])

    def test_known_articles(self):
        self.assertIn("fuentes del ordenamiento", self.by_label["Artículo 1"]["text"])
        self.assertIn("dos terceras partes", self.by_label["Artículo 808"]["text"])
        self.assertIn("Artículo 94 bis", self.by_label)

    def test_known_derogated_article_is_absent(self):
        self.assertNotIn("Artículo 315", self.by_label)


if __name__ == "__main__":
    unittest.main()
