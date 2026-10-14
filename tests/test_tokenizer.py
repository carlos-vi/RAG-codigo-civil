"""Tests del tokenizador de BM25 y de la deteccion de articulos citados
(src/retrieval.py). No necesitan red ni API key.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import cited_articles, tokenize  # noqa: E402


class TokenizeTest(unittest.TestCase):
    def test_numero_de_articulo_sin_puntuacion(self):
        # El encabezado del fragmento es "Artículo 808." y la pregunta "808".
        self.assertEqual(tokenize("Artículo 808."), ["articulo", "808"])
        self.assertEqual(tokenize("art. 808"), ["art", "808"])

    def test_minusculas_sin_tildes_ni_signos(self):
        self.assertEqual(tokenize("¿Qué es la LEGÍTIMA?"), ["que", "es", "la", "legitima"])

    def test_dieresis(self):
        self.assertEqual(tokenize("cigüeña"), ["cigueña"])

    def test_conserva_la_enie(self):
        self.assertEqual(tokenize("año"), ["año"])
        self.assertNotEqual(tokenize("año"), tokenize("ano"))

    def test_sufijos_bis_ter(self):
        self.assertEqual(tokenize("Artículo 94 bis."), ["articulo", "94", "bis"])


class CitedArticlesTest(unittest.TestCase):
    def test_formas_de_citar(self):
        # La pregunta real que fallaba: "del codigo civil" lo sacaba del top 6.
        self.assertEqual(cited_articles("¿Qué dice el Articulo 1 del codigo civil??"), ["1"])
        self.assertEqual(cited_articles("¿Qué dice el artículo 1902?"), ["1902"])
        self.assertEqual(cited_articles("art. 808"), ["808"])
        self.assertEqual(cited_articles("art 7"), ["7"])
        self.assertEqual(cited_articles("art.1255"), ["1255"])
        self.assertEqual(cited_articles("ARTÍCULO 3"), ["3"])

    def test_apartados_y_sufijos(self):
        self.assertEqual(cited_articles("según el artículo 1964.2"), ["1964"])
        self.assertEqual(cited_articles("¿Qué establece el artículo 94 bis?"), ["94 bis"])

    def test_varios_articulos(self):
        self.assertEqual(cited_articles("compara los artículos 1, 2 y 3"), ["1", "2", "3"])
        self.assertEqual(cited_articles("art. 29 y art. 30, y otra vez el art. 29"), ["29", "30"])

    def test_sin_cita(self):
        self.assertEqual(cited_articles("¿Qué es la legítima?"), [])
        # "parte 1" o "arte" no son citas de articulo.
        self.assertEqual(cited_articles("la parte 1 del contrato"), [])
        self.assertEqual(cited_articles("obras de arte 2"), [])


if __name__ == "__main__":
    unittest.main()
