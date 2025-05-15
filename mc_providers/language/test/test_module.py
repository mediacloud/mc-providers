import unittest
import mc_providers.language as language


class ModuleTest(unittest.TestCase):

    def test_min_word_length(self):
        test_title = "Rahul is Awesome"
        terms = language.terms_without_stopwords('en', test_title)
        assert len(terms) == 2
        assert "is" not in terms

    def test_remove_stopwords_es(self):
        test_text = "La izquierda hispanoamericana despide a Pepe Mujica: “¡Te vamos a extrañar mucho, viejo querido!"
        terms = language.terms_without_stopwords('es', test_text)
        assert len(terms) == 8
        assert "te" not in terms

    def test_remove_stopwords_list_en(self):
        texts = [
            "Trump receives lavish welcome in Qatar after meeting Syrian leader Bashar",
            "Israeli bombing wave kills dozens in Gaza including at least 22 children, say reports",
            "The good news from Kyiv: with or without a ceasefire, Ukraine has a newfound confidence"
        ]
        results = language.terms_without_stopwords_list('en', texts)
        assert len(results) == 3
        for term_list in results:
            assert len(term_list) > 0
            assert all(len(term) >= 2 for term in term_list)
            assert "with" not in term_list

