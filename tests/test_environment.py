#!/usr/bin/env python3
"""Property tests for the retrieval environment. Standard library only.

Run: PYTHONPATH=src python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeagent.evidence import evidence_view  # noqa: E402
from timeagent.search.local_backend import (BackendError, LocalCorpusBackend,  # noqa: E402
                                            build_index, normalize_corpus_record)

PILOT = ROOT / "events/wcep_pilot/corpus.jsonl"
QUERY = "flooding in Louisiana"


def build_pilot_index(directory):
    return build_index(PILOT, Path(directory) / "pilot.sqlite")


class IndexIntegrity(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="wtb-index-")
        self.manifest = build_pilot_index(self.dir)
        self.index = Path(self.dir) / "pilot.sqlite"

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_manifest_pins_corpus_and_index(self):
        self.assertEqual(self.manifest["document_count"], 199)
        for key in ("source_corpus_sha256", "index_sha256", "tokenizer_version"):
            self.assertIn(key, self.manifest)

    def test_digest_mismatch_refuses_to_load(self):
        sidecar = self.index.with_name(self.index.name + ".manifest.json")
        data = json.loads(sidecar.read_text())
        data["index_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            LocalCorpusBackend(self.index)

    def test_document_count_mismatch_refuses_to_load(self):
        with sqlite3.connect(self.index) as db:
            db.execute("DELETE FROM documents WHERE rowid=1")
        with self.assertRaises(ValueError):
            LocalCorpusBackend(self.index)

    def test_rebuild_over_existing_path_is_rejected(self):
        with self.assertRaises(ValueError):
            build_index(PILOT, self.index)


class CorpusHygiene(unittest.TestCase):
    def test_label_bearing_row_is_rejected(self):
        for field in ("gold", "golden_answers", "reward_model", "prompt"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    normalize_corpus_record({"corpus_id": "x", "content": "body text",
                                             field: "leak"})

    def test_empty_body_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_corpus_record({"corpus_id": "x", "content": "   "})

    def test_bad_publication_date_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_corpus_record({"corpus_id": "x", "content": "body",
                                     "publication_date": "2024-13-45"})


class RetrievalControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="wtb-controls-")
        build_pilot_index(cls.dir)
        cls.index = Path(cls.dir) / "pilot.sqlite"
        cls.bm25 = LocalCorpusBackend(cls.index)
        cls.random = LocalCorpusBackend(cls.index, retrieval="random")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_control_backend_has_distinct_identity(self):
        self.assertIn("fts5", self.bm25.backend_id)
        self.assertIn("random-control", self.random.backend_id)
        self.assertNotEqual(self.bm25.backend_id, self.random.backend_id)

    def test_random_control_is_deterministic_in_seed_and_query(self):
        first = [hit["corpus_id"] for hit in self.random.search(QUERY, 3)]
        again = [hit["corpus_id"] for hit in self.random.search(QUERY, 3)]
        self.assertEqual(first, again)
        other_seed = LocalCorpusBackend(self.index, retrieval="random", retrieval_seed=7)
        self.assertNotEqual(first, [h["corpus_id"] for h in other_seed.search(QUERY, 3)])

    def test_random_control_is_unrelated_to_bm25_ranking(self):
        lexical = {hit["corpus_id"] for hit in self.bm25.search(QUERY, 5)}
        control = {hit["corpus_id"] for hit in self.random.search(QUERY, 5)}
        self.assertLess(len(lexical & control), 5)

    def test_window_filter_excludes_later_evidence(self):
        cutoff = "2016-08-24"
        for backend in (self.bm25, self.random):
            with self.subTest(backend=backend.retrieval):
                for hit in backend.search(QUERY, 5, max_date=cutoff):
                    date = hit.get("publication_date")
                    if date is not None:
                        self.assertLessEqual(date, cutoff)

    def test_empty_query_fails_rather_than_returning_everything(self):
        for backend in (self.bm25, self.random):
            with self.subTest(backend=backend.retrieval):
                with self.assertRaises(ValueError):
                    backend.search("   ", 3)

    def test_topk_bounds_are_enforced(self):
        for bad in (0, 101, 3.0):
            with self.subTest(topk=bad):
                with self.assertRaises(ValueError):
                    self.bm25.search(QUERY, bad)


class EvidenceViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="wtb-views-")
        build_pilot_index(cls.dir)
        cls.backend = LocalCorpusBackend(Path(cls.dir) / "pilot.sqlite")
        cls.pool = cls.backend.search(QUERY, 3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_text_view_withholds_the_structured_date(self):
        for row in evidence_view(self.pool, view="text", query=QUERY):
            self.assertIsNone(row["publication_date_used"])
            self.assertTrue(row["document"]["contents"].startswith("withheld"))

    def test_dates_view_withholds_the_body(self):
        for row in evidence_view(self.pool, view="dates", query=QUERY):
            self.assertEqual(row["document"]["contents"].strip().count("\n"), 0)

    def test_shuffled_dates_is_a_permutation_of_the_pool_dates(self):
        full = sorted(str(r["publication_date_used"])
                      for r in evidence_view(self.pool, view="full", query=QUERY))
        shuffled = sorted(str(r["publication_date_used"])
                          for r in evidence_view(self.pool, view="shuffled-dates", query=QUERY))
        self.assertEqual(full, shuffled)

    def test_unknown_view_is_rejected(self):
        with self.assertRaises(ValueError):
            evidence_view(self.pool, view="no-such-view", query=QUERY)


if __name__ == "__main__":
    unittest.main()
