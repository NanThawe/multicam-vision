"""Synthetic regression suite. Does not require Market-1501, BoxMOT, weights or internet."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import encoder_benchmark as b


def row(pid, camera, name=None, folder="bounding_box_test"):
    name = name or f"{pid:04d}_c{camera}s1_000001_00.jpg"
    return {"pid":pid,"cam":camera,"rel":f"{folder}/{name}"}


class BenchmarkTests(unittest.TestCase):
    def test_parse_standard_name(self):
        self.assertEqual(b.parse_name("0001_c2s1_000001_00.jpg"),(1,2))

    def test_parse_junk(self):
        self.assertEqual(b.parse_name("-1_c6s1_000001_00.jpg"),(-1,6))
        self.assertEqual(b.parse_name("0000_c1s1_000001_00.jpg"),(0,1))

    def test_parse_invalid_camera(self):
        with self.assertRaises(ValueError):b.parse_name("0001_c7s1_000001_00.jpg")

    def test_parse_bad_filename(self):
        with self.assertRaises(ValueError):b.parse_name("0001c1.jpg")

    def test_hash_deterministic(self):
        self.assertEqual(b.stable_hash("a"),b.stable_hash("a"))
        self.assertNotEqual(b.stable_hash("a",1),b.stable_hash("a",2))

    def test_index_excludes_gallery_junk_but_keeps_zero_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder in ("query","bounding_box_test"):(root/folder).mkdir()
            for name in ("0001_c1s1_000001_00.jpg", "-1_c3s1_000001_00.jpg"):
                (root/"query"/name).write_bytes(b"x")
            for name in ("0001_c2s1_000001_00.jpg","-1_c4s1_000001_00.jpg","0000_c5s1_000001_00.jpg"):
                (root/"bounding_box_test"/name).write_bytes(b"x")
            pool=b.list_market(root)
            self.assertEqual(len(pool["query"]),1)
            self.assertEqual(sorted(g["pid"] for g in pool["gallery"]),[0,1])

    def test_prepared_pilot_repeatability(self):
        pool={"query":[row(1,1,folder="query"),row(2,3,folder="query"),row(3,2,folder="query")],
              "gallery":[row(1,2),row(2,1),row(3,4),row(4,5),row(0,6)]}
        a=b.select_samples(pool,"pilot",2,1,seed=14)
        self.assertEqual(a,b.select_samples(pool,"pilot",2,1,seed=14))
        self.assertEqual(len(a[0]),2)

    def test_dropped_unmatchable_query(self):
        pool={"query":[row(1,1,folder="query"),row(2,1,folder="query")],
              "gallery":[row(1,2),row(2,1)]}
        q,g=b.select_samples(pool,"pilot",2,1)
        self.assertEqual([r["pid"] for r in q],[1])

    def test_full_selection_retains_all_eligible(self):
        pool={"query":[row(1,1,folder="query"),row(2,1,folder="query")],
              "gallery":[row(1,2),row(2,1),row(0,1)]}
        q,g=b.select_samples(pool,"full")
        self.assertEqual(len(q),1)
        self.assertEqual(len(g),3)

    def test_empty_selection_rejected(self):
        with self.assertRaises(ValueError):b.select_samples({"query":[row(1,1)],"gallery":[row(1,1)]})

    def test_bad_sampling(self):
        with self.assertRaises(ValueError):b.select_samples({"query":[],"gallery":[]},queries=0)

    def test_rank_map_exact(self):
        queries=[row(1,1,folder="query")]
        gallery=[row(2,2),row(1,2),row(3,3)]
        q=np.array([[1.,0.]])
        g=np.array([[1.,0.],[.8,.6],[0.,1.]])
        val=b.calculate_metrics(queries,gallery,q,g)
        self.assertEqual(val["rank1"],0)
        self.assertEqual(val["rank5"],1)
        self.assertEqual(val["mAP"],.5)

    def test_same_pid_same_camera_is_junk(self):
        q=[row(1,1,folder="query")]
        g=[row(1,1),row(1,2),row(2,3)]
        score=b.calculate_metrics(q,g,np.array([[1.,0.]]),np.array([[1.,0.],[0.,1.],[.5,.5]]))
        self.assertEqual(score["rank1"],0)
        self.assertEqual(score["mAP"],.5)

    def test_zero_pid_not_filtered_as_junk(self):
        q=[row(1,1,folder="query")]
        g=[row(0,2),row(1,2)]
        score=b.calculate_metrics(q,g,np.array([[1.,0.]]),np.array([[1.,0.],[0.,1.]]))
        self.assertEqual(score["rank1"],0)

    def test_negative_pid_filtered_as_junk(self):
        q=[row(1,1,folder="query")]
        g=[row(-1,2),row(1,2)]
        score=b.calculate_metrics(q,g,np.array([[1.,0.]]),np.array([[1.,0.],[0.,1.]]))
        self.assertEqual(score["rank1"],1)

    def test_queries_without_positive_not_in_denominator(self):
        q=[row(1,1,folder="query"),row(3,1,folder="query")]
        g=[row(1,2),row(3,1)]
        scores=b.calculate_metrics(q,g,np.array([[1.,0.],[0.,1.]]),np.array([[1.,0.],[0.,1.]]))
        self.assertEqual(scores["eligible_queries"],1)
        self.assertEqual(scores["skipped_no_cross_camera_positive"],1)

    def test_camera_filter(self):
        q=[row(1,1,folder="query")]
        g=[row(1,2),row(1,3)]
        qs=np.array([[1.,0.]])
        gs=np.array([[1.,0.],[0.,1.]])
        self.assertEqual(b.calculate_metrics(q,g,qs,gs,2)["eligible_queries"],1)
        self.assertEqual(b.calculate_metrics(q,g,qs,gs,1)["eligible_queries"],0)

    def test_invalid_features_rejected(self):
        with self.assertRaises(ValueError):
            b.calculate_metrics([row(1,1)],[row(1,2)],np.array([[float('nan'),0]]),np.array([[1.,0.]]))

    def test_by_camera_outputs(self):
        q=[row(1,1,folder="query")]
        g=[row(1,2)]
        results=b.metrics_by_camera(q,g,np.array([[1.,0.]]),np.array([[1.,0.]]))
        self.assertEqual(len(results),5)
        self.assertEqual(sum(r["eligible_queries"] for r in results),1)

    def test_save_cache_detects_invalid_norm(self):
        with tempfile.TemporaryDirectory() as tmp:
            sel={"query":[row(1,1)],"gallery":[row(1,2)],"selection_sha256":"abc"}
            with self.assertRaises(ValueError):
                b.save_cache(tmp,"osnet_x1_0",np.array([[2.,0.],[1.,0.]]),sel,"xx","yy",1,{})

    def test_corrupt_cache_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"query").mkdir(); (root/"bounding_box_test").mkdir()
            q=row(1,1,folder="query"); g=row(1,2)
            (root/q["rel"]).write_bytes(b"first"); (root/g["rel"]).write_bytes(b"second")
            weight=root/"model.pt"; weight.write_bytes(b"weights")
            selection={"query":[q],"gallery":[g],"selection_sha256":"a"}
            feats=np.array([[1.,0.],[0.,1.]],dtype=np.float32)
            b.save_cache(tmp,"osnet_x1_0",feats,selection,b.hash_file(weight),b._sha_inputs(root,[q,g]),1,{})
            a,_=b.load_cache(tmp,"osnet_x1_0",selection,root,weight)
            self.assertTrue(np.array_equal(a,feats))
            (root/g["rel"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):b.load_cache(tmp,"osnet_x1_0",selection,root,weight)

    def test_different_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"query").mkdir();(root/"bounding_box_test").mkdir()
            q=row(1,1,folder="query");g=row(1,2)
            (root/q["rel"]).write_bytes(b"a");(root/g["rel"]).write_bytes(b"b")
            weights=root/"weight.pt";weights.write_bytes(b"1")
            selection={"query":[q],"gallery":[g],"selection_sha256":"a"}
            b.save_cache(root,"osnet_x1_0",np.array([[1.,0.],[0.,1.]],dtype=np.float32),selection,b.hash_file(weights),b._sha_inputs(root,[q,g]),1,{})
            weights.write_bytes(b"2")
            with self.assertRaises(ValueError):b.load_cache(root,"osnet_x1_0",selection,root,weights)


if __name__=="__main__": unittest.main()
