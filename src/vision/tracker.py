"""
    Greedy nearest-neighbour centroid tracker.
 
    Accepts any list of bbox dicts that contain a "centroid" key — works
    identically whether detections come from MOG2 or from GT bboxes.
    
    Uses scipy.spatial.distance.cdist for the full pairwise distance matrix
    instead of a hand-rolled nested loop.

"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from scipy.spatial.distance import cdist

import config as cfg

class CentroidTracker:
 
    def __init__(self):
        self.next_id = 0
        self.tracks  = {}   # id -> track dict
 
    def new_track(self, centroid):
        self.tracks[self.next_id] = {
            "id":          self.next_id,
            "centroid":    centroid,
            "trajectory":  [centroid],
            "disappeared": 0,
        }
        self.next_id += 1
 
    def update_track(self, track, centroid):
        track["centroid"]    = centroid
        track["disappeared"] = 0
        track["trajectory"].append(centroid)
        if len(track["trajectory"]) > cfg.MAX_TRAJECTORY_LEN:
            track["trajectory"] = track["trajectory"][-cfg.MAX_TRAJECTORY_LEN:]
 
    def update(self, detections):
        """
        Match new detections to existing tracks via scipy cdist.
 
        Parameters
        ----------
        detections : list of bbox dicts, each with a "centroid" key.
                     Works with both MOG2 detections and GT bboxes.
 
        Returns
        -------
        list of active track dicts
        """
        centroids = [d["centroid"] for d in detections]
 
        if not centroids:
            for t in self.tracks.values():
                t["disappeared"] += 1
            for tid in [tid for tid, t in self.tracks.items()
                        if t["disappeared"] > cfg.MAX_DISAPPEARED]:
                del self.tracks[tid]
            return list(self.tracks.values())
 
        if not self.tracks:
            for c in centroids:
                self.new_track(c)
            return list(self.tracks.values())
 
        track_ids   = list(self.tracks.keys())
        old_cents   = np.array([self.tracks[tid]["centroid"] for tid in track_ids],
                               dtype=float)
        new_cents   = np.array(centroids, dtype=float)
        dist_matrix = cdist(old_cents, new_cents, metric="euclidean")
        rows, cols  = np.unravel_index(np.argsort(dist_matrix, axis=None),
                                       dist_matrix.shape)
 
        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            self.update_track(self.tracks[track_ids[r]], centroids[c])
            used_rows.add(r)
            used_cols.add(c)
 
        for j, c in enumerate(centroids):
            if j not in used_cols:
                self.new_track(c)
 
        expired = []
        for i, tid in enumerate(track_ids):
            if i not in used_rows:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > cfg.MAX_DISAPPEARED:
                    expired.append(tid)
        for tid in expired:
            del self.tracks[tid]
 
        return list(self.tracks.values())
 
    def get_active_tracks(self):
        return list(self.tracks.values())
 
    def reset(self):
        self.tracks.clear()
        self.next_id = 0