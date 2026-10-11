"""New 2026 educational harness using historical KumarAg tracking primitives.

Detections are synthetic ground truth, not FCN predictions. No field data is used.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw
from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import bboxes_assignment_cost
from scpye.track.fruit_track import FruitTrack


class DemoCounter:
    """Minimal lifecycle demo; not a port of the full historical FruitTracker.

    A track retires on its first missed frame. Re-entry can therefore double count.
    This deliberately simple harness does not implement occlusion recovery.
    """
    def __init__(self, min_age=3):
        if min_age < 1:
            raise ValueError("min_age must be positive")
        self.min_age = min_age
        self.tracks = {}
        self.next_id = 1
        self.count = 0
        self.finished = False

    def _retire(self, ident):
        track = self.tracks.pop(ident)
        self.count += int(track.age >= self.min_age)

    def step(self, detections):
        if self.finished:
            raise RuntimeError("cannot step after finish")
        detections = np.asarray(detections, dtype=float).reshape(-1, 4)
        if not np.isfinite(detections).all() or (detections[:, 2:] <= 0).any():
            raise ValueError("detections require finite coordinates and positive sizes")
        ids = list(self.tracks)
        for track in self.tracks.values():
            track.predict()
        if ids and len(detections):
            costs = bboxes_assignment_cost(
                np.asarray([self.tracks[i].bbox for i in ids]), detections)
            matches, lost, new = hungarian_assignment(costs, unassigned_cost=0.75)
        else:
            matches = []
            lost, new = range(len(ids)), range(len(detections))
        for old, detection in matches:
            self.tracks[ids[old]].correct_bbox(detections[detection], [1., 1.])
        for old in lost:
            self._retire(ids[old])
        for detection in new:
            self.tracks[self.next_id] = FruitTrack(
                detections[detection], np.zeros(2), [4., 4., 4., 4.],
                [0.1, 0.1, 0.1, 0.1])
            self.next_id += 1
        return self.count

    def finish(self):
        for ident in list(self.tracks):
            self._retire(ident)
        self.finished = True
        return self.count


def sequence():
    for frame in range(18):
        boxes = []
        if frame < 13:
            boxes.append([55 + 7 * frame, 118, 32, 32])
        if 3 <= frame < 17:
            boxes.append([330 - 6 * frame, 212, 36, 36])
        if frame == 7:
            boxes.append([430, 132, 18, 18])
        if frame % 2:
            boxes.reverse()
        yield boxes


def render(destination):
    counter = DemoCounter()
    frames = []
    observations = 0
    for index, detections in enumerate(sequence()):
        observations += len(detections)
        counter.step(detections)
        image = Image.new("RGB", (640, 360), "#101c28")
        draw = ImageDraw.Draw(image)
        draw.text((24, 18), "GRASP / FRUIT TRACKING", fill="#d9f3f0", font_size=23)
        draw.text((24, 53), "Synthetic detections, not model predictions", fill="#abc2cc", font_size=16)
        for x, y, w, h in detections:
            draw.ellipse((x, y, x+w, y+h), fill="#dc8b39")
        for ident, track in counter.tracks.items():
            x, y, w, h = track.bbox
            mature = track.age >= counter.min_age
            color = "#7de1b9" if mature else "#ebca78"
            draw.rectangle((x-4, y-4, x+w+4, y+h+4), outline=color, width=2)
            draw.text((x-5, y-24), f"ID {ident} / age {track.age}", fill=color, font_size=14)
        draw.text((24, 300), f"Frame {index+1:02d}   Observations {observations:02d}   Counted {counter.count}", fill="white", font_size=19)
        draw.text((24, 331), "Count each mature track once at retirement. Reject one-frame noise.", fill="#abc2cc", font_size=14)
        frames.append(image)
    total = counter.finish()
    assert total == 2, total
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(destination, save_all=True, append_images=frames[1:],
                   duration=[250]*17+[1800], loop=0)
    frames[10].save(destination.with_suffix(".png"))
    print(f"{observations} frame observations -> {total} mature tracks; wrote {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/synthetic-tracking.gif")
    render(parser.parse_args().output)
