# Source provenance and credits

## Research role and authorship

Jai Ashar worked on computer-vision models and fruit-counting research at Penn’s GRASP Lab. This portfolio release presents a curated portion of the team’s KumarAg research code. It does not claim that Jai individually wrote every included module or that the original repository’s commit history establishes his authorship of them.

The original source headers credit **Chao Qu** for the five extracted tracking modules and **Chao Qu and Xu Liu** for `fruit_tracker.py`. The segmentation subtree carries the upstream **2016 Sarath Shekkizhar** MIT license. Those headers and license notices are preserved. The broader research and its upstream implementations remain collaborative work.

## Exact snapshot

Source repository: `Trefo/KumarAg` (private original).

Source commit: `a7f80b240dcf5bc2a8ef2f7ba667bf98cf52cda6`.

| Original path | Public destination | Treatment |
| --- | --- | --- |
| `FCN_Segmentation/*.py` | `historical/segmentation/` | All 11 Python files copied unchanged. |
| `FCN_Segmentation/LICENSE` | `historical/segmentation/LICENSE` | MIT license copied unchanged. |
| `tracking_localization/previous_version/tracking/scpye/track/kalman_filter.py` | `src/scpye/track/kalman_filter.py` | Copied unchanged. |
| Same directory: `bounding_box.py` | `src/scpye/track/bounding_box.py` | Copied unchanged. |
| Same directory: `fruit_track.py` | `src/scpye/track/fruit_track.py` | Copied unchanged. |
| Same directory: `assignment.py` | `src/scpye/track/assignment.py` | Compatibility and input-validation changes described below. |
| Same directory: `optical_flow.py` | `src/scpye/track/optical_flow.py` | Shape and empty-result changes described below. |
| Same directory: `fruit_tracker.py` | `historical/tracking/fruit_tracker.py` | Copied unchanged for inspection. |
| `LICENSE` | `LICENSE` | Root Apache 2.0 license copied unchanged. |

The selected tracking variant is explicitly the earlier `previous_version/tracking` implementation. Other variants in the original research tree include localization and trunk-tracking adaptations; those are not presented as part of this runnable extract.

[provenance-manifest.json](provenance-manifest.json) records the source path, destination, SHA-256 of each original file, SHA-256 of the released file, and whether it changed. Private source history is not imported. These hashes document the extraction; they are not an independent authorship certification.

## October 2026 release work

The following work was newly prepared for Jai’s public portfolio with Codex assistance. It is separate from the historical research:

- `assignment.py`: replace the removed private scikit-learn solver with SciPy’s supported solver; use infinite padding; validate matrix/cost inputs; support empty rectangular matrices; correct the explanation of two-sided unassignment cost.
- `optical_flow.py`: preserve consistent shapes for one point, handle empty input and failed OpenCV results, and return a finite zero displacement when there are no valid points.
- `examples/synthetic_demo.py`: a new minimal counting wrapper and generated illustration. It uses the extracted assignment, box-cost, and track/filter code. It does not port the full historical tracker, run the FCN, or claim to recover field results.
- `tests/`, package metadata, GitHub Actions, documentation, and generated demo assets: new release infrastructure.

Modified source files are marked at their start. Original author headers remain intact. Historical model code and the full tracker are deliberately kept outside the installed `src` package and test scope.

## Excluded material

No orchard images, training labels, checkpoint files, SQLite databases, pickles, deployment settings, credentials, or private Git history are included. The synthetic animation contains only generated shapes. Historical paths in the preserved model code are legacy configuration examples, not bundled datasets or working local setup instructions.

## Licensing

The original root Apache 2.0 license is retained for the tracking code. The FCN subtree retains its original MIT license and copyright notice. New demo, tests, and release tooling are distributed under Apache 2.0. This is a portfolio source release, not an official GRASP Lab distribution or institutional endorsement.
