# Demo video script (target runtime ≈ 4 min)

A shot-by-shot narration plan for screen-recording the walkthrough. Timings are
suggestions; total is under the 5-minute limit.

| # | Time | On screen | Narration |
|---|------|-----------|-----------|
| 1 | 0:00–0:25 | `README.md` in the browser / editor | "This is `roof-solar-optimized`. It reconstructs a house roof in 3D from a DSM and ortho imagery, then places flush-mounted solar panels. The repo is fully reproducible — one command regenerates every figure." |
| 2 | 0:25–0:50 | `data/` folder, then `outputs/01_overview.png` | "The inputs are a 10 cm DSM and a co-registered RGB ortho of a rural site. The overview shows the DSM, height-above-ground, hillshade, slope and aspect derived from it." |
| 3 | 0:50–1:15 | Terminal: `bash scripts/run.sh` | "Running the pipeline: it detects the target building, segments the roof into planar facets, finds obstructions, reconstructs the roof mesh and packs panels." |
| 4 | 1:15–1:45 | `outputs/02_roof_facets.png` | "Sequential RANSAC breaks the hipped roof into slope facets; azimuths are snapped to the building axes. The fitted planes have centimetre-scale RMS residuals — that's the reconstruction quality measure." |
| 5 | 1:45–2:15 | `outputs/03_obstructions.png` | "Obstructions come from two cues: raised blobs in the residual between the DSM and the fitted planes become chimneys and dormers; bright, low-saturation pixels become skylights and glass roofs." |
| 6 | 2:15–2:40 | `outputs/04_3d_roof.png`, then `outputs/roof_and_panels.html` | "The roof surface is exported as an OBJ mesh and an interactive 3D view. Note the dormers and the central roof-light void." |
| 7 | 2:40–3:20 | `outputs/05_panel_layout_2d.png` | "Panels are standard 1.72 × 1.13 m, 430 W modules, flush-mounted on each facet. Placement happens in true in-plane metres, so a panel can never straddle a ridge, cross an edge, or overlap an obstruction. Both module orientations are tried and the denser one wins." |
| 8 | 3:20–3:45 | `outputs/metrics.json` | "Result: 18 panels, 7.74 kWp, across three usable roof faces. Panels on the north-east face get the highest annual irradiance factor on this site." |
| 9 | 3:45–4:10 | `tests/` then `pytest` output | "The geometry and packing logic are unit-tested: slope/aspect against a synthetic plane, nodata inpainting, and a synthetic panel-packing case." |
| 10 | 4:10–4:25 | README "Assumptions" / "Limitations" sections | "The README documents assumptions — flush-mounting, no inter-row shading, standard modules — and limitations — wall geometry, the south-east hip, and occlusion handling." |

## Recording tips

* Terminal font ≥ 16 pt; hide the shell prompt clutter by running one command per shot.
* Pre-generate `outputs/` so figures are instant on screen.
* If time is short, cut shots 2 and 9; the core story is shots 4–8.
