# DPF Literature Search Strategy (2026)

## Search Infrastructure

| Source | Type | Auth | Rate Limit | Strength |
|--------|------|------|-----------|----------|
| **Semantic Scholar** | API | None (lower rate) | 1 req/sec | Best relevance for physics papers, citation graphs |
| **OpenAlex** | API | None | 100K/day | Largest corpus (250M+), good for broad sweeps |
| **arXiv** | API + Python | None | Reasonable | Preprints, fastest access to new work |
| **CrossRef** | API | None | Reasonable | DOI metadata, 130M+ works |
| **cortana-dpf-ref** | Local SQLite | N/A | N/A | 22 papers, 725 params, 214 formulas already extracted |

## Search Executed (2026-03-16)

**90 unique DPF/pinch papers found (2023-2026)** from combined Semantic Scholar + OpenAlex search.

Saved to: `docs/dpf_papers_2023_2026_combined.json`

## High-Priority Papers for DPF-Unified

### Tier 1 — Directly Validates/Challenges Our Implementation

| Paper | Year | Cites | Why It Matters |
|-------|------|-------|----------------|
| **Goyon et al.** "Neutron generation dynamics inside a MA-class dense plasma focus Z-pinch" | 2025 | 2 | MJOLNIR data — validates our Goyon instability timing formula. New neutron dynamics data. |
| **Auluck** "On the failure of neutron yield scaling in the dense plasma focus" | 2023 | 4 | Explains WHY alpha≠4 in our scaling validation (we got 2.96). Critical for Challenge 13. |
| **Auluck** "Generalized plasma focus problem and its application to space propulsion" | 2023 | 6 | Generalizes the Lee model EOM — may improve our snowplow physics. |
| **Wahbe et al.** "Numerical experiments on total D-D fusion neutron yield vs deuterium pressure" | 2023 | 3 | Lee model code experiments on Yn vs pressure — direct comparison for our validation. |
| **Schmidt et al.** "MJOLNIR Dense Plasma Focus Rebuild and High Current" | 2024 | 4 | MJOLNIR rebuild data — updates our MJOLNIR preset parameters. |
| **Damideh et al.** "Experimental results of FAETON-I 100kV dense plasma focus" | 2025 | 0 | FAETON-I data at 100kV/1MA — validates our FAETON preset. Already partially extracted. |

### Tier 2 — Informs Physics Modules We've Built

| Paper | Year | Cites | Relevant Module |
|-------|------|-------|----------------|
| **Lerner et al.** "Focus Fusion: Progress Towards p-B11 with Dense Plasma Focus" | 2023 | 15 | p-B11 yield module (pb11_yield.py). Highest-cited DPF paper 2023. |
| **Kubeš et al.** "Observation of filaments in MA dense plasma focus" | 2023 | 5 | Challenge 8 (filamentation), validates our metal_3d backend need. |
| **Kubeš et al.** "Evolution of filament-like structures in 3kJ plasma focus" | 2024 | 2 | Same — filament observation at smaller scale. |
| **Novotný et al.** "Effect of anode shape on neutron and x-ray emission" | 2023 | 4 | Electrode geometry effects — informs our preset geometry choices. |
| **Upadhyay et al.** "First survey of poloidal magnetic flux emission from DPF" | 2023 | 2 | Validates our plasmoid detection diagnostic (Challenge 14). |
| **Park et al.** "SPH method for pinch plasma with non-ideal MHD" | 2023 | 3 | Alternative MHD method — comparison target for our Metal solver. |

### Tier 3 — Background/Context

| Paper | Year | Cites | Context |
|-------|------|-------|---------|
| **Thompson et al.** "Electrode durability and sheared-flow-stabilized Z-pinch fusion" | 2023 | 6 | Z-pinch stability — relevant to Challenge 2 (instabilities). |
| **Auluck** "Poloidal magnetic field in the dense plasma focus" | 2024 | 1 | B-field structure — validates our electrode BC implementation. |
| **Auluck** "Symmetry and structure in the Generalized Plasma Focus problem" | 2024 | 1 | Mathematical structure of DPF EOM. |
| **Hosseinzadeh et al.** "Pre-ionization effect on pinch quality and neutron yield" | 2024 | 2 | Challenge 9 (reproducibility) — pre-ionization affects breakdown. |
| **Ahmed et al.** "Compact Plasma Focus with Tapered Anode" | 2023 | 2 | Electrode design variants. |
| **Barati** "Effects of electrode geometry on Ar soft X-ray from PF" | 2023 | 2 | High-Z radiation — relevant to our line_radiation.py module. |

## Papers to Extract Next

Priority order for adding to cortana-dpf-ref database:

1. **Goyon et al. 2025** — MJOLNIR neutron dynamics (already partially in our DB, need full extraction)
2. **Auluck 2023** — Neutron yield scaling failure (explains our alpha=2.96 result)
3. **Lerner et al. 2023** — p-B11 progress (validates/updates pb11_yield.py)
4. **Kubeš et al. 2023** — MA filament observation (validates 3D MHD approach)
5. **Wahbe et al. 2023** — Lee model Yn vs pressure (direct validation comparison)
6. **Damideh et al. 2025** — FAETON-I 100kV (update FAETON preset)

## Automated Literature Monitoring

### Recommended Workflow
```
# Weekly: search for new DPF papers
python3 -c "
import arxiv
search = arxiv.Search(query='dense plasma focus', max_results=10,
                       sort_by=arxiv.SortCriterion.SubmittedDate)
for r in arxiv.Client().results(search):
    print(f'[{r.published.date()}] {r.title}')
    print(f'  {r.entry_id}')
"

# Monthly: OpenAlex sweep for journal papers
# (see docs/dpf_papers_2023_2026_combined.json for format)
```

### Alert Keywords
Set up monitoring for these terms in new publications:
- "dense plasma focus"
- "Lee model" + "neutron yield"
- "z-pinch" + "simulation"
- "plasma focus" + "MHD"
- "Bennett" + "pinch" + "equilibrium"
- "Mather" OR "Filipov" + "plasma"

### Key Journals to Watch
- Physics of Plasmas (AIP) — most DPF papers land here
- IEEE Transactions on Plasma Science — engineering-focused DPF work
- Journal of Fusion Energy — Lee model papers, yield studies
- Plasma Physics and Controlled Fusion (IOP) — numerical experiments
- Nuclear Fusion (IAEA) — major fusion results
- Scientific Reports (Nature) — open access experimental results
