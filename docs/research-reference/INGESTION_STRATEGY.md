# Research Ingestion & Cross-Reference Strategy

## Database Schema

```
dpf_research.db (SQLite + FTS5)
├── papers (260 entries)          — metadata, DOI, journal, citations
├── formulas                      — extracted equations with variables
├── findings                      — claims, confidence, supporting/contradicting refs
├── experimental_data             — device measurements with uncertainties
├── cross_refs                    — paper-to-paper relationships and gaps
├── gaps                          — missing knowledge mapped to DPF challenges
├── implementation_map            — paper → DPF-Unified code mapping
├── tags                          — flexible categorization
└── papers_fts (FTS5)             — full-text search on title, authors, abstract
```

## Three-Phase Ingestion Pipeline

### Phase 1: Automated Extraction (AI Agent)
For each downloaded PDF:
1. Extract title, authors, abstract, references (if not already in DB)
2. Identify all equations with variable definitions
3. Extract experimental data points (device, parameter, value, unit, uncertainty)
4. Classify findings by type: measurement, model, prediction, review
5. Tag with DPF challenge numbers (1-15) where applicable
6. Store everything in `dpf_research.db`

**Agent prompt template:**
```
Read this paper and extract:
1. Every equation with all variable definitions and units
2. Every experimental measurement with device name, value, and uncertainty
3. Key findings (what does this paper prove/claim?)
4. What does this paper assume but not derive? (gaps)
5. Which of the 15 DPF simulation challenges does this address?
6. What existing DPF-Unified code implements (or should implement) this physics?
```

### Phase 2: Cross-Reference Analysis (AI Agent)
After Phase 1 completes for a batch of papers:
1. Compare formulas across papers — are they consistent?
2. Find gaps: Paper A needs X, Paper B provides X
3. Find contradictions: Paper A claims Y, Paper B claims not-Y
4. Map citation chains: who cites whom, and what claim travels between them
5. Identify orphaned findings (claims with no supporting evidence from other papers)

**Cross-reference queries:**
```sql
-- Find papers that share experimental devices
SELECT a.title, b.title, d.device
FROM experimental_data d
JOIN papers a ON d.paper_id = a.id
JOIN experimental_data d2 ON d.device = d2.device AND d.paper_id != d2.paper_id
JOIN papers b ON d2.paper_id = b.id;

-- Find formulas that appear in multiple papers (consistency check)
SELECT f.name, f.equation, GROUP_CONCAT(p.title, ' | ')
FROM formulas f JOIN papers p ON f.paper_id = p.id
GROUP BY f.name HAVING COUNT(*) > 1;

-- Find gaps where one paper identifies a need and another fills it
SELECT g.description, s.title as 'needs_it', r.title as 'provides_it'
FROM gaps g
JOIN papers s ON g.source_paper_id = s.id
LEFT JOIN papers r ON g.resolution_paper_id = r.id;

-- Find unimplemented findings by challenge
SELECT f.finding, p.title, f.dpf_challenge
FROM findings f JOIN papers p ON f.paper_id = p.id
WHERE f.implemented = 0 ORDER BY f.dpf_challenge;
```

### Phase 3: Synthesis & Action Items
Produce three deliverables:

#### A. Unified Research Spreadsheet
| Paper | Year | Challenge | Key Formula | Experimental Data | Gap Identified | Our Implementation | Status |
|-------|------|-----------|-------------|-------------------|----------------|-------------------|--------|

Export from database as CSV:
```sql
SELECT p.title, p.year, f.dpf_challenge,
       GROUP_CONCAT(DISTINCT fm.equation),
       GROUP_CONCAT(DISTINCT ed.parameter || '=' || ed.value || ed.unit),
       g.description,
       im.module_path,
       CASE WHEN im.needs_update THEN 'NEEDS UPDATE' ELSE 'OK' END
FROM papers p
LEFT JOIN findings f ON p.id = f.paper_id
LEFT JOIN formulas fm ON p.id = fm.paper_id
LEFT JOIN experimental_data ed ON p.id = ed.paper_id
LEFT JOIN gaps g ON p.id = g.source_paper_id
LEFT JOIN implementation_map im ON p.id = im.paper_id
GROUP BY p.id
ORDER BY p.year DESC, p.citations DESC;
```

#### B. Gap Analysis Document
For each of the 15 DPF challenges:
- What papers address it
- What formulas are available
- What experimental data validates it
- What's implemented in DPF-Unified
- What's missing (the gap)
- Recommended action

#### C. Cross-Reference Network
Mermaid diagram showing:
- Paper → paper citation relationships
- Paper → DPF-Unified module mappings
- Gap → resolution paths

## Prioritized Ingestion Order

### Batch 1: Core DPF (Tier 1 from strategy)
1. Goyon et al. 2025 — MJOLNIR neutron dynamics
2. Auluck 2023 — neutron yield scaling failure
3. Lerner et al. 2023 — p-B11 Focus Fusion
4. Wahbe et al. 2023 — Lee model Yn vs pressure
5. Schmidt et al. 2024 — MJOLNIR rebuild
6. Damideh et al. 2025 — FAETON-I 100kV

### Batch 2: Physics Modules
7. Kubeš et al. 2023 — MA filament observation
8. Novotný et al. 2023 — anode shape effects
9. Upadhyay et al. 2023 — poloidal magnetic flux
10. Park et al. 2023 — SPH for pinch plasma

### Batch 3: Adjacent Fields (highest impact)
11. NIF target gain (2024, 338 cites) — fusion diagnostics
12. Deep RL for tearing instability (2024, 114 cites) — AI+plasma
13. MPI-AMRVAC 3.0 (2023, 97 cites) — MHD solver comparison
14. MagLIF scaling (2023, 29 cites) — scaling laws
15. JET D-T results (2024, 83 cites) — beam-target validation

## CLI Tool for Database

```bash
# Search papers
python3 -c "
import sqlite3
conn = sqlite3.connect('docs/research-reference/dpf_research.db')
c = conn.cursor()
c.execute(\"SELECT title, year, citations FROM papers WHERE title LIKE '%dense plasma focus%' ORDER BY citations DESC LIMIT 10\")
for row in c.fetchall(): print(row)
"

# Export unified spreadsheet
python3 -c "
import sqlite3, csv
conn = sqlite3.connect('docs/research-reference/dpf_research.db')
c = conn.cursor()
c.execute('SELECT title, authors, year, doi, journal, category, citations, open_access FROM papers ORDER BY year DESC, citations DESC')
with open('docs/research-reference/unified_papers.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Title','Authors','Year','DOI','Journal','Category','Citations','Open Access'])
    w.writerows(c.fetchall())
print('Exported to docs/research-reference/unified_papers.csv')
"
```

## Automation

### Weekly Literature Scan
```bash
# Add to cortana cron or /loop
python3 scripts/scan_new_papers.py  # searches arXiv + OpenAlex for new DPF papers
```

### On Paper Download
```bash
# Extract and ingest automatically
python3 scripts/ingest_paper.py <pdf_path>  # AI extracts formulas, data, findings
```

### Monthly Cross-Reference
```bash
# Rebuild cross-references and identify new gaps
python3 scripts/cross_reference.py  # compares all papers, updates gaps table
```
