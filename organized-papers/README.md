# Research Database Browser

A professional, single-file interactive HTML application for browsing and analyzing your research paper database.

## File Location
- **`Research_Database_Browser.html`** - 7.54 MB, fully self-contained application

## Features

### 1. **Search & Filter Sidebar**
- **Full-text search** across titles, authors, abstracts, and key findings
- **Category dropdown** - filter by research category
- **Year range slider** - filter papers by publication year
- **Domain tags** - multi-select checkboxes for research domains
- **Methods** - filter by methodology type

### 2. **Papers Tab**
- Sortable table with clickable columns (Title, Authors, Year, Category, Equations)
- Click any paper to view full details
- Expandable paper detail panel with:
  - Abstract (truncated to 1000 chars)
  - Key findings
  - Domain tags with confidence scores
  - All equations from the paper
  - Methodology types
  - Related papers (via cross-references)

### 3. **Domain Network Tab** (Visual)
- SVG-based network graph showing all research domains
- Node size represents number of papers in that domain
- Edge thickness represents connection strength
- Click domains to explore relationships

### 4. **AI-Physics Links Tab**
- Dedicated view of all 5,198 AI-Physics crossover connections
- Shows paper pairs with shared domains and methods
- Direct links to jump between related papers

### 5. **Equations Tab**
- Searchable database of all 6,710 equations
- Search by equation text or context
- Shows source paper for each equation
- Organized by equation ID and paper

### 6. **Statistics Tab**
- Overview cards: Total papers, equations, cross-references, AI-Physics links
- Bar charts for:
  - Papers by year (top 15)
  - Papers by category
  - Papers by domain (top 20)
  - Most common research terms (top 30)

## Database Content

Extracted from SQLite database with intelligent filtering:

- **410 papers** with title, authors, year, DOI, arXiv ID, abstract, key findings
- **6,710 equations** with context and source paper
- **1,395 domain tags** with confidence scores (58 unique domains)
- **26,451 cross-references** (filtered from 32,886 total)
- **5,198 AI-Physics crossovers** (special connection type)
- **58 key terms** aggregated by frequency

## Data Optimization

- Full-text excerpts removed (too large)
- Abstracts truncated to 1000 characters
- Cross-references filtered to include only:
  - AI_PHYSICS_CROSSOVER connections (all 5,198)
  - strong_methodological connections
  - Weak connections excluded (27k+ filtered out)
- Key terms aggregated by total frequency across all papers

## Usage

1. **Open in any modern browser** - No plugins or external dependencies required
2. **Use filters in the sidebar** to narrow down papers
3. **Search** for specific terms, authors, or concepts
4. **Click paper titles** to view full details and related papers
5. **Switch tabs** to explore network, equations, AI-Physics links, and statistics

## Browser Compatibility

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (responsive design)

## Technical Details

- **Format**: Single HTML file with embedded JSON data
- **Size**: 7.54 MB (under 10 MB target)
- **Dependencies**: None (pure HTML, CSS, JavaScript)
- **Performance**: All data loaded in memory for instant filtering/search
- **Styling**: Modern CSS Grid/Flexbox with dark sidebar, clean white content area

## Tips for Best Results

1. **Search first** - Use the search bar to find papers by keyword
2. **Explore domains** - Check domain filters to see research areas
3. **Network graph** - Visual clusters show related research areas
4. **AI-Physics** - Check this tab to find interdisciplinary connections
5. **Statistics** - Understand trends in your research collection
6. **Paper details** - Click any paper to see full metadata and equations

---

Generated: 2026-03-15
Database: 410 papers, 6,710 equations, 5,198 AI-Physics crossovers
