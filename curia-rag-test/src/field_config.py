"""
Field-specific ingestion query configurations for CURIA Agent A.

When a user selects a field of study, Agent A fetches documents using
queries tailored to that discipline's industry signals.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-field ingestion parameters
# Keys must match the "field" values in university JSON files.
# ---------------------------------------------------------------------------

FIELD_INGESTION: dict[str, dict] = {

    # ── Engineering ──────────────────────────────────────────────────────
    "Computer Science": {
        "job_titles":      ["software engineer", "machine learning engineer", "devops", "cloud engineer", "security engineer"],
        "arxiv_cats":      ["cs.AI", "cs.LG", "cs.SE", "cs.CR", "cs.DC"],
        "so_tags":         ["machine-learning", "kubernetes", "python", "cybersecurity", "large-language-model"],
        "github_topics":   ["machine-learning", "llm", "rag", "cloud-native", "devsecops"],
        "remotive_queries":["machine learning", "LLM", "devops", "cloud native"],
        "arbeitnow_queries":["machine learning", "kubernetes", "devsecops"],
    },

    "Electrical Engineering": {
        "job_titles":      ["electrical engineer", "embedded systems", "FPGA engineer", "RF engineer", "power electronics"],
        "arxiv_cats":      ["eess.SP", "eess.SY", "cs.RO", "eess.IV"],
        "so_tags":         ["fpga", "verilog", "embedded", "signal-processing", "matlab"],
        "github_topics":   ["fpga", "embedded-systems", "signal-processing", "control-systems", "verilog"],
        "remotive_queries":["embedded engineer", "FPGA", "electronics"],
        "arbeitnow_queries":["electrical engineer", "embedded systems"],
    },

    "Mechanical Engineering": {
        "job_titles":      ["mechanical engineer", "robotics engineer", "manufacturing engineer", "CAD engineer", "simulation engineer"],
        "arxiv_cats":      ["cs.RO", "eess.SY", "physics.class-ph", "cond-mat.mtrl-sci"],
        "so_tags":         ["robotics", "matlab", "simulation", "control-systems", "ansys"],
        "github_topics":   ["robotics", "control-systems", "finite-element", "cad", "simulation"],
        "remotive_queries":["mechanical engineer", "robotics", "hardware engineer"],
        "arbeitnow_queries":["mechanical engineer", "hardware engineer", "manufacturing"],
    },

    "Civil Engineering": {
        "job_titles":      ["civil engineer", "structural engineer", "geotechnical engineer", "transportation engineer", "GIS analyst"],
        "arxiv_cats":      ["eess.SP", "cs.CV", "physics.geo-ph"],
        "so_tags":         ["gis", "autocad", "structural-analysis", "python", "remote-sensing"],
        "github_topics":   ["gis", "structural-analysis", "transportation", "smart-city", "remote-sensing"],
        "remotive_queries":["civil engineer", "GIS analyst", "structural engineer"],
        "arbeitnow_queries":["civil engineer", "infrastructure", "geotechnical"],
    },

    "Chemical Engineering": {
        "job_titles":      ["chemical engineer", "process engineer", "polymer engineer", "refinery engineer", "biotechnology engineer"],
        "arxiv_cats":      ["physics.chem-ph", "cond-mat.soft", "q-bio.BM", "cs.LG"],
        "so_tags":         ["python", "matlab", "simulation", "machine-learning", "data-analysis"],
        "github_topics":   ["process-simulation", "molecular-dynamics", "cheminformatics", "machine-learning"],
        "remotive_queries":["chemical engineer", "process engineer", "bioprocess"],
        "arbeitnow_queries":["chemical engineer", "process engineer", "materials engineer"],
    },

    "Aerospace Engineering": {
        "job_titles":      ["aerospace engineer", "systems engineer", "GNC engineer", "propulsion engineer", "UAV engineer"],
        "arxiv_cats":      ["cs.RO", "eess.SY", "physics.flu-dyn", "cs.AI"],
        "so_tags":         ["matlab", "python", "simulation", "control-systems", "ros"],
        "github_topics":   ["aerospace", "uav", "flight-dynamics", "gnc", "autonomous-systems"],
        "remotive_queries":["aerospace engineer", "drone", "autonomous systems", "systems engineer"],
        "arbeitnow_queries":["aerospace engineer", "systems engineer", "UAV"],
    },

    "Biomedical Engineering": {
        "job_titles":      ["biomedical engineer", "medical device engineer", "bioinformatics engineer", "clinical data scientist"],
        "arxiv_cats":      ["cs.LG", "q-bio.BM", "eess.IV", "cs.CV"],
        "so_tags":         ["python", "machine-learning", "bioinformatics", "image-processing", "tensorflow"],
        "github_topics":   ["medical-imaging", "bioinformatics", "health-informatics", "wearable", "deep-learning"],
        "remotive_queries":["biomedical engineer", "bioinformatics", "medical device", "health data"],
        "arbeitnow_queries":["biomedical engineer", "clinical engineer", "health technology"],
    },

    "Industrial & Systems Engineering": {
        "job_titles":      ["industrial engineer", "supply chain analyst", "operations research analyst", "quality engineer", "data analyst"],
        "arxiv_cats":      ["cs.LG", "math.OC", "eess.SY", "cs.AI"],
        "so_tags":         ["python", "optimization", "simulation", "data-analysis", "machine-learning"],
        "github_topics":   ["supply-chain", "optimization", "simulation", "operations-research", "digital-twin"],
        "remotive_queries":["supply chain", "operations research", "industrial engineer", "data analyst"],
        "arbeitnow_queries":["industrial engineer", "supply chain", "logistics engineer"],
    },

    "Materials Science & Engineering": {
        "job_titles":      ["materials engineer", "materials scientist", "metallurgist", "polymer scientist", "semiconductor engineer"],
        "arxiv_cats":      ["cond-mat.mtrl-sci", "cond-mat.mes-hall", "physics.app-ph", "cs.LG"],
        "so_tags":         ["python", "matlab", "materials-science", "machine-learning", "simulation"],
        "github_topics":   ["materials-informatics", "molecular-dynamics", "dft", "machine-learning", "semiconductor"],
        "remotive_queries":["materials engineer", "semiconductor engineer", "materials scientist"],
        "arbeitnow_queries":["materials engineer", "semiconductor", "nanotechnology"],
    },

    "Petroleum Engineering": {
        "job_titles":      ["petroleum engineer", "reservoir engineer", "drilling engineer", "production engineer", "data scientist"],
        "arxiv_cats":      ["physics.geo-ph", "cs.LG", "eess.SY", "physics.flu-dyn"],
        "so_tags":         ["python", "matlab", "data-analysis", "simulation", "machine-learning"],
        "github_topics":   ["reservoir-simulation", "geophysics", "drilling", "machine-learning"],
        "remotive_queries":["petroleum engineer", "energy data scientist", "reservoir engineer"],
        "arbeitnow_queries":["petroleum engineer", "energy engineer", "oil gas engineer"],
    },

    # ── Science ───────────────────────────────────────────────────────────
    "Mathematics": {
        "job_titles":      ["mathematician", "data scientist", "quantitative analyst", "research scientist", "actuary"],
        "arxiv_cats":      ["math.NA", "math.ST", "math.OC", "cs.LG", "math.PR"],
        "so_tags":         ["python", "numpy", "scipy", "optimization", "statistics"],
        "github_topics":   ["numerical-methods", "optimization", "statistics", "machine-learning", "scientific-computing"],
        "remotive_queries":["data scientist", "quantitative analyst", "research scientist", "mathematician"],
        "arbeitnow_queries":["data scientist", "quantitative analyst", "research scientist"],
    },

    "Statistics": {
        "job_titles":      ["statistician", "data scientist", "biostatistician", "quantitative researcher", "ML engineer"],
        "arxiv_cats":      ["stat.ML", "stat.AP", "stat.ME", "cs.LG"],
        "so_tags":         ["python", "r", "statistics", "machine-learning", "bayesian"],
        "github_topics":   ["statistics", "bayesian", "causal-inference", "time-series", "machine-learning"],
        "remotive_queries":["statistician", "data scientist", "biostatistician", "quantitative researcher"],
        "arbeitnow_queries":["statistician", "data scientist", "research analyst"],
    },

    "Physics": {
        "job_titles":      ["physicist", "research scientist", "quantum engineer", "data scientist", "simulation engineer"],
        "arxiv_cats":      ["quant-ph", "physics.comp-ph", "cond-mat", "cs.LG", "physics.data-an"],
        "so_tags":         ["python", "simulation", "quantum-computing", "numpy", "machine-learning"],
        "github_topics":   ["quantum-computing", "physics-simulation", "computational-physics", "machine-learning"],
        "remotive_queries":["physicist", "quantum engineer", "simulation engineer", "research scientist"],
        "arbeitnow_queries":["physicist", "quantum engineer", "research scientist"],
    },

    "Chemistry": {
        "job_titles":      ["chemist", "research scientist", "computational chemist", "drug discovery", "materials chemist"],
        "arxiv_cats":      ["physics.chem-ph", "q-bio.BM", "cond-mat.soft", "cs.LG"],
        "so_tags":         ["python", "rdkit", "cheminformatics", "molecular-dynamics", "machine-learning"],
        "github_topics":   ["cheminformatics", "drug-discovery", "molecular-dynamics", "computational-chemistry"],
        "remotive_queries":["computational chemist", "drug discovery", "cheminformatics", "research chemist"],
        "arbeitnow_queries":["chemist", "research scientist", "pharmaceutical engineer"],
    },

    "Biology": {
        "job_titles":      ["biologist", "bioinformatician", "research scientist", "computational biologist", "genomics analyst"],
        "arxiv_cats":      ["q-bio.GN", "q-bio.BM", "q-bio.CB", "cs.LG"],
        "so_tags":         ["python", "bioinformatics", "r", "machine-learning", "genomics"],
        "github_topics":   ["bioinformatics", "genomics", "computational-biology", "proteomics", "machine-learning"],
        "remotive_queries":["bioinformatician", "computational biologist", "genomics analyst"],
        "arbeitnow_queries":["biologist", "bioinformatics", "research scientist"],
    },

    # ── Business ──────────────────────────────────────────────────────────
    "Management Information Systems": {
        "job_titles":      ["IT manager", "systems analyst", "ERP consultant", "business analyst", "IT project manager"],
        "arxiv_cats":      ["cs.SE", "cs.DC", "cs.AI"],
        "so_tags":         ["sql", "python", "erp", "data-analysis", "cloud"],
        "github_topics":   ["enterprise-software", "erp", "business-intelligence", "data-analytics"],
        "remotive_queries":["systems analyst", "business analyst", "IT consultant", "ERP consultant"],
        "arbeitnow_queries":["business analyst", "systems analyst", "IT manager"],
    },

    "Business Analytics": {
        "job_titles":      ["business analyst", "data analyst", "analytics engineer", "BI developer", "product analyst"],
        "arxiv_cats":      ["cs.LG", "stat.AP", "cs.AI"],
        "so_tags":         ["python", "sql", "tableau", "power-bi", "machine-learning"],
        "github_topics":   ["business-intelligence", "data-analytics", "machine-learning", "visualization"],
        "remotive_queries":["business analyst", "data analyst", "BI developer", "analytics engineer"],
        "arbeitnow_queries":["data analyst", "business analyst", "BI developer"],
    },

    "Finance": {
        "job_titles":      ["financial analyst", "quantitative analyst", "risk analyst", "investment analyst", "fintech engineer"],
        "arxiv_cats":      ["q-fin.CP", "q-fin.RM", "cs.LG", "stat.ML"],
        "so_tags":         ["python", "r", "finance", "machine-learning", "sql"],
        "github_topics":   ["quantitative-finance", "algorithmic-trading", "risk-management", "machine-learning"],
        "remotive_queries":["quantitative analyst", "financial analyst", "fintech engineer", "risk analyst"],
        "arbeitnow_queries":["financial analyst", "quantitative analyst", "fintech"],
    },

    # ── Geosciences ───────────────────────────────────────────────────────
    "Atmospheric Science": {
        "job_titles":      ["atmospheric scientist", "climate scientist", "weather analyst", "data scientist", "remote sensing analyst"],
        "arxiv_cats":      ["physics.ao-ph", "physics.geo-ph", "cs.LG"],
        "so_tags":         ["python", "matlab", "climate-data", "machine-learning", "netcdf"],
        "github_topics":   ["climate-science", "weather-prediction", "atmospheric-modeling", "machine-learning"],
        "remotive_queries":["climate scientist", "atmospheric scientist", "environmental data scientist"],
        "arbeitnow_queries":["climate analyst", "environmental scientist", "data scientist"],
    },

    "Geography & Geospatial Sciences": {
        "job_titles":      ["GIS analyst", "geospatial engineer", "remote sensing analyst", "spatial data scientist"],
        "arxiv_cats":      ["cs.CV", "eess.IV", "physics.geo-ph", "cs.LG"],
        "so_tags":         ["gis", "python", "remote-sensing", "spatial-analysis", "qgis"],
        "github_topics":   ["gis", "remote-sensing", "geospatial", "satellite-imagery", "spatial-analysis"],
        "remotive_queries":["GIS analyst", "geospatial engineer", "remote sensing", "spatial data"],
        "arbeitnow_queries":["GIS analyst", "geospatial analyst", "remote sensing"],
    },

    # ── Agriculture ───────────────────────────────────────────────────────
    "Agricultural Data Science": {
        "job_titles":      ["agricultural data scientist", "precision agriculture engineer", "agtech engineer", "farm data analyst"],
        "arxiv_cats":      ["cs.LG", "eess.IV", "cs.CV", "physics.geo-ph"],
        "so_tags":         ["python", "machine-learning", "image-processing", "remote-sensing", "data-analysis"],
        "github_topics":   ["precision-agriculture", "agtech", "crop-disease-detection", "machine-learning"],
        "remotive_queries":["agtech engineer", "precision agriculture", "agricultural data scientist"],
        "arbeitnow_queries":["agricultural engineer", "agtech", "precision farming"],
    },

    "Biological & Agricultural Engineering": {
        "job_titles":      ["biological engineer", "agricultural engineer", "bioprocess engineer", "environmental engineer"],
        "arxiv_cats":      ["q-bio.BM", "physics.bio-ph", "cs.LG"],
        "so_tags":         ["python", "matlab", "simulation", "bioinformatics", "data-analysis"],
        "github_topics":   ["bioprocess", "agricultural-engineering", "environmental-modeling", "machine-learning"],
        "remotive_queries":["biological engineer", "bioprocess engineer", "agricultural engineer"],
        "arbeitnow_queries":["biological engineer", "environmental engineer", "bioprocess"],
    },
}


def get_ingestion_config(field: str) -> dict:
    """Return ingestion config for a field, falling back to Computer Science defaults."""
    return FIELD_INGESTION.get(field, FIELD_INGESTION["Computer Science"])
