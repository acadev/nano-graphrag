#!/usr/bin/env python3
"""
HAIQU literature seeding — Firecrawl-only search strategy.

Runs grouped Firecrawl /v1/search calls (with inline markdown scraping) to
populate a per-group paper corpus that the graph build pipeline can ingest.

Output layout:
    data/haiqu_corpus/
        corpus_index.json
        <group>/
            metadata.json          # queries used + per-paper records
            papers/
                <uuid>.md          # one paper per file (Firecrawl markdown)

Usage:
    export FIRECRAWL_API_KEY=...
    python seed_haiqu_corpus.py
    python seed_haiqu_corpus.py --groups haiqu_biosensor_detection haiqu_aerosol_exposure
    python seed_haiqu_corpus.py --max-per-query 20 --output-dir data/my_corpus
    python seed_haiqu_corpus.py --dry-run               # show queries, no API calls
"""
from __future__ import annotations

from _seed_corpus_shared import SearchQuery, PaperGroup, run_main

# ---------------------------------------------------------------------------
# HAIQU corpus definition — the ONLY thing that differs from seed_evodevo_corpus.py
# ---------------------------------------------------------------------------

GROUPS = [
    PaperGroup(
        name="haiqu_biosensor_detection",
        question=(
            "Can microfluidic droplet CRISPR biosensors detect pathogenic agents "
            "in hospital air, and at what limits?"
        ),
        schema="haiqu_biosensor_detection",
        queries=[
            SearchQuery(
                "microfluidic droplet CRISPR Cas13 airborne pathogen detection "
                "limit of detection site:pubmed.ncbi.nlm.nih.gov OR site:nature.com OR site:cell.com"
            ),
            SearchQuery(
                "SHERLOCK DETECTR CRISPR biosensor aerosol pathogen hospital "
                "site:pubmed.ncbi.nlm.nih.gov OR site:biorxiv.org",
                tbs="qdr:y",
            ),
            SearchQuery(
                "Cas13a Cas12a airborne virus detection sensitivity specificity "
                "site:pubmed.ncbi.nlm.nih.gov OR site:nature.com"
            ),
            SearchQuery(
                "digital droplet PCR airborne respiratory pathogen quantification "
                "hospital site:pubmed.ncbi.nlm.nih.gov OR site:plos.org"
            ),
            SearchQuery(
                "isothermal amplification RPA LAMP airborne pathogen point-of-care "
                "site:pubmed.ncbi.nlm.nih.gov OR site:biorxiv.org"
            ),
            SearchQuery(
                "biosensor validation aerosol spiked chamber clinical specimen "
                "respiratory pathogen site:pubmed.ncbi.nlm.nih.gov"
            ),
        ],
    ),
    PaperGroup(
        name="haiqu_aerosol_exposure",
        question=(
            "What pathogens are present in hospital air, at what concentrations, "
            "and under what environmental conditions?"
        ),
        schema="haiqu_aerosol_exposure",
        queries=[
            SearchQuery(
                "bioaerosol sampling hospital airborne pathogen concentration "
                "site:pubmed.ncbi.nlm.nih.gov OR site:ncbi.nlm.nih.gov/pmc"
            ),
            SearchQuery(
                "SARS-CoV-2 airborne concentration hospital sampling impactor cyclone "
                "site:pubmed.ncbi.nlm.nih.gov OR site:medrxiv.org OR site:biorxiv.org",
                tbs="qdr:y",
            ),
            SearchQuery(
                "aerosol viability respiratory pathogen relative humidity temperature "
                "site:pubmed.ncbi.nlm.nih.gov OR site:nature.com"
            ),
            SearchQuery(
                "particle size distribution respiratory aerosol droplet nucleus "
                "hospital infection site:pubmed.ncbi.nlm.nih.gov"
            ),
            SearchQuery(
                "Mycobacterium tuberculosis Aspergillus airborne hospital sampling "
                "concentration CFU copies site:pubmed.ncbi.nlm.nih.gov"
            ),
            SearchQuery(
                "influenza RSV MRSA airborne hospital environmental sampling "
                "site:pubmed.ncbi.nlm.nih.gov OR site:cdc.gov"
            ),
        ],
    ),
    PaperGroup(
        name="haiqu_hospital_environment",
        question=(
            "How does hospital room configuration and HVAC design affect "
            "airborne pathogen distribution?"
        ),
        schema="haiqu_hospital_environment",
        queries=[
            SearchQuery(
                "hospital HVAC design airborne pathogen distribution air changes "
                "site:pubmed.ncbi.nlm.nih.gov OR site:ashrae.org"
            ),
            SearchQuery(
                "negative pressure isolation room AIIR airborne infection control "
                "site:pubmed.ncbi.nlm.nih.gov OR site:cdc.gov"
            ),
            SearchQuery(
                "operating room ventilation airflow pathogen contamination "
                "site:pubmed.ncbi.nlm.nih.gov"
            ),
            SearchQuery(
                "tracer gas air distribution hospital ward CFD airflow "
                "site:pubmed.ncbi.nlm.nih.gov OR site:nature.com"
            ),
            SearchQuery(
                "hospital room configuration recirculation dead zone infection "
                "site:pubmed.ncbi.nlm.nih.gov",
                tbs="qdr:y",
            ),
            SearchQuery(
                "ASHRAE 170 ventilation healthcare facility infection control "
                "site:ashrae.org OR site:pubmed.ncbi.nlm.nih.gov"
            ),
        ],
    ),
    PaperGroup(
        name="haiqu_engineering_controls",
        question=(
            "Which engineering controls reduce airborne pathogen exposure "
            "and by how much?"
        ),
        schema="haiqu_engineering_controls",
        queries=[
            SearchQuery(
                "upper room UV-C UVGI airborne tuberculosis disinfection efficacy "
                "site:pubmed.ncbi.nlm.nih.gov"
            ),
            SearchQuery(
                "HEPA portable air cleaner hospital airborne pathogen reduction "
                "site:pubmed.ncbi.nlm.nih.gov OR site:biorxiv.org",
                tbs="qdr:y",
            ),
            SearchQuery(
                "engineering control airborne infection reduction hospital "
                "log reduction effectiveness site:pubmed.ncbi.nlm.nih.gov"
            ),
            SearchQuery(
                "increased ventilation ACH airborne infection risk reduction "
                "site:pubmed.ncbi.nlm.nih.gov OR site:plos.org"
            ),
            SearchQuery(
                "source control masking N95 respirator airborne hospital "
                "site:pubmed.ncbi.nlm.nih.gov OR site:cdc.gov"
            ),
            SearchQuery(
                "UV-C HEPA ventilation combined effectiveness hospital "
                "site:pubmed.ncbi.nlm.nih.gov"
            ),
        ],
    ),
    PaperGroup(
        name="haiqu_transmission_risk",
        question=(
            "What models predict real-time disease transmission risk from "
            "environmental measurements?"
        ),
        schema="haiqu_transmission_risk",
        queries=[
            SearchQuery(
                "Wells-Riley airborne infection risk model hospital real-time "
                "site:pubmed.ncbi.nlm.nih.gov OR site:plos.org"
            ),
            SearchQuery(
                "quantitative microbial risk assessment QMRA airborne hospital "
                "site:pubmed.ncbi.nlm.nih.gov"
            ),
            SearchQuery(
                "agent-based model nosocomial airborne transmission hospital simulation "
                "site:pubmed.ncbi.nlm.nih.gov OR site:nature.com"
            ),
            SearchQuery(
                "digital twin hospital airborne transmission risk prediction "
                "site:pubmed.ncbi.nlm.nih.gov OR site:nature.com",
                tbs="qdr:y",
            ),
            SearchQuery(
                "CO2 rebreathed air fraction proxy infection risk indoor "
                "site:pubmed.ncbi.nlm.nih.gov"
            ),
            SearchQuery(
                "dose-response model airborne pathogen infection probability "
                "site:pubmed.ncbi.nlm.nih.gov OR site:plos.org"
            ),
        ],
    ),
    PaperGroup(
        name="haiqu_cognitive_impact",
        question=(
            "How do respiratory infections affect cognitive function in "
            "healthcare workers?"
        ),
        schema="haiqu_cognitive_impact",
        queries=[
            SearchQuery(
                "post-COVID cognitive impairment healthcare worker "
                "neuropsychological assessment site:pubmed.ncbi.nlm.nih.gov OR site:nature.com"
            ),
            SearchQuery(
                "long COVID cognitive function executive memory processing speed "
                "longitudinal site:pubmed.ncbi.nlm.nih.gov OR site:medrxiv.org",
                tbs="qdr:y",
            ),
            SearchQuery(
                "respiratory infection cognitive deficit attention working memory "
                "site:pubmed.ncbi.nlm.nih.gov"
            ),
            SearchQuery(
                "neuropsychological test healthcare worker respiratory infection "
                "cognitive outcome site:pubmed.ncbi.nlm.nih.gov"
            ),
            SearchQuery(
                "ICU healthcare worker cognitive sequelae occupational exposure "
                "site:pubmed.ncbi.nlm.nih.gov"
            ),
        ],
    ),
]


if __name__ == "__main__":
    run_main(
        groups=GROUPS,
        default_output_dir="data/haiqu_corpus",
        default_max_per_query=10,
    )
