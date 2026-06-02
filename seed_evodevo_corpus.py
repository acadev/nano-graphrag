#!/usr/bin/env python3
"""
Evo-devo literature seeding — Firecrawl-only search strategy.

Populates a per-group paper corpus for 10 knowledge-graph topics covering
evolutionary developmental biology, body-plan innovation, and related
mechanistic/theoretical frameworks.

Output layout:
    data/evo_devo_corpus/
        corpus_index.json
        <group>/
            metadata.json
            papers/
                <uuid>.md

Usage:
    export FIRECRAWL_API_KEY=...
    python seed_evodevo_corpus.py
    python seed_evodevo_corpus.py --groups kg_symmetry_locomotion_manoeuvrability
    python seed_evodevo_corpus.py --max-per-query 100 --output-dir data/evo_devo_corpus
    python seed_evodevo_corpus.py --dry-run
"""
from __future__ import annotations

from _seed_corpus_shared import SearchQuery, PaperGroup, run_main

_SITES = (
    "site:pubmed.ncbi.nlm.nih.gov OR site:ncbi.nlm.nih.gov/pmc OR "
    "site:nature.com OR site:cell.com OR site:science.org OR "
    "site:biorxiv.org OR site:elifesciences.org OR site:plos.org"
)

# ---------------------------------------------------------------------------
# Evo-devo corpus definition — the ONLY thing that differs from seed_haiqu_corpus.py
# ---------------------------------------------------------------------------

GROUPS = [
    PaperGroup(
        name="kg_symmetry_locomotion_manoeuvrability",
        question=(
            "What is the relationship between body symmetry and locomotor performance, "
            "manoeuvrability, and hydrodynamic efficiency across animal taxa?"
        ),
        schema="kg_symmetry_locomotion_manoeuvrability",
        queries=[
            SearchQuery(
                query=(
                    f"bilateral symmetry aquatic locomotion hydrodynamic performance "
                    f"manoeuvrability fish swimming efficiency {_SITES}"
                ),
                notes="Core symmetry-locomotion link",
            ),
            SearchQuery(
                query=(
                    f"radial symmetry bilateral symmetry body plan evolutionary "
                    f"advantages locomotion animal movement {_SITES}"
                ),
                notes="Radial vs bilateral body plan comparison",
            ),
            SearchQuery(
                query=(
                    f"asymmetry fluctuating asymmetry locomotor performance fitness "
                    f"animal body plan evolution {_SITES}"
                ),
                notes="Asymmetry costs and locomotion",
            ),
            SearchQuery(
                query=(
                    f"evolutionary maintenance bilateral symmetry developmental "
                    f"constraint selection pressure body form {_SITES}"
                ),
                notes="Why symmetry is maintained evolutionarily",
                tbs="qdr:y",
            ),
        ],
    ),

    PaperGroup(
        name="kg_radial_vs_bilateral_signaling_geometry",
        question=(
            "How does body geometry (radial vs bilateral) constrain or enable "
            "morphogen gradient formation, positional information, and long-range "
            "developmental signaling?"
        ),
        schema="kg_radial_vs_bilateral_signaling_geometry",
        queries=[
            SearchQuery(
                query=(
                    f"morphogen gradient diffusion geometry radial symmetry "
                    f"bilateral symmetry positional information patterning {_SITES}"
                ),
                notes="Geometry-gradient interaction",
            ),
            SearchQuery(
                query=(
                    f"reaction diffusion patterning body plan geometry rotational "
                    f"symmetry synchronization developmental biology {_SITES}"
                ),
                notes="Reaction-diffusion in radial/bilateral contexts",
            ),
            SearchQuery(
                query=(
                    f"early animal embryo signaling coordination axis formation "
                    f"radial bilateral body plan evolution cnidaria {_SITES}"
                ),
                notes="Early animal signaling in radial organisms",
            ),
            SearchQuery(
                query=(
                    f"BMP Wnt gradient long range signaling tissue geometry "
                    f"body axis patterning invertebrate {_SITES}"
                ),
                notes="Specific morphogen gradients in body plan context",
                tbs="qdr:y",
            ),
        ],
    ),

    PaperGroup(
        name="kg_robustness_canalization_innovation",
        question=(
            "Do robust, canalized developmental networks constrain morphological "
            "innovation and ecological niche expansion, or do they enable it?"
        ),
        schema="kg_robustness_canalization_innovation",
        queries=[
            SearchQuery(
                query=(
                    f"developmental robustness canalization phenotypic innovation "
                    f"morphological novelty evolvability constraint {_SITES}"
                ),
                notes="Core robustness-innovation tension",
            ),
            SearchQuery(
                query=(
                    f"conserved developmental regulatory circuit evolvability "
                    f"phenotypic diversification developmental constraint {_SITES}"
                ),
                notes="Conserved circuits and diversification",
            ),
            SearchQuery(
                query=(
                    f"Waddington canalization genetic assimilation developmental "
                    f"buffering phenotypic variation evolution {_SITES}"
                ),
                notes="Waddington framework and empirical tests",
            ),
            SearchQuery(
                query=(
                    f"robustness evolvability trade-off gene regulatory network "
                    f"body plan innovation ecological niche {_SITES}"
                ),
                notes="Empirical robustness-evolvability trade-off",
                tbs="qdr:y",
            ),
        ],
    ),

    PaperGroup(
        name="kg_reaction_diffusion_body_plan_innovation",
        question=(
            "How do reaction-diffusion and Turing patterning mechanisms contribute "
            "to periodic structures, segmentation, and early animal body-plan innovation?"
        ),
        schema="kg_reaction_diffusion_body_plan_innovation",
        queries=[
            SearchQuery(
                query=(
                    f"reaction diffusion Turing pattern segmentation appendage "
                    f"patterning animal body plan {_SITES}"
                ),
                notes="Core RD patterning in development",
            ),
            SearchQuery(
                query=(
                    f"Turing mechanism stripe spot periodic pattern embryo "
                    f"vertebrate invertebrate comparative {_SITES}"
                ),
                notes="Comparative Turing patterning",
            ),
            SearchQuery(
                query=(
                    f"Cambrian body plan innovation evo-devo developmental "
                    f"mechanism gene duplication heterochrony ecological trigger {_SITES}"
                ),
                notes="Cambrian innovation mechanisms contrasted",
            ),
            SearchQuery(
                query=(
                    f"reaction diffusion vs gene regulatory network body plan "
                    f"segmentation growth dynamics alternative mechanisms {_SITES}"
                ),
                notes="RD vs alternative mechanisms",
                tbs="qdr:y",
            ),
        ],
    ),

    PaperGroup(
        name="kg_plasticity_modularity_gene_reuse",
        question=(
            "How do phenotypic plasticity, modularity, and gene reuse interact "
            "in the context-dependent morphology of marine invertebrates and "
            "other systems with environmentally variable development?"
        ),
        schema="kg_plasticity_modularity_gene_reuse",
        queries=[
            SearchQuery(
                query=(
                    f"phenotypic plasticity modularity developmental co-option "
                    f"gene reuse marine invertebrate morphology {_SITES}"
                ),
                notes="Core plasticity-modularity-reuse link",
            ),
            SearchQuery(
                query=(
                    f"developmental plasticity adaptive benefit environment "
                    f"context dependent morphology circuit reuse {_SITES}"
                ),
                notes="Plasticity vs adaptive benefit distinction",
            ),
            SearchQuery(
                query=(
                    f"modular gene regulatory network co-option novel trait "
                    f"evolution body plan diversification {_SITES}"
                ),
                notes="Modularity enabling co-option",
            ),
            SearchQuery(
                query=(
                    f"toolkit gene reuse deep homology convergent evolution "
                    f"morphological novelty invertebrate {_SITES}"
                ),
                notes="Gene reuse and deep homology",
                tbs="qdr:y",
            ),
        ],
    ),

    PaperGroup(
        name="kg_chromatin_accessibility_segment_identity",
        question=(
            "Does chromatin state constrain Hox binding and segment identity, "
            "or does Hox binding reshape chromatin — and what is the causal "
            "direction and timing?"
        ),
        schema="kg_chromatin_accessibility_segment_identity",
        queries=[
            SearchQuery(
                query=(
                    f"chromatin accessibility Hox binding segment identity "
                    f"embryonic patterning causal direction {_SITES}"
                ),
                notes="Core chromatin-Hox causality question",
            ),
            SearchQuery(
                query=(
                    f"chromatin remodeling Hox gene regulation ATAC-seq "
                    f"ChIP-seq embryo segmentation timing {_SITES}"
                ),
                notes="Perturbation/timing experiments",
            ),
            SearchQuery(
                query=(
                    f"pioneer transcription factor chromatin opening Hox "
                    f"homeodomain early development constraint {_SITES}"
                ),
                notes="Pioneer factor context",
            ),
            SearchQuery(
                query=(
                    f"Polycomb chromatin state Hox cluster regulation "
                    f"segment identity reprogramming perturbation {_SITES}"
                ),
                notes="Polycomb-Hox chromatin regulation",
                tbs="qdr:y",
            ),
        ],
    ),

    PaperGroup(
        name="kg_morphogen_gradients_organized_growth",
        question=(
            "How do morphogen gradients (SHH, FGF, BMP, Notch) separately "
            "control proliferation, differentiation, migration, and apoptosis "
            "to organize tissue architecture in neural and limb patterning?"
        ),
        schema="kg_morphogen_gradients_organized_growth",
        queries=[
            SearchQuery(
                query=(
                    f"morphogen gradient Sonic hedgehog FGF BMP proliferation "
                    f"differentiation tissue organization limb patterning {_SITES}"
                ),
                notes="Core morphogen gradient studies",
            ),
            SearchQuery(
                query=(
                    f"Notch signaling neural patterning proliferation zone "
                    f"tissue architecture organized growth {_SITES}"
                ),
                notes="Notch and neural organization",
            ),
            SearchQuery(
                query=(
                    f"morphogen gradient concentration threshold cell fate "
                    f"apoptosis migration separate roles tissue patterning {_SITES}"
                ),
                notes="Separating proliferation/diff/apoptosis/migration",
            ),
            SearchQuery(
                query=(
                    f"BMP gradient dorsal ventral patterning cell proliferation "
                    f"apoptosis embryo quantitative {_SITES}"
                ),
                notes="BMP gradient quantitative dissection",
                tbs="qdr:y",
            ),
        ],
    ),

    PaperGroup(
        name="kg_non_hox_novelty_sweep",
        question=(
            "What non-Hox mechanisms — biomechanics, ECM remodeling, planar cell "
            "polarity, Hippo, Notch, retinoic acid, heterochrony, metabolic "
            "constraints, and ecological selection — drive body-plan innovation?"
        ),
        schema="kg_non_hox_novelty_sweep",
        queries=[
            SearchQuery(
                query=(
                    f"body plan innovation non-Hox mechanism biomechanics "
                    f"extracellular matrix cell adhesion planar cell polarity {_SITES}"
                ),
                notes="Non-Hox structural mechanisms",
            ),
            SearchQuery(
                query=(
                    f"Hippo signaling Notch retinoic acid heterochrony body plan "
                    f"evolution morphological novelty {_SITES}"
                ),
                notes="Non-Hox signaling pathways",
            ),
            SearchQuery(
                query=(
                    f"metabolic constraint developmental timing ecological selection "
                    f"morphological innovation body plan evo-devo {_SITES}"
                ),
                notes="Metabolic and ecological drivers",
            ),
            SearchQuery(
                query=(
                    f"ECM remodeling tissue mechanics morphogenesis cell "
                    f"rearrangement body plan novelty evolution {_SITES}"
                ),
                notes="Mechanical and matrix contributions",
                tbs="qdr:y",
            ),
        ],
    ),

    PaperGroup(
        name="kg_contradictions_and_null_results",
        question=(
            "Where do dominant developmental and evolutionary explanations of "
            "body-plan innovation fail, produce null results, or face credible "
            "alternative interpretations?"
        ),
        schema="kg_contradictions_and_null_results",
        queries=[
            SearchQuery(
                query=(
                    f"null result rebuttal alternative mechanism developmental "
                    f"biology body plan evolution insufficient downstream {_SITES}"
                ),
                notes="Null results and rebuttals",
            ),
            SearchQuery(
                query=(
                    f"contradictory evidence evo-devo developmental mechanism "
                    f"context limited insufficient explanation review {_SITES}"
                ),
                notes="Contradictions and context limits",
            ),
            SearchQuery(
                query=(
                    f"failure dominant model morphogen Hox segmentation "
                    f"alternative explanation empirical challenge {_SITES}"
                ),
                notes="Challenges to dominant models",
            ),
            SearchQuery(
                query=(
                    f"replication failure inconsistent result developmental "
                    f"biology evolutionary mechanism critique {_SITES}"
                ),
                notes="Reproducibility and consistency issues",
                tbs="qdr:y",
            ),
        ],
    ),

    PaperGroup(
        name="kg_evidence_to_prediction",
        question=(
            "Which developmental biology papers specify explicit predictions, "
            "discriminating experiments, or measurable signatures that would "
            "distinguish mechanism A from mechanism B?"
        ),
        schema="kg_evidence_to_prediction",
        queries=[
            SearchQuery(
                query=(
                    f"developmental mechanism prediction discriminating experiment "
                    f"perturbation logic measurable outcome body plan {_SITES}"
                ),
                notes="Discriminating experiments",
            ),
            SearchQuery(
                query=(
                    f"comparative signature developmental biology mechanism test "
                    f"quantitative prediction evo-devo body plan {_SITES}"
                ),
                notes="Comparative and quantitative predictions",
            ),
            SearchQuery(
                query=(
                    f"falsifiable prediction developmental constraint body plan "
                    f"innovation mechanistic test alternative hypothesis {_SITES}"
                ),
                notes="Falsifiability and hypothesis testing",
            ),
            SearchQuery(
                query=(
                    f"perturbation experiment knockout rescue developmental "
                    f"mechanism causal test morphogenesis body plan {_SITES}"
                ),
                notes="Perturbation-based causal tests",
                tbs="qdr:y",
            ),
        ],
    ),
]


if __name__ == "__main__":
    run_main(
        groups=GROUPS,
        default_output_dir="data/evo_devo_corpus",
        default_max_per_query=100,
        only_main_content=True,
    )
