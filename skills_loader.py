"""Chargement des skills
(format Agent Skills : un dossier par skill avec un SKILL.md).

    skills/
      traitement-retour/
        SKILL.md

Principe de la « divulgation progressive » :
  1. au démarrage, seuls le NOM et la DESCRIPTION
     de chaque skill vont dans le prompt ;
  2. quand une demande correspond, l'agent appelle
     charger_skill(nom) et lit les instructions complètes,
     et seulement à ce moment-là.
"""

from pathlib import Path

import yaml
from langchain_core.tools import tool


# =========================================================
# DOSSIER DES SKILLS
# =========================================================

DOSSIER_SKILLS = (
    Path(__file__).parent
    / "skills"
)


# =========================================================
# LECTURE D'UN SKILL
# =========================================================

def lire_skill(
    fichier: Path
) -> tuple[dict, str]:
    """Sépare l'en-tête YAML
    du corps Markdown.
    """

    texte = fichier.read_text(
        encoding="utf-8"
    )

    _, entete, corps = texte.split(
        "---",
        2
    )

    return (
        yaml.safe_load(entete),
        corps.strip()
    )


# =========================================================
# CHARGEMENT DU CATALOGUE
# =========================================================

def charger_catalogue() -> dict[str, dict]:
    """Lit tous les skills/*/SKILL.md.

    Retour :
    {
        nom: {
            "description": ...,
            "instructions": ...
        }
    }
    """

    catalogue = {}


    for fichier in sorted(
        DOSSIER_SKILLS.glob(
            "*/SKILL.md"
        )
    ):

        entete, corps = lire_skill(
            fichier
        )

        catalogue[
            entete["name"]
        ] = {
            "description":
                entete["description"],

            "instructions":
                corps
        }


    return catalogue


# =========================================================
# CATALOGUE GLOBAL
# =========================================================

CATALOGUE = charger_catalogue()


# =========================================================
# RESUME DES SKILLS
# =========================================================

def resume_skills() -> str:
    """Texte ajouté au prompt système :
    uniquement nom + description.
    """

    lignes = [
        f"- {nom} : {skill['description']}"
        for nom, skill
        in CATALOGUE.items()
    ]


    return (
        "Skills disponibles "
        "(appelle charger_skill "
        "avec le nom exact AVANT d'agir "
        "si la demande correspond) :\n"
        + "\n".join(
            lignes
        )
    )


# =========================================================
# OUTIL LANGCHAIN
# =========================================================

@tool
def charger_skill(
    nom: str
) -> str:
    """Charge les instructions détaillées
    d'un skill
    (procédure à suivre pas à pas).
    """

    skill = CATALOGUE.get(
        nom
    )


    if skill is None:

        return (
            f"Skill inconnu : {nom}. "
            f"Skills disponibles : "
            f"{', '.join(CATALOGUE)}."
        )


    return skill[
        "instructions"
    ]