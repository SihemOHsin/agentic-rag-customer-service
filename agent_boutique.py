"""Agent service client : LangGraph + RAG (Chroma) + SQL + validation humaine.

    uv run indexer.py           # une fois, pour indexer la politique
    uv run agent_boutique.py
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from indexer import ouvrir_base
from outils_base import OUTILS_SQL, creer_base


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()


# =========================================================
# OUTILS
# =========================================================

vector_store = ouvrir_base()


@tool
def chercher_politique(question: str) -> str:
    """Cherche dans les règles de la boutique :
    livraison, retour, annulation, paiement.
    """

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    return "\n".join(
        d.page_content
        for d in docs
    )


OUTILS = [
    chercher_politique,
    *OUTILS_SQL
]


# Outils qui nécessitent une validation humaine
SENSIBLES = {
    "annuler_commande",
    "creer_retour"
}


# =========================================================
# PROMPT SYSTEME
# =========================================================

PROMPT = (
    "Tu es l'assistant du service client d'une boutique en ligne tunisienne. "
    "Réponds en français, brièvement et poliment. "
    "N'invente jamais d'information. "
    "Pour une question sur les règles de la boutique, "
    "utilise chercher_politique. "
    "Avant toute annulation ou tout retour : "
    "consulte la commande, vérifie la règle, "
    "puis agis seulement si la règle est respectée ; "
    "sinon explique pourquoi."
)


# =========================================================
# MODELE GEMINI
# =========================================================

llm = ChatGoogleGenerativeAI(
    model=os.getenv(
        "MODELE",
        "gemini-3.5-flash"
    ),
    temperature=0
).bind_tools(
    OUTILS
)


# =========================================================
# NOEUD AGENT
# =========================================================

def agent(
    state: MessagesState
):

    reponse = llm.invoke(
        [
            SystemMessage(PROMPT)
        ]
        + state["messages"]
    )

    return {
        "messages": [
            reponse
        ]
    }


# =========================================================
# ROUTAGE
# =========================================================

def router(
    state: MessagesState
):

    appels = (
        state["messages"][-1]
        .tool_calls
    )


    # Aucun outil demandé :
    # la réponse est terminée
    if not appels:

        return END


    # Au moins un outil sensible :
    # validation humaine obligatoire
    if any(
        a["name"] in SENSIBLES
        for a in appels
    ):

        return "validation"


    # Sinon outils de lecture normaux
    return "outils"


# =========================================================
# VALIDATION HUMAINE
# =========================================================

def validation(
    state: MessagesState
):

    appels = (
        state["messages"][-1]
        .tool_calls
    )


    # Le graphe se met en pause ici
    decision = interrupt(
        [
            f"{a['name']} {a['args']}"
            for a in appels
        ]
    )


    # -----------------------------------------------------
    # ACTION VALIDEE
    # -----------------------------------------------------

    if (
        decision
        .strip()
        .lower()
        in (
            "oui",
            "o"
        )
    ):

        return Command(
            goto="outils"
        )


    # -----------------------------------------------------
    # ACTION REFUSEE
    # -----------------------------------------------------

    refus = [
        ToolMessage(
            "Action refusée par le responsable.",
            tool_call_id=a["id"],
            name=a["name"],
            status="error"
        )

        for a in appels
    ]


    return Command(
        goto="agent",
        update={
            "messages": refus
        }
    )


# =========================================================
# GRAPHE LANGGRAPH
# =========================================================

graphe = StateGraph(
    MessagesState
)


# ---------------------------------------------------------
# NOEUDS
# ---------------------------------------------------------

graphe.add_node(
    "agent",
    agent
)

graphe.add_node(
    "validation",
    validation,
    destinations=(
        "outils",
        "agent"
    )
)

graphe.add_node(
    "outils",
    ToolNode(
        OUTILS
    )
)


# ---------------------------------------------------------
# EDGES
# ---------------------------------------------------------

graphe.add_edge(
    START,
    "agent"
)


graphe.add_conditional_edges(
    "agent",
    router,
    [
        "outils",
        "validation",
        END
    ]
)


graphe.add_edge(
    "outils",
    "agent"
)


# =========================================================
# COMPILATION
# =========================================================

app = graphe.compile(
    checkpointer=InMemorySaver()
)