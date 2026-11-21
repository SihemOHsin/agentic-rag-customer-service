
"""Interface Streamlit de l'agent service client.

Local :
    uv run indexer.py
    uv run streamlit run app_streamlit.py
"""

import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from indexer import ouvrir_base, main as indexer_politique
from outils_base import creer_base


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Service client boutique",
    page_icon="🛒"
)


# =========================================================
# INITIALISATION RAG + SQL
# =========================================================

@st.cache_resource
def preparer_ressources():
    """Prépare automatiquement ChromaDB et SQLite."""

    vector_store_temp = ouvrir_base()

    # Sur Streamlit Cloud, chroma_db n'existe pas au premier démarrage
    if not vector_store_temp.get(limit=1)["ids"]:
        indexer_politique()

    creer_base()


preparer_ressources()


# Import après préparation de ChromaDB
from agent_boutique import app, vector_store  # noqa: E402


# =========================================================
# SESSION
# =========================================================

def nouvelle_conversation():

    st.session_state.thread_id = str(
        uuid.uuid4()
    )

    st.session_state.messages = []

    st.session_state.actions = None


if "thread_id" not in st.session_state:
    nouvelle_conversation()


# =========================================================
# EXECUTION DU GRAPHE
# =========================================================

def lancer(entree):
    """Exécute le graphe et renvoie
    les étapes, la réponse et les actions à valider.
    """

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    etapes = []
    reponse = None

    for etape in app.stream(
        entree,
        config,
        stream_mode="updates"
    ):

        if "__interrupt__" in etape:

            return (
                etapes,
                reponse,
                etape["__interrupt__"][0].value
            )

        for sortie in etape.values():

            for msg in (
                sortie or {}
            ).get("messages", []):

                if msg.type == "tool":

                    etapes.append(
                        f"← {msg.name} : {msg.content}"
                    )

                elif msg.tool_calls:

                    etapes += [
                        f"→ {a['name']} {a['args']}"
                        for a in msg.tool_calls
                    ]

                else:

                    reponse = msg.text

    return etapes, reponse, None


def traiter(entree):
    """Lance le graphe et ajoute
    le résultat à l'historique.
    """

    with st.spinner("Thought..."):

        etapes, reponse, actions = lancer(
            entree
        )

    if etapes or reponse:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reponse or "",
                "etapes": etapes
            }
        )

    st.session_state.actions = actions

    st.rerun()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🛒 Boutique")

    st.caption(
        "Agent LangGraph : RAG + SQL + validation humaine"
    )

    if st.button(
        "Nouvelle conversation",
        use_container_width=True
    ):

        creer_base()

        nouvelle_conversation()

        st.rerun()

    st.subheader(
        "Questions à essayer"
    )

    st.markdown(
        "- Quels sont les délais de livraison ?\n"
        "- Où en est la commande 1002 ?\n"
        "- Annule ma commande 1001\n"
        "- J'ai reçu la commande 1003 mais la souris ne marche pas, "
        "je peux la retourner ?\n"
        "- Je veux retourner la commande 1004, le casque ne me plaît plus."
    )


# =========================================================
# CONVERSATION
# =========================================================

st.title("Service client")


for m in st.session_state.messages:

    with st.chat_message(
        m["role"]
    ):

        if m.get("etapes"):

            with st.expander(
                "Étapes de l'agent"
            ):

                st.code(
                    "\n".join(
                        m["etapes"]
                    ),
                    language=None
                )

        if m["content"]:

            st.markdown(
                m["content"]
            )


# =========================================================
# VALIDATION HUMAINE
# =========================================================

if st.session_state.actions:

    st.warning(
        "⚠️ Validation requise :\n\n"
        + "\n".join(
            f"- `{a}`"
            for a in st.session_state.actions
        )
    )

    col1, col2 = st.columns(2)

    if col1.button(
        "✅ Valider",
        use_container_width=True
    ):

        traiter(
            Command(
                resume="oui"
            )
        )

    if col2.button(
        "❌ Refuser",
        use_container_width=True
    ):

        traiter(
            Command(
                resume="non"
            )
        )


# =========================================================
# QUESTION
# =========================================================

question = st.chat_input(
    "Votre question...",
    disabled=bool(
        st.session_state.actions
    )
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    traiter(
        {
            "messages": [
                HumanMessage(
                    question
                )
            ]
        }
    )