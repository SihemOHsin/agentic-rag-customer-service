"""Graphe exposé à LangGraph Studio.

Le serveur LangGraph gère lui-même la persistance.
On compile donc une copie du graphe sans InMemorySaver.

L'agent utilisé par Streamlit reste inchangé.
"""

from agent_boutique import graphe


# Graphe destiné uniquement à LangGraph Studio
agent_boutique = graphe.compile()