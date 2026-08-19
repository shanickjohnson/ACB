"""
Language and canned-message tables shared across the API layer and every
agent graph node. Previously duplicated/scattered:
  - SUPPORTED_LANGUAGES, REFUSAL_MESSAGE(S), FALLBACK_MESSAGES lived in app.py
  - REFUSAL_MESSAGES was re-declared (identically) in guardrails.py
  - HANDOFF_NOTE lived in domain_agents.py
  - ESCALATION_MESSAGES lived in escalation.py
Consolidated here so adding a language is a one-file change.
"""

SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "native_name": "English",
        "flag": "🇬",
        "instruction": "Respond in English.",
    },
    "fr": {
        "name": "French",
        "native_name": "Français",
        "flag": "🇫🇷",
        "instruction": "Respond entirely in French (Français). Use natural, idiomatic French.",
    },
    "es": {
        "name": "Spanish",
        "native_name": "Español",
        "flag": "🇪",
        "instruction": "Respond entirely in Spanish (Español). Use natural, idiomatic Spanish.",
    },
    "nl": {
        "name": "Dutch",
        "native_name": "Nederlands",
        "flag": "🇳🇱",
        "instruction": "Respond entirely in Dutch (Nederlands). Use natural, idiomatic Dutch.",
    },
}
DEFAULT_LANGUAGE = "en"

REFUSAL_MESSAGE = (
    "I can't help with that request. I'm here to answer general questions about "
    "loans, accounts, cards, branch locations, and hours."
)

REFUSAL_MESSAGES = {
    "en": REFUSAL_MESSAGE,
    "fr": "Je ne peux pas répondre à cette demande. Je suis là pour répondre aux questions générales sur les prêts, les comptes, les cartes, les agences et les horaires.",
    "es": "No puedo ayudar con esa solicitud. Estoy aquí para responder preguntas generales sobre préstamos, cuentas, tarjetas, sucursales y horarios.",
    "nl": "Ik kan niet helpen met dat verzoek. Ik ben hier om algemene vragen te beantwoorden over leningen, rekeningen, kaarten, filialen en openingstijden.",
}

FALLBACK_MESSAGES = {
    "rate_limited": {
        "en": "I'm getting a lot of questions right now and can't keep up — please try again in a minute.",
        "fr": "Je reçois beaucoup de questions en ce moment — veuillez réessayer dans une minute.",
        "es": "Estoy recibiendo muchas preguntas en este momento; inténtelo de nuevo en un minuto.",
        "nl": "Ik krijg momenteel veel vragen en kan het niet bijhouden — probeer het over een minuut opnieuw.",
    },
    "error": {
        "en": "Sorry, I'm having trouble thinking right now. Try again in a moment!",
        "fr": "Désolé, j'ai du mal à réfléchir en ce moment. Réessayez dans un instant !",
        "es": "Lo siento, tengo problemas para pensar en este momento. ¡Inténtalo de nuevo en un momento!",
        "nl": "Sorry, ik heb momenteel moeite met nadenken. Probeer het zo opnieuw!",
    },
}

HANDOFF_NOTE = {
    "en": "For anything involving your specific account, please log into Online Banking or call 1-800-222-2265.",
    "fr": "Pour toute question concernant votre compte, veuillez vous connecter à Online Banking ou appeler le 1-800-222-2265.",
    "es": "Para cualquier asunto relacionado con su cuenta específica, inicie sesión en Online Banking o llame al 1-800-222-2265.",
    "nl": "Voor alles wat met uw specifieke rekening te maken heeft, logt u in op Online Banking of belt u 1-800-222-2265.",
}

ESCALATION_MESSAGES = {
    "en": "I'm connecting you with a team member who can help further. Someone will follow up shortly, or you can call 1-800-222-2265 now.",
    "fr": "Je vous mets en relation avec un membre de l'équipe qui pourra vous aider davantage. Vous pouvez aussi appeler le 1-800-222-2265.",
    "es": "Le voy a poner en contacto con un miembro del equipo que puede ayudarle más. También puede llamar al 1-800-222-2265.",
    "nl": "Ik verbind u door met een medewerker die verder kan helpen. U kunt ook bellen naar 1-800-222-2265.",
}
