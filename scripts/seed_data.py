"""
Seed script — populates the local FAISS index with curated Sydney insider tips.

Usage (run from the project root):
    python scripts/seed_data.py

Requires OPENAI_API_KEY to be set in .env (used for embedding the documents).
"""

import sys
import os

# Ensure `app` package is importable when the script is run from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from app.rag.retriever import save_documents_to_store


SYDNEY_INSIDER_DOCS = [
    Document(
        page_content=(
            "Fisher Library at USYD is open until midnight during semester but gets absolutely packed after 6 PM — "
            "noise level rises sharply from 7 PM as stressed students fill every seat. "
            "The move: head to the Law Library (on Fisher Road) instead. It's open to all USYD students, consistently "
            "quieter, has better natural light from the north-facing windows, and most people don't know about it. "
            "Arrive before 9 AM to claim a window seat overlooking the quad. "
            "Skip the ground-floor vending machines — overpriced and the coffee is bad. "
            "The nearest decent flat white is Taste Baguette on City Road, about a 7-minute walk."
        ),
        metadata={"source": "aura_seed", "category": "study", "location": "USYD, Camperdown"},
    ),
    Document(
        page_content=(
            "Victoria Park, wedged between USYD and Broadway Shopping Centre, is the inner west's most underrated "
            "weekday lunch spot. The eastern strip along City Road is fully exposed to afternoon sun — great for tanning, "
            "brutal for reading a laptop screen after 1 PM. "
            "The shaded zone under the Moreton Bay fig trees on the western side is the real gem: cool, quiet, and "
            "almost always has bench space even at peak lunch hour. "
            "Critical warning: the ibises — locals call them 'bin chickens' — are bold and WILL take your food if "
            "it's unattended for even a moment. Keep your bag zipped. "
            "Best visit windows: Tuesday and Thursday 11:30 AM–1 PM. "
            "The free gas BBQs near the playground work well for group hangs; allow 20 minutes warm-up time."
        ),
        metadata={"source": "aura_seed", "category": "outdoor", "location": "Camperdown / Glebe"},
    ),
    Document(
        page_content=(
            "Campos Coffee on King Street, Newtown is the best study café in the inner west — but only if you follow "
            "the unwritten rules. Arrive before 9:30 AM on weekdays; seating fills up by 10 AM and staff quietly "
            "discourage laptop campers lingering past 11. "
            "The Wi-Fi password is printed on your receipt and rotates weekly. "
            "Fastest signal: grab a seat near the back wall, away from the espresso machine interference. "
            "Order the single-origin pour-over rather than the filter — the filter batch can sit for hours in the "
            "afternoon and loses its brightness. "
            "Noise level: pleasant conversational buzz until midday, then it gets loud and echoey. "
            "Transport: parking on King St is a nightmare. Take the 422 bus or walk 10 minutes from Newtown station. "
            "Vibe: creative, independent, slightly hipster — great for focused solo work or a casual first meeting."
        ),
        metadata={"source": "aura_seed", "category": "cafe", "location": "Newtown, King St"},
    ),
    Document(
        page_content=(
            "The Bondi to Coogee coastal walk (6 km, roughly 90 minutes at a relaxed pace) is one of Sydney's iconic "
            "routes — but most people do it backwards. Start from Coogee and walk north toward Bondi to avoid the "
            "tourist crush; about 90% of walkers go the other direction, so you'll have the trail noticeably quieter. "
            "Best light: 7–9 AM heading north gives you the sun at your back for photos. "
            "Bronte ocean pool (halfway) is the highlight — less crowded than Bondi Icebergs, free entry, and the "
            "water is calmer. "
            "Avoid the walk between 11 AM–2 PM in summer: zero shade, full UV exposure, brutal. "
            "Hidden gem: Gordons Bay underwater nature trail just past Coogee — bring a snorkel on calm days, it's "
            "genuinely spectacular and almost no tourists find it. "
            "Tamarama ('Glamarama') Beach is tiny and locals-only — strong rip current, no flags on weekdays, "
            "don't swim there unless conditions are perfect."
        ),
        metadata={"source": "aura_seed", "category": "outdoor", "location": "Bondi to Coogee"},
    ),
    Document(
        page_content=(
            "Carriageworks Farmers Market runs every Saturday, 8 AM–1 PM, on Wilson Street in Eveleigh. "
            "It's Sydney's best market — but it turns chaotic after 9:30 AM. "
            "The secret: arrive right at 8 AM and walk straight to the small-producer stalls at the back. "
            "These aren't on the Instagram maps, they sell out first, and they're where the serious regulars go. "
            "The paella stall in the centre aisle is gone by 10:30 AM. "
            "The Blue Mountains honey stall near the left entrance does free tastings all morning — don't skip it. "
            "Bring cash; at least half the stalls are cash-only. "
            "Transport: Redfern station is a 5-minute walk, or the L2 light rail to Jubilee Park stop. "
            "Street parking on Wilson St is 2-hour limited. "
            "Vibe: loud, community-heavy, genuinely dog-friendly, great for a solo Saturday morning or a relaxed "
            "first date — the energy is warm and unhurried before 9 AM."
        ),
        metadata={"source": "aura_seed", "category": "market", "location": "Eveleigh / Redfern"},
    ),
]


def main() -> None:
    print(f"Seeding {len(SYDNEY_INSIDER_DOCS)} Sydney insider tip documents...")

    try:
        chunks_stored = save_documents_to_store(SYDNEY_INSIDER_DOCS)
    except Exception as exc:
        print(f"ERROR: Seeding failed — {exc}")
        print("Make sure OPENAI_API_KEY is set in your .env file.")
        sys.exit(1)

    print(f"Done. {chunks_stored} chunks embedded and saved to the FAISS index.")
    print("You can now start the API: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
