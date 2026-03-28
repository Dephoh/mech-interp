"""
Download all Riftbound cards from the official Riot card gallery.
Saves card metadata to cards/cards.json and images to cards/images/.

Source: https://riftbound.leagueoflegends.com/en-us/card-gallery/
The page is SSR'd with prefetchAll=True, so all card data is embedded
in the __NEXT_DATA__ script tag under page.blades[riftboundCardGallery].cards.items
"""

import json
import os
import time
import sys

import requests
from bs4 import BeautifulSoup

GALLERY_URL = "https://riftbound.leagueoflegends.com/en-us/card-gallery/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CARDS_JSON = os.path.join(OUTPUT_DIR, "cards.json")


def fetch_all_cards() -> list[dict]:
    """Fetch all cards embedded in the SSR page HTML."""
    all_cards: dict[str, dict] = {}

    # The main page has prefetchAll=True — all cards are in the HTML
    pages_to_fetch = [GALLERY_URL]

    for url in pages_to_fetch:
        print(f"Fetching {url} ...")
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        script = soup.find("script", {"id": "__NEXT_DATA__"})
        if not script:
            print(f"  WARNING: No __NEXT_DATA__ found at {url}")
            continue

        data = json.loads(script.string)
        blades = data["props"]["pageProps"]["page"]["blades"]

        gallery_blade = next(
            (b for b in blades if b.get("type") == "riftboundCardGallery"),
            None,
        )
        if not gallery_blade:
            print("  WARNING: No riftboundCardGallery blade found")
            continue

        cards_data = gallery_blade.get("cards", {})
        items = cards_data.get("items", [])
        async_meta = cards_data.get("async", {}).get("metadata", {})
        total = async_meta.get("totalItems", "?")

        print(f"  Got {len(items)} cards (total reported: {total})")

        for card in items:
            uid = card.get("id") or card.get("publicCode") or card.get("name", "")
            if uid:
                all_cards[uid] = card

    return list(all_cards.values())


def get_image_url(card: dict) -> str | None:
    card_image = card.get("cardImage")
    if not card_image:
        return None
    if isinstance(card_image, str):
        return card_image
    return card_image.get("url")


def get_card_id(card: dict) -> str:
    return (
        card.get("id")
        or card.get("publicCode")
        or str(card.get("collectorNumber", ""))
        or card.get("name", "unknown")
    ).replace("/", "-").replace("\\", "-")


def download_images(card_list: list[dict]) -> tuple[int, int]:
    session = requests.Session()
    ok = 0
    fail = 0

    for i, card in enumerate(card_list, 1):
        card_id = get_card_id(card)
        out_path = os.path.join(IMAGES_DIR, f"{card_id}.png")

        if os.path.exists(out_path):
            ok += 1
            continue

        image_url = get_image_url(card)
        if not image_url:
            print(f"  WARN: No image URL for {card_id} ({card.get('name', '?')})")
            fail += 1
            continue

        try:
            resp = session.get(image_url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            ok += 1
        except Exception as e:
            print(f"  WARN: Failed to download {card_id}: {e}")
            fail += 1

        if i % 50 == 0:
            print(f"  {i}/{len(card_list)} images ({ok} OK, {fail} failed)...")

        time.sleep(0.05)

    return ok, fail


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Step 1: Collect all card metadata
    print("=== Collecting card data ===")
    card_list = fetch_all_cards()
    print(f"\nTotal unique cards collected: {len(card_list)}")

    if not card_list:
        print("ERROR: No cards found. Exiting.")
        sys.exit(1)

    # Step 2: Save metadata
    with open(CARDS_JSON, "w", encoding="utf-8") as f:
        json.dump(card_list, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata -> {CARDS_JSON}")

    # Print set breakdown
    sets: dict[str, int] = {}
    for card in card_list:
        set_id = card.get("set", {}).get("value", {}).get("id", "?")
        sets[set_id] = sets.get(set_id, 0) + 1
    print("Cards per set:", sets)

    # Step 3: Download images
    print(f"\n=== Downloading {len(card_list)} card images ===")
    ok, fail = download_images(card_list)

    print(f"\nDone!")
    print(f"  {ok} images saved to {IMAGES_DIR}")
    if fail:
        print(f"  {fail} images failed")
    print(f"  {len(card_list)} cards in {CARDS_JSON}")


if __name__ == "__main__":
    main()
