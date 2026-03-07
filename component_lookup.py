import json
import csv
import os
import re
import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


# ── STEP 1: Try fetching from DigiKey search ─────────────────────────────────
def fetch_from_web(component_name):
    """
    Tries to scrape basic info from DigiKey search results.
    Returns a dict or None if it fails.
    """
    try:
        search_url = f"https://www.digikey.com/en/products/result?keywords={component_name.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=headers, timeout=6)

        if response.status_code == 200:
            # Try to extract a price from the page text using regex
            price_match = re.search(r'\$(\d+\.\d+)', response.text)
            price = float(price_match.group(1)) if price_match else 1.00

            print(f"  🌐 Web fetch succeeded for '{component_name}' (DigiKey)")
            return {
                "part_number": component_name.upper().replace(" ", "-"),
                "voltage": 5.0,   # Default - web scrape can't always get this
                "price_usd": price,
                "package": "THT",
                "in_stock": True,
                "digikey_url": search_url
            }
    except Exception as e:
        print(f"  ⚠️  Web fetch failed for '{component_name}': {e}")

    return None


# ── STEP 2: Fallback — Ask Gemini AI ─────────────────────────────────────────
def fetch_from_gemini(component_name):
    """
    Uses Gemini AI to generate component details as JSON.
    Returns a dict or None if it fails.
    """
    try:
        model = genai.GenerativeModel(
            model_name="gemini-3.1-flash-lite-preview",
            system_instruction="""
            You are an electronics component database.
            When given a component name, return ONLY raw JSON with these exact fields:
            {
              "part_number": "...",
              "voltage": 3.3,
              "price_usd": 0.50,
              "package": "THT or SMD",
              "in_stock": true,
              "digikey_url": "https://www.digikey.com/..."
            }
            Return ONLY raw JSON. No markdown, no explanation.
            """
        )

        response = model.generate_content(f"Give me component details for: {component_name}")
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        print(f"  🤖 Gemini fallback succeeded for '{component_name}'")
        return data

    except Exception as e:
        print(f"  ❌ Gemini fallback also failed for '{component_name}': {e}")
        return None


# ── STEP 3: Auto-fetch unknown components ────────────────────────────────────
def auto_fetch_component(component_name):
    """
    Tries web first, then Gemini. Returns best available data.
    """
    print(f"\n🔍 Auto-fetching unknown component: '{component_name}'")

    # Try web first
    result = fetch_from_web(component_name)
    if result:
        return result

    # Fallback to Gemini
    result = fetch_from_gemini(component_name)
    if result:
        return result

    # Last resort - return a placeholder so BOM still works
    print(f"  ⚠️  Could not fetch '{component_name}' — using placeholder.")
    return {
        "part_number": "UNKNOWN",
        "voltage": 0,
        "price_usd": 0.00,
        "package": "UNKNOWN",
        "in_stock": False,
        "digikey_url": "N/A"
    }


# ── STEP 4: Enrich components ────────────────────────────────────────────────
def enrich_components(component_names):
    """
    Looks up each component. Auto-fetches any unknown ones.
    Returns enriched list (temporary, not saved to DB).
    """
    enriched = []

    for name in component_names:
        if name in COMPONENTS:
            # Found in static DB
            row = COMPONENTS[name].copy()
            row["name"] = name
            enriched.append(row)
        else:
            # Not in DB — auto-fetch it
            fetched = auto_fetch_component(name)
            fetched["name"] = name
            enriched.append(fetched)

    return enriched


# ── STEP 5: Generate BOM ─────────────────────────────────────────────────────
def generate_bom(json_path="output.json", csv_path="BOM.csv"):
    """
    Reads output.json, enriches all components (auto-fetching unknowns),
    and writes BOM.csv.
    """
    with open(json_path) as f:
        data = json.load(f)

    names = [c["name"] for c in data["components"]]
    enriched = enrich_components(names)

    fields = ["name", "part_number", "voltage", "price_usd", "package", "in_stock", "digikey_url"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(enriched)

    total = round(sum(c["price_usd"] for c in enriched), 2)
    print(f"\n✅ BOM.csv written — {len(enriched)} components, estimated cost: ${total}")

    unknown = [c["name"] for c in enriched if c["part_number"] == "UNKNOWN"]
    if unknown:
        print(f"⚠️  Could not resolve: {unknown}")

    return enriched


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists("output.json"):
        print("📝 No output.json found — creating a test one...")
        test_data = {
            "components": [
                {"name": "ESP32"},
                {"name": "Arduino-Uno-R3"},
                {"name": "Red-LED"},
                {"name": "Resistor-220R"},
                {"name": "USB-5V-Supply"},
            ]
        }
        with open("output.json", "w") as f:
            json.dump(test_data, f, indent=2)

    generate_bom()
