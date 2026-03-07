import os
import json
import csv
import re
import time
import requests
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv


# ==============================================================
#  STEP 1: SETUP
# ==============================================================
load_dotenv()
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

MODEL = "gemini-2.0-flash-lite"

NETLIST_SYSTEM = """
You are a Senior Hardware Engineer (PCB Specialist).
Your job is to parse descriptions into a JSON Netlist.
Rules:
- Return ONLY raw JSON. No markdown blocks (no ```).
- Include: MCU, Power Stage (LDO/Buck), Motor Drivers, and Passives.
- Format: {"components": [{"id": "U1", "name": "ESP32-S3", "type": "MCU", "voltage": 3.3}], "connections": [{"from": "LiPo", "to": "ESP32"}]}
- Each component MUST include: id, name, type, and voltage fields.
- connections must use "from" and "to" keys matching component names.
- Output ONLY raw JSON. Do not include backtick markdown tags or any introductory text.
"""

SAFETY_SYSTEM = """
You are a Senior Electrical Safety Engineer.
You will be given a circuit JSON with components and connections.

Your job:
1. Identify what type of project this is (e.g. traffic light, motor controller, IoT sensor, etc.)
2. Generate and run 5-8 safety checks that are SPECIFIC and RELEVANT to that project type.
3. For each check, evaluate it against the actual components and connections in the JSON.

Return ONLY a raw JSON array like this (no markdown, no explanation):
[
  {
    "check": "Check Name",
    "status": "PASS or FAIL or WARN or SKIP",
    "detail": "Specific explanation referencing actual components in the design."
  }
]

Status rules:
- PASS = everything looks good
- FAIL = serious issue that must be fixed
- WARN = potential issue worth noting
- SKIP = check not applicable to this design

Return ONLY the raw JSON array. No markdown. No extra text.
"""

FIX_SYSTEM = """
You are a Senior Hardware Engineer and circuit repair specialist.
You will be given a circuit JSON and a list of FAIL safety checks.

Your job:
- Fix EVERY fail by modifying the circuit JSON
- Add missing components and connections
- Do NOT remove any existing components or connections
- Keep the same JSON format

Return ONLY raw JSON in this exact format (no markdown, no extra text):
{
  "fixed_circuit": { ...the full updated circuit JSON... },
  "fixes_applied": ["Fix 1: ...", "Fix 2: ..."]
}
"""

LOOKUP_SYSTEM = """
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


# ==============================================================
#  GEMINI RATE-LIMIT SAFE CALLER
# ==============================================================
def gemini_call_with_retry(system_instruction, prompt, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                )
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                wait_match = re.search(r'seconds:\s*(\d+)', error_str)
                wait_seconds = int(wait_match.group(1)) + 5 if wait_match else 60
                print(f"\n Rate limit hit. Waiting {wait_seconds}s before retry (attempt {attempt}/{max_retries})...")
                time.sleep(wait_seconds)
            else:
                raise e
    raise Exception(f"Gemini call failed after {max_retries} retries.")


# ==============================================================
#  STEP 2: SAMPLE DATA FALLBACK
# ==============================================================
SAMPLE_JSON = {
    "components": [
        {"id": "U1", "name": "Arduino-Uno-R3", "type": "MCU",         "voltage": 5.0},
        {"id": "D1", "name": "Red-LED",         "type": "LED",         "voltage": 2.0},
        {"id": "D2", "name": "Yellow-LED",      "type": "LED",         "voltage": 2.1},
        {"id": "D3", "name": "Green-LED",       "type": "LED",         "voltage": 2.2},
        {"id": "R1", "name": "Resistor-220R",   "type": "Resistor",    "voltage": 0},
        {"id": "R2", "name": "Resistor-220R",   "type": "Resistor",    "voltage": 0},
        {"id": "R3", "name": "Resistor-220R",   "type": "Resistor",    "voltage": 0},
        {"id": "P1", "name": "USB-5V-Supply",   "type": "PowerSupply", "voltage": 5.0},
    ],
    "connections": [
        {"from": "USB-5V-Supply",  "to": "Arduino-Uno-R3"},
        {"from": "Arduino-Uno-R3", "to": "Red-LED"},
        {"from": "Arduino-Uno-R3", "to": "Yellow-LED"},
        {"from": "Arduino-Uno-R3", "to": "Green-LED"},
        {"from": "Red-LED",        "to": "Resistor-220R"},
        {"from": "Yellow-LED",     "to": "Resistor-220R"},
        {"from": "Green-LED",      "to": "Resistor-220R"},
    ]
}


# ==============================================================
#  STEP 3: STATIC COMPONENT DATABASE
# ==============================================================
COMPONENTS = {
    "ESP32": {"part_number": "ESP32-WROOM-32E", "voltage": 3.3, "price_usd": 3.50, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/espressif-systems/ESP32-WROOM-32E/11613142"},
    "L298N": {"part_number": "L298N", "voltage": 5, "price_usd": 1.80, "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/stmicroelectronics/L298N/585918"},
    "TP4056": {"part_number": "TP4056-SOT25", "voltage": 4.2, "price_usd": 0.30, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/tc-charger/TP4056/7353588"},
    "LiPo Battery": {"part_number": "PRT-13854", "voltage": 3.7, "price_usd": 9.95, "package": "THT", "in_stock": True, "digikey_url": "https://www.sparkfun.com/products/13854"},
    "AMS1117-3.3": {"part_number": "AMS1117-3.3", "voltage": 3.3, "price_usd": 0.25, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/advanced-monolithic-systems-inc/AMS1117-3-3/5010163"},
    "Decoupling Capacitor": {"part_number": "C0402C104K5RACTU", "voltage": 10, "price_usd": 0.05, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/kemet/C0402C104K5RACTU/411388"},
    "ESD Diode": {"part_number": "PRTR5V0U2X", "voltage": 5, "price_usd": 0.40, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/nexperia-usa-inc/PRTR5V0U2X/1177477"},
}


# ==============================================================
#  STEP 4: GENERATE JSON WITH GEMINI
# ==============================================================
def generate_and_save_json(user_request):
    try:
        print(f"Sending to Gemini: '{user_request}'\n")
        text = gemini_call_with_retry(NETLIST_SYSTEM, user_request)
        with open("output.json", "w") as f:
            f.write(text)
        print("Success! Design logic saved to output.json.")
        return True
    except Exception as e:
        print(f"Error: {e}. Make sure your API key is in the .env file.")
        return False


# ==============================================================
#  STEP 5: LOAD JSON
# ==============================================================
def load_json():
    if os.path.exists("output.json"):
        print("Found output.json - using real data!\n")
        with open("output.json", "r") as f:
            return json.load(f)
    else:
        print("output.json not found - using sample data for now.\n")
        return SAMPLE_JSON


# ==============================================================
#  STEP 6: SMART AI-POWERED SAFETY CHECKS
# ==============================================================
def run_checks(data):
    print("\n Asking Gemini to analyse the circuit and run smart safety checks...\n")
    try:
        circuit_summary = json.dumps(data, indent=2)
        raw = gemini_call_with_retry(SAFETY_SYSTEM, f"Analyse this circuit and run smart safety checks:\n\n{circuit_summary}")
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        results = json.loads(raw)
        print(f" Gemini generated {len(results)} smart safety checks for this project.")
        return results
    except Exception as e:
        print(f"  Smart checks failed ({e}). Returning error result.")
        return [{"check": "AI Safety Check", "status": "WARN", "detail": f"Could not run smart checks: {e}"}]


# ==============================================================
#  STEP 7: PRINT & SAVE SAFETY REPORT
# ==============================================================
def print_and_save_report(results, data):
    lines = []
    lines.append("=" * 60)
    lines.append("  KEVIN THE WIZARD'S ELECTRICAL SAFETY REPORT")
    lines.append("  Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("=" * 60)
    lines.append("")
    lines.append("COMPONENTS FOUND:")
    for c in data.get("components", []):
        lines.append("  - " + c["name"] + " (" + c.get("type", "?") + ") - " + str(c.get("voltage", "?")) + "V")
    lines.append("")
    lines.append("CONNECTIONS:")
    for conn in data.get("connections", []):
        lines.append("  " + conn["from"] + "  -->  " + conn["to"])
    lines.append("")
    lines.append("-" * 60)
    lines.append("SAFETY CHECK RESULTS:")
    lines.append("-" * 60)

    pass_count = 0
    fail_count = 0
    for r in results:
        lines.append("")
        lines.append("[" + r["status"] + "]  " + r["check"])
        lines.append("   -> " + r["detail"])
        if r["status"] == "PASS":
            pass_count += 1
        elif r["status"] == "FAIL":
            fail_count += 1

    lines.append("")
    lines.append("=" * 60)
    lines.append("  SUMMARY: " + str(pass_count) + " passed,  " + str(fail_count) + " failed,  "
                 + str(len(results) - pass_count - fail_count) + " warnings/skipped")
    if fail_count == 0:
        lines.append("  OVERALL: DESIGN LOOKS SAFE TO PROCEED!")
    else:
        lines.append("  OVERALL: " + str(fail_count) + " ISSUE(S) NEED FIXING BEFORE MANUFACTURE.")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)
    with open("safety_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("\nReport saved to safety_report.txt")


# ==============================================================
#  STEP 8: AUTO-FETCH UNKNOWN COMPONENTS
# ==============================================================
def fetch_from_web(component_name):
    try:
        search_url = f"https://www.digikey.com/en/products/result?keywords={component_name.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=headers, timeout=6)
        if response.status_code == 200:
            price_match = re.search(r'\$(\d+\.\d+)', response.text)
            price = float(price_match.group(1)) if price_match else 1.00
            print(f"   Web fetch succeeded for '{component_name}' (DigiKey)")
            return {"part_number": component_name.upper().replace(" ", "-"), "voltage": 5.0, "price_usd": price, "package": "THT", "in_stock": True, "digikey_url": search_url}
    except Exception as e:
        print(f"    Web fetch failed for '{component_name}': {e}")
    return None


def fetch_from_gemini(component_name):
    try:
        raw = gemini_call_with_retry(LOOKUP_SYSTEM, f"Give me component details for: {component_name}")
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        print(f"   Gemini fallback succeeded for '{component_name}'")
        return data
    except Exception as e:
        print(f"   Gemini fallback also failed for '{component_name}': {e}")
        return None


def auto_fetch_component(component_name):
    print(f"\n Auto-fetching unknown component: '{component_name}'")
    result = fetch_from_web(component_name)
    if result:
        return result
    result = fetch_from_gemini(component_name)
    if result:
        return result
    print(f"    Could not fetch '{component_name}' - using placeholder.")
    return {"part_number": "UNKNOWN", "voltage": 0, "price_usd": 0.00, "package": "UNKNOWN", "in_stock": False, "digikey_url": "N/A"}


# ==============================================================
#  STEP 9: GENERATE BOM
# ==============================================================
def generate_bom(data, csv_path="BOM.csv"):
    print("\n" + "=" * 60)
    print("  GENERATING BILL OF MATERIALS (BOM)")
    print("=" * 60)

    names = [c["name"] for c in data.get("components", [])]
    enriched = []

    for name in names:
        if name in COMPONENTS:
            row = COMPONENTS[name].copy()
            row["name"] = name
            enriched.append(row)
            print(f"   Found in DB: '{name}'")
        else:
            fetched = auto_fetch_component(name)
            fetched["name"] = name
            enriched.append(fetched)

    fields = ["name", "part_number", "voltage", "price_usd", "package", "in_stock", "digikey_url"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(enriched)

    total = round(sum(c["price_usd"] for c in enriched), 2)
    print(f"\n BOM.csv written - {len(enriched)} components, estimated cost: ${total}")

    unknown = [c["name"] for c in enriched if c["part_number"] == "UNKNOWN"]
    if unknown:
        print(f"  Could not resolve: {unknown}")

    return enriched


# ==============================================================
#  STEP 10: AUTO-FIX FAILS
# ==============================================================
def auto_fix(data, results):
    fails = [r for r in results if r["status"] == "FAIL"]

    if not fails:
        print("\n No FAILs found - nothing to auto-fix!")
        return data

    print("\n" + "=" * 60)
    print("  AUTO-FIX: GEMINI IS FIXING THE FAILS...")
    print("=" * 60)

    for f in fails:
        print(f"   Fixing: [{f['check']}]  {f['detail'][:60]}...")

    try:
        prompt = f"Circuit JSON:\n{json.dumps(data, indent=2)}\n\nFAIL checks to fix:\n{json.dumps(fails, indent=2)}\n\nFix all the FAILs and return the updated circuit with explanation."
        raw = gemini_call_with_retry(FIX_SYSTEM, prompt)
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        fixed_circuit = result.get("fixed_circuit", data)
        fixes_applied = result.get("fixes_applied", [])

        print("\n FIXES APPLIED:")
        print("-" * 60)
        for i, fix in enumerate(fixes_applied, 1):
            print(f"  {i}. {fix}")

        with open("output.json", "w") as f:
            json.dump(fixed_circuit, f, indent=2)

        print("\n output.json overwritten with fixed circuit design.")
        return fixed_circuit

    except Exception as e:
        print(f"\n Auto-fix failed: {e}")
        return data


# ==============================================================
#  MAIN PIPELINE
# ==============================================================
if __name__ == "__main__":
    print("Kevin the Wizard's Safety Checker starting...\n")

    user_request = "traffic signal circuit using arduino uno r3"
    MAX_FIX_ROUNDS = 5

    generate_and_save_json(user_request)
    data = load_json()
    results = run_checks(data)
    print_and_save_report(results, data)

    fix_round = 0
    while True:
        fails = [r for r in results if r["status"] == "FAIL"]

        if not fails:
            print("\n All FAILs resolved! Design is clean.")
            break

        if fix_round >= MAX_FIX_ROUNDS:
            print(f"\n  Reached max fix rounds ({MAX_FIX_ROUNDS}). Stopping auto-fix.")
            print(f"   Remaining FAILs: {[f['check'] for f in fails]}")
            break

        fix_round += 1
        print(f"\n FIX ROUND {fix_round} - {len(fails)} FAIL(s) remaining...")
        data = auto_fix(data, results)
        print(f"\n Re-running safety checks after round {fix_round}...\n")
        results = run_checks(data)
        print_and_save_report(results, data)

    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE - {fix_round} fix round(s) applied.")
    print("=" * 60)

    generate_bom(data)
