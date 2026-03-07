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
#  - Global throttle: 5s minimum gap between all calls (~12/min)
#  - Exponential backoff on 429: 60s, 120s, 180s...
#  - 7 retries (up from original 3)
# ==============================================================
_last_call_time = 0.0
FREE_TIER_MIN_GAP = 5.0


def gemini_call_with_retry(system_instruction, prompt, max_retries=7):
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < FREE_TIER_MIN_GAP:
        wait = FREE_TIER_MIN_GAP - elapsed
        print(f"  ⏸  Throttling {wait:.1f}s (free tier gap)...")
        time.sleep(wait)
    for attempt in range(1, max_retries + 1):
        try:
            _last_call_time = time.time()
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
                wait_seconds = int(wait_match.group(1)) + 5 if wait_match else (60 * attempt)
                print(f"\n  ⏳ Rate limit hit. Waiting {wait_seconds}s before retry "
                      f"(attempt {attempt}/{max_retries})...")
                print(f"     Tip: upgrade to a paid API key to avoid these delays.")
                time.sleep(wait_seconds)
                _last_call_time = time.time()
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
    "ESP32":               {"part_number": "ESP32-WROOM-32E",   "voltage": 3.3, "price_usd": 3.50,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/espressif-systems/ESP32-WROOM-32E/11613142"},
    "L298N":               {"part_number": "L298N",             "voltage": 5,   "price_usd": 1.80,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/stmicroelectronics/L298N/585918"},
    "TP4056":              {"part_number": "TP4056-SOT25",      "voltage": 4.2, "price_usd": 0.30,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/tc-charger/TP4056/7353588"},
    "LiPo Battery":        {"part_number": "PRT-13854",         "voltage": 3.7, "price_usd": 9.95,  "package": "THT", "in_stock": True, "digikey_url": "https://www.sparkfun.com/products/13854"},
    "AMS1117-3.3":         {"part_number": "AMS1117-3.3",       "voltage": 3.3, "price_usd": 0.25,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/advanced-monolithic-systems-inc/AMS1117-3-3/5010163"},
    "Decoupling Capacitor":{"part_number": "C0402C104K5RACTU", "voltage": 10,  "price_usd": 0.05,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/kemet/C0402C104K5RACTU/411388"},
    "ESD Diode":           {"part_number": "PRTR5V0U2X",        "voltage": 5,   "price_usd": 0.40,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/nexperia-usa-inc/PRTR5V0U2X/1177477"},
    "Arduino-Uno-R3":      {"part_number": "A000066",           "voltage": 5.0, "price_usd": 27.60, "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/arduino/A000066/2784006"},
    "Red-LED":             {"part_number": "HLMP-EG08-Y2000",   "voltage": 2.0, "price_usd": 0.35,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/broadcom-limited/HLMP-EG08-Y2000/3906329"},
    "Yellow-LED":          {"part_number": "TLHY4200",          "voltage": 2.1, "price_usd": 0.30,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/vishay-semiconductor-opto-division/TLHY4200/1805986"},
    "Green-LED":           {"part_number": "TLHG4200",          "voltage": 2.2, "price_usd": 0.30,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/vishay-semiconductor-opto-division/TLHG4200/1806003"},
    "Resistor-220R":       {"part_number": "CF14JT220R",        "voltage": 0,   "price_usd": 0.10,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/stackpole-electronics-inc/CF14JT220R/1741547"},
    "USB-5V-Supply":       {"part_number": "GENERIC-USB-5V",    "voltage": 5.0, "price_usd": 5.00,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/filter/usb-cables/469"},
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
        print("✅ Success! Design logic saved to output.json.")
        return True
    except Exception as e:
        print(f"❌ Error: {e}. Make sure your API key is in the .env file.")
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
    print("\n🔍 Asking Gemini to analyse the circuit and run smart safety checks...\n")
    try:
        circuit_summary = json.dumps(data, indent=2)
        raw = gemini_call_with_retry(
            SAFETY_SYSTEM,
            f"Analyse this circuit and run smart safety checks:\n\n{circuit_summary}"
        )
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        results = json.loads(raw)
        print(f"✅ Gemini generated {len(results)} smart safety checks for this project.")
        return results
    except Exception as e:
        print(f"⚠️  Smart checks failed ({e}). Returning error result.")
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
        lines.append("  OVERALL: ✅ DESIGN LOOKS SAFE TO PROCEED!")
    else:
        lines.append("  OVERALL: ❌ " + str(fail_count) + " ISSUE(S) NEED FIXING BEFORE MANUFACTURE.")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)
    with open("safety_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("\nReport saved to safety_report.txt")


# ==============================================================
#  STEP 8: AUTO-FETCH UNKNOWN COMPONENTS
#  Batches ALL unknown components into ONE Gemini call
# ==============================================================
def fetch_from_gemini_batch(component_names):
    if not component_names:
        return {}
    print(f"\n🔄 Batch-fetching {len(component_names)} unknown component(s) in ONE Gemini call...")
    batch_prompt = (
        "For each component in the list below, return a JSON object.\n"
        "Return ONLY a raw JSON array (no markdown) where each element has:\n"
        '  {"name": "...", "part_number": "...", "voltage": 0, "price_usd": 0.00, '
        '"package": "THT or SMD", "in_stock": true, "digikey_url": "https://www.digikey.com/..."}\n\n'
        "Components:\n" + "\n".join(f"- {n}" for n in component_names)
    )
    try:
        raw = gemini_call_with_retry(LOOKUP_SYSTEM, batch_prompt)
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        items = json.loads(raw)
        result = {item.get("name", ""): item for item in items if item.get("name")}
        print(f"✅ Batch lookup returned {len(result)} component(s).")
        return result
    except Exception as e:
        print(f"⚠️  Batch Gemini lookup failed: {e}")
        return {}


# ==============================================================
#  STEP 9: GENERATE BOM
# ==============================================================
def generate_bom(data, csv_path="BOM.csv"):
    print("\n" + "=" * 60)
    print("  GENERATING BILL OF MATERIALS (BOM)")
    print("=" * 60)

    names = [c["name"] for c in data.get("components", [])]
    db_hits = {}
    unknowns = []

    for name in names:
        if name in COMPONENTS:
            db_hits[name] = COMPONENTS[name].copy()
            db_hits[name]["name"] = name
            print(f"  ✅ Found in DB: '{name}'")
        else:
            unknowns.append(name)

    gemini_results = fetch_from_gemini_batch(unknowns) if unknowns else {}

    enriched = []
    for name in names:
        if name in db_hits:
            enriched.append(db_hits[name])
        elif name in gemini_results:
            row = gemini_results[name]
            row["name"] = name
            enriched.append(row)
        else:
            enriched.append({"name": name, "part_number": "UNKNOWN", "voltage": 0,
                              "price_usd": 0.00, "package": "UNKNOWN",
                              "in_stock": False, "digikey_url": "N/A"})

    fields = ["name", "part_number", "voltage", "price_usd", "package", "in_stock", "digikey_url"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)

    total = round(sum(c.get("price_usd", 0) for c in enriched), 2)
    print(f"\n✅ BOM.csv written - {len(enriched)} components, estimated cost: ${total}")

    unknown_names = [c["name"] for c in enriched if c.get("part_number") == "UNKNOWN"]
    if unknown_names:
        print(f"  ⚠️  Could not resolve: {unknown_names}")

    return enriched


# ==============================================================
#  STEP 10: AUTO-FIX FAILS
# ==============================================================
def auto_fix(data, results):
    fails = [r for r in results if r["status"] == "FAIL"]

    if not fails:
        print("\n✅ No FAILs found - nothing to auto-fix!")
        return data

    print("\n" + "=" * 60)
    print("  AUTO-FIX: GEMINI IS FIXING THE FAILS...")
    print("=" * 60)

    for f in fails:
        print(f"  🔧 Fixing: [{f['check']}]  {f['detail'][:60]}...")

    try:
        prompt = (
            f"Circuit JSON:\n{json.dumps(data, indent=2)}\n\n"
            f"FAIL checks to fix:\n{json.dumps(fails, indent=2)}\n\n"
            "Fix all the FAILs and return the updated circuit with explanation."
        )
        raw = gemini_call_with_retry(FIX_SYSTEM, prompt)
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        fixed_circuit = result.get("fixed_circuit", data)
        fixes_applied = result.get("fixes_applied", [])

        print("\n✅ FIXES APPLIED:")
        print("-" * 60)
        for i, fix in enumerate(fixes_applied, 1):
            print(f"  {i}. {fix}")

        with open("output.json", "w") as f:
            json.dump(fixed_circuit, f, indent=2)

        print("\n✅ output.json overwritten with fixed circuit design.")
        return fixed_circuit

    except Exception as e:
        print(f"\n❌ Auto-fix failed: {e}")
        return data


# ==============================================================
#  STEP 11: KICAD NETLIST GENERATOR
#  Converts circuit JSON → KiCad .net file (no extra file needed)
# ==============================================================
FOOTPRINT_MAP = {
    # MCUs
    "Arduino-Uno-R3":       ("MCU_Module",       "Arduino_Uno_R3",       "Package_DIP:DIP-28_W15.24mm"),
    "Arduino_Uno_R3":       ("MCU_Module",       "Arduino_Uno_R3",       "Package_DIP:DIP-28_W15.24mm"),
    "ESP32":                ("RF_Module",        "ESP32-WROOM-32",        "RF_Module:ESP32-WROOM-32"),
    "ESP32-S3":             ("RF_Module",        "ESP32-S3-WROOM-1",      "RF_Module:ESP32-S3-WROOM-1"),
    "STM32":                ("MCU_ST_STM32",     "STM32F103C8Tx",         "Package_QFP:LQFP-48_7x7mm"),
    # Power
    "AMS1117-3.3":          ("Device",           "Regulator_Linear",      "Package_TO_SOT_SMD:SOT-223-3_TabPin2"),
    "TP4056":               ("Battery_Management","TP4056",               "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"),
    "USB-5V-Supply":        ("Connector",        "USB_B",                 "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"),
    "LiPo":                 ("Device",           "Battery",               "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"),
    "LiPo-Battery":         ("Device",           "Battery",               "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"),
    # Drivers
    "L298N":                ("Motor_Control",    "L298N",                 "Package_TO_SOT_THT:TO-220-15"),
    # LEDs
    "Red-LED":              ("Device",           "LED",                   "LED_THT:LED_D5.0mm"),
    "Yellow-LED":           ("Device",           "LED",                   "LED_THT:LED_D5.0mm"),
    "Green-LED":            ("Device",           "LED",                   "LED_THT:LED_D5.0mm"),
    "Blue-LED":             ("Device",           "LED",                   "LED_THT:LED_D5.0mm"),
    # Passives
    "Resistor-220R":        ("Device",           "R",                     "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"),
    "Resistor-1K":          ("Device",           "R",                     "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"),
    "Resistor-10K":         ("Device",           "R",                     "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"),
    "Decoupling-Capacitor": ("Device",           "C",                     "Capacitor_SMD:C_0402_1005Metric"),
    "ESD-Diode":            ("Device",           "D_Zener",               "Diode_THT:D_DO-35_SOD27_P7.62mm_Horizontal"),
    # Generic fallbacks by type
    "__MCU__":              ("MCU_Module",       "Generic_MCU",           "Package_DIP:DIP-28_W15.24mm"),
    "__LED__":              ("Device",           "LED",                   "LED_THT:LED_D5.0mm"),
    "__Resistor__":         ("Device",           "R",                     "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"),
    "__Capacitor__":        ("Device",           "C",                     "Capacitor_SMD:C_0402_1005Metric"),
    "__PowerSupply__":      ("Connector",        "USB_B",                 "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"),
    "__Motor__":            ("Motor_Control",    "L298N",                 "Package_TO_SOT_THT:TO-220-15"),
    "__Default__":          ("Device",           "Generic_Component",     "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"),
}


def get_footprint(name, comp_type):
    clean = name.replace(" ", "-")
    if clean in FOOTPRINT_MAP:
        return FOOTPRINT_MAP[clean]
    if name in FOOTPRINT_MAP:
        return FOOTPRINT_MAP[name]
    for key in FOOTPRINT_MAP:
        if key.startswith("__"):
            continue
        if key.lower() in name.lower() or name.lower() in key.lower():
            return FOOTPRINT_MAP[key]
    type_key = f"__{comp_type}__"
    if type_key in FOOTPRINT_MAP:
        return FOOTPRINT_MAP[type_key]
    return FOOTPRINT_MAP["__Default__"]


def build_net_map(connections, components):
    name_to_id  = {c["name"]: c["id"] for c in components}
    net_map     = {}
    pin_counter = {}

    def get_pin(ref):
        pin_counter[ref] = pin_counter.get(ref, 0) + 1
        return pin_counter[ref]

    for conn in connections:
        from_name = conn["from"]
        to_name   = conn["to"]
        from_ref  = name_to_id.get(from_name, from_name)
        to_ref    = name_to_id.get(to_name,   to_name)

        net_name = f"Net-{re.sub(r'[^A-Za-z0-9_]', '_', from_name)}_to_{re.sub(r'[^A-Za-z0-9_]', '_', to_name)}"

        if "gnd" in from_name.lower() or "ground" in from_name.lower():
            net_name = "GND"
        elif "gnd" in to_name.lower() or "ground" in to_name.lower():
            net_name = "GND"
        elif "vcc" in from_name.lower() or "5v" in from_name.lower() or "power" in from_name.lower():
            net_name = "VCC_5V"
        elif "3.3" in from_name or "3v3" in from_name.lower():
            net_name = "VCC_3V3"

        if net_name not in net_map:
            net_map[net_name] = []
        net_map[net_name].append((from_ref, get_pin(from_ref)))
        net_map[net_name].append((to_ref,   get_pin(to_ref)))

    return net_map


def generate_kicad_netlist(data, output_path="design.net"):
    components  = data.get("components",  [])
    connections = data.get("connections", [])
    title       = data.get("title", "PCB_Project")

    print(f"\n{'='*60}")
    print("  GENERATING KICAD NETLIST")
    print(f"{'='*60}")
    print(f"  Components : {len(components)}")
    print(f"  Connections: {len(connections)}")

    lines = [
        "(export (version D)",
        "  (design",
        f"    (source {title}.sch)",
        f"    (date \"{datetime.now().strftime('%Y-%m-%d')}\")",
        "    (tool \"Kevin the Wizard - PCB Co-Pilot AI\"))",
        "  (components",
    ]

    for comp in components:
        ref  = comp["id"]
        name = comp["name"]
        lib, part, footprint = get_footprint(name, comp.get("type", "Default"))
        lines += [
            f"    (comp (ref {ref})",
            f"      (value {name})",
            f"      (libsource (lib {lib}) (part {part}))",
            f"      (footprint {footprint}))",
        ]
        print(f"  ✅ {ref} → {name}  [{footprint}]")

    # Always add power symbols
    lines += [
        "    (comp (ref PWR_GND)",
        "      (value GND)",
        "      (libsource (lib power) (part GND))",
        "      (footprint TestPoint:TestPoint_Pad_1.0x1.0mm))",
        "    (comp (ref PWR_VCC)",
        "      (value VCC)",
        "      (libsource (lib power) (part VCC))",
        "      (footprint TestPoint:TestPoint_Pad_1.0x1.0mm))",
        "  )",
        "  (nets",
    ]

    net_map = build_net_map(connections, components)
    if "GND"    not in net_map: net_map["GND"]    = []
    if "VCC_5V" not in net_map: net_map["VCC_5V"] = []
    net_map["GND"].append(("PWR_GND", 1))
    net_map["VCC_5V"].append(("PWR_VCC", 1))

    for code, (net_name, nodes) in enumerate(net_map.items(), start=1):
        lines.append(f"    (net (code {code}) (name \"{net_name}\")")
        seen = set()
        for ref, pin in nodes:
            if (ref, pin) not in seen:
                seen.add((ref, pin))
                lines.append(f"      (node (ref {ref}) (pin {pin}))")
        lines.append("    )")

    lines += ["  )", ")"]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n✅ KiCad netlist saved → {output_path}")
    print(f"   Nets generated: {len(net_map)}")
    print(f"{'='*60}\n")


# ==============================================================
#  MAIN PIPELINE
# ==============================================================
if __name__ == "__main__":
    print("🧙 Kevin the Wizard's Safety Checker starting...\n")

    user_request = "traffic signal circuit using arduino uno r3"
    MAX_FIX_ROUNDS = 3

    # 1. Generate netlist JSON
    generate_and_save_json(user_request)

    # 2. Load it
    data = load_json()

    # 3. First safety check
    results = run_checks(data)
    print_and_save_report(results, data)

    # 4. Auto-fix loop
    fix_round = 0
    while True:
        fails = [r for r in results if r["status"] == "FAIL"]

        if not fails:
            print("\n✅ All FAILs resolved! Design is clean.")
            break

        if fix_round >= MAX_FIX_ROUNDS:
            print(f"\n⚠️  Reached max fix rounds ({MAX_FIX_ROUNDS}). Stopping auto-fix.")
            print(f"   Remaining FAILs: {[f['check'] for f in fails]}")
            break

        fix_round += 1
        print(f"\n🔧 FIX ROUND {fix_round} - {len(fails)} FAIL(s) remaining...")
        new_data = auto_fix(data, results)

        if new_data != data:
            data = new_data
        else:
            print("⚠️  Fix returned unchanged data — stopping to avoid infinite loop.")
            break

        print(f"\n🔍 Re-running safety checks after round {fix_round}...\n")
        results = run_checks(data)
        print_and_save_report(results, data)

    # 5. Summary
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE - {fix_round} fix round(s) applied.")
    print("=" * 60)

    # 6. Generate BOM
    generate_bom(data)

    # 7. Generate KiCad netlist
    generate_kicad_netlist(data, "design.net")

    print("\n" + "=" * 60)
    print("  ✅ ALL FILES GENERATED:")
    print("     📄 output.json       - Circuit design")
    print("     📄 safety_report.txt - Safety analysis")
    print("     📄 BOM.csv           - Bill of materials")
    print("     📄 design.net        - KiCad netlist")
    print("  📦 Import design.net into KiCad:")
    print("     PCB Editor → File → Import → Netlist → Update PCB")
    print("=" * 60)
