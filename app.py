import os
import json
import csv
import re
import time
import zipfile
import tempfile
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

app = Flask(__name__, static_folder="static")
CORS(app)

# ==============================================================
#  GEMINI MODELS
# ==============================================================
netlist_model = genai.GenerativeModel(
    model_name="gemini-3.1-flash-lite-preview",
    system_instruction="""
    You are a Senior Hardware Engineer (PS1 Specialist).
    Your job is to parse descriptions into a JSON Netlist.
    Rules:
    - Return ONLY raw JSON. No markdown blocks (no ```).
    - Include: MCU, Power Stage (LDO/Buck), Motor Drivers, and Passives.
    - Format: {"components": [{"id": "U1", "name": "ESP32-S3", "type": "MCU", "voltage": 3.3}], "connections": [{"from": "LiPo", "to": "ESP32"}]}
    - Each component MUST include: id, name, type, and voltage fields.
    - connections must use "from" and "to" keys matching component names.
    - Output ONLY raw JSON. No markdown, no extra text.
    """
)


# ==============================================================
#  VALIDATION MODEL - checks if prompt is feasible BEFORE generating
# ==============================================================
validation_model = genai.GenerativeModel(
    model_name="gemini-3.1-flash-lite-preview",
    system_instruction="""
    You are a Senior Hardware Engineer and circuit feasibility expert.
    Your job is to evaluate whether a circuit design request is feasible and buildable.

    A request is IMPOSSIBLE or INVALID if it:
    - Violates fundamental laws of physics (e.g. perpetual motion, free energy)
    - Contains contradictory requirements (e.g. "5V input but 100V output with no boost converter")
    - Is completely unrelated to electronics or circuits (e.g. "make me a sandwich")
    - Requests components that cannot physically work together as described
    - Is completely vague with no actionable circuit information (e.g. "make something cool")

    A request is POSSIBLE even if it is:
    - Simple (e.g. "blink an LED with Arduino")
    - Complex (e.g. "motor controller with battery management")
    - Unconventional but physically valid

    Return ONLY raw JSON in this exact format (no markdown, no extra text):
    {
      "feasible": true or false,
      "reason": "One sentence explaining why it is or is not feasible.",
      "suggestions": ["Specific suggestion 1", "Specific suggestion 2", "Specific suggestion 3"]
    }

    If feasible is true, suggestions should list optional improvements.
    If feasible is false, suggestions MUST be specific actionable alternatives the user can try instead.
    """
)


def validate_request(user_request):
    """
    Runs a feasibility check on the user prompt before any generation.
    Returns (is_feasible, reason, suggestions).
    """
    try:
        response = gemini_call_with_retry(validation_model, f"Evaluate this circuit request: {user_request}")
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return result.get("feasible", True), result.get("reason", ""), result.get("suggestions", [])
    except Exception as e:
        # If validation itself fails, allow the request through
        return True, "", []

# ==============================================================
#  STATIC COMPONENT DATABASE
# ==============================================================
COMPONENTS = {
    "ESP32":               {"part_number": "ESP32-WROOM-32E",   "voltage": 3.3, "price_usd": 3.50, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/espressif-systems/ESP32-WROOM-32E/11613142"},
    "L298N":               {"part_number": "L298N",             "voltage": 5,   "price_usd": 1.80, "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/stmicroelectronics/L298N/585918"},
    "TP4056":              {"part_number": "TP4056-SOT25",      "voltage": 4.2, "price_usd": 0.30, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/tc-charger/TP4056/7353588"},
    "AMS1117-3.3":         {"part_number": "AMS1117-3.3",       "voltage": 3.3, "price_usd": 0.25, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/advanced-monolithic-systems-inc/AMS1117-3-3/5010163"},
    "Decoupling Capacitor":{"part_number": "C0402C104K5RACTU", "voltage": 10,  "price_usd": 0.05, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/kemet/C0402C104K5RACTU/411388"},
    "Arduino-Uno-R3":      {"part_number": "A000066",           "voltage": 5.0, "price_usd": 27.60,"package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/arduino/A000066/2784006"},
    "Red-LED":             {"part_number": "HLMP-EG08-Y2000",   "voltage": 2.0, "price_usd": 0.35, "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/broadcom-limited/HLMP-EG08-Y2000/3906329"},
    "Yellow-LED":          {"part_number": "TLHY4200",          "voltage": 2.1, "price_usd": 0.30, "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/vishay-semiconductor-opto-division/TLHY4200/1805986"},
    "Green-LED":           {"part_number": "TLHG4200",          "voltage": 2.2, "price_usd": 0.30, "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/vishay-semiconductor-opto-division/TLHG4200/1806003"},
    "Resistor-220R":       {"part_number": "CF14JT220R",        "voltage": 0,   "price_usd": 0.10, "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/stackpole-electronics-inc/CF14JT220R/1741547"},
    "USB-5V-Supply":       {"part_number": "GENERIC-USB-5V",    "voltage": 5.0, "price_usd": 5.00, "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/filter/usb-cables/469"},
}

# ==============================================================
#  KICAD FOOTPRINT MAP
# ==============================================================
FOOTPRINT_MAP = {
    "Arduino-Uno-R3":       ("MCU_Module",        "Arduino_Uno_R3",      "Package_DIP:DIP-28_W15.24mm"),
    "Arduino_Uno_R3":       ("MCU_Module",        "Arduino_Uno_R3",      "Package_DIP:DIP-28_W15.24mm"),
    "ESP32":                ("RF_Module",         "ESP32-WROOM-32",       "RF_Module:ESP32-WROOM-32"),
    "ESP32-S3":             ("RF_Module",         "ESP32-S3-WROOM-1",     "RF_Module:ESP32-S3-WROOM-1"),
    "STM32":                ("MCU_ST_STM32",      "STM32F103C8Tx",        "Package_QFP:LQFP-48_7x7mm"),
    "AMS1117-3.3":          ("Device",            "Regulator_Linear",     "Package_TO_SOT_SMD:SOT-223-3_TabPin2"),
    "TP4056":               ("Battery_Management","TP4056",               "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"),
    "USB-5V-Supply":        ("Connector",         "USB_B",                "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"),
    "LiPo":                 ("Device",            "Battery",              "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"),
    "LiPo-Battery":         ("Device",            "Battery",              "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"),
    "L298N":                ("Motor_Control",     "L298N",                "Package_TO_SOT_THT:TO-220-15"),
    "Red-LED":              ("Device",            "LED",                  "LED_THT:LED_D5.0mm"),
    "Yellow-LED":           ("Device",            "LED",                  "LED_THT:LED_D5.0mm"),
    "Green-LED":            ("Device",            "LED",                  "LED_THT:LED_D5.0mm"),
    "Blue-LED":             ("Device",            "LED",                  "LED_THT:LED_D5.0mm"),
    "Resistor-220R":        ("Device",            "R",                    "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"),
    "Resistor-1K":          ("Device",            "R",                    "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"),
    "Resistor-10K":         ("Device",            "R",                    "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"),
    "Decoupling-Capacitor": ("Device",            "C",                    "Capacitor_SMD:C_0402_1005Metric"),
    "ESD-Diode":            ("Device",            "D_Zener",              "Diode_THT:D_DO-35_SOD27_P7.62mm_Horizontal"),
    "__MCU__":              ("MCU_Module",        "Generic_MCU",          "Package_DIP:DIP-28_W15.24mm"),
    "__LED__":              ("Device",            "LED",                  "LED_THT:LED_D5.0mm"),
    "__Resistor__":         ("Device",            "R",                    "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"),
    "__Capacitor__":        ("Device",            "C",                    "Capacitor_SMD:C_0402_1005Metric"),
    "__PowerSupply__":      ("Connector",         "USB_B",                "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"),
    "__Motor__":            ("Motor_Control",     "L298N",                "Package_TO_SOT_THT:TO-220-15"),
    "__Default__":          ("Device",            "Generic_Component",    "Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal"),
}


def get_footprint(name, comp_type):
    clean = name.replace(" ", "-")
    if clean in FOOTPRINT_MAP: return FOOTPRINT_MAP[clean]
    if name  in FOOTPRINT_MAP: return FOOTPRINT_MAP[name]
    for key in FOOTPRINT_MAP:
        if key.startswith("__"): continue
        if key.lower() in name.lower() or name.lower() in key.lower():
            return FOOTPRINT_MAP[key]
    type_key = f"__{comp_type}__"
    if type_key in FOOTPRINT_MAP: return FOOTPRINT_MAP[type_key]
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
        if   "gnd"   in from_name.lower() or "ground" in from_name.lower(): net_name = "GND"
        elif "gnd"   in to_name.lower()   or "ground" in to_name.lower():   net_name = "GND"
        elif "vcc"   in from_name.lower() or "5v"     in from_name.lower() or "power" in from_name.lower(): net_name = "VCC_5V"
        elif "3.3"   in from_name         or "3v3"    in from_name.lower(): net_name = "VCC_3V3"

        if net_name not in net_map: net_map[net_name] = []
        net_map[net_name].append((from_ref, get_pin(from_ref)))
        net_map[net_name].append((to_ref,   get_pin(to_ref)))

    return net_map


def generate_kicad_netlist_content(data):
    """Generates KiCad .net file content as a string."""
    components  = data.get("components",  [])
    connections = data.get("connections", [])
    title       = data.get("title", "PCB_Project")

    lines = [
        "(export (version D)",
        "  (design",
        f"    (source {title}.sch)",
        f"    (date \"{datetime.now().strftime('%Y-%m-%d')}\")",
        "    (tool \"Kevin the Wizard - PCB Co-Pilot AI\"))",
        "  (components",
    ]

    for comp in components:
        lib, part, footprint = get_footprint(comp["name"], comp.get("type", "Default"))
        lines += [
            f"    (comp (ref {comp['id']})",
            f"      (value {comp['name']})",
            f"      (libsource (lib {lib}) (part {part}))",
            f"      (footprint {footprint}))",
        ]

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
    return "\n".join(lines)


# ==============================================================
#  HELPERS
# ==============================================================
def gemini_call_with_retry(model, prompt, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            return response
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                wait_match = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', error_str)
                wait_seconds = int(wait_match.group(1)) + 5 if wait_match else 60
                time.sleep(wait_seconds)
            else:
                raise e
    raise Exception(f"Gemini call failed after {max_retries} retries.")


def generate_circuit_json(user_request):
    response = gemini_call_with_retry(netlist_model, user_request)
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def run_smart_checks(data):
    safety_model = genai.GenerativeModel(
        model_name="gemini-3.1-flash-lite-preview",
        system_instruction="""
        You are a Senior Electrical Safety Engineer.
        Given a circuit JSON, identify the project type and run 5-8 SPECIFIC safety checks relevant to it.
        Return ONLY a raw JSON array:
        [{"check": "Name", "status": "PASS|FAIL|WARN|SKIP", "detail": "explanation referencing actual components"}]
        No markdown, no extra text.
        """
    )
    response = gemini_call_with_retry(safety_model, f"Analyse this circuit:\n{json.dumps(data, indent=2)}")
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def auto_fix_fails(data, results):
    fails = [r for r in results if r["status"] == "FAIL"]
    if not fails:
        return data, []

    fix_model = genai.GenerativeModel(
        model_name="gemini-3.1-flash-lite-preview",
        system_instruction="""
        You are a circuit repair specialist.
        Fix all FAILs by modifying the circuit JSON.
        Add missing components and connections. Do NOT remove existing ones.
        Return ONLY raw JSON:
        {"fixed_circuit": {...}, "fixes_applied": ["Fix 1: ...", "Fix 2: ..."]}
        No markdown, no extra text.
        """
    )
    prompt = f"Circuit:\n{json.dumps(data, indent=2)}\n\nFAILs:\n{json.dumps(fails, indent=2)}\n\nFix all FAILs."
    response = gemini_call_with_retry(fix_model, prompt)
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    return result.get("fixed_circuit", data), result.get("fixes_applied", [])


def fetch_component_gemini(name):
    try:
        lookup_model = genai.GenerativeModel(
            model_name="gemini-3.1-flash-lite-preview",
            system_instruction="""
            You are an electronics component database.
            Return ONLY raw JSON: {"part_number": "...", "voltage": 3.3, "price_usd": 0.50, "package": "THT", "in_stock": true, "digikey_url": "https://..."}
            No markdown, no extra text.
            """
        )
        response = gemini_call_with_retry(lookup_model, f"Component details for: {name}")
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except:
        return {"part_number": "UNKNOWN", "voltage": 0, "price_usd": 0.00, "package": "UNKNOWN", "in_stock": False, "digikey_url": "N/A"}


def generate_safety_report_txt(results, data, fix_log):
    lines = []
    lines.append("=" * 60)
    lines.append("  KEVIN THE WIZARD'S ELECTRICAL SAFETY REPORT")
    lines.append("  Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("=" * 60)
    lines.append("")
    lines.append("COMPONENTS FOUND:")
    for c in data.get("components", []):
        lines.append(f"  - {c['name']} ({c.get('type','?')}) - {c.get('voltage','?')}V")
    lines.append("")
    lines.append("CONNECTIONS:")
    for conn in data.get("connections", []):
        lines.append(f"  {conn['from']}  -->  {conn['to']}")
    if fix_log:
        lines.append("")
        lines.append("AUTO-FIXES APPLIED:")
        for i, fix in enumerate(fix_log, 1):
            lines.append(f"  {i}. {fix}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("FINAL SAFETY CHECK RESULTS:")
    lines.append("-" * 60)
    pass_count = fail_count = 0
    for r in results:
        lines.append(f"\n[{r['status']}]  {r['check']}")
        lines.append(f"   -> {r['detail']}")
        if r["status"] == "PASS": pass_count += 1
        elif r["status"] == "FAIL": fail_count += 1
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  SUMMARY: {pass_count} passed, {fail_count} failed, {len(results)-pass_count-fail_count} warnings/skipped")
    lines.append("  OVERALL: " + ("DESIGN LOOKS SAFE TO PROCEED!" if fail_count == 0 else f"{fail_count} ISSUE(S) NEED FIXING."))
    lines.append("=" * 60)
    return "\n".join(lines)


def generate_mermaid_md(data):
    lines = ["# Circuit Diagram", "", "```mermaid", "graph TD"]
    for conn in data.get("connections", []):
        f = conn["from"].replace(" ", "_").replace("-", "_")
        t = conn["to"].replace(" ", "_").replace("-", "_")
        lines.append(f"    {f} --> {t}")
    lines += ["```", "", "---", "_Paste at https://mermaid.live to view the diagram_"]
    return "\n".join(lines)


def generate_bom_csv_content(data):
    names = [c["name"] for c in data.get("components", [])]
    enriched = []
    for name in names:
        if name in COMPONENTS:
            row = COMPONENTS[name].copy()
            row["name"] = name
        else:
            row = fetch_component_gemini(name)
            row["name"] = name
        enriched.append(row)

    lines = ["name,part_number,voltage,price_usd,package,in_stock,digikey_url"]
    for c in enriched:
        lines.append(f"{c.get('name','')},{c.get('part_number','')},{c.get('voltage','')},{c.get('price_usd','')},{c.get('package','')},{c.get('in_stock','')},{c.get('digikey_url','')}")
    total = round(sum(float(c.get("price_usd", 0)) for c in enriched), 2)
    lines.append(f",,,,,,TOTAL: ${total}")
    return "\n".join(lines)


# ==============================================================
#  MAIN ROUTE
# ==============================================================
@app.route("/generate", methods=["POST"])
def generate():
    try:
        body = request.get_json()
        user_request = body.get("request", "").strip()
        if not user_request:
            return jsonify({"error": "No circuit description provided."}), 400

        MAX_FIX_ROUNDS = 4
        all_fixes = []

        # Step 0: Validate the request before doing anything
        is_feasible, reason, suggestions = validate_request(user_request)
        if not is_feasible:
            return jsonify({
                "error": "impossible_request",
                "message": reason,
                "suggestions": suggestions
            }), 422

        # Step 1: Generate circuit JSON
        data = generate_circuit_json(user_request)

        # Step 2: Smart safety checks + auto-fix loop
        results = run_smart_checks(data)
        fix_round = 0
        while fix_round < MAX_FIX_ROUNDS:
            fails = [r for r in results if r["status"] == "FAIL"]
            if not fails:
                break
            fix_round += 1
            data, fixes = auto_fix_fails(data, results)
            all_fixes.extend(fixes)
            results = run_smart_checks(data)

        # Step 3: Generate all output files
        report_txt  = generate_safety_report_txt(results, data, all_fixes)
        diagram_md  = generate_mermaid_md(data)
        bom_csv     = generate_bom_csv_content(data)
        kicad_net   = generate_kicad_netlist_content(data)   # ← NEW

        # Step 4: Pack into ZIP (now includes design.net)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        with zipfile.ZipFile(tmp.name, "w") as zf:
            zf.writestr("output.json",       json.dumps(data, indent=2))
            zf.writestr("safety_report.txt", report_txt)
            zf.writestr("diagram.md",        diagram_md)
            zf.writestr("BOM.csv",           bom_csv)
            zf.writestr("design.net",        kicad_net)       # ← NEW

        return send_file(tmp.name, as_attachment=True, download_name="kevin_wizard_output.zip", mimetype="application/zip")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
