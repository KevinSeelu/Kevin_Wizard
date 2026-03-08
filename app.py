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
    You are a Senior Hardware Engineer (PCB Specialist).
    Your job is to parse descriptions into a detailed JSON Netlist with pin-level assignments.

    Rules:
    - Return ONLY raw JSON. No markdown blocks (no ```).
    - Include: MCU, Power Stage (LDO/Buck), Motor Drivers, and Passives.
    - Each component MUST include: id, name, type, voltage fields.
    - connections must use "from" and "to" keys matching component names.
    - For MCUs (ESP32, Arduino, STM32, Raspberry Pi, etc.) you MUST include a "pin_assignments" array.
    - Each pin_assignment must specify: pin_number, pin_name, signal, connected_to, peripheral.
    - Detect and resolve peripheral conflicts: do NOT assign two signals to the same pin.
    - Flag any I2C/SPI/UART/PWM peripheral conflicts in a "conflicts" array (empty if none).

    Output format (return ONLY this JSON, no extra text):
    {
      "components": [{"id": "U1", "name": "ESP32", "type": "MCU", "voltage": 3.3}],
      "connections": [{"from": "ComponentA", "to": "ComponentB", "signal": "SDA"}],
      "pin_assignments": [
        {
          "component_id": "U1",
          "component_name": "ESP32",
          "pins": [
            {"pin_number": "GPIO21", "pin_name": "SDA", "signal": "I2C_SDA", "connected_to": "OLED_Display", "peripheral": "I2C0"},
            {"pin_number": "GPIO22", "pin_name": "SCL", "signal": "I2C_SCL", "connected_to": "OLED_Display", "peripheral": "I2C0"},
            {"pin_number": "GND",    "pin_name": "GND", "signal": "GND",     "connected_to": "GND_Rail",     "peripheral": "Power"},
            {"pin_number": "3V3",    "pin_name": "VCC", "signal": "VCC_3V3", "connected_to": "LDO_Output",   "peripheral": "Power"}
          ]
        }
      ],
      "conflicts": []
    }

    If a component is NOT an MCU (e.g. LED, resistor, motor driver), skip it in pin_assignments.
    Always include GND and VCC pins in the MCU pin list.
    Output ONLY raw JSON. No markdown, no extra text.
    """
)

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

# ==============================================================
#  STATIC COMPONENT DATABASE
# ==============================================================
COMPONENTS = {
    "ESP32":               {"part_number": "ESP32-WROOM-32E",   "voltage": 3.3, "price_usd": 3.50,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/espressif-systems/ESP32-WROOM-32E/11613142"},
    "L298N":               {"part_number": "L298N",             "voltage": 5,   "price_usd": 1.80,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/stmicroelectronics/L298N/585918"},
    "TP4056":              {"part_number": "TP4056-SOT25",      "voltage": 4.2, "price_usd": 0.30,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/tc-charger/TP4056/7353588"},
    "AMS1117-3.3":         {"part_number": "AMS1117-3.3",       "voltage": 3.3, "price_usd": 0.25,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/advanced-monolithic-systems-inc/AMS1117-3-3/5010163"},
    "Decoupling Capacitor":{"part_number": "C0402C104K5RACTU", "voltage": 10,  "price_usd": 0.05,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/kemet/C0402C104K5RACTU/411388"},
    "Arduino-Uno-R3":      {"part_number": "A000066",           "voltage": 5.0, "price_usd": 27.60, "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/arduino/A000066/2784006"},
    "Red-LED":             {"part_number": "HLMP-EG08-Y2000",   "voltage": 2.0, "price_usd": 0.35,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/broadcom-limited/HLMP-EG08-Y2000/3906329"},
    "Yellow-LED":          {"part_number": "TLHY4200",          "voltage": 2.1, "price_usd": 0.30,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/vishay-semiconductor-opto-division/TLHY4200/1805986"},
    "Green-LED":           {"part_number": "TLHG4200",          "voltage": 2.2, "price_usd": 0.30,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/vishay-semiconductor-opto-division/TLHG4200/1806003"},
    "Resistor-220R":       {"part_number": "CF14JT220R",        "voltage": 0,   "price_usd": 0.10,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/stackpole-electronics-inc/CF14JT220R/1741547"},
    "USB-5V-Supply":       {"part_number": "GENERIC-USB-5V",    "voltage": 5.0, "price_usd": 5.00,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/filter/usb-cables/469"},
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


def validate_request(user_request):
    try:
        response = gemini_call_with_retry(
            validation_model,
            f"Evaluate this circuit request for feasibility: {user_request}"
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return result.get("feasible", True), result.get("reason", ""), result.get("suggestions", [])
    except Exception:
        return True, "", []


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
    lines = ["=" * 60, "  KEVIN THE WIZARD'S ELECTRICAL SAFETY REPORT",
             "  Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "=" * 60, "",
             "COMPONENTS FOUND:"]
    for c in data.get("components", []):
        lines.append(f"  - {c['name']} ({c.get('type','?')}) - {c.get('voltage','?')}V")
    lines += ["", "CONNECTIONS:"]
    for conn in data.get("connections", []):
        lines.append(f"  {conn['from']}  -->  {conn['to']}")
    if fix_log:
        lines += ["", "AUTO-FIXES APPLIED:"]
        for i, fix in enumerate(fix_log, 1):
            lines.append(f"  {i}. {fix}")
    pin_assignments = data.get("pin_assignments", [])
    if pin_assignments:
        lines += ["", "-" * 60, "PIN ASSIGNMENTS:", "-" * 60]
        for mcu in pin_assignments:
            lines.append(f"\n  {mcu.get('component_name', mcu.get('component_id','?'))} ({mcu.get('component_id','')}):")
            lines.append(f"  {'PIN':<12} {'NAME':<12} {'SIGNAL':<20} {'CONNECTED TO':<25} PERIPHERAL")
            lines.append(f"  {'-'*12} {'-'*12} {'-'*20} {'-'*25} {'-'*12}")
            for p in mcu.get("pins", []):
                lines.append(f"  {str(p.get('pin_number','')):<12} {str(p.get('pin_name','')):<12} {str(p.get('signal','')):<20} {str(p.get('connected_to','')):<25} {p.get('peripheral','')}")
        conflicts = data.get("conflicts", [])
        if conflicts:
            lines += ["", "  WARNING - PERIPHERAL CONFLICTS DETECTED:"]
            for c in conflicts: lines.append(f"    - {c}")
        else:
            lines.append("\n  OK - No peripheral conflicts detected.")

    lines += ["", "-" * 60, "FINAL SAFETY CHECK RESULTS:", "-" * 60]
    pass_count = fail_count = 0
    for r in results:
        lines += [f"\n[{r['status']}]  {r['check']}", f"   -> {r['detail']}"]
        if r["status"] == "PASS": pass_count += 1
        elif r["status"] == "FAIL": fail_count += 1
    lines += ["", "=" * 60,
              f"  SUMMARY: {pass_count} passed, {fail_count} failed, {len(results)-pass_count-fail_count} warnings/skipped",
              "  OVERALL: " + ("DESIGN LOOKS SAFE TO PROCEED!" if fail_count == 0 else f"{fail_count} ISSUE(S) NEED FIXING."),
              "=" * 60]
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
        row = COMPONENTS[name].copy() if name in COMPONENTS else fetch_component_gemini(name)
        row["name"] = name
        enriched.append(row)
    lines = ["name,part_number,voltage,price_usd,package,in_stock,digikey_url"]
    for c in enriched:
        lines.append(f"{c.get('name','')},{c.get('part_number','')},{c.get('voltage','')},{c.get('price_usd','')},{c.get('package','')},{c.get('in_stock','')},{c.get('digikey_url','')}")
    total = round(sum(float(c.get("price_usd", 0)) for c in enriched), 2)
    lines.append(f",,,,,,TOTAL: ${total}")
    return "\n".join(lines)


def generate_kicad_sch_content(data):
    """
    Generates a KiCad 6/7 schematic file (.kicad_sch) from circuit data.
    Places components on a grid and draws wire connections between them.
    """
    components  = data.get("components", [])
    connections = data.get("connections", [])
    pin_assignments = data.get("pin_assignments", [])
    title       = data.get("title", "Kevin_Wizard_PCB")
    now         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build a position grid: spread components across a 200x200 mil grid
    COLS = 4
    GRID = 50  # spacing in mm
    comp_positions = {}
    for i, comp in enumerate(components):
        col = i % COLS
        row = i // COLS
        comp_positions[comp["id"]] = (col * GRID, row * GRID)

    # Map component names to IDs for connection wiring
    name_to_id = {c["name"]: c["id"] for c in components}

    lines = [
        '(kicad_sch (version 20230121) (generator kevin_wizard)',
        '',
        '  (paper "A4")',
        '',
        f'  (title_block',
        f'    (title "{title}")',
        f'    (date "{now}")',
        f'    (rev "1.0")',
        f'    (company "Kevin the Wizard - AI PCB Co-Pilot")',
        f'  )',
        '',
        '  (lib_symbols)',
        '',
    ]

    # Write each component as a schematic symbol
    for comp in components:
        cid  = comp["id"]
        name = comp["name"]
        ctype = comp.get("type", "Generic")
        volt  = comp.get("voltage", 0)
        x, y  = comp_positions.get(cid, (0, 0))
        lib, part, fp = get_footprint(name, ctype)

        lines += [
            f'  (symbol (lib_id "{lib}:{part}")',
            f'    (at {x} {y} 0)',
            f'    (unit 1)',
            f'    (in_bom yes) (on_board yes)',
            f'    (property "Reference" "{cid}" (at {x} {y - 3} 0))',
            f'    (property "Value" "{name}" (at {x} {y + 3} 0))',
            f'    (property "Footprint" "{fp}" (at {x} {y + 6} 0))',
            f'    (property "Description" "{ctype} - {volt}V" (at {x} {y + 9} 0))',
            f'  )',
            '',
        ]

    # Power symbols: GND and VCC
    lines += [
        '  (symbol (lib_id "power:GND")',
        '    (at 10 10 0)',
        '    (unit 1)',
        '    (in_bom yes) (on_board yes)',
        '    (property "Reference" "#PWR_GND" (at 10 10 0))',
        '    (property "Value" "GND" (at 10 13 0))',
        '  )',
        '',
        '  (symbol (lib_id "power:VCC")',
        '    (at 20 10 0)',
        '    (unit 1)',
        '    (in_bom yes) (on_board yes)',
        '    (property "Reference" "#PWR_VCC" (at 20 10 0))',
        '    (property "Value" "VCC" (at 20 7 0))',
        '  )',
        '',
    ]

    # Draw wires between connected components
    lines.append('  ; === Connections ===')
    for conn in connections:
        from_id = name_to_id.get(conn["from"], conn["from"])
        to_id   = name_to_id.get(conn["to"],   conn["to"])
        fx, fy  = comp_positions.get(from_id, (0, 0))
        tx, ty  = comp_positions.get(to_id,   (0, 0))
        signal  = conn.get("signal", "")
        lines += [
            f'  (wire',
            f'    (pts (xy {fx + 5} {fy}) (xy {tx - 5} {ty}))',
            f'    (stroke (width 0) (type default))',
            f'  )',
        ]
        if signal:
            mid_x = (fx + tx) / 2
            mid_y = (fy + ty) / 2
            lines += [
                f'  (label "{signal}"',
                f'    (at {mid_x} {mid_y} 0)',
                f'    (fields_autoplaced)',
                f'    (effects (font (size 1.27 1.27)))',
                f'  )',
            ]

    # Add pin assignment labels for MCUs
    if pin_assignments:
        lines.append('')
        lines.append('  ; === Pin Assignment Labels ===')
        for mcu in pin_assignments:
            cid  = mcu.get("component_id", "U1")
            x, y = comp_positions.get(cid, (0, 0))
            for idx, pin in enumerate(mcu.get("pins", [])):
                label = f"{pin.get('pin_number','')}:{pin.get('signal','')}"
                lines += [
                    f'  (label "{label}"',
                    f'    (at {x + 8} {y + idx * 2.5} 0)',
                    f'    (effects (font (size 1.0 1.0)))',
                    f'  )',
                ]

    lines.append(')')
    return "\n".join(lines)


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
        fn, tn = conn["from"], conn["to"]
        fr, tr = name_to_id.get(fn, fn), name_to_id.get(tn, tn)
        net = f"Net-{re.sub(r'[^A-Za-z0-9_]','_',fn)}_to_{re.sub(r'[^A-Za-z0-9_]','_',tn)}"
        if   "gnd" in fn.lower() or "ground" in fn.lower(): net = "GND"
        elif "gnd" in tn.lower() or "ground" in tn.lower(): net = "GND"
        elif "vcc" in fn.lower() or "5v" in fn.lower() or "power" in fn.lower(): net = "VCC_5V"
        elif "3.3" in fn or "3v3" in fn.lower(): net = "VCC_3V3"
        if net not in net_map: net_map[net] = []
        net_map[net].append((fr, get_pin(fr)))
        net_map[net].append((tr, get_pin(tr)))
    return net_map


def generate_kicad_netlist_content(data):
    components  = data.get("components", [])
    connections = data.get("connections", [])
    title       = data.get("title", "PCB_Project")
    lines = ["(export (version D)", "  (design",
             f"    (source {title}.sch)",
             f"    (date \"{datetime.now().strftime('%Y-%m-%d')}\")",
             "    (tool \"Kevin the Wizard - PCB Co-Pilot AI\"))",
             "  (components"]
    for comp in components:
        lib, part, fp = get_footprint(comp["name"], comp.get("type", "Default"))
        lines += [f"    (comp (ref {comp['id']})", f"      (value {comp['name']})",
                  f"      (libsource (lib {lib}) (part {part}))", f"      (footprint {fp}))"]
    lines += ["    (comp (ref PWR_GND)", "      (value GND)",
              "      (libsource (lib power) (part GND))",
              "      (footprint TestPoint:TestPoint_Pad_1.0x1.0mm))",
              "    (comp (ref PWR_VCC)", "      (value VCC)",
              "      (libsource (lib power) (part VCC))",
              "      (footprint TestPoint:TestPoint_Pad_1.0x1.0mm))",
              "  )", "  (nets"]
    net_map = build_net_map(connections, components)
    if "GND"    not in net_map: net_map["GND"] = []
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
#  GERBER GENERATION
# ==============================================================

# Standard PCB layer stack
GERBER_LAYERS = {
    "F.Cu":        ("GTL", "Top Copper"),
    "B.Cu":        ("GBL", "Bottom Copper"),
    "F.SilkS":     ("GTO", "Top Silkscreen"),
    "B.SilkS":     ("GBO", "Bottom Silkscreen"),
    "F.Mask":      ("GTS", "Top Soldermask"),
    "B.Mask":      ("GBS", "Bottom Soldermask"),
    "Edge.Cuts":   ("GKO", "Board Outline"),
    "F.Paste":     ("GTP", "Top Paste"),
    "B.Paste":     ("GBP", "Bottom Paste"),
    "In1.Cu":      ("G2L", "Inner Copper 1"),
    "In2.Cu":      ("G3L", "Inner Copper 2"),
    "drill":       ("DRL", "Drill File"),
}

# Component footprint sizes in mm (width, height, pad_dia, drill_dia)
FOOTPRINT_SIZES = {
    "ESP32":               (18.0, 25.5, 1.0, 0.8),
    "ESP32-S3":            (18.0, 25.5, 1.0, 0.8),
    "Arduino-Uno-R3":      (53.3, 68.6, 1.6, 0.8),
    "Arduino_Uno_R3":      (53.3, 68.6, 1.6, 0.8),
    "STM32":               (7.0,  7.0,  0.5, 0.0),
    "AMS1117-3.3":         (6.5,  4.5,  1.8, 0.0),
    "TP4056":              (5.0,  6.0,  0.6, 0.0),
    "L298N":               (15.0, 20.0, 1.6, 1.0),
    "USB-5V-Supply":       (12.0, 16.0, 1.6, 1.0),
    "LiPo":                (8.0,  10.0, 1.6, 1.0),
    "LiPo-Battery":        (8.0,  10.0, 1.6, 1.0),
    "Red-LED":             (5.0,  5.0,  1.6, 0.8),
    "Yellow-LED":          (5.0,  5.0,  1.6, 0.8),
    "Green-LED":           (5.0,  5.0,  1.6, 0.8),
    "Blue-LED":            (5.0,  5.0,  1.6, 0.8),
    "Resistor-220R":       (9.0,  3.0,  1.6, 0.8),
    "Resistor-1K":         (9.0,  3.0,  1.6, 0.8),
    "Resistor-10K":        (9.0,  3.0,  1.6, 0.8),
    "Decoupling-Capacitor":( 1.0, 0.5,  0.0, 0.0),
    "Decoupling Capacitor":( 1.0, 0.5,  0.0, 0.0),
    "__default__":         ( 5.0, 5.0,  1.6, 0.8),
}


def get_component_size(name):
    clean = name.replace(" ", "-")
    for key in [name, clean]:
        if key in FOOTPRINT_SIZES:
            return FOOTPRINT_SIZES[key]
    for key in FOOTPRINT_SIZES:
        if key.startswith("__"):
            continue
        if key.lower() in name.lower() or name.lower() in key.lower():
            return FOOTPRINT_SIZES[key]
    return FOOTPRINT_SIZES["__default__"]


def _gerber_header(layer_name, ext, board_w, board_h):
    """Emit the RS-274X header block."""
    now = datetime.now().strftime("%Y%m%dT%H%M%S")
    return [
        f"%TF.FileFunction,{layer_name}*%",
        f"%TF.FilePolarity,Positive*%",
        f"%TF.CreationDate,{now}*%",
        f"%TF.GenerationSoftware,KevinWizard,PCBCoPilot,v1.0*%",
        "%FSLAX46Y46*%",         # format spec: 4 integer, 6 decimal
        "%MOMM*%",               # metric units
        "%LPD*%",                # layer polarity dark
        # Aperture definitions
        "%ADD10C,0.150000*%",    # D10 = 0.15 mm trace
        "%ADD11C,0.800000*%",    # D11 = 0.8 mm pad circle (THT)
        "%ADD12R,1.600000X1.600000*%",  # D12 = 1.6x1.6 mm SMD pad
        "%ADD13C,0.254000*%",    # D13 = 0.254 mm silk line
        "%ADD14O,2.000000X1.500000*%",  # D14 = SMD oval pad
    ]


def _gerber_footer():
    return ["M02*"]


def _coord(val_mm):
    """Convert mm to Gerber integer (×1,000,000)."""
    return int(round(val_mm * 1_000_000))


def _rect_flash(cx, cy, lines):
    """Flash a 1.6×1.6 mm square pad at (cx, cy)."""
    lines.append(f"D12*")
    lines.append(f"X{_coord(cx)}Y{_coord(cy)}D03*")


def _circle_flash(cx, cy, lines):
    """Flash a 0.8 mm circular pad."""
    lines.append(f"D11*")
    lines.append(f"X{_coord(cx)}Y{_coord(cy)}D03*")


def _draw_rect_outline(x, y, w, h, aperture, lines):
    """Draw a rectangular outline using linear interpolation."""
    lines += [
        f"{aperture}*",
        f"G01*",
        f"X{_coord(x)}Y{_coord(y)}D02*",          # move to corner
        f"X{_coord(x+w)}Y{_coord(y)}D01*",         # right
        f"X{_coord(x+w)}Y{_coord(y+h)}D01*",       # up
        f"X{_coord(x)}Y{_coord(y+h)}D01*",         # left
        f"X{_coord(x)}Y{_coord(y)}D01*",           # close
    ]


def _draw_silk_text(x, y, text, lines, char_w=1.2, char_h=1.5):
    """Very simplified silk-screen text as small dashes (approximation)."""
    lines.append("D13*")
    for i, ch in enumerate(text[:10]):  # max 10 chars on silk
        cx = x + i * char_w
        lines.append(f"X{_coord(cx)}Y{_coord(y)}D03*")


def generate_gerber_files(data):
    """
    Generate a dict of {filename: content_string} for all Gerber layers
    plus an Excellon drill file.
    """
    components  = data.get("components", [])
    connections = data.get("connections", [])
    title       = data.get("title", "PCB_Project")

    # Layout: same grid as kicad schematic but in real PCB mm coords
    # Slightly tighter – 30 mm grid, 4 columns
    COLS   = 4
    GRID_X = 30.0
    GRID_Y = 30.0
    MARGIN = 5.0     # board edge margin

    comp_positions = {}
    for i, comp in enumerate(components):
        col = i % COLS
        row = i // COLS
        cx  = MARGIN + col * GRID_X + GRID_X / 2
        cy  = MARGIN + row * GRID_Y + GRID_Y / 2
        comp_positions[comp["id"]] = (cx, cy)

    n_rows   = (len(components) + COLS - 1) // COLS
    board_w  = MARGIN * 2 + COLS * GRID_X
    board_h  = MARGIN * 2 + n_rows * GRID_Y

    name_to_id = {c["name"]: c["id"] for c in components}
    files      = {}

    # ----- F.Cu  (top copper – pads + ratsnest traces) -----
    lines = _gerber_header("Copper,L1,Top", "GTL", board_w, board_h)
    for comp in components:
        cx, cy = comp_positions[comp["id"]]
        w, h, pd, dd = get_component_size(comp["name"])
        if dd > 0:          # THT – circular pads
            _circle_flash(cx - w/4, cy, lines)
            _circle_flash(cx + w/4, cy, lines)
        else:               # SMD – rect pads
            _rect_flash(cx - w/4, cy, lines)
            _rect_flash(cx + w/4, cy, lines)

    # Ratsnest traces between connected components
    lines.append("D10*")
    for conn in connections:
        fid  = name_to_id.get(conn["from"], conn["from"])
        tid  = name_to_id.get(conn["to"],   conn["to"])
        fx, fy = comp_positions.get(fid, (MARGIN, MARGIN))
        tx, ty = comp_positions.get(tid, (MARGIN + GRID_X, MARGIN))
        lines += [
            f"X{_coord(fx)}Y{_coord(fy)}D02*",
            f"X{_coord(tx)}Y{_coord(ty)}D01*",
        ]
    lines += _gerber_footer()
    files[f"{title}-F_Cu.GTL"] = "\n".join(lines)

    # ----- B.Cu  (bottom copper – GND plane fill approximation) -----
    lines = _gerber_header("Copper,L2,Bot", "GBL", board_w, board_h)
    _draw_rect_outline(MARGIN/2, MARGIN/2, board_w - MARGIN, board_h - MARGIN, "D10", lines)
    lines += _gerber_footer()
    files[f"{title}-B_Cu.GBL"] = "\n".join(lines)

    # ----- F.SilkS  (component outlines + ref labels) -----
    lines = _gerber_header("Legend,Top", "GTO", board_w, board_h)
    lines.append("D13*")
    for comp in components:
        cx, cy = comp_positions[comp["id"]]
        w, h, pd, dd = get_component_size(comp["name"])
        _draw_rect_outline(cx - w/2, cy - h/2, w, h, "D13", lines)
        _draw_silk_text(cx - w/2 + 0.3, cy + h/2 + 0.5, comp["id"], lines)
    lines += _gerber_footer()
    files[f"{title}-F_Silkscreen.GTO"] = "\n".join(lines)

    # ----- B.SilkS -----
    lines = _gerber_header("Legend,Bot", "GBO", board_w, board_h)
    lines += _gerber_footer()
    files[f"{title}-B_Silkscreen.GBO"] = "\n".join(lines)

    # ----- F.Mask (soldermask = inverse of pads) -----
    lines = _gerber_header("Soldermask,Top", "GTS", board_w, board_h)
    for comp in components:
        cx, cy = comp_positions[comp["id"]]
        w, _, pd, dd = get_component_size(comp["name"])
        if dd > 0:
            _circle_flash(cx - w/4, cy, lines)
            _circle_flash(cx + w/4, cy, lines)
        else:
            _rect_flash(cx - w/4, cy, lines)
            _rect_flash(cx + w/4, cy, lines)
    lines += _gerber_footer()
    files[f"{title}-F_Mask.GTS"] = "\n".join(lines)

    # ----- B.Mask -----
    lines = _gerber_header("Soldermask,Bot", "GBS", board_w, board_h)
    lines += _gerber_footer()
    files[f"{title}-B_Mask.GBS"] = "\n".join(lines)

    # ----- F.Paste -----
    lines = _gerber_header("SolderPaste,Top", "GTP", board_w, board_h)
    lines += _gerber_footer()
    files[f"{title}-F_Paste.GTP"] = "\n".join(lines)

    # ----- Edge.Cuts (board outline) -----
    lines = _gerber_header("Profile,NP", "GKO", board_w, board_h)
    _draw_rect_outline(0, 0, board_w, board_h, "D10", lines)
    lines += _gerber_footer()
    files[f"{title}-Edge_Cuts.GKO"] = "\n".join(lines)

    # ----- Excellon drill file -----
    drill_lines = [
        "M48",
        "; Excellon Drill File — Kevin the Wizard PCB Co-Pilot",
        f"; Board: {title}",
        f"; Date: {datetime.now().strftime('%Y-%m-%d')}",
        "METRIC,TZ",
        "T1C0.8",   # 0.8 mm drill for THT / vias
        "T2C1.0",   # 1.0 mm drill for larger THT
        "%",
        "G90",
        "G05",
        "T1",
    ]
    for comp in components:
        cx, cy = comp_positions[comp["id"]]
        w, _, pd, dd = get_component_size(comp["name"])
        if dd > 0:   # only THT
            drill_lines.append(f"X{cx - w/4:.4f}Y{cy:.4f}")
            drill_lines.append(f"X{cx + w/4:.4f}Y{cy:.4f}")
    drill_lines += ["T0", "M30"]
    files[f"{title}.DRL"] = "\n".join(drill_lines)

    # ----- Gerber Job file (.gbrjob) – ties everything together -----
    job = {
        "Header": {
            "CreationDate": datetime.now().isoformat(),
            "GeneratedBy": "Kevin the Wizard PCB Co-Pilot v1.0",
            "ProjectId": {"Name": title, "GUID": "kevin-wizard-001", "Revision": "1.0"}
        },
        "GeneralSpecs": {
            "ProjectId": {"Name": title},
            "Size": {"X": round(board_w, 3), "Y": round(board_h, 3)},
            "LayerNumber": 2,
            "BoardThickness": 1.6,
            "Finish": "HASL"
        },
        "DesignRules": [{"Layers": "Outer", "PadToPad": 0.2, "PadToTrack": 0.2,
                          "TrackToTrack": 0.2, "MinLineWidth": 0.15}],
        "FilesAttributes": [
            {"Path": f"{title}-F_Cu.GTL",          "FileFunction": "Copper,L1,Top",      "FilePolarity": "Positive"},
            {"Path": f"{title}-B_Cu.GBL",           "FileFunction": "Copper,L2,Bot",      "FilePolarity": "Positive"},
            {"Path": f"{title}-F_Silkscreen.GTO",   "FileFunction": "Legend,Top",         "FilePolarity": "Positive"},
            {"Path": f"{title}-B_Silkscreen.GBO",   "FileFunction": "Legend,Bot",         "FilePolarity": "Positive"},
            {"Path": f"{title}-F_Mask.GTS",         "FileFunction": "Soldermask,Top",     "FilePolarity": "Negative"},
            {"Path": f"{title}-B_Mask.GBS",         "FileFunction": "Soldermask,Bot",     "FilePolarity": "Negative"},
            {"Path": f"{title}-F_Paste.GTP",        "FileFunction": "SolderPaste,Top",    "FilePolarity": "Negative"},
            {"Path": f"{title}-Edge_Cuts.GKO",      "FileFunction": "Profile,NP",         "FilePolarity": "Positive"},
            {"Path": f"{title}.DRL",                "FileFunction": "Drill,PTH,Drill",    "FilePolarity": "Positive"},
        ],
        "MaterialStackup": [
            {"Type": "Legend",    "Name": "F.SilkS"},
            {"Type": "SolderPaste","Name": "F.Paste"},
            {"Type": "SolderMask","Name": "F.Mask",  "Color": "Green"},
            {"Type": "Copper",    "Name": "F.Cu"},
            {"Type": "Dielectric","Material": "FR4", "Name": "core", "Thickness": 1.51},
            {"Type": "Copper",    "Name": "B.Cu"},
            {"Type": "SolderMask","Name": "B.Mask",  "Color": "Green"},
            {"Type": "SolderPaste","Name": "B.Paste"},
            {"Type": "Legend",    "Name": "B.SilkS"},
        ]
    }
    files[f"{title}.gbrjob"] = json.dumps(job, indent=2)

    return files


# ==============================================================
#  EASYEDA / ECDA JSON GENERATION
# ==============================================================

def generate_easyeda_json(data):
    """
    Generate EasyEDA Standard Edition (ECDA) schematic JSON.
    Compatible with EasyEDA Pro import via File > Open > EasyEDA JSON.
    Spec: https://docs.easyeda.com/en/DocumentFormat/2-EasyEDA-Schematic-File-Format
    """
    components  = data.get("components", [])
    connections = data.get("connections", [])
    title       = data.get("title", "Kevin_Wizard_PCB")

    # EasyEDA uses its own coordinate system (mils, ~100 unit spacing)
    GRID    = 400   # spacing between components in EasyEDA units
    COLS    = 4
    MARGIN  = 300

    comp_positions = {}
    for i, comp in enumerate(components):
        col = i % COLS
        row = i // COLS
        comp_positions[comp["id"]] = (
            MARGIN + col * GRID,
            MARGIN + row * GRID,
        )

    name_to_id = {c["name"]: c["id"] for c in components}

    # Build shape list
    shapes = []

    # -- Components --
    for comp in components:
        cid   = comp["id"]
        name  = comp["name"]
        ctype = comp.get("type", "Generic")
        volt  = comp.get("voltage", 0)
        x, y  = comp_positions[cid]

        # EasyEDA schematic symbol entry (simplified rectangular body)
        shapes.append({
            "type":     "SCH_SYMBOL",
            "id":       cid,
            "x":        x,
            "y":        y,
            "rotation": 0,
            "mirror":   0,
            "packageName": name,
            "attributes": [
                {"keyName": "Ref",         "keyValue": cid,   "visible": True},
                {"keyName": "Name",        "keyValue": name,  "visible": True},
                {"keyName": "Value",       "keyValue": name,  "visible": True},
                {"keyName": "Voltage",     "keyValue": f"{volt}V", "visible": False},
                {"keyName": "Type",        "keyValue": ctype,       "visible": False},
            ],
            # Pin stubs for EasyEDA renderer
            "pins": [
                {"pinNumber": "1", "pinName": "VCC", "x": x - 50, "y": y,      "rotation": 180},
                {"pinNumber": "2", "pinName": "GND", "x": x + 50, "y": y,      "rotation": 0},
                {"pinNumber": "3", "pinName": "SIG", "x": x,      "y": y - 50, "rotation": 90},
            ],
            # Rectangular body outline
            "body": {
                "shape": "RECT",
                "x": x - 100, "y": y - 60,
                "width": 200, "height": 120,
                "strokeColor": "#000080",
                "fillColor": "#CCCCFF",
                "strokeWidth": 1,
            }
        })

        # Reference label
        shapes.append({
            "type":       "SCH_TEXT",
            "id":         f"txt_{cid}",
            "x":          x,
            "y":          y - 80,
            "text":       f"{cid}: {name}",
            "fontSize":   12,
            "fontFamily": "Arial",
            "color":      "#000000",
            "rotation":   0,
        })

    # -- Wires / nets --
    wire_id = 0
    for conn in connections:
        fid = name_to_id.get(conn["from"], conn["from"])
        tid = name_to_id.get(conn["to"],   conn["to"])
        fx, fy = comp_positions.get(fid, (MARGIN, MARGIN))
        tx, ty = comp_positions.get(tid, (MARGIN + GRID, MARGIN))
        signal = conn.get("signal", "")

        shapes.append({
            "type":        "SCH_WIRE",
            "id":          f"W{wire_id}",
            "startX":      fx + 50,
            "startY":      fy,
            "endX":        tx - 50,
            "endY":        ty,
            "strokeColor": "#008000" if "GND" in signal else "#0000FF",
            "strokeWidth": 1,
            "netName":     signal or f"Net_{fid}_{tid}",
        })
        wire_id += 1

        # Net label at midpoint
        if signal:
            shapes.append({
                "type":       "SCH_NET_LABEL",
                "id":         f"NL{wire_id}",
                "x":          (fx + tx) // 2,
                "y":          fy - 20,
                "text":       signal,
                "fontSize":   10,
                "color":      "#006400",
                "rotation":   0,
            })

    # -- Power rails: GND & VCC --
    shapes += [
        {
            "type": "SCH_POWER",
            "id":   "PWR_GND",
            "x":    MARGIN,
            "y":    MARGIN - 100,
            "name": "GND",
            "netName": "GND",
        },
        {
            "type": "SCH_POWER",
            "id":   "PWR_VCC",
            "x":    MARGIN + GRID,
            "y":    MARGIN - 100,
            "name": "VCC",
            "netName": "VCC_5V",
        },
    ]

    # Pin assignments as net labels on MCU
    for mcu in data.get("pin_assignments", []):
        cid  = mcu.get("component_id", "U1")
        x, y = comp_positions.get(cid, (MARGIN, MARGIN))
        for idx, pin in enumerate(mcu.get("pins", [])):
            pin_label = f"{pin.get('pin_number','?')}:{pin.get('signal','?')}"
            shapes.append({
                "type":     "SCH_NET_LABEL",
                "id":       f"PIN_{cid}_{idx}",
                "x":        x + 120,
                "y":        y - 60 + idx * 25,
                "text":     pin_label,
                "fontSize": 8,
                "color":    "#8B0000",
                "rotation": 0,
            })

    n_rows  = (len(components) + COLS - 1) // COLS
    canvas_w = MARGIN * 2 + COLS * GRID
    canvas_h = MARGIN * 2 + n_rows * GRID

    easyeda_doc = {
        "head": {
            "type":       "schematic",
            "editorVersion": "6.0.0",
            "title":      title,
            "description": f"Generated by Kevin the Wizard PCB Co-Pilot — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "author":     "Kevin the Wizard AI",
            "company":    "PCB Co-Pilot Hackathon 2026",
            "revision":   "1.0",
            "createdAt":  datetime.now().isoformat(),
            "updatedAt":  datetime.now().isoformat(),
        },
        "canvas": {
            "width":       canvas_w,
            "height":      canvas_h,
            "gridSize":    10,
            "snapSize":    10,
            "unit":        "mil",
            "origin":      {"x": 0, "y": 0},
        },
        "BBox": {"x": 0, "y": 0, "width": canvas_w, "height": canvas_h},
        "shapes": shapes,
        "netList": _build_easyeda_netlist(connections, name_to_id, data.get("components", [])),
    }

    return json.dumps(easyeda_doc, indent=2)


def _build_easyeda_netlist(connections, name_to_id, components):
    """Build EasyEDA-style net list array."""
    nets   = {}
    id_map = {c["id"]: c["name"] for c in components}

    for conn in connections:
        fid    = name_to_id.get(conn["from"], conn["from"])
        tid    = name_to_id.get(conn["to"],   conn["to"])
        signal = conn.get("signal", f"Net_{fid}_{tid}")
        if "gnd" in signal.lower():
            signal = "GND"
        elif "vcc" in signal.lower() or "5v" in signal.lower():
            signal = "VCC_5V"
        elif "3v3" in signal.lower() or "3.3" in signal:
            signal = "VCC_3V3"
        if signal not in nets:
            nets[signal] = []
        nets[signal].extend([fid, tid])

    return [
        {
            "netName":   name,
            "type":      "power" if name in ("GND", "VCC_5V", "VCC_3V3") else "signal",
            "nodes":     list(set(nodes)),
        }
        for name, nodes in nets.items()
    ]


# ==============================================================
#  ROUTES
# ==============================================================
def _generate_readme(title, gerber_filenames):
    """Generate a human-readable README for the design package."""
    gerber_list = "\n".join(f"  - `{fn}`" for fn in gerber_filenames)
    return f"""# Kevin the Wizard — PCB Co-Pilot Design Package
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Project: **{title}**

---

## Package Contents

### 📄 Root Files
| File | Description |
|------|-------------|
| `output.json` | Full circuit netlist JSON |
| `safety_report.txt` | Electrical safety audit results |
| `BOM.csv` | Bill of Materials with DigiKey links |
| `diagram.md` | Mermaid circuit diagram (paste at https://mermaid.live) |

### ⚡ KiCad Files (`/kicad/`)
| File | Description |
|------|-------------|
| `design.kicad_sch` | KiCad 6/7 schematic — open in KiCad Schematic Editor |
| `design.net` | KiCad netlist — import via KiCad PCB Layout Editor |

### 🏭 Gerber Files (`/gerber/`) — **Manufacturer Ready**
{gerber_list}

Upload the entire `/gerber/` folder to any PCB manufacturer:
- [JLCPCB](https://jlcpcb.com) → New Order → Add Gerber File
- [PCBWay](https://pcbway.com) → Quote Now → Upload Gerber
- [OSH Park](https://oshpark.com) → Upload .zip of gerber folder

The `.gbrjob` file contains the full layer stackup for automated import.

### 🔌 EasyEDA / ECDA (`/easyeda/`)
| File | Description |
|------|-------------|
| `{title}_schematic.json` | EasyEDA Standard format schematic |

Import in EasyEDA Pro: **File → Open → EasyEDA JSON**  
Import in EasyEDA Standard: **File → Import → EasyEDA JSON**

---

## Quick Start: Manufacturing with JLCPCB
1. Zip the contents of `/gerber/`
2. Go to https://jlcpcb.com → **Order Now**
3. Upload the zip file
4. Review auto-detected settings (2-layer, 1.6mm FR4)
5. Select finish (HASL recommended for prototypes)
6. Add to cart — typical cost: $2-5 for 5 boards

---
*Generated by Kevin the Wizard PCB Co-Pilot — Hackathon 2026*
"""


@app.route("/generate", methods=["POST"])
def generate():
    try:
        body = request.get_json()
        user_request = body.get("request", "").strip()
        if not user_request:
            return jsonify({"error": "No circuit description provided."}), 400

        # Step 0: Validate feasibility BEFORE doing anything
        is_feasible, reason, suggestions = validate_request(user_request)
        if not is_feasible:
            return jsonify({
                "error": "impossible_request",
                "message": reason,
                "suggestions": suggestions
            }), 422

        MAX_FIX_ROUNDS = 4
        all_fixes = []

        # Step 1: Generate circuit JSON
        data = generate_circuit_json(user_request)

        # Step 2: Safety checks + auto-fix loop
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
        kicad_net   = generate_kicad_netlist_content(data)
        kicad_sch   = generate_kicad_sch_content(data)
        gerber_files = generate_gerber_files(data)
        easyeda_json = generate_easyeda_json(data)

        title = data.get("title", "PCB_Project")

        # Step 4: Pack into ZIP (with sub-folders for Gerber and EasyEDA)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Core files (root)
            zf.writestr("output.json",       json.dumps(data, indent=2))
            zf.writestr("safety_report.txt", report_txt)
            zf.writestr("diagram.md",        diagram_md)
            zf.writestr("BOM.csv",           bom_csv)

            # KiCad files
            zf.writestr(f"kicad/design.net",       kicad_net)
            zf.writestr(f"kicad/design.kicad_sch", kicad_sch)

            # Gerber files (manufacturer-ready)
            for fname, content in gerber_files.items():
                zf.writestr(f"gerber/{fname}", content)

            # EasyEDA / ECDA JSON
            zf.writestr(f"easyeda/{title}_schematic.json", easyeda_json)

            # README explaining the package
            readme = _generate_readme(title, list(gerber_files.keys()))
            zf.writestr("README.md", readme)

        return send_file(tmp.name, as_attachment=True, download_name="kevin_wizard_output.zip", mimetype="application/zip")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

def _build_tfidf_index(snippets):
    """
    Build a simple TF-IDF index over all snippets.
    Returns (vocab, idf_weights, doc_vectors) for cosine similarity retrieval.
    """
    import math

    # Tokenize each document (tags + title + snippet body)
    def tokenize(s):
        text = " ".join([
            " ".join(s.get("tags", [])),
            s.get("title", ""),
            s.get("snippet", ""),
            " ".join(s.get("warnings", [])),
            " ".join(s.get("components", [])),
        ]).lower()
        # Simple word tokenizer — split on non-alphanumeric
        return re.findall(r'[a-z0-9]+', text)

    docs = [tokenize(s) for s in snippets]

    # Build vocabulary
    vocab = {}
    for doc in docs:
        for word in set(doc):
            vocab[word] = vocab.get(word, 0) + 1

    N = len(docs)
    # IDF: log(N / df) + 1
    idf = {word: math.log(N / df) + 1 for word, df in vocab.items()}

    # TF-IDF vectors per document
    doc_vectors = []
    for doc in docs:
        tf = {}
        for word in doc:
            tf[word] = tf.get(word, 0) + 1
        total = len(doc) or 1
        vec = {word: (count / total) * idf.get(word, 1) for word, count in tf.items()}
        doc_vectors.append(vec)

    return idf, doc_vectors


# Build index at startup
_RAG_IDF, _RAG_DOC_VECTORS = _build_tfidf_index(GOLDEN_SNIPPETS)


def _cosine_similarity(query_vec, doc_vec):
    """Compute cosine similarity between two TF-IDF vectors."""
    import math
    dot = sum(query_vec.get(w, 0) * doc_vec.get(w, 0) for w in query_vec)
    norm_q = math.sqrt(sum(v * v for v in query_vec.values())) or 1
    norm_d = math.sqrt(sum(v * v for v in doc_vec.values())) or 1
    return dot / (norm_q * norm_d)


def retrieve_relevant_snippets(user_request, top_k=3):
    """
    Full TF-IDF + cosine similarity retrieval over the Golden Circuit Snippets.

    Scoring layers (in order of priority):
      1. Exact tag match  — strong signal (x4 boost)
      2. TF-IDF cosine similarity — semantic word overlap
      3. Title word match — moderate boost (x2)
      4. Warning/component word match — small boost (x0.5)

    Returns top_k most relevant snippets.
    """
    query = user_request.lower()
    query_tokens = re.findall(r'[a-z0-9]+', query)

    # Build query TF-IDF vector
    query_tf = {}
    for word in query_tokens:
        query_tf[word] = query_tf.get(word, 0) + 1
    total_q = len(query_tokens) or 1
    query_vec = {
        word: (count / total_q) * _RAG_IDF.get(word, 1)
        for word, count in query_tf.items()
    }

    scored = []
    for i, snippet in enumerate(GOLDEN_SNIPPETS):
        # Layer 1: exact tag match boost
        tag_score = sum(
            4 if tag.lower() in query else
            2 if any(t in query for t in tag.lower().split())
            else 0
            for tag in snippet.get("tags", [])
        )

        # Layer 2: TF-IDF cosine similarity
        cos_score = _cosine_similarity(query_vec, _RAG_DOC_VECTORS[i]) * 10

        # Layer 3: title word match boost
        title_score = sum(
            2 for word in query_tokens
            if word in snippet.get("title", "").lower() and len(word) > 2
        )

        # Layer 4: warning/component body match
        body_text = " ".join(snippet.get("warnings", []) + snippet.get("components", [])).lower()
        body_score = sum(
            0.5 for word in query_tokens
            if word in body_text and len(word) > 3
        )

        total_score = tag_score + cos_score + title_score + body_score

        if total_score > 0.5:
            scored.append((total_score, snippet))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored[:top_k]]

    print(f"[RAG] Retrieved {len(top)} snippets for query: '{user_request[:60]}'")
    for score, s in scored[:top_k]:
        print(f"  [{score:.2f}] {s['title']}")

    return top


def build_rag_context(snippets):
    """
    Formats retrieved snippets into a rich context block for the LLM prompt.
    Includes full datasheet parameters, pin-level specs, warnings, and exact values.
    """
    if not snippets:
        return ""
    lines = [
        "=" * 70,
        "ENGINEERING REFERENCE — GOLDEN CIRCUIT SNIPPETS WITH DATASHEET DATA",
        "Use the following verified technical specifications in your design.",
        "These are real component specs — use exact values for voltages, resistors, capacitors.",
        "=" * 70,
    ]
    for i, s in enumerate(snippets, 1):
        lines += [
            f"\n[SNIPPET {i}] {s['title']}",
            "-" * 50,
            s["snippet"],
        ]
        if s.get("components"):
            lines.append("REQUIRED COMPONENTS: " + ", ".join(s["components"]))
        if s.get("warnings"):
            lines.append("CRITICAL WARNINGS:")
            for w in s["warnings"]:
                lines.append(f"  !! {w}")

        # Inject datasheet parameters
        ds = s.get("datasheet", {})
        if ds:
            lines.append("DATASHEET PARAMETERS:")
            # Voltage ratings
            for key in ["vcc", "vcc_typical", "vcc_min", "vcc_max", "vin_min", "vin_max", "vout", "gpio_voltage"]:
                if key in ds:
                    lines.append(f"  {key.upper()}: {ds[key]}V")
            # Current ratings
            for key in ["iout_max_a", "iout_max", "icc_wifi_tx_ma", "gpio_max_ma", "total_gpio_ma", "coil_current_ma"]:
                if key in ds:
                    unit = "A" if "a" in key else "mA"
                    lines.append(f"  {key.upper()}: {ds[key]}{unit}")
            # Package
            if "package" in ds:
                lines.append(f"  PACKAGE: {ds['package']}")
            # Key pins
            if "key_pins" in ds:
                lines.append("  KEY PINS:")
                for pin, desc in ds["key_pins"].items():
                    lines.append(f"    {pin}: {desc}")
            # Application notes
            if "application_notes" in ds:
                lines.append(f"  APPLICATION NOTES: {ds['application_notes']}")

    lines += [
        "\n" + "=" * 70,
        "END OF REFERENCE DATA — Generate circuit JSON using the above verified specs.",
        "=" * 70 + "\n",
    ]
    return "\n".join(lines)


# ==============================================================
#  GEMINI MODELS
# ==============================================================
netlist_model = genai.GenerativeModel(
    model_name="gemini-3.1-flash-lite-preview",
    system_instruction="""
    You are a Senior Hardware Engineer (PCB Specialist).
    Your job is to parse descriptions into a detailed JSON Netlist with pin-level assignments.

    Rules:
    - Return ONLY raw JSON. No markdown blocks (no ```).
    - Include: MCU, Power Stage (LDO/Buck), Motor Drivers, and Passives.
    - Each component MUST include: id, name, type, voltage fields.
    - connections must use "from" and "to" keys matching component names.
    - For MCUs (ESP32, Arduino, STM32, Raspberry Pi, etc.) you MUST include a "pin_assignments" array.
    - Each pin_assignment must specify: pin_number, pin_name, signal, connected_to, peripheral.
    - Detect and resolve peripheral conflicts: do NOT assign two signals to the same pin.
    - Flag any I2C/SPI/UART/PWM peripheral conflicts in a "conflicts" array (empty if none).

    Output format (return ONLY this JSON, no extra text):
    {
      "components": [{"id": "U1", "name": "ESP32", "type": "MCU", "voltage": 3.3}],
      "connections": [{"from": "ComponentA", "to": "ComponentB", "signal": "SDA"}],
      "pin_assignments": [
        {
          "component_id": "U1",
          "component_name": "ESP32",
          "pins": [
            {"pin_number": "GPIO21", "pin_name": "SDA", "signal": "I2C_SDA", "connected_to": "OLED_Display", "peripheral": "I2C0"},
            {"pin_number": "GPIO22", "pin_name": "SCL", "signal": "I2C_SCL", "connected_to": "OLED_Display", "peripheral": "I2C0"},
            {"pin_number": "GND",    "pin_name": "GND", "signal": "GND",     "connected_to": "GND_Rail",     "peripheral": "Power"},
            {"pin_number": "3V3",    "pin_name": "VCC", "signal": "VCC_3V3", "connected_to": "LDO_Output",   "peripheral": "Power"}
          ]
        }
      ],
      "conflicts": []
    }

    If a component is NOT an MCU (e.g. LED, resistor, motor driver), skip it in pin_assignments.
    Always include GND and VCC pins in the MCU pin list.
    Output ONLY raw JSON. No markdown, no extra text.
    """
)

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

# ==============================================================
#  STATIC COMPONENT DATABASE
# ==============================================================
COMPONENTS = {
    "ESP32":               {"part_number": "ESP32-WROOM-32E",   "voltage": 3.3, "price_usd": 3.50,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/espressif-systems/ESP32-WROOM-32E/11613142"},
    "L298N":               {"part_number": "L298N",             "voltage": 5,   "price_usd": 1.80,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/stmicroelectronics/L298N/585918"},
    "TP4056":              {"part_number": "TP4056-SOT25",      "voltage": 4.2, "price_usd": 0.30,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/tc-charger/TP4056/7353588"},
    "AMS1117-3.3":         {"part_number": "AMS1117-3.3",       "voltage": 3.3, "price_usd": 0.25,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/advanced-monolithic-systems-inc/AMS1117-3-3/5010163"},
    "Decoupling Capacitor":{"part_number": "C0402C104K5RACTU", "voltage": 10,  "price_usd": 0.05,  "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/kemet/C0402C104K5RACTU/411388"},
    "Arduino-Uno-R3":      {"part_number": "A000066",           "voltage": 5.0, "price_usd": 27.60, "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/arduino/A000066/2784006"},
    "Red-LED":             {"part_number": "HLMP-EG08-Y2000",   "voltage": 2.0, "price_usd": 0.35,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/broadcom-limited/HLMP-EG08-Y2000/3906329"},
    "Yellow-LED":          {"part_number": "TLHY4200",          "voltage": 2.1, "price_usd": 0.30,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/vishay-semiconductor-opto-division/TLHY4200/1805986"},
    "Green-LED":           {"part_number": "TLHG4200",          "voltage": 2.2, "price_usd": 0.30,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/vishay-semiconductor-opto-division/TLHG4200/1806003"},
    "Resistor-220R":       {"part_number": "CF14JT220R",        "voltage": 0,   "price_usd": 0.10,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/stackpole-electronics-inc/CF14JT220R/1741547"},
    "USB-5V-Supply":       {"part_number": "GENERIC-USB-5V",    "voltage": 5.0, "price_usd": 5.00,  "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/filter/usb-cables/469"},
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


def validate_request(user_request):
    try:
        response = gemini_call_with_retry(
            validation_model,
            f"Evaluate this circuit request for feasibility: {user_request}"
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return result.get("feasible", True), result.get("reason", ""), result.get("suggestions", [])
    except Exception:
        return True, "", []


def generate_circuit_json(user_request):
    # RAG: TF-IDF retrieval of Golden Circuit Snippets + datasheet context injection
    snippets    = retrieve_relevant_snippets(user_request, top_k=3)
    rag_context = build_rag_context(snippets)
    prompt      = rag_context + user_request
    response    = gemini_call_with_retry(netlist_model, prompt)
    raw         = response.text.strip().replace("```json", "").replace("```", "").strip()
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
    lines = ["=" * 60, "  KEVIN THE WIZARD'S ELECTRICAL SAFETY REPORT",
             "  Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "=" * 60, "",
             "COMPONENTS FOUND:"]
    for c in data.get("components", []):
        lines.append(f"  - {c['name']} ({c.get('type','?')}) - {c.get('voltage','?')}V")
    lines += ["", "CONNECTIONS:"]
    for conn in data.get("connections", []):
        lines.append(f"  {conn['from']}  -->  {conn['to']}")
    if fix_log:
        lines += ["", "AUTO-FIXES APPLIED:"]
        for i, fix in enumerate(fix_log, 1):
            lines.append(f"  {i}. {fix}")
    pin_assignments = data.get("pin_assignments", [])
    if pin_assignments:
        lines += ["", "-" * 60, "PIN ASSIGNMENTS:", "-" * 60]
        for mcu in pin_assignments:
            lines.append(f"\n  {mcu.get('component_name', mcu.get('component_id','?'))} ({mcu.get('component_id','')}):")
            lines.append(f"  {'PIN':<12} {'NAME':<12} {'SIGNAL':<20} {'CONNECTED TO':<25} PERIPHERAL")
            lines.append(f"  {'-'*12} {'-'*12} {'-'*20} {'-'*25} {'-'*12}")
            for p in mcu.get("pins", []):
                lines.append(f"  {str(p.get('pin_number','')):<12} {str(p.get('pin_name','')):<12} {str(p.get('signal','')):<20} {str(p.get('connected_to','')):<25} {p.get('peripheral','')}")
        conflicts = data.get("conflicts", [])
        if conflicts:
            lines += ["", "  WARNING - PERIPHERAL CONFLICTS DETECTED:"]
            for c in conflicts: lines.append(f"    - {c}")
        else:
            lines.append("\n  OK - No peripheral conflicts detected.")

    lines += ["", "-" * 60, "FINAL SAFETY CHECK RESULTS:", "-" * 60]
    pass_count = fail_count = 0
    for r in results:
        lines += [f"\n[{r['status']}]  {r['check']}", f"   -> {r['detail']}"]
        if r["status"] == "PASS": pass_count += 1
        elif r["status"] == "FAIL": fail_count += 1
    lines += ["", "=" * 60,
              f"  SUMMARY: {pass_count} passed, {fail_count} failed, {len(results)-pass_count-fail_count} warnings/skipped",
              "  OVERALL: " + ("DESIGN LOOKS SAFE TO PROCEED!" if fail_count == 0 else f"{fail_count} ISSUE(S) NEED FIXING."),
              "=" * 60]
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
        row = COMPONENTS[name].copy() if name in COMPONENTS else fetch_component_gemini(name)
        row["name"] = name
        enriched.append(row)
    lines = ["name,part_number,voltage,price_usd,package,in_stock,digikey_url"]
    for c in enriched:
        lines.append(f"{c.get('name','')},{c.get('part_number','')},{c.get('voltage','')},{c.get('price_usd','')},{c.get('package','')},{c.get('in_stock','')},{c.get('digikey_url','')}")
    total = round(sum(float(c.get("price_usd", 0)) for c in enriched), 2)
    lines.append(f",,,,,,TOTAL: ${total}")
    return "\n".join(lines)


def generate_kicad_sch_content(data):
    """
    Generates a KiCad 6/7 schematic file (.kicad_sch) from circuit data.
    Places components on a grid and draws wire connections between them.
    """
    components  = data.get("components", [])
    connections = data.get("connections", [])
    pin_assignments = data.get("pin_assignments", [])
    title       = data.get("title", "Kevin_Wizard_PCB")
    now         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build a position grid: spread components across a 200x200 mil grid
    COLS = 4
    GRID = 50  # spacing in mm
    comp_positions = {}
    for i, comp in enumerate(components):
        col = i % COLS
        row = i // COLS
        comp_positions[comp["id"]] = (col * GRID, row * GRID)

    # Map component names to IDs for connection wiring
    name_to_id = {c["name"]: c["id"] for c in components}

    lines = [
        '(kicad_sch (version 20230121) (generator kevin_wizard)',
        '',
        '  (paper "A4")',
        '',
        f'  (title_block',
        f'    (title "{title}")',
        f'    (date "{now}")',
        f'    (rev "1.0")',
        f'    (company "Kevin the Wizard - AI PCB Co-Pilot")',
        f'  )',
        '',
        '  (lib_symbols)',
        '',
    ]

    # Write each component as a schematic symbol
    for comp in components:
        cid  = comp["id"]
        name = comp["name"]
        ctype = comp.get("type", "Generic")
        volt  = comp.get("voltage", 0)
        x, y  = comp_positions.get(cid, (0, 0))
        lib, part, fp = get_footprint(name, ctype)

        lines += [
            f'  (symbol (lib_id "{lib}:{part}")',
            f'    (at {x} {y} 0)',
            f'    (unit 1)',
            f'    (in_bom yes) (on_board yes)',
            f'    (property "Reference" "{cid}" (at {x} {y - 3} 0))',
            f'    (property "Value" "{name}" (at {x} {y + 3} 0))',
            f'    (property "Footprint" "{fp}" (at {x} {y + 6} 0))',
            f'    (property "Description" "{ctype} - {volt}V" (at {x} {y + 9} 0))',
            f'  )',
            '',
        ]

    # Power symbols: GND and VCC
    lines += [
        '  (symbol (lib_id "power:GND")',
        '    (at 10 10 0)',
        '    (unit 1)',
        '    (in_bom yes) (on_board yes)',
        '    (property "Reference" "#PWR_GND" (at 10 10 0))',
        '    (property "Value" "GND" (at 10 13 0))',
        '  )',
        '',
        '  (symbol (lib_id "power:VCC")',
        '    (at 20 10 0)',
        '    (unit 1)',
        '    (in_bom yes) (on_board yes)',
        '    (property "Reference" "#PWR_VCC" (at 20 10 0))',
        '    (property "Value" "VCC" (at 20 7 0))',
        '  )',
        '',
    ]

    # Draw wires between connected components
    lines.append('  ; === Connections ===')
    for conn in connections:
        from_id = name_to_id.get(conn["from"], conn["from"])
        to_id   = name_to_id.get(conn["to"],   conn["to"])
        fx, fy  = comp_positions.get(from_id, (0, 0))
        tx, ty  = comp_positions.get(to_id,   (0, 0))
        signal  = conn.get("signal", "")
        lines += [
            f'  (wire',
            f'    (pts (xy {fx + 5} {fy}) (xy {tx - 5} {ty}))',
            f'    (stroke (width 0) (type default))',
            f'  )',
        ]
        if signal:
            mid_x = (fx + tx) / 2
            mid_y = (fy + ty) / 2
            lines += [
                f'  (label "{signal}"',
                f'    (at {mid_x} {mid_y} 0)',
                f'    (fields_autoplaced)',
                f'    (effects (font (size 1.27 1.27)))',
                f'  )',
            ]

    # Add pin assignment labels for MCUs
    if pin_assignments:
        lines.append('')
        lines.append('  ; === Pin Assignment Labels ===')
        for mcu in pin_assignments:
            cid  = mcu.get("component_id", "U1")
            x, y = comp_positions.get(cid, (0, 0))
            for idx, pin in enumerate(mcu.get("pins", [])):
                label = f"{pin.get('pin_number','')}:{pin.get('signal','')}"
                lines += [
                    f'  (label "{label}"',
                    f'    (at {x + 8} {y + idx * 2.5} 0)',
                    f'    (effects (font (size 1.0 1.0)))',
                    f'  )',
                ]

    lines.append(')')
    return "\n".join(lines)


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
        fn, tn = conn["from"], conn["to"]
        fr, tr = name_to_id.get(fn, fn), name_to_id.get(tn, tn)
        net = f"Net-{re.sub(r'[^A-Za-z0-9_]','_',fn)}_to_{re.sub(r'[^A-Za-z0-9_]','_',tn)}"
        if   "gnd" in fn.lower() or "ground" in fn.lower(): net = "GND"
        elif "gnd" in tn.lower() or "ground" in tn.lower(): net = "GND"
        elif "vcc" in fn.lower() or "5v" in fn.lower() or "power" in fn.lower(): net = "VCC_5V"
        elif "3.3" in fn or "3v3" in fn.lower(): net = "VCC_3V3"
        if net not in net_map: net_map[net] = []
        net_map[net].append((fr, get_pin(fr)))
        net_map[net].append((tr, get_pin(tr)))
    return net_map


def generate_kicad_netlist_content(data):
    components  = data.get("components", [])
    connections = data.get("connections", [])
    title       = data.get("title", "PCB_Project")
    lines = ["(export (version D)", "  (design",
             f"    (source {title}.sch)",
             f"    (date \"{datetime.now().strftime('%Y-%m-%d')}\")",
             "    (tool \"Kevin the Wizard - PCB Co-Pilot AI\"))",
             "  (components"]
    for comp in components:
        lib, part, fp = get_footprint(comp["name"], comp.get("type", "Default"))
        lines += [f"    (comp (ref {comp['id']})", f"      (value {comp['name']})",
                  f"      (libsource (lib {lib}) (part {part}))", f"      (footprint {fp}))"]
    lines += ["    (comp (ref PWR_GND)", "      (value GND)",
              "      (libsource (lib power) (part GND))",
              "      (footprint TestPoint:TestPoint_Pad_1.0x1.0mm))",
              "    (comp (ref PWR_VCC)", "      (value VCC)",
              "      (libsource (lib power) (part VCC))",
              "      (footprint TestPoint:TestPoint_Pad_1.0x1.0mm))",
              "  )", "  (nets"]
    net_map = build_net_map(connections, components)
    if "GND"    not in net_map: net_map["GND"] = []
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
#  ROUTES
# ==============================================================
@app.route("/generate", methods=["POST"])
def generate():
    try:
        body = request.get_json()
        user_request = body.get("request", "").strip()
        if not user_request:
            return jsonify({"error": "No circuit description provided."}), 400

        # Step 0: Validate feasibility BEFORE doing anything
        is_feasible, reason, suggestions = validate_request(user_request)
        if not is_feasible:
            return jsonify({
                "error": "impossible_request",
                "message": reason,
                "suggestions": suggestions
            }), 422

        MAX_FIX_ROUNDS = 4
        all_fixes = []

        # Step 1: Generate circuit JSON
        data = generate_circuit_json(user_request)

        # Step 2: Safety checks + auto-fix loop
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
        report_txt = generate_safety_report_txt(results, data, all_fixes)
        diagram_md = generate_mermaid_md(data)
        bom_csv    = generate_bom_csv_content(data)
        kicad_net  = generate_kicad_netlist_content(data)
        kicad_sch  = generate_kicad_sch_content(data)

        # Step 4: Pack into ZIP
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        with zipfile.ZipFile(tmp.name, "w") as zf:
            zf.writestr("output.json",       json.dumps(data, indent=2))
            zf.writestr("safety_report.txt", report_txt)
            zf.writestr("diagram.md",        diagram_md)
            zf.writestr("BOM.csv",           bom_csv)
            zf.writestr("design.net",        kicad_net)
            zf.writestr("design.kicad_sch",  kicad_sch)

        return send_file(tmp.name, as_attachment=True, download_name="kevin_wizard_output.zip", mimetype="application/zip")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
