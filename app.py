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
#  RAG PIPELINE — Golden Circuit Snippets Knowledge Base
#  Full TF-IDF weighted retrieval with datasheet parameters
# ==============================================================
_SNIPPETS_PATH = os.path.join(os.path.dirname(__file__), "golden_snippets.json")


def _load_snippets():
    try:
        with open(_SNIPPETS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


GOLDEN_SNIPPETS = _load_snippets()


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
