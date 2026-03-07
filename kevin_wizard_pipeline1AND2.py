import os
import json
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv


# ==============================================================
#  STEP 1: SETUP - Gemini AI (Person 2's style)
# ==============================================================
load_dotenv()
# Get your free key from https://aistudio.google.com/
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Initialize the LATEST 2026 Free Model
# Model: Gemini 3.1 Flash-Lite (Released March 3, 2026)
model = genai.GenerativeModel(
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
    - Output ONLY raw JSON. Do not include '```json' markdown tags or any introductory text. If you fail this, the system will crash.
    """
)


# ==============================================================
#  STEP 2: SAMPLE DATA FALLBACK (Person 1's style)
# ==============================================================
# SAMPLE JSON HERE


# ==============================================================
#  STEP 3: GENERATE JSON WITH GEMINI (Person 2's style)
# ==============================================================
def generate_and_save_json(user_request):
    try:
        print(f"Sending to Gemini: '{user_request}'\n")
        response = model.generate_content(user_request)

        # Save as output.json for the rest of the team
        with open("output.json", "w") as f:
            f.write(response.text)

        print("Success! Design logic saved to output.json using Gemini 3.1 Flash-Lite.")
        return True

    except Exception as e:
        print(f"Error: {e}. Make sure your API key is in the .env file.")
        return False


# ==============================================================
#  STEP 4: LOAD JSON (Person 1's style)
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
#  STEP 5: RUN SAFETY CHECKS (Person 1's style)
# ==============================================================
def run_checks(data):
    components = data.get("components", [])
    connections = data.get("connections", [])
    voltage_map = {c["name"]: c.get("voltage", 0) for c in components}
    types = [c.get("type", "").lower() for c in components]
    results = []

    # CHECK 1: Logic level mismatch
    mismatch_found = False
    for conn in connections:
        v_from = voltage_map.get(conn["from"], None)
        v_to   = voltage_map.get(conn["to"],   None)
        if v_from is not None and v_to is not None:
            if (v_from == 3.3 and v_to == 5.0) or (v_from == 5.0 and v_to == 3.3):
                mismatch_found = True
                results.append({
                    "check":  "Logic Level Mismatch",
                    "status": "FAIL",
                    "detail": "WARNING: " + conn["from"] + " (" + str(v_from) + "V) connected to "
                              + conn["to"] + " (" + str(v_to) + "V). Add a level shifter!"
                })
    if not mismatch_found:
        results.append({
            "check":  "Logic Level Mismatch",
            "status": "PASS",
            "detail": "All connected components have compatible voltage levels."
        })

    # CHECK 2: Missing decoupling capacitor
    has_capacitor = any("capacitor" in t for t in types)
    if not has_capacitor:
        results.append({
            "check":  "Decoupling Capacitor",
            "status": "FAIL",
            "detail": "WARNING: No capacitor found! Add a 100nF capacitor near the MCU power pin."
        })
    else:
        results.append({
            "check":  "Decoupling Capacitor",
            "status": "PASS",
            "detail": "Capacitor found in the design."
        })

    # CHECK 3: Battery voltage vs components
    battery = next((c for c in components if c.get("type") == "Battery"), None)
    if battery:
        batt_v = battery.get("voltage", 0)
        overpowered = [c["name"] for c in components
                       if c.get("type") != "Battery" and c.get("voltage", 0) > batt_v]
        if overpowered:
            results.append({
                "check":  "Battery Voltage Sufficiency",
                "status": "FAIL",
                "detail": "WARNING: Battery is " + str(batt_v) + "V but these need more: " + ", ".join(overpowered)
            })
        else:
            results.append({
                "check":  "Battery Voltage Sufficiency",
                "status": "PASS",
                "detail": "Battery voltage (" + str(batt_v) + "V) is sufficient for all components."
            })
    else:
        results.append({
            "check":  "Battery Voltage Sufficiency",
            "status": "WARN",
            "detail": "No battery found. If USB-powered, ignore this warning."
        })

    # CHECK 4: LED resistor
    has_led = any("Red-LED-Resistor" in t for t in types)
    has_resistor = any("resistor" in t for t in types)
    if has_led and not has_resistor:
        results.append({
            "check":  "LED Current Limiting Resistor",
            "status": "FAIL",
            "detail": "WARNING: LED found but no resistor! Add a 220-470 ohm resistor in series."
        })
    elif has_led and has_resistor:
        results.append({
            "check":  "LED Current Limiting Resistor",
            "status": "PASS",
            "detail": "LED and resistor both present."
        })
    else:
        results.append({
            "check":  "LED Current Limiting Resistor",
            "status": "SKIP",
            "detail": "No LED in design - check skipped."
        })

    # CHECK 5: MCU present
    has_mcu = any("mcu" in t or "microcontroller" in t for t in types)
    if not has_mcu:
        results.append({"check": "MCU Present", "status": "WARN", "detail": "No MCU detected. Is this intentional?"})
    else:
        results.append({"check": "MCU Present", "status": "PASS", "detail": "MCU detected in the design."})

    return results


# ==============================================================
#  STEP 6: PRINT & SAVE REPORT (Person 1's style)
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
#  MAIN PIPELINE
# ==============================================================
if __name__ == "__main__":
    print("Kevin the Wizard's Safety Checker starting...\n")

    # The Prompt - describe your circuit here
    user_request = "traffic signal circuit using arduino uno r3"

    # Generate JSON from Gemini, then load and check it
    generate_and_save_json(user_request)
    data = load_json()
    results = run_checks(data)
    print_and_save_report(results, data)
