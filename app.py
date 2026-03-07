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
#  STATIC COMPONENT DATABASE
# ==============================================================
COMPONENTS = {
    "ESP32": {"part_number": "ESP32-WROOM-32E", "voltage": 3.3, "price_usd": 3.50, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/espressif-systems/ESP32-WROOM-32E/11613142"},
    "L298N": {"part_number": "L298N", "voltage": 5, "price_usd": 1.80, "package": "THT", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/stmicroelectronics/L298N/585918"},
    "TP4056": {"part_number": "TP4056-SOT25", "voltage": 4.2, "price_usd": 0.30, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/tc-charger/TP4056/7353588"},
    "AMS1117-3.3": {"part_number": "AMS1117-3.3", "voltage": 3.3, "price_usd": 0.25, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/advanced-monolithic-systems-inc/AMS1117-3-3/5010163"},
    "Decoupling Capacitor": {"part_number": "C0402C104K5RACTU", "voltage": 10, "price_usd": 0.05, "package": "SMD", "in_stock": True, "digikey_url": "https://www.digikey.com/en/products/detail/kemet/C0402C104K5RACTU/411388"},
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

        # Step 3: Generate all 3 output files
        report_txt = generate_safety_report_txt(results, data, all_fixes)
        diagram_md = generate_mermaid_md(data)
        bom_csv = generate_bom_csv_content(data)

        # Step 4: Pack into ZIP
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        with zipfile.ZipFile(tmp.name, "w") as zf:
            zf.writestr("output.json", json.dumps(data, indent=2))
            zf.writestr("safety_report.txt", report_txt)
            zf.writestr("diagram.md", diagram_md)
            zf.writestr("BOM.csv", bom_csv)

        return send_file(tmp.name, as_attachment=True, download_name="kevin_wizard_output.zip", mimetype="application/zip")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 5000)))
