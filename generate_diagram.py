import json
import os

# ==============================================================
#  STEP 1: LOAD JSON (from Person 1's output)
# ==============================================================
def load_json():
    if os.path.exists("output.json"):
        print("Found output.json - using real data!\n")
        with open("output.json", "r") as f:
            return json.load(f)
    else:
        print("output.json not found - using sample data for now.\n")
        return {
            "components": [
                {"name": "ESP32",  "type": "MCU",         "voltage": 3.3},
                {"name": "L298N",  "type": "MotorDriver", "voltage": 5.0},
                {"name": "LiPo",   "type": "Battery",     "voltage": 7.4},
                {"name": "C1",     "type": "Capacitor",   "voltage": 3.3},
            ],
            "connections": [
                {"from": "LiPo",  "to": "ESP32"},
                {"from": "ESP32", "to": "L298N"},
                {"from": "ESP32", "to": "C1"},
            ]
        }


# ==============================================================
#  STEP 2: CONVERT CONNECTIONS TO MERMAID FORMAT
# ==============================================================
def generate_mermaid(data):
    connections = data.get("connections", [])

    lines = []
    lines.append("```mermaid")
    lines.append("graph TD")

    # Loop through each connection and write a line
    for conn in connections:
        from_node = conn["from"].replace(" ", "_")  # Mermaid doesn't like spaces
        to_node   = conn["to"].replace(" ", "_")
        lines.append(f"    {from_node} --> {to_node}")

    lines.append("```")

    return "\n".join(lines)


# ==============================================================
#  STEP 3: SAVE AS diagram.md
# ==============================================================
def save_diagram(mermaid_text):
    with open("diagram.md", "w", encoding="utf-8") as f:
        f.write("# Circuit Diagram\n\n")
        f.write("Generated from output.json\n\n")
        f.write(mermaid_text)
        f.write("\n\n---\n")
        f.write("_Paste the mermaid block at https://mermaid.live to view the diagram!_\n")

    print("Diagram saved to diagram.md")
    print("Paste the contents at https://mermaid.live to see the picture!\n")
    print("=" * 60)
    print(mermaid_text)
    print("=" * 60)


# ==============================================================
#  MAIN
# ==============================================================
if __name__ == "__main__":
    print("Person 4's Mermaid Diagram Generator starting...\n")

    data = load_json()

    print("CONNECTIONS FOUND:")
    for conn in data.get("connections", []):
        print(f"  {conn['from']}  -->  {conn['to']}")
    print()

    mermaid_text = generate_mermaid(data)
    save_diagram(mermaid_text)
