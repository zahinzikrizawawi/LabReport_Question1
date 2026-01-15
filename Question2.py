import json
from typing import List, Dict, Any, Tuple
import operator
import streamlit as st

# ----------------------------
# 1) Rule Engine Configuration
# ----------------------------

OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
}

# Your specific AC rules
DEFAULT_RULES: List[Dict[str, Any]] = [
    {
        "name": "Windows open → turn AC off",
        "priority": 100,
        "conditions": [["windows_open", "==", True]],
        "action": {"ac_mode": "OFF", "fan_speed": "LOW", "setpoint": None, "reason": "Windows are open"}
    },
    {
        "name": "No one home → eco mode",
        "priority": 90,
        "conditions": [
            ["occupancy", "==", "EMPTY"],
            ["temperature", ">=", 24]
        ],
        "action": {"ac_mode": "ECO", "fan_speed": "LOW", "setpoint": 27, "reason": "Home empty; save energy"}
    },
    {
        "name": "Hot & humid (occupied) → cool strong",
        "priority": 80,
        "conditions": [
            ["occupancy", "==", "OCCUPIED"],
            ["temperature", ">=", 30],
            ["humidity", ">=", 70]
        ],
        "action": {"ac_mode": "COOL", "fan_speed": "HIGH", "setpoint": 23, "reason": "Hot and humid"}
    },
    {
        "name": "Hot (occupied) → cool",
        "priority": 70,
        "conditions": [
            ["occupancy", "==", "OCCUPIED"],
            ["temperature", ">=", 28]
        ],
        "action": {"ac_mode": "COOL", "fan_speed": "MEDIUM", "setpoint": 24, "reason": "Temperature high"}
    },
    {
        "name": "Slightly warm (occupied) → gentle cool",
        "priority": 60,
        "conditions": [
            ["occupancy", "==", "OCCUPIED"],
            ["temperature", ">=", 26],
            ["temperature", "<", 28]
        ],
        "action": {"ac_mode": "COOL", "fan_speed": "LOW", "setpoint": 25, "reason": "Slightly warm"}
    },
    {
        "name": "Night (occupied) → sleep mode",
        "priority": 75,
        "conditions": [
            ["occupancy", "==", "OCCUPIED"],
            ["time_of_day", "==", "NIGHT"],
            ["temperature", ">=", 26]
        ],
        "action": {"ac_mode": "SLEEP", "fan_speed": "LOW", "setpoint": 26, "reason": "Night comfort"}
    },
    {
        "name": "Too cold → turn off",
        "priority": 85,
        "conditions": [["temperature", "<=", 22]],
        "action": {"ac_mode": "OFF", "fan_speed": "LOW", "setpoint": None, "reason": "Already cold"}
    }
]

def evaluate_condition(facts: Dict[str, Any], cond: List[Any]) -> bool:
    if len(cond) != 3: return False
    field, op, value = cond
    if field not in facts or op not in OPS: return False
    try:
        return OPS[op](facts[field], value)
    except Exception:
        return False

def rule_matches(facts: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    return all(evaluate_condition(facts, c) for c in rule.get("conditions", []))

def run_rules(facts: Dict[str, Any], rules: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    fired = [r for r in rules if rule_matches(facts, r)]
    if not fired:
        return ({"ac_mode": "FAN ONLY", "reason": "No rule matched"}, [])
    fired_sorted = sorted(fired, key=lambda r: r.get("priority", 0), reverse=True)
    return fired_sorted[0].get("action", {}), fired_sorted

# ----------------------------
# 2) Streamlit UI
# ----------------------------
st.set_page_config(page_title="Rule-Based AC Controller", page_icon="❄️", layout="wide")
st.title("Question 2 : Rule-based System")
st.write("Enter current home conditions:")

with st.sidebar:
    st.header("Home Scenario")
    temp = st.number_input("Temperature (°C)", value=31.0, step=0.5)
    humidity = st.number_input("Humidity (%)", value=75.0, step=1.0)
    occupancy = st.text_input("Occupancy (OCCUPIED / EMPTY)", value="OCCUPIED").strip().upper()
    time_of_day = st.text_input("Time of Day (MORNING / AFTERNOON / EVENING / NIGHT)", value="AFTERNOON").strip().upper()
    windows = st.selectbox("Windows Open?", [True, False], index=1)

    st.divider()
    st.header("Rules (JSON)")
    default_json = json.dumps(DEFAULT_RULES, indent=2)
    rules_text = st.text_area("Edit rules here", value=default_json, height=300)
    run_btn = st.button("Evaluate System", type="primary")

facts = {
    "temperature": float(temp),
    "humidity": float(humidity),
    "occupancy": occupancy,
    "time_of_day": time_of_day,
    "windows_open": windows,
}

st.subheader("Current Home Condition")
st.json(facts)

# Parse rules
try:
    rules = json.loads(rules_text)
except Exception:
    rules = DEFAULT_RULES

st.divider()

if run_btn:
    action, fired = run_rules(facts, rules)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Decision Result")
        mode = action.get("ac_mode", "N/A")
        setpoint = action.get("setpoint")
        reason = action.get("reason", "-")
        
        st.metric("AC Mode", mode)
        if setpoint:
            st.metric("Setpoint", f"{setpoint}°C")
        st.info(f"**Reason:** {reason}")

    with col2:
        st.subheader("Triggered Rules")
        if not fired:
            st.info("No rules matched.")
        else:
            for i, r in enumerate(fired, start=1):
                st.write(f"**{i}. {r.get('name')}** (Priority: {r.get('priority')})")
                with st.expander("Show Details"):
                    st.json(r["action"])
else:
    st.info("Fill in the data and click **Evaluate System**.")