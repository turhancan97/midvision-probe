from __future__ import annotations

import json
from typing import Any, Dict, List


PERSISTENCE_STORAGE_KEY = "sparrta_ui_state_v2"

PERSISTENT_CONTROL_IDS: List[str] = [
    "ctl-show-experimental",
    "ctl-perspective",
    "ctl-backbone",
    "ctl-triplet",
    "ctl-source-mode",
    "ctl-sample-name",
    "ctl-map-name",
    "ctl-alpha",
    "ctl-compare-enabled",
    "ctl-compare-map",
    "ctl-view-mode",
    "ctl-sharpen",
    "ctl-clip-percentile",
    "ctl-gamma",
]


def serialize_settings(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def build_persistence_script(control_ids: List[str], storage_key: str = PERSISTENCE_STORAGE_KEY) -> str:
    controls_json = json.dumps(control_ids)
    return f"""
<script>
(() => {{
  const STORAGE_KEY = {json.dumps(storage_key)};
  const CONTROL_IDS = {controls_json};

  const findRoot = (id) => document.getElementById(id);

  const findField = (root) => {{
    if (!root) return null;
    const checkbox = root.querySelector('input[type="checkbox"]');
    if (checkbox) return checkbox;
    const radioChecked = root.querySelector('input[type="radio"]:checked');
    if (radioChecked) return radioChecked;
    return root.querySelector('input, textarea, select');
  }};

  const getValue = (id) => {{
    const root = findRoot(id);
    if (!root) return null;
    const checkbox = root.querySelector('input[type="checkbox"]');
    if (checkbox) return !!checkbox.checked;

    const radioChecked = root.querySelector('input[type="radio"]:checked');
    if (radioChecked) return radioChecked.value;

    const field = findField(root);
    if (!field) return null;
    return field.value;
  }};

  const dispatchValueEvent = (el) => {{
    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
    el.dispatchEvent(new Event('change', {{ bubbles: true }}));
  }};

  const setValue = (id, value) => {{
    const root = findRoot(id);
    if (!root || value === null || value === undefined) return;

    const checkbox = root.querySelector('input[type="checkbox"]');
    if (checkbox) {{
      checkbox.checked = !!value;
      dispatchValueEvent(checkbox);
      return;
    }}

    const radio = root.querySelector(`input[type="radio"][value="${{value}}"]`);
    if (radio) {{
      radio.checked = true;
      dispatchValueEvent(radio);
      return;
    }}

    const field = findField(root);
    if (!field) return;
    field.value = String(value);
    dispatchValueEvent(field);
  }};

  const saveAll = () => {{
    const payload = {{}};
    for (const id of CONTROL_IDS) {{
      const val = getValue(id);
      if (val !== null) payload[id] = val;
    }}
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  }};

  const restoreAll = () => {{
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try {{
      const payload = JSON.parse(raw);
      for (const id of CONTROL_IDS) {{
        if (Object.prototype.hasOwnProperty.call(payload, id)) {{
          setValue(id, payload[id]);
        }}
      }}
    }} catch (_) {{
      // Ignore parse/set errors.
    }}
  }};

  const bindAll = () => {{
    for (const id of CONTROL_IDS) {{
      const root = findRoot(id);
      if (!root || root.dataset.sparrtaBound === "1") continue;
      root.addEventListener('change', saveAll, true);
      root.addEventListener('input', saveAll, true);
      root.dataset.sparrtaBound = "1";
    }}
  }};

  let tries = 0;
  const timer = setInterval(() => {{
    bindAll();
    if (tries === 0) restoreAll();
    tries += 1;
    if (tries > 40) clearInterval(timer);
  }}, 250);

  window.addEventListener('beforeunload', saveAll);
}})();
</script>
"""
