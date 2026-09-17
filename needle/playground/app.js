var PRESETS = {"home": {"tools": [{"name": "set_lights", "description": "Turn lights on or off or dim them.", "parameters": {"type": "object", "properties": {"room": {"type": "string", "description": "Which room."}, "state": {"type": "string", "enum": ["on", "off"]}, "brightness": {"type": "integer", "description": "Percent 1-100."}}, "required": ["room", "state"]}}, {"name": "set_thermostat", "description": "Set the target temperature.", "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "temperature": {"type": "number"}, "mode": {"type": "string", "enum": ["heat", "cool", "auto"]}}, "required": ["room", "temperature"]}}, {"name": "lock_door", "description": "Lock a door.", "parameters": {"type": "object", "properties": {"door": {"type": "string", "description": "Which door."}}, "required": ["door"]}}], "q": "dim the bedroom lights to 20 percent and lock the front door"}, "robot": {"tools": [{"name": "move", "description": "Drive the robot in a direction.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["forward", "backward", "left", "right"]}, "distance_m": {"type": "number", "description": "Distance in meters."}}, "required": ["direction", "distance_m"]}}, {"name": "rotate", "description": "Rotate the robot in place.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["left", "right"]}, "degrees": {"type": "number"}}, "required": ["direction", "degrees"]}}, {"name": "gripper", "description": "Open or close the gripper.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}}], "q": "move forward 2 meters, turn left 90 degrees, then close the gripper"}, "device": {"tools": [{"name": "open_website", "description": "Open a website in a new tab.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The site to open."}}, "required": ["url"]}}, {"name": "set_theme", "description": "Change the page accent color.", "parameters": {"type": "object", "properties": {"accent_color": {"type": "string", "description": "A color name or hex."}}, "required": ["accent_color"]}}, {"name": "speak", "description": "Say something out loud through the device speakers.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}], "q": "make the accent green, open github.com, and say hello"}, "gallery": {"tools": [{"name": "create_album", "description": "Create a new photo album.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}, {"name": "move_photos", "description": "Move photos into an album.", "parameters": {"type": "object", "properties": {"album": {"type": "string"}, "filter": {"type": "string", "description": "Which photos, e.g. 'last weekend', 'screenshots'."}}, "required": ["album", "filter"]}}, {"name": "delete_photos", "description": "Delete photos.", "parameters": {"type": "object", "properties": {"filter": {"type": "string"}}, "required": ["filter"]}}], "q": "create an album called Summer 2026 and move the photos from last weekend into it"}, "email": {"tools": [{"name": "send_email", "description": "Send an email.", "parameters": {"type": "object", "properties": {"to": {"type": "string", "description": "Email address."}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}], "q": "send the receipt to finance@cactus.dev with subject expenses: Blue Bottle Coffee, $14.50, August 4th"}, "currency": {"tools": [{"name": "convert_currency", "description": "Convert an amount between currencies.", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string", "description": "ISO code, e.g. USD."}, "to_currency": {"type": "string", "description": "ISO code, e.g. EUR."}}, "required": ["amount", "from_currency", "to_currency"]}}], "q": "how much is 250 dollars in euros"}, "document": {"tools": [{"name": "record_booking", "description": "Record the details of a hotel booking.", "parameters": {"type": "object", "properties": {"hotel": {"type": "string"}, "confirmation_number": {"type": "string"}, "guest_name": {"type": "string"}, "room_type": {"type": "string"}, "check_in": {"type": "string"}, "check_out": {"type": "string"}, "total": {"type": "number"}, "currency": {"type": "string"}}, "required": ["confirmation_number", "check_in", "check_out", "total"]}}], "q": "extract the booking from this email: Dear Mr Okafor, thank you for choosing the Harbor Light Hotel. This email confirms reservation HL-88213 for a deluxe sea-view room, checking in on 14 September 2026 and checking out on 18 September 2026. The total for your stay is 642.50 euros, payable at checkout. Breakfast is included from 7am, and our shuttle meets the 9:40 ferry on request. We look forward to welcoming you. Warm regards, Reception."}, "sentiment": {"tools": [{"name": "classify_sentiment", "description": "Classify the sentiment of a message.", "parameters": {"type": "object", "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral", "mixed"]}}, "required": ["sentiment"]}}], "q": "classify the sentiment of this message: this is the worst purchase I have ever made, it broke in a day and support ignored me"}, "crowded": {"tools": [{"name": "set_lights", "description": "Control lights.", "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "state": {"type": "string", "enum": ["on", "off"]}}, "required": ["room", "state"]}}, {"name": "set_thermostat", "description": "Set temperature.", "parameters": {"type": "object", "properties": {"temperature": {"type": "number"}}, "required": ["temperature"]}}, {"name": "lock_door", "description": "Lock a door.", "parameters": {"type": "object", "properties": {"door": {"type": "string"}}, "required": ["door"]}}, {"name": "play_music", "description": "Play music.", "parameters": {"type": "object", "properties": {"genre": {"type": "string"}}, "required": []}}, {"name": "set_timer", "description": "Set a timer.", "parameters": {"type": "object", "properties": {"time_human": {"type": "string"}}, "required": ["time_human"]}}, {"name": "send_message", "description": "Send a text.", "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "message": {"type": "string"}}, "required": ["recipient", "message"]}}, {"name": "create_note", "description": "Create a note.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}, {"name": "start_vacuum", "description": "Start the robot vacuum.", "parameters": {"type": "object", "properties": {"room": {"type": "string"}}, "required": []}}, {"name": "water_plants", "description": "Run the irrigation system.", "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}, "required": []}}, {"name": "open_blinds", "description": "Open or close the blinds.", "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "position": {"type": "string", "enum": ["open", "closed", "half"]}}, "required": ["room", "position"]}}, {"name": "preheat_oven", "description": "Preheat the oven.", "parameters": {"type": "object", "properties": {"temperature_c": {"type": "number"}}, "required": ["temperature_c"]}}, {"name": "charge_car", "description": "Start charging the car.", "parameters": {"type": "object", "properties": {"limit_percent": {"type": "integer"}}, "required": []}}], "q": "close the bedroom blinds halfway"}, "repeat": {"tools": [{"name": "add_to_list", "description": "Add one item to the shopping list.", "parameters": {"type": "object", "properties": {"item": {"type": "string"}, "quantity": {"type": "integer"}}, "required": ["item"]}}], "q": "add 2 milk and 6 eggs to the shopping list"}, "array": {"tools": [{"name": "add_items", "description": "Add items to the shopping list.", "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}}}, "required": ["items"]}}], "q": "add milk, eggs, and bread to the shopping list"}, "flight": {"tools": [{"name": "search_flights", "description": "Search for flights.", "parameters": {"type": "object", "properties": {"from_city": {"type": "string"}, "to_city": {"type": "string"}, "date": {"type": "string"}, "direct_only": {"type": "boolean", "description": "Only non-stop flights."}, "passengers": {"type": "integer"}}, "required": ["from_city", "to_city"]}}], "q": "find non-stop flights from lagos to nairobi on december 3rd for two passengers"}, "refuse": {"tools": [{"name": "move", "description": "Drive the robot in a direction.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["forward", "backward", "left", "right"]}, "distance_m": {"type": "number", "description": "Distance in meters."}}, "required": ["direction", "distance_m"]}}, {"name": "rotate", "description": "Rotate the robot in place.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["left", "right"]}, "degrees": {"type": "number"}}, "required": ["direction", "degrees"]}}, {"name": "gripper", "description": "Open or close the gripper.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}}], "q": "write me a short poem about the moon"}};
var LABELS = {"home": "Smart home", "robot": "Robot", "gallery": "Gallery", "device": "Device control", "email": "Extract email", "currency": "Currency", "document": "Document extraction", "sentiment": "Sentiment", "crowded": "12 tools routing", "repeat": "Repeated calls", "array": "Array argument", "flight": "Flight form", "refuse": "Off-topic refusal"};

var _toastTimer = null;

function showError(msg) {
  if (_toastTimer) clearTimeout(_toastTimer);
  document.getElementById("toastMsg").textContent = msg;
  document.getElementById("toast").classList.add("visible");
  _toastTimer = setTimeout(dismissToast, 8000);
}

function dismissToast() {
  document.getElementById("toast").classList.remove("visible");
}

function loadToolsFile(input) {
  var file = input.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function () {
    try {
      JSON.parse(reader.result);
      document.getElementById("tools").value = reader.result;
      newChat();
    } catch (e) {
      showError("Invalid JSON file");
    }
  };
  reader.readAsText(file);
  input.value = "";
}

function fetchModelName() {
  fetch("/model").then(function (r) { return r.json(); })
    .then(function (d) { document.getElementById("modelName").textContent = d.name || ""; })
    .catch(function () {});
}

function loadModelFile(input) {
  var file = input.files[0];
  if (!file) return;
  var name = document.getElementById("modelName");
  name.textContent = "loading " + file.name + "...";
  fetch("/load-model", { method: "POST", headers: { "X-Filename": file.name }, body: file })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      if (d.error) { showError(d.error); fetchModelName(); }
      else { name.textContent = d.name; newChat(); }
    })
    .catch(function (e) { showError("Upload failed: " + e.message); fetchModelName(); });
  input.value = "";
}

var conversation = document.getElementById("conversation");
var emptyState = document.getElementById("emptyState");

function addTurn(query, data) {
  if (emptyState) { emptyState.remove(); emptyState = null; }
  var turn = document.createElement("div");
  turn.className = "turn";

  var q = document.createElement("div");
  q.className = "turn-query";
  q.textContent = query;
  turn.appendChild(q);

  var pre = document.createElement("pre");
  pre.className = "turn-result";
  var calls = data.function_calls;
  if (data.type === "refuse" || (Array.isArray(calls) && calls.length === 0)) {
    pre.classList.add("refused");
    pre.textContent = "no tool call (off-topic / refused)";
  } else {
    pre.textContent = JSON.stringify(calls || data, null, 2);
  }
  turn.appendChild(pre);

  if (data.reasoning) {
    var reason = document.createElement("div");
    reason.className = "turn-reasoning";
    var label = document.createElement("div");
    label.className = "turn-reasoning-label";
    label.textContent = "Reasoning trace";
    var body = document.createElement("div");
    body.className = "turn-reasoning-body";
    body.textContent = data.reasoning;
    reason.appendChild(label);
    reason.appendChild(body);
    turn.appendChild(reason);
  }

  var bits = [];
  if (data.confidence !== undefined && data.confidence !== null)
    bits.push("confidence " + Number(data.confidence).toFixed(4));
  if (data.decode_tps) bits.push(Math.round(data.decode_tps) + " tok/s");
  if (bits.length) {
    var meta = document.createElement("div");
    meta.className = "turn-meta";
    meta.textContent = bits.join("  ·  ");
    turn.appendChild(meta);
  }

  conversation.appendChild(turn);
  conversation.scrollTop = conversation.scrollHeight;
}

async function send() {
  var input = document.getElementById("query");
  var btn = document.getElementById("sendBtn");
  var query = input.value.trim();
  if (!query) return;
  var tools = document.getElementById("tools").value.trim() || "[]";
  try { JSON.parse(tools); } catch (e) { showError("Invalid tools JSON"); return; }

  input.disabled = true;
  btn.disabled = true;
  try {
    var r = await fetch("/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, tools: tools }),
    });
    var data = await r.json();
    if (data.error) { showError(data.error); }
    else { addTurn(query, data); input.value = ""; }
  } catch (e) {
    showError("Request failed: " + e.message);
  } finally {
    input.disabled = false;
    btn.disabled = false;
    input.focus();
  }
}

function newChat() {
  fetch("/reset", { method: "POST" }).catch(function () {});
  conversation.innerHTML = "";
  emptyState = document.createElement("div");
  emptyState.className = "empty";
  emptyState.id = "emptyState";
  emptyState.textContent = "Pick a preset or type a query, then Run.";
  conversation.appendChild(emptyState);
}

function applyPreset(key) {
  var p = PRESETS[key];
  document.getElementById("tools").value = JSON.stringify(p.tools, null, 2);
  document.getElementById("query").value = p.q;
  newChat();
  document.getElementById("query").focus();
}

var presetBox = document.getElementById("presets");
Object.keys(LABELS).forEach(function (key) {
  if (!PRESETS[key]) return;
  var b = document.createElement("button");
  b.className = "preset-btn";
  b.textContent = LABELS[key];
  b.onclick = function () { applyPreset(key); };
  presetBox.appendChild(b);
});

function togglePanel() {
  var sidebar = document.getElementById("sidebar");
  sidebar.classList.toggle("open");
  document.querySelector(".tools-toggle span").textContent =
    sidebar.classList.contains("open") ? "Query" : "Tools";
}

var _pollTimer = null;
var _ftRunning = false;

// ---------------------------------------------------------------------------
// Providers, credentials and the model catalog.
// The browser only ever names a provider and a model. Credentials stay on the
// server: a key pasted here is forwarded once and then reported back masked.
// ---------------------------------------------------------------------------

var _providers = [];
var _catalog = null;
var _modelValue = "";

// The modality the current prompt actually carries. It is the only thing that
// decides which models the selector may offer: "image" is set by the attachment
// toggle, never guessed from a model name.
var _inputModality = null;

// Monotonic attempt id. Every async reply carries the generation it belongs to
// and is dropped if a newer attempt has started, so a late URL or key from one
// login can never surface under the next.
var _gen = 0;
var _connect = { attempt: 0, status: "idle", flow: null, timer: null };
var _modelRequest = 0;

function _bumpGeneration() {
  _gen += 1;
  return _gen;
}

function fetchProviders() {
  return fetch("/providers").then(function (r) { return r.json(); })
    .then(function (d) {
      _providers = d.providers || [];
      var sel = document.getElementById("ftProvider");
      sel.innerHTML = "";
      _providers.forEach(function (p) {
        var o = document.createElement("option");
        o.value = p.id;
        o.textContent = p.label;
        sel.appendChild(o);
      });
      var preferred = _providers.filter(function (p) { return p.connected; })[0]
        || _providers[0];
      if (preferred) sel.value = preferred.id;
      onProviderChange();
    })
    .catch(function () {});
}

function currentProvider() {
  var id = document.getElementById("ftProvider").value;
  return _providers.filter(function (p) { return p.id === id; })[0] || null;
}

function onProviderChange() {
  var p = currentProvider();
  if (!p) return;
  var canConnect = (p.methods || []).indexOf("pkce") !== -1;
  document.getElementById("ftConnectMethod").hidden = !canConnect;
  document.getElementById("ftApiKey").placeholder = p.id === "orcarouter"
    ? "sk-orca-..." : "sk-or-...";
  showCredentialState(p);
  loadModels(p.id);
}

// The durable statement of what the panel is holding: which method produced
// the credential and its masked form. Always masked — the full key lives on
// the server and is never rendered here.
function showCredentialState(p) {
  var el = document.getElementById("ftCredState");
  if (!p || !p.connected || !p.masked_key) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.className = "credential-state" + (p.connection_degraded ? " warn" : " ok");
  el.textContent = "Connected via " + (p.key_source === "pkce" ? "OrcaRouter sign-in"
    : "API key") + " · " + p.masked_key
    + (p.connection_degraded ? " · needs reauthentication" : "");
}

function showModelStatus(text, kind) {
  var el = document.getElementById("ftModelStatus");
  el.textContent = text || "";
  el.className = "model-status" + (kind ? " " + kind : "");
}

function loadModels(providerId) {
  var request = ++_modelRequest;
  var select = document.getElementById("ftModel");
  select.disabled = true;
  showModelStatus("Loading models...", "");
  fetch("/provider/models?provider=" + encodeURIComponent(providerId)
        + "&capability=chat"
        + (_inputModality ? "&input_modality=" + encodeURIComponent(_inputModality) : ""))
    .then(function (r) { return r.json(); })
    .then(function (d) {
      if (request !== _modelRequest) return;   // a newer provider won
      _catalog = d;
      renderModels(d);
    })
    .catch(function (e) {
      if (request !== _modelRequest) return;
      _catalog = null;
      showModelStatus("Could not load models: " + e.message, "warn");
      select.innerHTML = "";
      select.disabled = true;
    });
}

function renderModels(d) {
  var select = document.getElementById("ftModel");
  select.innerHTML = "";
  (d.models || []).forEach(function (m) {
    var o = document.createElement("option");
    o.value = m.id;
    o.textContent = m.id + (m.context_length ? "  ·  " + m.context_length : "");
    select.appendChild(o);
  });
  select.disabled = (d.models || []).length === 0;
  if (d.degraded) {
    showModelStatus("Live catalog unavailable (" + (d.error || "unknown")
      + "). Showing " + d.source + " models — refresh when the network is back.",
      "warn");
  } else {
    showModelStatus(d.models.length + " of " + d.total + " models for " + d.capability
      + " · live from " + d.api_base, "ok");
  }
  // A previously chosen model is restored only if it is still in the filtered
  // list; otherwise the selection is cleared rather than silently kept.
  if (_modelValue && (d.models || []).some(function (m) { return m.id === _modelValue; })) {
    select.value = _modelValue;
  } else {
    _modelValue = "";
    if ((d.models || []).length) select.value = d.models[0].id;
  }
  saveModelChoice(select.value);
}

function saveModelChoice(value) {
  _modelValue = value || "";
  try { localStorage.setItem("needle.model", _modelValue); } catch (e) {}
}

function restoreModelChoice() {
  try { _modelValue = localStorage.getItem("needle.model") || ""; } catch (e) {}
}

// The attachment toggle is the only thing that may widen what the selector
// offers. Turning it on makes the requirement stricter, so the catalog is
// re-requested with the modality and any model that does not declare an
// image input is dropped from the list (and from the current selection).
function onInputModalityChange() {
  var on = document.getElementById("ftImageToggle").checked;
  _inputModality = on ? "image" : null;
  document.getElementById("ftImageUrl").hidden = !on;
  showModalityStatus(on
    ? "Only models that declare an image input are listed."
    : "", "");
  var p = currentProvider();
  if (p) loadModels(p.id);
}

function showModalityStatus(text, kind) {
  var el = document.getElementById("ftModalityStatus");
  el.textContent = text || "";
  el.className = "model-status" + (kind ? " " + kind : "");
}

// Second line of defence. The selector never offers an incompatible model, so
// this only fires if the catalog changed under a stale selection.
function attachmentBlocksSend() {
  if (!_inputModality) return null;
  var p = currentProvider();
  if (!p || p.id !== "orcarouter") return null;   // other providers keep their behaviour
  var imageUrl = document.getElementById("ftImageUrl").value.trim();
  if (!imageUrl) return "Add the image URL to send, or turn the image input off";
  var m = (_catalog && _catalog.models || []).filter(function (x) {
    return x.id === selectedModel(); })[0];
  if (!m) return null;
  if ((m.input_modalities || []).indexOf(_inputModality) === -1) {
    return "The selected model does not declare an image input; pick another";
  }
  return null;
}

// The model always comes from the provider's own catalog: there is no free-text
// path, so a user cannot type a model the workspace cannot reach.
function selectedModel() {
  return document.getElementById("ftModel").value;
}

function startConnect() {
  var p = currentProvider();
  if (!p) return;
  var cxl = document.getElementById("ftConnectCxl");
  var btn = document.getElementById("ftConnectBtn");
  btn.disabled = true;
  cxl.hidden = false;
  document.getElementById("ftConnectPanel").hidden = false;
  document.getElementById("ftOobRow").hidden = true;
  document.getElementById("ftConnectUrl").textContent = "";
  document.getElementById("ftConnectHint").textContent = "Requesting an authorization URL...";
  var generation = _bumpGeneration();
  fetch("/provider/connect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider: p.id, flow: "loopback", app_name: "Needle",
                           scope: "api" }),
  }).then(function (r) { return r.json(); }).then(function (d) {
    if (generation !== _gen) return;            // superseded: leave state alone
    _connect.attempt = d.attempt;
    _connect.status = d.status;
    _connect.flow = d.flow;
    if (d.error) { _connectFailed(d.error); return; }
    _showAuthorization(d);
    _connect.timer = setInterval(pollConnect, 1500);
  }).catch(function (e) {
    if (generation !== _gen) return;
    _connectFailed("Could not start the connection: " + e.message);
  });
  try { window.open("", "_blank"); } catch (e) {}
}

function _showAuthorization(d) {
  document.getElementById("ftConnectHint").textContent =
    "Approve access in the browser window. This panel updates on its own.";
  var link = document.getElementById("ftConnectUrl");
  link.textContent = d.url || "";
  link.href = d.url || "#";
  if (d.url) { try { window.open(d.url, "_blank"); } catch (e) {} }
  // The consent screen also offers "show me a code", which a human then pastes
  // back, so the out-of-band input is always available.
  document.getElementById("ftOobRow").hidden = false;
}

function pollConnect() {
  var generation = _gen;
  fetch("/provider/connect/status").then(function (r) { return r.json(); })
    .then(function (d) {
      if (generation !== _gen) return;
      if (d.attempt !== _connect.attempt) return;
      if (d.status === "done") {
        _connectDone(d);
      } else if (d.status === "failed" || d.status === "cancelled") {
        _connectFailed(d.error || "the connection was cancelled");
      } else if (d.hint) {
        document.getElementById("ftConnectHint").textContent = d.hint;
      }
    })
    .catch(function () {});
}

function _connectDone(d) {
  _stopConnectTimer();
  _connect.status = "done";
  // The panel stays visible so the "connected via" line below the methods can
  // report what was stored, always masked.
  document.getElementById("ftConnectPanel").hidden = true;
  document.getElementById("ftConnectBtn").disabled = false;
  document.getElementById("ftConnectCxl").hidden = true;
  document.getElementById("ftApiKey").value = "";
  showModelStatus("Connected · " + (d.masked_key || "key stored on the server"),
                  "ok");
  fetchProviders();
}

function _connectFailed(message) {
  _stopConnectTimer();
  _connect.status = "failed";
  document.getElementById("ftConnectBtn").disabled = false;
  document.getElementById("ftConnectCxl").hidden = true;
  document.getElementById("ftConnectPanel").hidden = true;
  showModelStatus(message || "connection failed", "warn");
}

function _stopConnectTimer() {
  if (_connect.timer) { clearInterval(_connect.timer); _connect.timer = null; }
}

function cancelConnect() {
  // Cancelling invalidates the generation first, so the cancelled attempt's
  // own finally-handler cannot write over the state we are clearing now.
  _bumpGeneration();
  _stopConnectTimer();
  document.getElementById("ftConnectPanel").hidden = true;
  document.getElementById("ftOobRow").hidden = true;
  document.getElementById("ftConnectBtn").disabled = false;
  document.getElementById("ftConnectCxl").hidden = true;
  showModelStatus("Connection cancelled", "");
  var attempt = _connect.attempt;
  _connect.status = "cancelled";
  fetch("/provider/connect/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ attempt: attempt }),
  }).catch(function () {});
}

function submitOobCode() {
  var code = document.getElementById("ftOobCode").value.trim();
  if (!code) { showError("Paste the code shown on the consent screen"); return; }
  document.getElementById("ftOobCode").value = "";
  document.getElementById("ftConnectHint").textContent = "Exchanging the code...";
  fetch("/provider/connect/code", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: code }),
  }).catch(function () {});
}

function saveApiKey() {
  var key = document.getElementById("ftApiKey").value.trim();
  var p = currentProvider();
  if (!p) return;
  if (!key) { showError("Enter an API key, or use Connect instead"); return; }
  fetch("/provider/connect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider: p.id, api_key: key }),
  }).then(function (r) { return r.json(); }).then(function (d) {
    document.getElementById("ftApiKey").value = "";
    if (d.error) { showModelStatus(d.error, "warn"); return; }
    showModelStatus(p.label + " key stored · " + (d.masked_key || ""), "ok");
    loadModels(p.id);
    fetchProviders();   // refresh the masked credential line from the server
  }).catch(function (e) { showModelStatus("Could not store the key: " + e.message, "warn"); });
}

// pagehide fires when the page enters the back-forward cache. The generation
// guard below would (correctly) refuse to mutate state afterwards, so the busy
// flags and hint are cleared synchronously here, and the server-side login is
// cancelled with keepalive. Without this, a restored page stays busy forever.
window.addEventListener("pagehide", function () {
  _bumpGeneration();
  _stopConnectTimer();
  _connect.status = "idle";
  _connect.attempt = 0;
  document.getElementById("ftConnectBtn").disabled = false;
  document.getElementById("ftConnectCxl").hidden = true;
  document.getElementById("ftConnectPanel").hidden = true;
  document.getElementById("ftConnectHint").textContent = "";
  try {
    fetch("/provider/connect/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
      keepalive: true,
    });
  } catch (e) {}
});

function openFinetuneModal() {
  var tools = document.getElementById("tools").value.trim() || "[]";
  try { JSON.parse(tools); } catch (e) { showError("Invalid tools JSON"); return; }
  _resetModal();
  restoreModelChoice();
  document.getElementById("modalOverlay").classList.add("visible");
  fetchProviders();
  document.getElementById("ftApiKey").focus();
}

function closeModal(e) {
  if (e && e.target && e.target !== document.getElementById("modalOverlay")) return;
  if (_ftRunning) return;
  // Closing the modal is a terminal path for a login in progress; release the
  // server-side lock rather than leaving it held.
  if (_connect.status === "pending") cancelConnect();
  document.getElementById("modalOverlay").classList.remove("visible");
}

function _resetModal() {
  document.getElementById("ftSteps").classList.remove("visible");
  document.getElementById("ftProgress").textContent = "";
  var start = document.getElementById("ftStartBtn");
  start.disabled = false;
  start.textContent = "Start Finetune";
  start.style.display = "";
  document.getElementById("modalCloseBtn").style.display = "";
  var dl = document.getElementById("ftDownload");
  if (dl) dl.remove();
  document.querySelectorAll(".modal-step").forEach(function (s) {
    s.classList.remove("active", "done");
  });
}

async function startFinetune() {
  var p = currentProvider();
  if (!p) { showError("Pick a provider"); return; }
  var tools = document.getElementById("tools").value.trim() || "[]";
  try { JSON.parse(tools); } catch (e) { showError("Invalid tools JSON"); return; }
  var samples = parseInt(document.getElementById("ftSamples").value, 10) || 200;
  var model = selectedModel();
  if (!model) {
    showError("Pick a model — the list comes from the provider's own catalog");
    return;
  }
  var blocked = attachmentBlocksSend();
  if (blocked) { showError(blocked); return; }

  var btn = document.getElementById("ftStartBtn");
  btn.disabled = true;
  btn.textContent = "Starting...";
  document.getElementById("modalCloseBtn").style.display = "none";
  document.getElementById("ftSteps").classList.add("visible");
  _ftRunning = true;

  try {
    // The key is not sent: the server uses the credential it already holds for
    // this provider, whichever of the two methods produced it.
    var r = await fetch("/finetune", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tools: tools, provider: p.id, model: model,
                             samples: samples,
                             image_url: _inputModality
                               ? document.getElementById("ftImageUrl").value.trim()
                               : "" }),
    });
    var data = await r.json();
    if (data.error) { showError(data.error); _ftRunning = false; _resetModal(); return; }
    _pollTimer = setInterval(pollFinetune, 2000);
  } catch (e) {
    showError("Request failed: " + e.message);
    _ftRunning = false;
    _resetModal();
  }
}

function _updateSteps(current) {
  var steps = document.querySelectorAll(".modal-step");
  var past = true;
  steps.forEach(function (s) {
    var name = s.getAttribute("data-step");
    s.classList.remove("active", "done");
    if (name === current) { s.classList.add("active"); past = false; }
    else if (past) { s.classList.add("done"); }
  });
}

var _stepLabels = {
  "generating data": "Generating data...",
  "training": "Training...",
  "building": "Building .cact...",
};

async function pollFinetune() {
  try {
    var r = await fetch("/finetune/status");
    var data = await r.json();
    _updateSteps(data.step);
    var btn = document.getElementById("ftStartBtn");
    btn.textContent = _stepLabels[data.step] || data.step;
    if (data.log && data.log.length)
      document.getElementById("ftProgress").textContent = data.log[data.log.length - 1];

    if (!data.running) {
      clearInterval(_pollTimer);
      _pollTimer = null;
      _ftRunning = false;
      document.getElementById("modalCloseBtn").style.display = "";
      if (data.step === "done") {
        document.querySelectorAll(".modal-step").forEach(function (s) {
          s.classList.remove("active"); s.classList.add("done");
        });
        btn.style.display = "none";
        fetchModelName();
        if (data.checkpoint) {
          var dl = document.createElement("a");
          dl.id = "ftDownload";
          dl.className = "modal-download";
          dl.href = "/download/" + data.checkpoint;
          dl.download = data.checkpoint;
          dl.textContent = "Download " + data.checkpoint;
          document.getElementById("ftFooter").appendChild(dl);
        }
      } else {
        showError("Finetune failed — " + (data.error || "unknown error"));
        btn.textContent = "Retry";
        btn.disabled = false;
      }
    }
  } catch (e) {
    clearInterval(_pollTimer);
    _pollTimer = null;
    _ftRunning = false;
    document.getElementById("modalCloseBtn").style.display = "";
    showError("Lost connection to server");
    document.getElementById("ftStartBtn").textContent = "Retry";
    document.getElementById("ftStartBtn").disabled = false;
  }
}

document.getElementById("query").addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeModal(); });

(function () {
  var handle = document.getElementById("resizeHandle");
  var sidebar = document.getElementById("sidebar");
  var dragging = false;
  handle.addEventListener("mousedown", function (e) {
    if (window.innerWidth <= 768) return;
    e.preventDefault();
    dragging = true;
    handle.classList.add("active");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });
  window.addEventListener("mousemove", function (e) {
    if (!dragging) return;
    sidebar.style.width = Math.min(Math.max(e.clientX, 200), window.innerWidth * 0.6) + "px";
  });
  window.addEventListener("mouseup", function () {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("active");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  });
  window.addEventListener("resize", function () {
    if (window.innerWidth <= 768) sidebar.style.width = "";
  });
})();

fetchModelName();
applyPreset("home");
