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
    } catch (e) {
      showError("Invalid JSON file");
    }
  };
  reader.readAsText(file);
  input.value = "";
}

function fetchModelName() {
  fetch("/model-info")
    .then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    })
    .then(function (data) {
      document.getElementById("modelName").textContent = data.name || "";
    })
    .catch(function () {});
}

fetchModelName();

function loadModelFile(input) {
  var file = input.files[0];
  if (!file) return;
  var formData = new FormData();
  formData.append("file", file);
  document.getElementById("result").textContent = "Loading model...";
  fetch("/load-model", { method: "POST", body: formData })
    .then(function (r) {
      if (!r.ok && r.headers.get("content-type") !== "application/json; charset=utf-8") throw new Error("Server error " + r.status);
      return r.json();
    })
    .then(function (data) {
      if (data.error) {
        showError(data.error);
        document.getElementById("result").textContent = "";
      } else {
        document.getElementById("result").className = "result-box has-result";
        document.getElementById("result").textContent = "Model loaded: " + data.name;
        document.getElementById("modelName").textContent = data.name;
      }
    })
    .catch(function (e) { showError("Upload failed: " + e.message); });
  input.value = "";
}

async function send() {
  var input = document.getElementById("query");
  var btn = document.getElementById("sendBtn");
  var result = document.getElementById("result");
  var query = input.value.trim();
  if (!query) return;

  input.disabled = true;
  btn.disabled = true;
  result.className = "result-box";
  result.textContent = "Running...";

  try {
    var r = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        tools: document.getElementById("tools").value.trim() || "[]",
        seed: 0,
        max_gen_len: 512,
        constrained: true,
      }),
    });
    if (!r.ok && !(r.headers.get("content-type") || "").includes("json")) throw new Error("Server error " + r.status);
    var data = await r.json();
    if (data.error) {
      result.textContent = "";
      showError(data.error);
    } else if (data.result === null || data.result === undefined || data.result === "") {
      result.textContent = "";
      showError("Empty response from model");
    } else {
      result.className = "result-box has-result";
      result.textContent = data.result;
    }
  } catch (e) {
    result.textContent = "";
    showError("Request failed: " + e.message);
  } finally {
    input.disabled = false;
    btn.disabled = false;
    input.focus();
  }
}

function togglePanel() {
  var sidebar = document.getElementById("sidebar");
  sidebar.classList.toggle("open");
  var label = document.querySelector(".tools-toggle span");
  label.textContent = sidebar.classList.contains("open") ? "Query" : "Tools";
}

var _pollTimer = null;
var _ftRunning = false;

function openFinetuneModal() {
  var tools = document.getElementById("tools").value.trim() || "[]";
  try {
    JSON.parse(tools);
  } catch (e) {
    showError("Invalid tools JSON");
    return;
  }
  _resetModal();
  document.getElementById("modalOverlay").classList.add("visible");
  document.getElementById("ftApiKey").focus();
}

function closeModal(e) {
  if (e && e.target && e.target !== document.getElementById("modalOverlay")) return;
  if (_ftRunning) return;
  document.getElementById("modalOverlay").classList.remove("visible");
}

function _resetModal() {
  document.getElementById("ftSteps").classList.remove("visible");
  document.getElementById("ftStartBtn").disabled = false;
  document.getElementById("ftStartBtn").textContent = "Start Finetune";
  document.getElementById("ftStartBtn").style.display = "";
  document.getElementById("modalCloseBtn").style.display = "";
  var dl = document.getElementById("ftDownload");
  if (dl) dl.remove();
  document.getElementById("ftEvalResults").innerHTML = "";
  var steps = document.querySelectorAll(".modal-step");
  for (var i = 0; i < steps.length; i++) {
    steps[i].classList.remove("active", "done");
  }
}

async function startFinetune() {
  var apiKey = document.getElementById("ftApiKey").value.trim();
  if (!apiKey) {
    showError("Gemini API key is required");
    return;
  }

  var tools = document.getElementById("tools").value.trim() || "[]";
  try {
    JSON.parse(tools);
  } catch (e) {
    showError("Invalid tools JSON");
    return;
  }

  var btn = document.getElementById("ftStartBtn");
  btn.disabled = true;
  btn.textContent = "Starting...";
  document.getElementById("modalCloseBtn").style.display = "none";
  document.getElementById("ftSteps").classList.add("visible");
  var steps = document.querySelectorAll(".modal-step");
  for (var i = 0; i < steps.length; i++) steps[i].classList.remove("active", "done");
  _ftRunning = true;

  try {
    var r = await fetch("/finetune", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tools: tools,
        api_key: apiKey,
      }),
    });
    if (!r.ok && !(r.headers.get("content-type") || "").includes("json")) throw new Error("Server error " + r.status);
    var data = await r.json();
    if (data.error) {
      showError(data.error);
      _ftRunning = false;
      _resetModal();
      return;
    }
    _pollTimer = setInterval(pollFinetune, 2000);
  } catch (e) {
    showError("Request failed: " + e.message);
    _ftRunning = false;
    _resetModal();
  }
}

function _updateSteps(currentStep) {
  var steps = document.querySelectorAll(".modal-step");
  var found = false;
  for (var i = 0; i < steps.length; i++) {
    if (steps[i].getAttribute("data-step") === currentStep) found = true;
  }
  if (!found) return;

  var past = true;
  for (var i = 0; i < steps.length; i++) {
    var stepName = steps[i].getAttribute("data-step");
    steps[i].classList.remove("active", "done");
    if (stepName === currentStep) {
      steps[i].classList.add("active");
      past = false;
    } else if (past) {
      steps[i].classList.add("done");
    }
  }
}

function _renderEvalResults(ft, base) {
  var el = document.getElementById("ftEvalResults");
  if (!ft && !base) { el.innerHTML = ""; return; }
  var metrics = ["call_f1", "name_f1", "exact_match", "parse_rate", "args_acc"];
  var labels = {"call_f1": "Call F1", "name_f1": "Name F1", "exact_match": "Exact Match", "parse_rate": "Parse Rate", "args_acc": "Args Accuracy"};
  var hasBase = base && base.call_f1 != null;
  var rows = metrics.map(function (k) {
    var bv = base && base[k] != null ? (base[k] * 100).toFixed(1) + '%' : '-';
    var fv = ft && ft[k] != null ? (ft[k] * 100).toFixed(1) + '%' : '-';
    if (hasBase) {
      return '<tr><td>' + labels[k] + '</td><td>' + bv + '</td><td>' + fv + '</td></tr>';
    }
    return '<tr><td>' + labels[k] + '</td><td>' + fv + '</td></tr>';
  }).join("");
  var header = hasBase
    ? '<tr><th>Metric</th><th>Base</th><th>Finetuned</th></tr>'
    : '<tr><th>Metric</th><th>Score</th></tr>';
  el.innerHTML =
    '<div class="ft-section-title">Evaluation Results</div>' +
    '<table class="ft-table"><thead>' + header + '</thead><tbody>' + rows + '</tbody></table>';
}

async function pollFinetune() {
  try {
    var r = await fetch("/finetune/status");
    if (!r.ok) throw new Error("Server error " + r.status);
    var data = await r.json();

    _updateSteps(data.step);

    var btn = document.getElementById("ftStartBtn");
    var labels = {
      "starting": "Starting...",
      "generating data": "Generating data...",
      "evaluating base": "Training...",
      "training": "Training...",
      "evaluating finetuned": "Training...",
    };
    var label = labels[data.step] || data.step;
    btn.textContent = label;

    _renderEvalResults(data.eval_results, data.base_eval);

    if (!data.running) {
      clearInterval(_pollTimer);
      _pollTimer = null;
      _ftRunning = false;
      document.getElementById("modalCloseBtn").style.display = "";

      if (data.step === "done") {
        var steps = document.querySelectorAll(".modal-step");
        for (var i = 0; i < steps.length; i++) {
          steps[i].classList.remove("active");
          steps[i].classList.add("done");
        }
        btn.style.display = "none";
        if (data.checkpoint) {
          var footer = document.getElementById("ftFooter");
          var dl = document.createElement("a");
          dl.id = "ftDownload";
          dl.className = "modal-download";
          dl.href = "/download/" + data.checkpoint;
          dl.download = data.checkpoint;
          dl.textContent = "Download " + data.checkpoint;
          footer.appendChild(dl);
        }
      } else if (data.step === "failed") {
        var last = data.log.length > 0 ? data.log[data.log.length - 1] : "Unknown error";
        showError("Finetune failed — " + last);
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
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") closeModal();
});

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
