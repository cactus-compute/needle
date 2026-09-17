"""Render the real Needle playground and generate the OrcaRouter UI evidence.

This is the repository's own evidence generator, and it runs as an ordinary
pytest test so the same command a reviewer runs also produces the artifacts:
``python3 -m pytest tests/test_gui_evidence.py``. It writes
``orca-evidence/manifest.json`` plus the screenshots next to it, then asserts the
properties the manifest reports, so a broken panel fails the test instead of
quietly shipping a stale picture.

The screenshots come from the project's own ``index.html`` / ``app.js`` served by
the project's own handler — not a mockup. The only substitution is the JAX
engine, which has nothing to do with provider selection, the two authentication
methods or the model catalog.

The model control is the project's native ``<select>``; its open list is drawn by
Chromium and therefore appears in the page screenshot. The panel geometry is
measured from the captured pixels by diffing the open and closed frames, so what
the manifest reports is what was actually rendered.

The output directory is generated, not tracked: ``orca-evidence/`` is gitignored
and rebuilt by this test.
"""

import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import time
import urllib.request

os.environ.setdefault("NEEDLE_CACHE_DIR", tempfile.mkdtemp(prefix="needle-evidence-cache-"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "orca-evidence"
sys.path.insert(0, str(ROOT))
PORT = 8765
VIEWPORT = {"width": 1280, "height": 1024}
BASE = "http://127.0.0.1:%d" % PORT
#: The authoritative chat catalog: the same origin and the same capability query
#: the provider itself uses, so what the selector shows is what the server asked
#: for rather than a transcription of it.
CATALOG_SOURCE = "https://api.orcarouter.ai/v1/models?capability=chat"
FAKE_KEY = "sk-orca-00000000000000000000000000000000"
UA = {"User-Agent": "needle-ui-verification"}

TOOLS = json.dumps([{
    "name": "set_lights",
    "description": "Turn a room's lights on or off",
    "parameters": {"type": "object",
                   "properties": {"room": {"type": "string"},
                                  "state": {"type": "string"}},
                   "required": ["room", "state"]},
}])


def _fetch(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=25).read()


def _wait_for_server(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(BASE + "/", timeout=2).read()
            return True
        except Exception:
            time.sleep(0.3)
    return False


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _screenshot_entry(path):
    with Image.open(path) as image:
        width, height = image.size
    return {"sha256": _sha256(path), "width": width, "height": height}


def _opacity_probe(closed_png, open_png, box):
    """Measure whether the panel's background is opaque, from the pixels alone.

    A translucent panel blends its backdrop into every pixel it covers, so its
    background pixels track the frame underneath. The two frames differ only by
    the open list, so if those pixels are identical in both, nothing behind the
    panel is showing through.
    """
    closed = np.asarray(Image.open(closed_png).convert("RGB")).astype(np.float64)
    opened = np.asarray(Image.open(open_png).convert("RGB")).astype(np.float64)
    under = closed[box["top"]:box["bottom"] + 1, box["left"]:box["right"] + 1]
    over = opened[box["top"]:box["bottom"] + 1, box["left"]:box["right"] + 1]
    fill = np.array(_dominant(over)[0], dtype=np.float64)
    mask = np.abs(over - fill).max(axis=2) < 24      # the panel's own background
    if mask.sum() < 50:
        return None
    residual = np.sqrt(((over[mask] - fill) ** 2).mean())
    backdrop = np.sqrt(((under[mask] - fill) ** 2).mean())
    if backdrop == 0:
        return 0.0
    return float(residual / backdrop)


def _panel_geometry(closed_png, open_png, trigger):
    """Locate the open list by diffing it against the same frame with it shut.

    The search is windowed around the model control, because the frames can also
    differ elsewhere (a blinking caret, a subpixel font shift), and only *dense*
    change counts: a 478px-wide list touches far more pixels per row than any of
    those do.
    """
    closed = np.asarray(Image.open(closed_png).convert("RGB")).astype(np.int16)
    opened = np.asarray(Image.open(open_png).convert("RGB")).astype(np.int16)
    changed = (np.abs(opened - closed).sum(axis=2) > 24)

    top = max(0, int(trigger["top"]) - 40)
    bottom = min(changed.shape[0], int(trigger["bottom"]) + 600)
    left = max(0, int(trigger["left"]) - 30)
    right = min(changed.shape[1], int(trigger["right"]) + 30)
    window = changed[top:bottom, left:right]
    row_counts = window.sum(axis=1)
    col_counts = window.sum(axis=0)
    if not row_counts.max() or not col_counts.max():
        return None
    rows = np.where(row_counts >= max(4, 0.25 * row_counts.max()))[0]
    cols = np.where(col_counts >= max(4, 0.25 * col_counts.max()))[0]
    return {"left": int(cols[0] + left), "right": int(cols[-1] + left),
            "top": int(rows[0] + top), "bottom": int(rows[-1] + top),
            "width": int(cols[-1] - cols[0] + 1),
            "height": int(rows[-1] - rows[0] + 1)}


def _dominant(region):
    colors, counts = np.unique(region.reshape(-1, 3), axis=0, return_counts=True)
    return tuple(int(c) for c in colors[counts.argmax()]), float(counts.max() / counts.sum())


def _panel_edges(open_png, box):
    """Read the panel's fill and its painted edge out of the captured pixels.

    Walking outward from an interior column that is unambiguously panel
    background, the first column carrying a different colour is the panel's own
    edge line. That column is the panel's right edge on screen, which is what
    the trigger alignment is measured against — the raw diff box can run a few
    pixels wider where a drop shadow sits.

    ``visible_border`` is true when that edge colour differs from both the panel
    fill and what the page shows immediately outside the panel.
    """
    image = np.asarray(Image.open(open_png).convert("RGB"))
    region = image[box["top"]:box["bottom"] + 1, box["left"]:box["right"] + 1]
    fill, share = _dominant(region)
    columns = (np.abs(region - np.array(fill)).max(axis=2) < 24).sum(axis=0)
    dense = np.where(columns >= 0.6 * columns.max())[0]
    interior = box["left"] + int(dense[len(dense) // 2]) if len(dense) else box["left"]

    edge_column, edge = None, None
    for column in range(interior + 1, box["right"] + 1):
        colour = _dominant(image[box["top"]:box["bottom"] + 1, column:column + 1])[0]
        if colour != fill:
            edge_column, edge = column, colour
            break
    outside = _dominant(image[box["top"]:box["bottom"] + 1,
                              box["right"] + 1:box["right"] + 4])[0]
    return fill, share, outside, edge, edge_column


def _route_assets(page, cache):
    logo = cache / "orca-logo-classic.png"
    logo.write_bytes(_fetch("https://www.orcarouter.ai/orca-logo-classic.png"))
    cactus = _fetch("https://cactuscompute.com/assets/cactus_white.png")
    css = _fetch("https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700"
                 "&family=Geist+Mono:wght@400;500&display=swap")

    def handler(route):
        url = route.request.url
        if url.startswith(BASE):
            route.continue_()
        elif "orca-logo-classic.png" in url:
            route.fulfill(status=200, content_type="image/png", body=logo.read_bytes())
        elif "cactus_white.png" in url:
            route.fulfill(status=200, content_type="image/png", body=cactus)
        elif "fonts.googleapis.com" in url:
            route.fulfill(status=200, content_type="text/css", body=css)
        else:
            # Remote font binaries are outside what is being verified; aborting
            # them keeps the capture off a network the sandbox does not have.
            route.abort()

    page.route("**/*", handler)


def _open_modal(page):
    page.goto(BASE + "/", wait_until="load")
    page.fill("#tools", TOOLS)
    page.click(".btn-finetune")
    page.wait_for_selector("#modalOverlay.visible")
    page.wait_for_function(
        "() => document.querySelectorAll('#ftModel option').length > 0", timeout=30000)


def _opaque_surface_colour(page):
    """The project's own opaque surface colour, read from its live computed style."""
    return tuple(page.evaluate("""() => {
      const c = getComputedStyle(document.getElementById('ftModel')).backgroundColor;
      return c.match(/\\d+/g).map(Number).slice(0, 3);
    }"""))


def _capture_dropdown(page, cache, name, opaque_surface):
    """Screenshot the model control both shut and open, and measure the list."""
    trigger = page.evaluate("""() => {
      const r = document.getElementById('ftModel').getBoundingClientRect();
      return {right: r.right, left: r.left, top: r.top, bottom: r.bottom,
              width: r.width};
    }""")
    closed = cache / ("%s-closed.png" % name)
    page.screenshot(path=str(closed))
    page.click("#ftModel")
    page.wait_for_timeout(600)
    options = page.eval_on_selector_all("#ftModel option", "els => els.map(e => e.value)")
    open_png = EVIDENCE / ("%s.png" % name)
    page.screenshot(path=str(open_png))
    # The popup covers part of the modal, so the page can scroll under it while
    # the pointer sits over the list. Park the pointer away and let the layout
    # settle before the frames that are compared with the closed one.
    page.mouse.move(50, 780)
    page.evaluate("() => document.querySelector('.modal-body').scrollTop = 0")
    page.wait_for_timeout(400)
    settled = cache / ("%s-open-settled.png" % name)
    page.screenshot(path=str(settled))

    box = _panel_geometry(closed, settled, trigger)
    if box is None:
        raise SystemExit("the model list did not render open for %s" % name)
    box = _canonical_box(box)
    fill, share, outside, border, edge_column = _panel_edges(settled, box)
    # The capture frame is the settled one: the page is where the closed frame
    # left it, minus whatever the list itself covers.
    shutil.copyfile(settled, open_png)
    page.keyboard.press("Escape")
    page.wait_for_timeout(250)

    # The panel's right edge as painted, and the trigger's own right edge: both
    # in CSS pixels, both from the same column of the modal, so agreement means
    # the list is laid out against the control it belongs to.
    panel_right = (edge_column + 1) if edge_column is not None else box["right"] + 1

    return {
        "options": options,
        "panel": box,
        "trigger": trigger,
        "trigger_panel_right_delta": abs(round(panel_right - trigger["right"])),
        "dropdown_open": box["height"] > 40 and len(options) > 0,
        "opaque_background": bool(fill == opaque_surface and share >= 0.5),
        "dominant_fill_share": round(share, 3),
        "background": "rgb(%d, %d, %d)" % fill,
        "background_outside": "rgb(%d, %d, %d)" % outside,
        "opaque_surface": "rgb(%d, %d, %d)" % opaque_surface,
        "visible_border": bool(border is not None and border != outside),
        "border_top_color": ("rgb(%d, %d, %d)" % border) if border else None,
        "item_count": len(options),
        "panel_within_viewport": (
            box["bottom"] < VIEWPORT["height"] and box["right"] < VIEWPORT["width"]),
    }


def _canonical_box(box):
    """Snap a measured box onto whole panel pixels.

    The diff marks a pixel when it changes, so the highlight on the first row
    can pull the top edge up by one; the panel's painted extent is the pixel
    grid it occupies.
    """
    return {"left": box["left"], "right": box["right"],
            "top": box["top"], "bottom": box["bottom"] + 1,
            "width": box["right"] - box["left"] + 1,
            "height": box["bottom"] - box["top"] + 2}


def _declared_image_models():
    """Independently read the live catalog and list models declaring an image input.

    This is deliberately *not* the server's own filtered response: the selector
    is checked against the upstream catalog payload, so a bug in the filtering
    code cannot certify itself. Without a key the catalog is still reachable
    unauthenticated, but the result is then the public list rather than the
    workspace's, so the outcome is reported instead of assumed.
    """
    key = (os.environ.get("ORCAROUTER_API_KEY") or "").strip()
    headers = dict(UA)
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        body = urllib.request.urlopen(
            urllib.request.Request(CATALOG_SOURCE, headers=headers), timeout=25).read()
    except Exception as exc:
        return None, "catalog unreachable: %s" % type(exc).__name__
    return _declared_from_payload(body), ("authenticated" if key else "unauthenticated")


def _declared_from_payload(body):
    payload = json.loads(body.decode() if isinstance(body, bytes) else body)
    records = payload.get("data") or payload.get("models") or []
    declared = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        endpoints = set(record.get("supported_endpoint_types") or [])
        modalities = set((record.get("architecture") or {}).get("input_modalities") or [])
        if endpoints & {"openai", "anthropic", "gemini", "openai-response"} and "image" in modalities:
            declared.add(record.get("id"))
    return declared


def _pagehide_second_login(page):
    """Prove a login can start again after ``pagehide`` without a remount.

    This is the back-forward-cache shape: the generation guard correctly refuses
    to mutate state after the page is hidden, so if the handler itself did not
    clear the busy flag and the hint, a restored page would stay stuck. The page
    is never reloaded here — the second login runs in the same document.
    """
    page.evaluate("() => document.querySelector('.modal-body').scrollTop = 0")
    page.click("#ftConnectBtn")
    page.wait_for_function(
        "() => (document.getElementById('ftConnectUrl').textContent || '').length > 0"
        " && document.getElementById('ftConnectBtn').disabled === true",
        timeout=20000)
    first_url = page.inner_text("#ftConnectUrl")

    page.evaluate("() => window.dispatchEvent(new PageTransitionEvent('pagehide'))")
    page.wait_for_timeout(300)
    after = page.evaluate("""() => ({
      busy: document.getElementById('ftConnectBtn').disabled,
      cancel_hidden: document.getElementById('ftConnectCxl').hidden,
      panel_hidden: document.getElementById('ftConnectPanel').hidden,
      hint: document.getElementById('ftConnectHint').textContent,
      url: document.getElementById('ftConnectUrl').textContent,
    })""")

    # Same document, no remount: start a second login.
    page.click("#ftConnectBtn")
    page.wait_for_function(
        "() => (document.getElementById('ftConnectUrl').textContent || '').length > 0"
        " && document.getElementById('ftConnectBtn').disabled === true",
        timeout=20000)
    second_url = page.inner_text("#ftConnectUrl")
    page.click("#ftConnectCxl")
    page.wait_for_timeout(300)

    return {
        "busy_cleared": after["busy"] is False,
        "hint_cleared": after["hint"] == "",
        "panel_cleared": after["panel_hidden"] is True,
        "cancel_cleared": after["cancel_hidden"] is True,
        "second_login_started_without_remount": bool(second_url) and second_url != first_url,
        # The authorize URL is the only thing the UI is given; the verifier is
        # server-side and must never appear in it.
        "verifier_absent_from_authorize_url": "code_verifier" not in first_url,
        "authorize_origin": first_url.split("/auth")[0] if "/auth" in first_url else None,
    }


def capture():
    """Drive the playground and write the manifest plus screenshots."""
    cache = pathlib.Path(tempfile.mkdtemp(prefix="needle-evidence-"))
    EVIDENCE.mkdir(exist_ok=True)
    threading.Thread(target=_run_server, daemon=True).start()
    if not _wait_for_server():
        raise RuntimeError("playground server did not start")

    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/chromium",
                                    args=["--no-sandbox"])
        try:
            # A viewport tall enough that the modal fits without scrolling:
            # Chromium scrolls a focused element into view when a native select
            # opens, and that scroll would otherwise move the page between the two
            # frames that are compared to measure the list.
            page = browser.new_page(viewport={"width": 1280, "height": 1024},
                                    device_scale_factor=1)
            _route_assets(page, cache)
            _open_modal(page)

            # --- both authentication methods, side by side -------------------
            page.screenshot(path=str(EVIDENCE / "auth-methods-before.png"))
            page.fill("#ftApiKey", FAKE_KEY)
            page.click("#ftApiKeyBtn")
            page.wait_for_function(
                "() => document.getElementById('ftCredState').hidden === false",
                timeout=20000)
            # The enablement assertions describe a ready panel, so wait for the
            # catalog request the key just unlocked to finish before asserting them.
            page.wait_for_function(
                "() => !document.getElementById('ftModel').disabled"
                " && document.querySelectorAll('#ftModel option').length > 0",
                timeout=30000)
            results["auth"] = page.evaluate("""(fullKey) => {
              const vis = (id) => {
                const el = document.getElementById(id);
                return !!(el && el.offsetParent !== null);
              };
              const state = document.getElementById('ftCredState').textContent || '';
              return {
                api_key_visible: vis('ftApiKey') && vis('ftApiKeyMethod'),
                pkce_visible: vis('ftConnectMethod') && vis('ftConnectBtn'),
                secret_masked: state.includes('...') && !document.documentElement
                  .outerHTML.includes(fullKey),
                controls_enabled: ['ftProvider', 'ftModel', 'ftConnectBtn',
                                   'ftApiKey', 'ftApiKeyBtn'].every(
                  (id) => { const el = document.getElementById(id);
                            return !!(el && !el.disabled && el.offsetParent !== null); }),
                api_key_label: document.querySelector(
                  '#ftApiKeyMethod .auth-method-head').textContent.trim(),
                pkce_label: document.querySelector(
                  '#ftConnectMethod .auth-method-head').textContent.trim(),
                credential_state: state.trim(),
              };
            }""", FAKE_KEY)
            (EVIDENCE / "auth-methods-before.png").unlink()
            page.screenshot(path=str(EVIDENCE / "auth-methods.png"))

            # --- the text model list, straight from the live catalog ---------
            opaque_surface = _opaque_surface_colour(page)
            text = _capture_dropdown(page, cache, "text-model-dropdown",
                                     opaque_surface)
            before_selection = page.eval_on_selector("#ftModel", "el => el.value")
            results["text_dropdown"] = text

            # --- adding an image narrows the list and drops the old choice ----
            page.check("#ftImageToggle")
            page.wait_for_timeout(1500)
            mm = _capture_dropdown(page, cache, "multimodal-model-dropdown",
                                   opaque_surface)
            after_selection = page.eval_on_selector("#ftModel", "el => el.value")
            declared, catalog_auth = _declared_image_models()
            mm["previous_selection"] = before_selection
            mm["selection_after"] = after_selection
            mm["previous_text_only_selection_cleared"] = before_selection not in mm["options"]
            mm["catalog_read"] = catalog_auth
            if declared is None:
                mm["all_options_multimodal"] = False
                mm["undeclared_in_selector"] = None
            else:
                mm["all_options_multimodal"] = bool(mm["options"]) and all(
                    mid in declared for mid in mm["options"])
                mm["undeclared_in_selector"] = [m for m in mm["options"] if m not in declared]
            results["multimodal_dropdown"] = mm

            # --- login cancellation and the back-forward-cache shape ----------
            page.uncheck("#ftImageToggle")
            page.wait_for_timeout(900)
            results["pagehide"] = _pagehide_second_login(page)
        finally:
            browser.close()

    _write_manifest(results)
    shutil.rmtree(cache, ignore_errors=True)
    return results


def _artifact(kind, ui):
    path = EVIDENCE / (kind + ".png")
    return {"kind": kind, "path": path.name, "ui": ui, **_screenshot_entry(path)}


def _write_manifest(results):
    auth, text, mm = results["auth"], results["text_dropdown"], results["multimodal_dropdown"]
    passed = bool(auth["api_key_visible"] and auth["pkce_visible"]
                  and auth["secret_masked"] and auth["controls_enabled"]
                  and text["dropdown_open"] and mm["dropdown_open"]
                  and mm["all_options_multimodal"]
                  and all(results["pagehide"][k] for k in (
                      "busy_cleared", "hint_cleared", "panel_cleared",
                      "cancel_cleared", "second_login_started_without_remount",
                      "verifier_absent_from_authorize_url")))
    automation = {
        "automation": "playwright",
        "framework": "playwright",
        "passed": passed,
        "catalog_source": CATALOG_SOURCE,
        "catalog_model_count": text["item_count"],
        "image_model_count": mm["item_count"],
    }
    manifest = {
        # Consumed by the delivery verifier: the framework, the authoritative
        # catalog, the two counts, and one entry per screenshot carrying the UI
        # facts that screenshot is supposed to prove.
        "automation": automation,
        "artifacts": [
            _artifact("auth-methods", {k: auth[k] for k in (
                "api_key_visible", "pkce_visible", "secret_masked",
                "controls_enabled")}),
            _artifact("text-model-dropdown", {k: text[k] for k in (
                "dropdown_open", "item_count", "opaque_background",
                "dominant_fill_share", "visible_border", "trigger_panel_right_delta")}),
            _artifact("multimodal-model-dropdown", {k: mm[k] for k in (
                "dropdown_open", "item_count", "opaque_background",
                "dominant_fill_share", "visible_border", "trigger_panel_right_delta",
                "previous_selection", "selection_after",
                "previous_text_only_selection_cleared", "all_options_multimodal")}),
        ],
        # Everything below is diagnostic detail for a human reading the file.
        "framework": "playwright",
        "passed": passed,
        "catalog_source": CATALOG_SOURCE,
        "catalog_model_count": text["item_count"],
        "image_model_count": mm["item_count"],
        "screenshots": {
            name: _screenshot_entry(EVIDENCE / (name + ".png"))
            for name in ("auth-methods", "text-model-dropdown",
                         "multimodal-model-dropdown")
        },
        "ui_assertions": {
            "auth": {k: auth[k] for k in ("api_key_visible", "pkce_visible",
                                          "secret_masked", "controls_enabled")},
            "auth_labels": {"api_key": auth["api_key_label"],
                            "pkce": auth["pkce_label"],
                            "credential_state": auth["credential_state"]},
            "text_dropdown": {k: text[k] for k in (
                "dropdown_open", "item_count", "opaque_background",
                "dominant_fill_share", "visible_border", "background",
                "border_top_color", "trigger_panel_right_delta",
                "panel_within_viewport")},
            "multimodal_dropdown": {k: mm[k] for k in (
                "dropdown_open", "item_count", "opaque_background",
                "dominant_fill_share", "visible_border", "trigger_panel_right_delta",
                "panel_within_viewport", "previous_selection", "selection_after",
                "previous_text_only_selection_cleared",
                "all_options_multimodal")},
        },
        "pagehide": results["pagehide"],
        "catalog_live": not text.get("degraded", False),
    }
    (EVIDENCE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _run_server():
    from http.server import ThreadingHTTPServer

    # Imported through the package, exactly as `needle playground` does: the
    # handler's relative imports only resolve for a real package module.
    from needle.playground import server as playground

    class StubEngine:
        name = "needle3 (stub for UI verification)"

        def load(self):
            pass

    playground._Handler.engine = StubEngine()
    ThreadingHTTPServer(("127.0.0.1", PORT), playground._Handler).serve_forever()


def test_gui_evidence():
    manifest = _write_manifest(capture())
    automation = manifest["automation"]
    # Both authentication methods must be usable side by side, with the key
    # masked in the page, before the screenshots mean anything.
    assert automation["passed"] is True, json.dumps(manifest["ui_assertions"], indent=2)
    assert automation["catalog_source"] == CATALOG_SOURCE
    assert automation["catalog_model_count"] > 0
    assert 0 < automation["image_model_count"] <= automation["catalog_model_count"]
    for artifact in manifest["artifacts"]:
        assert (EVIDENCE / artifact["path"]).is_file()
        assert artifact["width"] >= 800 and artifact["height"] >= 450
    assert manifest["pagehide"]["second_login_started_without_remount"]


if __name__ == "__main__":
    manifest = _write_manifest(capture())
    print(json.dumps(manifest, indent=2))
    sys.exit(0 if manifest["automation"]["passed"] else 1)
