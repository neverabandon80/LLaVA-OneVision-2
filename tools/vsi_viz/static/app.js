const $ = (id) => document.getElementById(id);
const state = {
  shards: [],
  shard: null,
  keys: [],
  key: null,
  subCount: 0,
  subIdx: 0,
  view: null,
};

function setStatus(msg, cls = "") {
  const el = $("status");
  el.textContent = msg;
  el.className = "status " + cls;
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} for ${url}`);
  return r.json();
}

async function loadShards() {
  setStatus("loading shards…");
  const data = await fetchJSON("/api/shards");
  state.shards = data.shards;
  const sel = $("shard-sel");
  sel.innerHTML = state.shards
    .map((s) => `<option value="${s}">${s}</option>`)
    .join("");
  setStatus(`${data.count} shards`, "ok");
}

async function loadKeys(shard) {
  setStatus(`indexing ${shard}…`);
  const data = await fetchJSON(`/api/shard/${shard}/keys`);
  state.keys = data.keys;
  const sel = $("key-sel");
  sel.innerHTML = state.keys
    .map((k) => `<option value="${k}">${k}</option>`)
    .join("");
  setStatus(`${data.count} packed samples`, "ok");
}

async function loadSample(shard, key) {
  setStatus(`loading ${key}…`);
  const data = await fetchJSON(`/api/shard/${shard}/sample/${key}`);
  state.subCount = data.sub_count;
  const sel = $("sub-sel");
  sel.innerHTML = Array.from({ length: data.sub_count }, (_, i) => {
    const f = data.frames_per_sub[i];
    const t = data.turns_per_sub[i];
    return `<option value="${i}">sub ${i} · ${f}f · ${t}q</option>`;
  }).join("");
  setStatus(`${data.sub_count} sub-samples`, "ok");
}

async function loadSub(shard, key, subIdx) {
  setStatus(`rendering sub ${subIdx}…`);
  const view = await fetchJSON(
    `/api/shard/${shard}/sample/${key}/sub/${subIdx}`,
  );
  state.view = view;
  renderMeta(view);
  renderDialog(view);
  renderFrames(view);
  setStatus(`sub ${subIdx} ready`, "ok");
}

function renderMeta(view) {
  const td = view.timestamp_decimal;
  const fps = view.fps;
  const patchPct = view.patch_summary.total
    ? ((100 * view.patch_summary.unique) / view.patch_summary.total).toFixed(1)
    : "0";
  $("meta-body").innerHTML = `
    <dl>
      <dt>shard</dt><dd>${view.shard}</dd>
      <dt>packed key</dt><dd>${view.key}</dd>
      <dt>sub-sample</dt><dd>${view.sub_idx} / ${view.sub_count - 1}</dd>
    </dl>
    <div class="kv-block">
      <dl>
        <dt>fps</dt><dd>${fps ?? "—"}</dd>
        <dt>timestamp_decimal</dt><dd>${td}</dd>
        <dt># frames</dt><dd>${view.num_frames}</dd>
        <dt># Q/A turns</dt><dd>${view.num_turns}</dd>
      </dl>
    </div>
    <div class="kv-block">
      <dl>
        <dt>unique patch frames (RoPE)</dt>
        <dd>${view.patch_summary.unique} / ${view.patch_summary.total} (${patchPct}%)</dd>
        <dt>note</dt>
        <dd style="color: var(--fg-dim); font-family: inherit;">
          frames with empty patch_positions reuse the previous frame's RoPE positions
          (highlighted green in the grid).
        </dd>
      </dl>
    </div>
  `;
  const fmInline =
    fps != null
      ? `fps=${fps}, td=${td}, duration≈${(view.num_frames / fps).toFixed(2)}s`
      : "";
  $("frame-meta-inline").textContent = fmInline;
}

function escapeHTML(s) {
  return s.replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

function renderDialog(view) {
  const root = $("dialog");
  $("turn-count").textContent = `${view.num_turns} turn${view.num_turns === 1 ? "" : "s"}`;
  if (!view.turns.length) {
    root.innerHTML = `<div class="empty">no Q/A turns</div>`;
    return;
  }
  root.innerHTML = view.turns
    .map((t, i) => {
      const safe = escapeHTML(t.content).replace(
        /&lt;image&gt;/g,
        '<span class="img-tok">&lt;image&gt;</span>',
      );
      return `
        <div class="bubble ${t.role}">
          <span class="role-tag">${t.role} #${Math.floor(i / 2)}</span>
          <div>${safe}</div>
        </div>`;
    })
    .join("");
}

function renderFrames(view) {
  $("frame-count").textContent = `${view.num_frames} frame${view.num_frames === 1 ? "" : "s"}`;
  const showTs = $("show-ts").checked;
  const showPatch = $("show-patch").checked;
  const root = $("frames");
  root.innerHTML = view.images
    .map((img) => {
      const url = `/api/shard/${view.shard}/image/${view.key}/${img.tar_member}`;
      const ts =
        showTs && img.timestamp_str
          ? `<div class="ts">${img.timestamp_str}</div>`
          : "";
      const patchCls = showPatch && img.has_patch ? " has-patch" : "";
      return `
        <div class="frame${patchCls}" data-frame="${img.frame}" data-url="${url}" data-ts="${img.timestamp_str ?? ""}">
          <img loading="lazy" src="${url}" alt="frame ${img.frame}" />
          <div class="idx">#${img.frame}</div>
          ${ts}
        </div>`;
    })
    .join("");
}

function applyThumbSize(v) {
  document.documentElement.style.setProperty("--thumb", `${v}px`);
  $("size-val").textContent = v;
}

function bindEvents() {
  $("shard-sel").addEventListener("change", async (e) => {
    state.shard = e.target.value;
    await loadKeys(state.shard);
    state.key = state.keys[0];
    $("key-sel").value = state.key;
    await loadSample(state.shard, state.key);
    state.subIdx = 0;
    $("sub-sel").value = "0";
    await loadSub(state.shard, state.key, 0);
  });
  $("key-sel").addEventListener("change", async (e) => {
    state.key = e.target.value;
    await loadSample(state.shard, state.key);
    state.subIdx = 0;
    $("sub-sel").value = "0";
    await loadSub(state.shard, state.key, 0);
  });
  $("sub-sel").addEventListener("change", async (e) => {
    state.subIdx = parseInt(e.target.value, 10);
    await loadSub(state.shard, state.key, state.subIdx);
  });

  $("prev-sub").addEventListener("click", () => stepSub(-1));
  $("next-sub").addEventListener("click", () => stepSub(+1));
  $("prev-key").addEventListener("click", () => stepKey(-1));
  $("next-key").addEventListener("click", () => stepKey(+1));

  $("size-range").addEventListener("input", (e) =>
    applyThumbSize(e.target.value),
  );
  $("show-ts").addEventListener("change", () => state.view && renderFrames(state.view));
  $("show-patch").addEventListener("change", () => state.view && renderFrames(state.view));

  document.addEventListener("click", (e) => {
    const f = e.target.closest(".frame");
    if (!f) return;
    $("zoom-img").src = f.dataset.url;
    $("zoom-cap").textContent = `frame #${f.dataset.frame}  ${f.dataset.ts}`;
    $("zoom").showModal();
  });
  $("zoom").addEventListener("click", (e) => {
    if (e.target.id === "zoom") $("zoom").close();
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowRight") stepSub(+1);
    else if (e.key === "ArrowLeft") stepSub(-1);
    else if (e.key === "PageDown") stepKey(+1);
    else if (e.key === "PageUp") stepKey(-1);
  });
}

async function stepSub(delta) {
  const next = state.subIdx + delta;
  if (next < 0 || next >= state.subCount) return;
  state.subIdx = next;
  $("sub-sel").value = String(next);
  await loadSub(state.shard, state.key, next);
}

async function stepKey(delta) {
  const idx = state.keys.indexOf(state.key);
  const next = idx + delta;
  if (next < 0 || next >= state.keys.length) return;
  state.key = state.keys[next];
  $("key-sel").value = state.key;
  await loadSample(state.shard, state.key);
  state.subIdx = 0;
  $("sub-sel").value = "0";
  await loadSub(state.shard, state.key, 0);
}

(async function main() {
  bindEvents();
  applyThumbSize(120);
  try {
    await loadShards();
    state.shard = state.shards[0];
    await loadKeys(state.shard);
    state.key = state.keys[0];
    await loadSample(state.shard, state.key);
    state.subIdx = 0;
    await loadSub(state.shard, state.key, 0);
  } catch (e) {
    setStatus(e.message, "err");
    console.error(e);
  }
})();
