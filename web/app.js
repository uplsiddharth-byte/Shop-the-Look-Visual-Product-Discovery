const $ = (id) => document.getElementById(id);
const body = document.body;
const els = { ext: $("ext"), extLabel: $("extLabel"), drop: $("drop"), file: $("file"), scene: $("scene"), img: $("sceneImg"), boxes: $("boxes"), frame: $("frame"),
  status: $("status"), results: $("results"), samples: $("samples"), again: $("again"), theme: $("theme") };
const main = $("main");
const state = (s) => { body.dataset.state = s; };
state("idle");

// element helper: text goes in via textContent, so server strings are never parsed as HTML
function h(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  kids.flat().forEach((c) => c != null && e.append(c));
  return e;
}
const icon = (n) => h("i", { class: `ph ph-${n}`, "aria-hidden": "true" });

// theme: three dots (dark, light, red). Red is the default; the choice is remembered
const root = document.documentElement;
const setTheme = (t) => {
  root.dataset.theme = t;
  els.theme.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.t === t)));
  try { localStorage.setItem("theme", t); } catch {}
};
let saved = "blush"; try { saved = localStorage.getItem("theme") || "blush"; } catch {}
setTheme(["dark", "light", "blush"].includes(saved) ? saved : "blush");
els.theme.addEventListener("click", (e) => { const t = e.target.closest("button")?.dataset.t; if (t) setTheme(t); });

// extended catalog switch: remembered, and re-runs the current photo
try { els.ext.checked = localStorage.getItem("ext") !== "0"; } catch {}
let lastFile = null;
els.ext.addEventListener("change", () => {
  try { localStorage.setItem("ext", els.ext.checked ? "1" : "0"); } catch {}
  if (lastFile) run(lastFile);
});
fetch("/api/info").then((r) => r.json()).then((i) => {
  els.extLabel.textContent = `Extended catalog (+${i.extended.toLocaleString()})`;
  if (!i.extended) els.ext.closest("label").hidden = true;
}).catch(() => {});

// upload
els.file.addEventListener("change", () => els.file.files[0] && run(els.file.files[0]));
els.drop.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), els.file.click()));
["dragenter", "dragover"].forEach((t) => els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.remove("over"); }));
els.drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && run(e.dataTransfer.files[0]));
addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])][0]; if (f) run(f); });
els.again.addEventListener("click", () => { els.file.value = ""; els.scene.hidden = true; els.results.hidden = true; els.boxes.replaceChildren(); state("idle"); main.style.removeProperty("--rows"); });

fetch("/api/samples").then((r) => r.json()).then((list) => {
  els.samples.replaceChildren(...list.slice(0, 6).map((u) => h("button", { type: "button", "aria-label": "Try this sample photo",
    onclick: async () => { const b = await (await fetch(u)).blob(); run(new File([b], "sample.jpg", { type: b.type })); } },
    h("img", { src: u, alt: "", loading: "lazy" }))));
}).catch(() => { $("samplesWrap").hidden = true; });

async function run(file) {
  if (!file.type.startsWith("image/")) return fail("That file is not an image. Try a JPG or PNG.");
  if (file.size > 10 * 1024 * 1024) return fail("That image is over 10 MB. Try a smaller one.");
  lastFile = file;
  state("loading");
  els.boxes.replaceChildren();
  els.img.src = URL.createObjectURL(file);
  els.scene.hidden = false;
  els.results.hidden = false;
  els.status.textContent = "Looking for clothing items...";
  main.style.setProperty("--rows", 1);
  els.results.replaceChildren(...[0, 1].map(() => h("div", { class: "skel" })));
  await els.img.decode().catch(() => {});
  els.frame.style.setProperty("--h", els.frame.clientHeight - 90 + "px");
  try {
    const fd = new FormData(); fd.append("file", file);
    const r = await fetch(`/api/search?extended=${els.ext.checked}`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Server error (${r.status})`);
    render(await r.json());
  } catch (e) { fail(e.message || "Something went wrong. Please try again."); }
}

function fail(msg) {
  state("error");
  els.scene.hidden = true;
  els.results.hidden = false;
  els.results.replaceChildren(h("div", { class: "error", role: "alert" }, icon("warning-circle"),
    h("h3", { text: "We could not read that photo" }), h("p", { text: msg }),
    h("button", { class: "cta", type: "button", text: "Try again", onclick: () => els.again.click() })));
  body.dataset.state = "idle";
  // in the idle layout the results block sits below the samples, off screen: bring the error into view
  els.results.firstElementChild.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
}

function render(d) {
  state("done");
  const items = d.items.filter((i) => i.matches.length);
  main.style.setProperty("--rows", Math.max(1, Math.ceil(items.length / 2)));
  els.status.textContent = `${items.length} item${items.length === 1 ? "" : "s"} found in ${d.elapsed}s`;
  els.boxes.replaceChildren(...items.map((it, i) => {
    const [x0, y0, x1, y1] = it.box;
    return h("button", { class: "box", type: "button", "aria-label": `Item ${i + 1}: ${it.label}`, onclick: () => focusItem(i),
      style: `--i:${i};left:${x0 / d.width * 100}%;top:${y0 / d.height * 100}%;width:${(x1 - x0) / d.width * 100}%;height:${(y1 - y0) / d.height * 100}%` },
      h("span", { class: "n", text: i + 1 }));
  }));
  if (!items.length) {
    const none = d.reason === "no_close_match";
    els.results.replaceChildren(h("div", { class: "empty" }, icon("magnifying-glass"),
      h("h3", { text: none ? "No close matches in the catalog" : "No clothing found in this photo" }),
      h("p", { text: none ? "We found items, but nothing in the catalog is similar enough to show." : "Try a sharper photo where the outfit is clearly visible. Blurry, dark or non-clothing images cannot be matched." }),
      h("button", { class: "cta", type: "button", text: "Try another photo", onclick: () => els.again.click() })));
    return;
  }
  els.results.replaceChildren(...items.map(card));
}

// one card per item: a large current match, a strip of the others, click a thumb to swap
function card(it, i) {
  const img = h("img", { loading: "lazy" }), badge = h("span", { class: "badge" }), score = h("span", { class: "score" });
  const name = h("span", { class: "pname" }), why = h("p", { class: "why" });
  const thumbs = it.matches.map((m, k) => h("button", { class: "thumb", type: "button", "aria-label": `Show match ${k + 1}`, onclick: () => show(k) },
    h("img", { src: m.image, alt: "", loading: "lazy" })));
  function show(k) {
    const m = it.matches[k];
    img.src = m.image; img.alt = `${it.label} match, ${Math.round(m.score * 100)}% similar`;
    badge.textContent = m.exact ? "Exact match" : m.source === "extended" ? "Extended" : "Similar";
    badge.className = "badge" + (m.exact ? " exact" : m.source === "extended" ? " ext" : "");
    score.textContent = `${Math.round(m.score * 100)}% match`; name.textContent = m.name || "";
    why.textContent = m.explanation.text;
    thumbs.forEach((t, j) => t.setAttribute("aria-pressed", String(j === k)));
  }
  show(0);
  const tags = ["color", "pattern", "material"].map((k) => it.tags[k]).filter(Boolean).join(" / ");
  return h("article", { class: "item", id: `item-${i}`, style: `--i:${i}`, onmouseenter: () => mark(i, true), onmouseleave: () => mark(i, false) },
    h("header", {}, h("span", { class: "n", text: i + 1 }), h("div", {}, h("h3", { text: it.label }), tags ? h("p", { class: "sub", text: tags }) : null),
      h("span", { class: "pill" + (it.exact_found ? " ok" : ""), text: it.exact_found ? "Exact in catalog" : "Closest matches" })),
    h("div", { class: "big" }, img, badge),
    h("div", { class: "row" }, score, name), why,
    it.matches.length > 1 ? h("div", { class: "strip" }, thumbs) : null);
}

function mark(i, on) { els.boxes.children[i]?.classList.toggle("on", on); $(`item-${i}`)?.classList.toggle("on", on); }
function focusItem(i) { $(`item-${i}`)?.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" }); mark(i, true); setTimeout(() => mark(i, false), 1600); }
