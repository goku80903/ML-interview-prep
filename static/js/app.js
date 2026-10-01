/* Loaded from the JSON <script> tag the template embeds (no inline data). */
const CONCEPTS = JSON.parse(document.getElementById("concepts-data").textContent);
const STUDY_SECTIONS = JSON.parse(document.getElementById("study-sections-data").textContent);
const STUDY_CATEGORIES = JSON.parse(document.getElementById("study-categories-data").textContent);
const STUDY_SECTIONS_BY_ID = {};
STUDY_SECTIONS.forEach(s => { STUDY_SECTIONS_BY_ID[s.id] = s; });

/* ---------------------------- persistence (server-side, data/progress.json) ---------------------------- */
/* Progress lives on disk via the Flask backend -- not just localStorage -- so it
   survives clearing browser data, switching browsers, or moving machines. */

let currentId = CONCEPTS[0].id;
let currentLang = "numpy";
let currentView = "home"; // "home" | "dashboard" | "practice" | "study" -- the app lands on a chooser
const codeState = {};   // `${id}::${lang}` -> code string
const testState = {};   // `${id}::${lang}` -> {results, ranOnce}

/* ---------------------------- study tab state (read-progress saved to disk via /api/study/read) ---------------------------- */
let currentStudyId = STUDY_SECTIONS[0].id;
let studyCategoryFilter = "all";
const quizState = {}; // sectionId -> {qIndex, revealed}
const studyReadState = {}; // sectionId -> true, populated from /api/progress on load

function loadStudyRead() {
  return studyReadState;
}
function toggleStudyRead(id) {
  const next = !studyReadState[id];
  if (next) studyReadState[id] = true; else delete studyReadState[id];
  fetch("/api/study/read", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, read: next }),
  }).catch(() => {});
  renderNav();
  if (currentView === "study") renderApp();
}

function keyFor(id, lang) { return id + "::" + lang; }

function groupByCategory(list) {
  const map = new Map();
  for (const c of list) {
    const cat = c.category || "Misc";
    if (!map.has(cat)) map.set(cat, []);
    map.get(cat).push(c);
  }
  return map;
}

/* ---------------------------- confirm modal ---------------------------- */
function showConfirm(message, onConfirm, confirmLabel) {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };

  const box = document.createElement("div");
  box.className = "modal-box";
  const msg = document.createElement("div");
  msg.className = "modal-message";
  msg.textContent = message;
  box.appendChild(msg);

  const row = document.createElement("div");
  row.className = "modal-btnrow";
  const cancelBtn = document.createElement("button");
  cancelBtn.className = "action";
  cancelBtn.textContent = "Cancel";
  cancelBtn.onclick = () => overlay.remove();
  const confirmBtn = document.createElement("button");
  confirmBtn.className = "action danger";
  confirmBtn.textContent = confirmLabel || "Reset";
  confirmBtn.onclick = () => { overlay.remove(); onConfirm(); };
  row.appendChild(cancelBtn);
  row.appendChild(confirmBtn);
  box.appendChild(row);

  overlay.appendChild(box);
  document.body.appendChild(overlay);

  const onKey = (e) => {
    if (e.key === "Escape") { overlay.remove(); document.removeEventListener("keydown", onKey); }
  };
  document.addEventListener("keydown", onKey);
}

function getCode(id, lang) {
  const k = keyFor(id, lang);
  if (!(k in codeState)) {
    const c = CONCEPTS.find(x => x.id === id);
    codeState[k] = c[lang] ? c[lang].stub : "";
  }
  return codeState[k];
}

let saveTimer = null;
let pendingSave = null; // {id, lang, code}

function scheduleSave(id, lang, code) {
  codeState[keyFor(id, lang)] = code;
  pendingSave = { id, lang, code };
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flushSave, 800);
}

function flushSave() {
  clearTimeout(saveTimer);
  if (!pendingSave) return;
  const body = JSON.stringify(pendingSave);
  pendingSave = null;
  fetch("/api/save", { method: "POST", headers: { "Content-Type": "application/json" }, body }).catch(() => {});
}

// Best-effort: flush a pending debounced save if the tab is closed before it fires.
window.addEventListener("beforeunload", () => {
  if (!pendingSave) return;
  const blob = new Blob([JSON.stringify(pendingSave)], { type: "application/json" });
  navigator.sendBeacon("/api/save", blob);
});

async function loadProgress() {
  try {
    const resp = await fetch("/api/progress");
    const progress = await resp.json();
    for (const [key, entry] of Object.entries(progress)) {
      if (key.startsWith("study::")) {
        if (entry.read) studyReadState[key.slice("study::".length)] = true;
        continue;
      }
      if (typeof entry.code === "string") codeState[key] = entry.code;
      if (entry.ran_once) {
        testState[key] = { results: entry.results && entry.results.length ? entry.results : [{ passed: false }], ranOnce: true };
      }
    }
  } catch (e) {
    // Backend unreachable / fresh install with no data/progress.json yet -- start empty.
  }
}

/* ---------------------------- nav ---------------------------- */
const nav = document.getElementById("nav");

function renderNav() {
  nav.innerHTML = "";

  const home = document.createElement("button");
  home.className = "q nav-home" + (currentView === "dashboard" ? " active" : "");
  home.textContent = "≡ Progress";
  home.onclick = () => { flushSave(); currentView = "dashboard"; renderNav(); renderApp(); };
  nav.appendChild(home);

  const studyBtn = document.createElement("button");
  studyBtn.className = "q nav-home" + (currentView === "study" ? " active" : "");
  studyBtn.textContent = "📖 Study Guide";
  studyBtn.onclick = () => { flushSave(); currentView = "study"; renderNav(); renderApp(); };
  nav.appendChild(studyBtn);

  if (currentView === "study") {
    renderStudyNavList();
    return;
  }
  if (currentView === "home") {
    return; // let the chooser cards in the main panel drive the pick
  }

  for (const [category, items] of groupByCategory(CONCEPTS)) {
    const header = document.createElement("div");
    header.className = "nav-category";
    header.textContent = category;
    nav.appendChild(header);

    for (const c of items) {
      const btn = document.createElement("button");
      btn.className = "q" + (currentView === "practice" && c.id === currentId ? " active" : "");

      const label = document.createElement("span");
      label.textContent = c.title;
      btn.appendChild(label);

      const tags = document.createElement("span");
      tags.className = "tags";
      if (c.numpy) {
        const t = document.createElement("span");
        t.className = "tag";
        const st = testState[keyFor(c.id, "numpy")];
        t.style.background = st && st.ranOnce ? (st.results.every(r => r.passed) ? "var(--ok)" : "var(--bad)") : "var(--numpy)";
        t.style.opacity = st && st.ranOnce ? "1" : "0.35";
        tags.appendChild(t);
      }
      if (c.torch) {
        const t = document.createElement("span");
        t.className = "tag";
        const st = testState[keyFor(c.id, "torch")];
        t.style.background = st && st.ranOnce ? (st.results.every(r => r.passed) ? "var(--ok)" : "var(--bad)") : "var(--torch)";
        t.style.opacity = st && st.ranOnce ? "1" : "0.35";
        tags.appendChild(t);
      }
      btn.appendChild(tags);

      btn.onclick = () => {
        flushSave();
        currentId = c.id;
        currentLang = c.numpy ? "numpy" : "torch";
        currentView = "practice";
        renderNav();
        renderApp();
      };
      nav.appendChild(btn);
    }
  }
}

function renderStudyNavList() {
  const filterRow = document.createElement("div");
  filterRow.className = "study-filter-row";
  const allChip = document.createElement("button");
  allChip.className = "study-chip" + (studyCategoryFilter === "all" ? " active" : "");
  allChip.textContent = "All";
  allChip.onclick = () => { studyCategoryFilter = "all"; renderNav(); };
  filterRow.appendChild(allChip);
  for (const cat of STUDY_CATEGORIES) {
    const chip = document.createElement("button");
    chip.className = "study-chip" + (studyCategoryFilter === cat.id ? " active" : "");
    chip.textContent = cat.label;
    chip.onclick = () => { studyCategoryFilter = cat.id; renderNav(); };
    filterRow.appendChild(chip);
  }
  nav.appendChild(filterRow);

  const readSet = loadStudyRead();
  for (const s of STUDY_SECTIONS) {
    if (studyCategoryFilter !== "all" && s.category !== studyCategoryFilter) continue;
    const btn = document.createElement("button");
    btn.className = "q" + (currentView === "study" && s.id === currentStudyId ? " active" : "");

    const check = document.createElement("span");
    check.className = "study-check" + (readSet[s.id] ? " checked" : "");
    check.onclick = (e) => { e.stopPropagation(); toggleStudyRead(s.id); };
    btn.appendChild(check);

    const label = document.createElement("span");
    label.textContent = s.title;
    btn.appendChild(label);

    btn.onclick = () => { currentStudyId = s.id; renderNav(); renderApp(); };
    nav.appendChild(btn);
  }
}

/* ---------------------------- Monaco editor (loaded lazily from CDN on first use) ---------------------------- */
const MONACO_CDN_BASE = "https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs";
let monacoLoadPromise = null;

function loadMonaco() {
  if (monacoLoadPromise) return monacoLoadPromise;
  monacoLoadPromise = new Promise((resolve, reject) => {
    if (window.monaco) { resolve(window.monaco); return; }
    const script = document.createElement("script");
    script.src = MONACO_CDN_BASE + "/loader.js";
    script.onload = () => {
      window.require.config({ paths: { vs: MONACO_CDN_BASE } });
      window.require(["vs/editor/editor.main"], () => {
        window.monaco.editor.defineTheme("dracula", {
          base: "vs-dark",
          inherit: true,
          rules: [
            { token: "comment", foreground: "6272a4", fontStyle: "italic" },
            { token: "string", foreground: "f1fa8c" },
            { token: "number", foreground: "bd93f9" },
            { token: "keyword", foreground: "ff79c6" },
            { token: "type", foreground: "8be9fd", fontStyle: "italic" },
            { token: "delimiter", foreground: "f8f8f2" },
            { token: "identifier", foreground: "f8f8f2" },
          ],
          colors: {
            "editor.background": "#21222c",
            "editor.foreground": "#f8f8f2",
            "editorLineNumber.foreground": "#6272a4",
            "editorLineNumber.activeForeground": "#f8f8f2",
            "editor.selectionBackground": "#44475a",
            "editor.lineHighlightBackground": "#343746",
            "editorCursor.foreground": "#f8f8f2",
            "editorIndentGuide.background": "#343746",
            "editorIndentGuide.activeBackground": "#6272a4",
          },
        });
        resolve(window.monaco);
      });
    };
    script.onerror = () => reject(new Error("Couldn't load the editor from the CDN -- check your internet connection."));
    document.head.appendChild(script);
  });
  return monacoLoadPromise;
}

/* ---------------------------- dashboard (landing page) ---------------------------- */
const main = document.getElementById("main");
let cmInstance = null;
let renderGeneration = 0;

function disposeEditor() {
  if (cmInstance) {
    cmInstance.dispose();
    cmInstance = null;
  }
}

function progressStats(list) {
  let total = 0, passed = 0;
  for (const c of (list || CONCEPTS)) {
    for (const lang of ["numpy", "torch"]) {
      if (!c[lang]) continue;
      total++;
      const st = testState[keyFor(c.id, lang)];
      if (st && st.ranOnce && st.results.every(r => r.passed)) passed++;
    }
  }
  return { total, passed };
}

function resetAllProgress() {
  showConfirm(
    "Reset ALL progress? This clears every exercise's saved code and pass/fail state. This cannot be undone.",
    async () => {
      await fetch("/api/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) }).catch(() => {});
      for (const k of Object.keys(codeState)) delete codeState[k];
      for (const k of Object.keys(testState)) delete testState[k];
      renderNav();
      renderApp();
    },
    "Reset everything"
  );
}

function openConcept(id, lang) {
  flushSave();
  currentId = id;
  currentLang = lang;
  currentView = "practice";
  renderNav();
  renderApp();
}

function renderDashboard() {
  main.innerHTML = "";
  const { total, passed } = progressStats();
  const pct = total ? Math.round((passed / total) * 100) : 0;

  const headerRow = document.createElement("div");
  headerRow.className = "dashboard-header";
  const h2 = document.createElement("h2");
  h2.textContent = "Your progress";
  const resetAllBtn = document.createElement("button");
  resetAllBtn.className = "action danger small";
  resetAllBtn.textContent = "Reset all progress";
  resetAllBtn.onclick = resetAllProgress;
  headerRow.appendChild(h2);
  headerRow.appendChild(resetAllBtn);
  main.appendChild(headerRow);

  const summary = document.createElement("div");
  summary.className = "progress-summary";
  const stat = document.createElement("div");
  stat.className = "progress-stat";
  stat.textContent = `${passed} / ${total} exercises passed (${pct}%)`;
  const barOuter = document.createElement("div");
  barOuter.className = "progress-bar-outer";
  const barInner = document.createElement("div");
  barInner.className = "progress-bar-inner";
  barInner.style.width = pct + "%";
  barOuter.appendChild(barInner);
  summary.appendChild(stat);
  summary.appendChild(barOuter);
  main.appendChild(summary);

  for (const [category, items] of groupByCategory(CONCEPTS)) {
    const catStats = progressStats(items);
    const catHeader = document.createElement("div");
    catHeader.className = "category-header";
    catHeader.textContent = `${category} — ${catStats.passed}/${catStats.total}`;
    main.appendChild(catHeader);

    const grid = document.createElement("div");
    grid.className = "concept-grid";
    for (const c of items) {
      const card = document.createElement("button");
      card.className = "concept-card";
      card.onclick = () => openConcept(c.id, c.numpy ? "numpy" : "torch");

      const title = document.createElement("div");
      title.className = "title";
      title.textContent = c.title;
      card.appendChild(title);

      const tags = document.createElement("div");
      tags.className = "tags";
      for (const lang of ["numpy", "torch"]) {
        if (!c[lang]) continue;
        const t = document.createElement("span");
        t.className = "tag";
        const st = testState[keyFor(c.id, lang)];
        t.style.background = st && st.ranOnce ? (st.results.every(r => r.passed) ? "var(--ok)" : "var(--bad)") : `var(--${lang})`;
        t.style.opacity = st && st.ranOnce ? "1" : "0.35";
        t.title = lang;
        t.onclick = (e) => { e.stopPropagation(); openConcept(c.id, lang); };
        tags.appendChild(t);
      }
      card.appendChild(tags);
      grid.appendChild(card);
    }
    main.appendChild(grid);
  }
}

/* ---------------------------- practice view ---------------------------- */
function renderPractice() {
  const c = CONCEPTS.find(x => x.id === currentId);
  if (!c[currentLang]) currentLang = c.numpy ? "numpy" : "torch";
  const v = c[currentLang];

  main.innerHTML = "";

  const split = document.createElement("div");
  split.className = "practice-split";
  main.appendChild(split);

  // ---- left pane: problem description (language-agnostic) ----
  const left = document.createElement("div");
  left.className = "practice-left";
  split.appendChild(left);

  const catTag = document.createElement("div");
  catTag.className = "cat-tag";
  catTag.textContent = c.category;
  left.appendChild(catTag);

  const h2 = document.createElement("h2");
  h2.textContent = c.title;
  left.appendChild(h2);

  // ---- right pane: language switch + editor + output (always built so the
  // switch works even when this language has no version) ----
  const right = document.createElement("div");
  right.className = "practice-right";
  split.appendChild(right);

  const langRow = document.createElement("div");
  langRow.className = "practice-lang-row";
  const sw = document.createElement("div");
  sw.className = "switch";
  const npBtn = document.createElement("button");
  npBtn.textContent = "NumPy";
  npBtn.className = "lang-numpy" + (currentLang === "numpy" ? " active" : "");
  npBtn.disabled = !c.numpy;
  npBtn.onclick = () => { flushSave(); currentLang = "numpy"; renderApp(); };
  const ptBtn = document.createElement("button");
  ptBtn.textContent = "PyTorch";
  ptBtn.className = "lang-torch" + (currentLang === "torch" ? " active" : "");
  ptBtn.disabled = !c.torch;
  ptBtn.onclick = () => { flushSave(); currentLang = "torch"; renderApp(); };
  sw.appendChild(npBtn); sw.appendChild(ptBtn);
  langRow.appendChild(sw);
  right.appendChild(langRow);

  if (!v) {
    const note = document.createElement("div");
    note.className = "torch-only-note";
    note.textContent = "No equivalent version for this language.";
    right.appendChild(note);
    return;
  }

  const prompt = document.createElement("div");
  prompt.className = "prompt";
  const promptLabel = document.createElement("div");
  promptLabel.className = "prompt-label";
  promptLabel.textContent = "Problem";
  const promptText = document.createElement("div");
  promptText.textContent = v.prompt;
  prompt.appendChild(promptLabel);
  prompt.appendChild(promptText);
  left.appendChild(prompt);

  const tabBar = document.createElement("div");
  tabBar.className = "editor-tabbar";
  const tab = document.createElement("span");
  tab.className = "editor-tab active lang-" + currentLang;
  tab.textContent = "your_submission.py";
  tabBar.appendChild(tab);
  right.appendChild(tabBar);

  const editorWrap = document.createElement("div");
  editorWrap.className = "editor-wrap";
  right.appendChild(editorWrap);

  const output = document.createElement("div");
  output.className = "practice-output";
  right.appendChild(output);

  const btnrow = document.createElement("div");
  btnrow.className = "btnrow";
  const runBtn = document.createElement("button");
  runBtn.className = "action primary lang-" + currentLang;
  runBtn.textContent = "▶ Run tests";
  const solutionBtn = document.createElement("button");
  solutionBtn.className = "action";
  solutionBtn.textContent = "💡 Reveal solution";
  const resetBtn = document.createElement("button");
  resetBtn.className = "action";
  resetBtn.textContent = "↺ Reset to stub";
  const clearProgressBtn = document.createElement("button");
  clearProgressBtn.className = "action danger";
  clearProgressBtn.textContent = "🗑 Clear progress";
  btnrow.appendChild(runBtn); btnrow.appendChild(solutionBtn); btnrow.appendChild(resetBtn); btnrow.appendChild(clearProgressBtn);
  output.appendChild(btnrow);

  const resultsDiv = document.createElement("div");
  resultsDiv.className = "results";
  output.appendChild(resultsDiv);

  const consoleDiv = document.createElement("div");
  consoleDiv.className = "console-panel";
  consoleDiv.style.display = "none";
  const consoleLabel = document.createElement("div");
  consoleLabel.className = "console-label";
  consoleLabel.textContent = "console";
  const consoleBody = document.createElement("pre");
  consoleBody.className = "console-body";
  consoleDiv.appendChild(consoleLabel);
  consoleDiv.appendChild(consoleBody);
  output.appendChild(consoleDiv);

  // Monaco (the editor engine behind VS Code) loads async from a CDN on
  // first use; show a placeholder until it's ready. myGeneration guards
  // against the user navigating away before the load finishes.
  const myGeneration = renderGeneration;
  editorWrap.innerHTML = `<div class="note">Loading editor…</div>`;

  loadMonaco().then((monaco) => {
    if (myGeneration !== renderGeneration) return; // navigated away while loading
    editorWrap.innerHTML = "";

    cmInstance = monaco.editor.create(editorWrap, {
      value: getCode(c.id, currentLang),
      language: "python",
      theme: "dracula",
      automaticLayout: true,
      minimap: { enabled: false },
      fontSize: 13,
      fontFamily: "SFMono-Regular, Consolas, Menlo, monospace",
      tabSize: 4,
      insertSpaces: true,
      scrollBeyondLastLine: false,
      renderLineHighlight: "all",
    });
    cmInstance.onDidChangeModelContent(() => scheduleSave(c.id, currentLang, cmInstance.getValue()));
    cmInstance.addCommand(monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => runBtn.click());

    resetBtn.onclick = () => { cmInstance.setValue(v.stub); scheduleSave(c.id, currentLang, v.stub); flushSave(); };
    solutionBtn.onclick = () => { cmInstance.setValue(v.solution); scheduleSave(c.id, currentLang, v.solution); flushSave(); };
    runBtn.onclick = () => runTests(c.id, currentLang, cmInstance.getValue(), resultsDiv, consoleDiv, runBtn);
  }).catch((err) => {
    if (myGeneration !== renderGeneration) return;
    editorWrap.innerHTML = `<div class="note error">${escapeHtml(err.message)}</div>`;
  });
  clearProgressBtn.onclick = () => {
    const lang = currentLang;
    showConfirm(
      `Clear saved progress for "${c.title}" (${lang})? This removes its saved code and pass/fail result, and resets the editor to the stub.`,
      async () => {
        const key = keyFor(c.id, lang);
        await fetch("/api/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: c.id, lang }) }).catch(() => {});
        delete codeState[key];
        delete testState[key];
        if (pendingSave && pendingSave.id === c.id && pendingSave.lang === lang) { pendingSave = null; clearTimeout(saveTimer); }
        renderNav();
        renderApp();
      },
      "Clear progress"
    );
  };
}

/* ---------------------------- study panel ---------------------------- */
function pickRandomQuizIndex(section, excludeIndex) {
  const n = section.quiz.length;
  if (n <= 1) return 0;
  let idx;
  do { idx = Math.floor(Math.random() * n); } while (idx === excludeIndex);
  return idx;
}

function renderQuizCard(section) {
  if (!section.quiz || !section.quiz.length) return document.createDocumentFragment();
  if (!quizState[section.id]) {
    quizState[section.id] = { qIndex: Math.floor(Math.random() * section.quiz.length), revealed: false };
  }
  const state = quizState[section.id];
  const q = section.quiz[state.qIndex];

  const card = document.createElement("div");
  card.className = "quiz-card";

  const label = document.createElement("div");
  label.className = "quiz-label";
  label.textContent = `Quiz yourself — question ${state.qIndex + 1} of ${section.quiz.length}`;
  card.appendChild(label);

  const question = document.createElement("div");
  question.className = "quiz-question";
  question.textContent = q.q;
  card.appendChild(question);

  if (state.revealed) {
    const answer = document.createElement("div");
    answer.className = "quiz-answer";
    answer.textContent = q.a;
    card.appendChild(answer);
  }

  const row = document.createElement("div");
  row.className = "quiz-btnrow";
  const revealBtn = document.createElement("button");
  revealBtn.className = "action";
  revealBtn.textContent = state.revealed ? "Hide answer" : "Reveal answer";
  revealBtn.onclick = () => { state.revealed = !state.revealed; renderApp(); };
  const nextQBtn = document.createElement("button");
  nextQBtn.className = "action";
  nextQBtn.textContent = "Next question";
  nextQBtn.onclick = () => {
    state.qIndex = pickRandomQuizIndex(section, state.qIndex);
    state.revealed = false;
    renderApp();
  };
  row.appendChild(revealBtn);
  row.appendChild(nextQBtn);
  card.appendChild(row);

  return card;
}

function renderStudyPanel() {
  const s = STUDY_SECTIONS_BY_ID[currentStudyId] || STUDY_SECTIONS[0];
  currentStudyId = s.id;
  main.innerHTML = "";

  const catInfo = STUDY_CATEGORIES.find(c => c.id === s.category);
  const tag = document.createElement("div");
  tag.className = "cat-tag";
  tag.textContent = catInfo ? catInfo.label : s.category;
  main.appendChild(tag);

  const h2 = document.createElement("h2");
  h2.textContent = s.title;
  main.appendChild(h2);

  const body = document.createElement("div");
  body.className = "study-content";
  body.innerHTML = s.html;
  main.appendChild(body);

  if (s.videos && s.videos.length) {
    const videoBox = document.createElement("div");
    videoBox.className = "study-videos";
    const label = document.createElement("div");
    label.className = "study-videos-label";
    label.textContent = "📺 Recommended videos";
    videoBox.appendChild(label);
    for (const v of s.videos) {
      const link = document.createElement("a");
      link.className = "study-video-link";
      link.href = v.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = v.title;
      videoBox.appendChild(link);
    }
    main.appendChild(videoBox);
  }

  main.appendChild(renderQuizCard(s));

  const footer = document.createElement("div");
  footer.className = "study-footer";
  const idx = STUDY_SECTIONS.findIndex(x => x.id === s.id);

  const prevBtn = document.createElement("button");
  prevBtn.className = "action";
  prevBtn.textContent = "← Previous";
  prevBtn.disabled = idx <= 0;
  prevBtn.onclick = () => { currentStudyId = STUDY_SECTIONS[idx - 1].id; renderNav(); renderApp(); };

  const readSet = loadStudyRead();
  const markBtn = document.createElement("button");
  markBtn.className = "action" + (readSet[s.id] ? " done" : "");
  markBtn.textContent = readSet[s.id] ? "Marked as read ✓" : "Mark as read";
  markBtn.onclick = () => toggleStudyRead(s.id);

  const nextBtn = document.createElement("button");
  nextBtn.className = "action";
  nextBtn.textContent = "Next →";
  nextBtn.disabled = idx >= STUDY_SECTIONS.length - 1;
  nextBtn.onclick = () => { currentStudyId = STUDY_SECTIONS[idx + 1].id; renderNav(); renderApp(); };

  footer.appendChild(prevBtn);
  footer.appendChild(markBtn);
  footer.appendChild(nextBtn);
  main.appendChild(footer);
}

function renderHome() {
  main.innerHTML = "";
  const h2 = document.createElement("h2");
  h2.textContent = "What do you want to do?";
  main.appendChild(h2);

  const grid = document.createElement("div");
  grid.className = "home-grid";

  function makeCard(icon, title, desc, onClick) {
    const card = document.createElement("button");
    card.className = "home-card";
    const iconEl = document.createElement("div");
    iconEl.className = "home-card-icon";
    iconEl.textContent = icon;
    const titleEl = document.createElement("div");
    titleEl.className = "home-card-title";
    titleEl.textContent = title;
    const descEl = document.createElement("div");
    descEl.className = "home-card-desc";
    descEl.textContent = desc;
    card.appendChild(iconEl); card.appendChild(titleEl); card.appendChild(descEl);
    card.onclick = onClick;
    return card;
  }

  const { total } = progressStats();
  grid.appendChild(makeCard(
    "≡", "Practice coding exercises",
    `${CONCEPTS.length} concepts / ${total} exercises — hands-on NumPy & PyTorch implementations with real tests, softmax to GRPO.`,
    () => { currentView = "dashboard"; renderNav(); renderApp(); }
  ));
  grid.appendChild(makeCard(
    "📖", "Study guide",
    `${STUDY_SECTIONS.length} sections of conceptual reference material, with self-check quizzes and video recommendations.`,
    () => { currentView = "study"; renderNav(); renderApp(); }
  ));

  main.appendChild(grid);
}

function renderApp() {
  disposeEditor();
  renderGeneration++;
  main.classList.toggle("practice-mode", currentView === "practice");
  if (currentView === "home") renderHome();
  else if (currentView === "dashboard") renderDashboard();
  else if (currentView === "study") renderStudyPanel();
  else renderPractice();
}

document.getElementById("homeLink").addEventListener("click", (e) => {
  e.preventDefault(); // stay in the SPA instead of a full page reload
  flushSave();
  currentView = "home";
  renderNav();
  renderApp();
});

/* ---------------------------- run tests via Flask ---------------------------- */
async function runTests(id, lang, code, resultsDiv, consoleDiv, runBtn) {
  flushSave();
  runBtn.disabled = true;
  resultsDiv.innerHTML = `<div class="note">Running locally…</div>`;
  consoleDiv.style.display = "none";
  consoleDiv.querySelector(".console-body").textContent = "";

  let payload;
  try {
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, lang, code }),
    });
    payload = await resp.json();
  } catch (e) {
    resultsDiv.innerHTML = `<div class="note error">Request failed: ${escapeHtml(String(e))}</div>`;
    runBtn.disabled = false;
    return;
  }

  resultsDiv.innerHTML = "";

  if (payload.console) {
    consoleDiv.style.display = "block";
    consoleDiv.querySelector(".console-body").textContent = payload.console;
  }

  if (payload.error) {
    const note = document.createElement("div");
    note.className = "note error";
    note.textContent = payload.error;
    resultsDiv.appendChild(note);
    testState[keyFor(id, lang)] = { results: [{ passed: false }], ranOnce: true };
  } else {
    for (const r of payload.results) {
      const row = document.createElement("div");
      row.className = "result-row " + (r.passed ? "pass" : "fail");

      const header = document.createElement("div");
      header.className = "result-header";
      const status = document.createElement("span");
      status.textContent = r.passed ? "PASS" : "FAIL";
      const name = document.createElement("span");
      name.textContent = r.name;
      header.appendChild(status);
      header.appendChild(name);
      row.appendChild(header);

      // Multi-line messages are real Python tracebacks -- render them like a
      // terminal, not squeezed into one line, so they're actually debuggable.
      if (r.message && r.message.includes("\n")) {
        const tb = document.createElement("pre");
        tb.className = "result-traceback";
        tb.textContent = r.message;
        row.appendChild(tb);
      } else if (r.message) {
        const msg = document.createElement("span");
        msg.className = "msg";
        msg.textContent = "— " + r.message;
        header.appendChild(msg);
      }

      resultsDiv.appendChild(row);
    }
    testState[keyFor(id, lang)] = { results: payload.results, ranOnce: true };
  }
  // Server already persisted this run's result (see /api/run); just refresh local nav/dashboard state.
  renderNav();
  runBtn.disabled = false;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

(async function init() {
  main.innerHTML = `<div class="note">Loading…</div>`;
  await loadProgress();
  renderNav();
  renderApp();
})();
