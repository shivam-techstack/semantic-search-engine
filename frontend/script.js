/**
 * script.js v2 — Semantic Search Engine frontend
 *
 * New in v2 (from google-but-smarter reference):
 *  - 4 search modes: semantic, keyword, expanded, rocchio
 *  - All-4-modes comparison page
 *  - Inverted index term lookup page
 *  - More-Like-This button on every result card
 *  - Expanded query terms display strip
 */

const API = "http://127.0.0.1:5000";

// ── Tab navigation ──────────────────────────────────────────
document.querySelectorAll(".nav-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("page-" + tab.dataset.page).classList.add("active");
    if (tab.dataset.page === "about") loadStatus();
  });
});

// ── Helpers ──────────────────────────────────────────────────
const $ = id => document.getElementById(id);
function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function show(el) { el.style.display = ""; el.classList.add("visible"); }
function hide(el) { el.style.display = "none"; el.classList.remove("visible"); }
function scoreCls(s) {
  if (s >= 0.5) return "badge-score";
  if (s >= 0.25) return "badge-score medium";
  return "badge-score low";
}
function renderError(el, msg) {
  el.innerHTML = `<div class="message-box"><div class="icon">⚠️</div>
    <h4>Something went wrong</h4><p>${esc(msg)}</p></div>`;
}
function renderEmpty(el, q) {
  el.innerHTML = `<div class="message-box"><div class="icon">🔍</div>
    <h4>No results for "${esc(q)}"</h4>
    <p>Try different keywords or a broader query.</p></div>`;
}

// ── Mode pill highlight + description ────────────────────────
const MODE_DESC = {
  semantic: "🧠 <strong>Semantic:</strong> Averaged Word2Vec vectors + cosine similarity. Understands meaning, not just keywords.",
  keyword:  "🔑 <strong>Keyword (TF-IDF):</strong> Matches exact query terms weighted by rarity. Fast but vocabulary-dependent.",
  expanded: "🌐 <strong>Query Expansion:</strong> Adds semantically similar words to your query using Word2Vec before searching.",
  rocchio:  "🎯 <strong>Rocchio Feedback:</strong> Runs initial search, moves query vector toward top results, then re-ranks. Classic IR algorithm.",
};
const PILL_IDS = ["semantic","keyword","expanded","rocchio"];

function updatePills() {
  const val = document.querySelector('input[name="stype"]:checked').value;
  PILL_IDS.forEach(m => {
    const pill = $("pill-" + m);
    pill.className = "mode-pill";
    if (m === val) pill.classList.add("checked-" + m);
  });
  $("mode-desc").innerHTML = MODE_DESC[val] || "";
}

document.querySelectorAll('input[name="stype"]').forEach(r =>
  r.addEventListener("change", updatePills)
);
updatePills(); // set initial state

// ── PAGE 1 — SEARCH ─────────────────────────────────────────

async function runSearch() {
  const query = $("search-input").value.trim();
  if (!query) { $("search-input").focus(); return; }

  const type = document.querySelector('input[name="stype"]:checked').value;
  const k    = $("top-k").value;

  show($("search-spinner"));
  $("search-results").innerHTML = "";
  $("similar-panel").innerHTML  = "";

  try {
    const res  = await fetch(`${API}/api/search?q=${enc(query)}&type=${type}&k=${k}`);
    const data = await res.json();
    hide($("search-spinner"));

    if (!res.ok) { renderError($("search-results"), data.error || "Unknown error."); return; }
    if (!data.result_count) { renderEmpty($("search-results"), query); return; }

    renderResults(data, type);
  } catch {
    hide($("search-spinner"));
    renderError($("search-results"), "Cannot connect to server. Make sure app.py is running.");
  }
}

function renderResults(data, type) {
  const labels = {
    semantic: "🧠 Semantic (Word2Vec)",
    keyword:  "🔑 Keyword (TF-IDF)",
    expanded: "🌐 Query Expansion",
    rocchio:  "🎯 Rocchio Feedback",
  };

  let html = `<div class="results-header">
    <h3>${data.result_count} result${data.result_count!==1?"s":""} for
    "<strong>${esc(data.query)}</strong>" — ${labels[type]||type}</h3></div>`;

  // Show expanded terms if present
  if (type === "expanded" && data.expanded_terms && data.expanded_terms.length) {
    html += `<div class="expanded-info">
      <strong>🌐 Query expanded with:</strong>
      ${data.expanded_terms.map(t=>`<span class="term-tag">${esc(t)}</span>`).join("")}
    </div>`;
  }

  data.results.forEach((doc, i) => {
    html += `
      <div class="result-card mode-${type}">
        <div class="result-meta">
          <span class="result-rank">#${i+1}</span>
          <div class="result-badges">
            <span class="badge badge-category">${esc(doc.category)}</span>
            <span class="badge ${scoreCls(doc.score)}">Similarity: ${esc(doc.score_pct)}</span>
          </div>
        </div>
        <div class="result-title">${esc(doc.title)}</div>
        <div class="result-preview">${esc(doc.preview)}</div>
        <button class="btn-similar" data-id="${doc.id}" data-title="${esc(doc.title)}">
          🔗 More like this
        </button>
      </div>`;
  });

  $("search-results").innerHTML = html;

  // Attach More-Like-This handlers
  document.querySelectorAll(".btn-similar").forEach(btn => {
    btn.addEventListener("click", () => loadSimilar(+btn.dataset.id, btn.dataset.title));
  });
}

// ── More-Like-This (from google-but-smarter) ─────────────────
async function loadSimilar(docId, docTitle) {
  const panel = $("similar-panel");
  panel.innerHTML = `<div class="similar-panel">
    <h3>🔗 More like: "${esc(docTitle)}"</h3>
    <div class="spinner-ring" style="margin:16px auto;"></div>
  </div>`;
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });

  try {
    const res  = await fetch(`${API}/api/similar/${docId}?k=5`);
    const data = await res.json();

    if (!res.ok || !data.similar || !data.similar.length) {
      panel.innerHTML = `<div class="similar-panel"><p style="color:var(--muted)">No similar documents found.</p></div>`;
      return;
    }

    let html = `<div class="similar-panel">
      <h3>🔗 More like: "${esc(data.source_title)}"</h3>`;
    data.similar.forEach((doc, i) => {
      html += `<div class="similar-card">
        <div>
          <div class="sc-title">${i+1}. ${esc(doc.title)}</div>
          <div class="sc-meta"><span class="badge badge-category">${esc(doc.category)}</span></div>
        </div>
        <span class="badge ${scoreCls(doc.score)}">${esc(doc.score_pct)}</span>
      </div>`;
    });
    html += `</div>`;
    panel.innerHTML = html;
  } catch {
    panel.innerHTML = `<div class="similar-panel"><p style="color:var(--muted)">Error loading similar documents.</p></div>`;
  }
}

$("search-btn").addEventListener("click", runSearch);
$("search-input").addEventListener("keydown", e => { if (e.key==="Enter") runSearch(); });
$("clear-btn").addEventListener("click", () => {
  $("search-input").value = "";
  $("search-results").innerHTML = "";
  $("similar-panel").innerHTML  = "";
  $("search-input").focus();
});
$("sample-chips").addEventListener("click", e => {
  if (e.target.classList.contains("query-chip")) {
    $("search-input").value = e.target.textContent;
    runSearch();
  }
});

// ── PAGE 2 — COMPARE ALL 4 MODES ─────────────────────────────

async function runCompare() {
  const query = $("compare-input").value.trim();
  if (!query) { $("compare-input").focus(); return; }

  show($("compare-spinner"));
  $("compare-results").innerHTML = "";

  try {
    const res  = await fetch(`${API}/api/compare/all?q=${enc(query)}&k=5`);
    const data = await res.json();
    hide($("compare-spinner"));

    if (!res.ok) { renderError($("compare-results"), data.error||"Error"); return; }

    renderCompareAll(data);
  } catch {
    hide($("compare-spinner"));
    renderError($("compare-results"), "Cannot connect to server.");
  }
}

function renderCompareCol(results, type, label) {
  if (!results || !results.length)
    return `<div style="padding:14px 0;color:var(--muted);font-size:.85rem;text-align:center;">No results</div>`;

  return results.map((doc, i) => `
    <div class="compare-card">
      <div class="cc-rank">#${i+1} · <strong>${esc(doc.score_pct)}</strong></div>
      <div class="cc-title">${esc(doc.title)}</div>
      <div class="cc-score"><span class="badge badge-category">${esc(doc.category)}</span></div>
    </div>`).join("");
}

function renderCompareAll(data) {
  const cols = [
    { key: "semantic", label: "🧠 Semantic (Word2Vec)" },
    { key: "keyword",  label: "🔑 Keyword (TF-IDF)" },
    { key: "expanded", label: "🌐 Query Expansion" },
    { key: "rocchio",  label: "🎯 Rocchio Feedback" },
  ];

  let html = `<p style="font-size:.86rem;color:var(--muted);margin-bottom:10px;">
    Query: <strong>"${esc(data.query)}"</strong></p>`;

  // Show expansion terms if present
  if (data.expanded_terms && data.expanded_terms.length) {
    html += `<div class="expanded-info" style="margin-bottom:12px;">
      <strong>🌐 Query expanded with:</strong>
      ${data.expanded_terms.map(t=>`<span class="term-tag">${esc(t)}</span>`).join("")}
    </div>`;
  }

  html += `<div class="compare-grid-4">`;
  cols.forEach(({ key, label }) => {
    html += `<div class="compare-col ${key}">
      <h3>${label}</h3>
      ${renderCompareCol(data[key], key, label)}
    </div>`;
  });
  html += `</div>
    <div class="info-card" style="margin-top:14px;background:#f8fafc;">
      <p style="font-size:.83rem;color:var(--muted);">
        <strong>Reading this:</strong>
        Semantic finds by meaning. Keyword only matches exact words.
        Query Expansion adds related terms (e.g. "detect" → "diagnose", "identify").
        Rocchio shifts the query vector toward initial top results for a second-pass re-ranking.
      </p>
    </div>`;

  $("compare-results").innerHTML = html;
}

$("compare-btn").addEventListener("click", runCompare);
$("compare-input").addEventListener("keydown", e => { if (e.key==="Enter") runCompare(); });
$("compare-chips").addEventListener("click", e => {
  if (e.target.classList.contains("query-chip")) {
    $("compare-input").value = e.target.textContent;
    runCompare();
  }
});

// ── PAGE 3 — INVERTED INDEX ───────────────────────────────────

async function runIndexLookup() {
  const word = $("index-input").value.trim();
  if (!word) { $("index-input").focus(); return; }

  show($("index-spinner"));
  $("index-results").innerHTML = "";

  try {
    const res  = await fetch(`${API}/api/index/lookup?word=${enc(word)}`);
    const data = await res.json();
    hide($("index-spinner"));

    if (!res.ok) { renderError($("index-results"), data.error||"Error"); return; }

    renderIndexResult(data);
  } catch {
    hide($("index-spinner"));
    renderError($("index-results"), "Cannot connect to server.");
  }
}

function renderIndexResult(data) {
  if (!data.count) {
    $("index-results").innerHTML = `<div class="message-box">
      <div class="icon">📋</div>
      <h4>Term not found: "${esc(data.word)}"</h4>
      <p>${data.message || "This word does not appear in any document after preprocessing."}</p>
    </div>`;
    return;
  }

  let html = `<div class="index-result">
    <h3>📋 Posting list for: <strong>"${esc(data.token)}"</strong>
      ${data.word !== data.token ? `<span style="color:var(--muted);font-size:.85rem;font-weight:400;">(lemmatized from "${esc(data.word)}")</span>` : ""}
    </h3>
    <div class="index-stats">Found in <strong>${data.count}</strong> document${data.count!==1?"s":""} — Document IDs: [${data.doc_ids.join(", ")}]</div>
    <div class="posting-list">`;

  data.documents.forEach(doc => {
    html += `<div class="posting-item">
      <strong>Doc #${doc.id}</strong>
      ${esc(doc.title)}<br/>
      <span style="font-size:.76rem;color:var(--muted);">${esc(doc.category)}</span>
    </div>`;
  });

  html += `</div>
    <div style="margin-top:12px;padding:10px 14px;background:#eff6ff;border-radius:8px;font-size:.83rem;">
      <strong>How it works:</strong> When documents were indexed, each was tokenized and
      preprocessed. For every unique token, the document's ID was added to its posting list.
      At search time, posting lists are intersected to find documents matching all query terms.
    </div>
  </div>`;

  $("index-results").innerHTML = html;
}

$("index-btn").addEventListener("click", runIndexLookup);
$("index-input").addEventListener("keydown", e => { if (e.key==="Enter") runIndexLookup(); });
$("index-chips").addEventListener("click", e => {
  if (e.target.classList.contains("query-chip")) {
    $("index-input").value = e.target.textContent;
    runIndexLookup();
  }
});

// ── PAGE 4 — EVALUATION ──────────────────────────────────────

async function runEval() {
  const k = $("eval-k").value;
  show($("eval-spinner"));
  $("eval-results").innerHTML = "";

  try {
    const res  = await fetch(`${API}/api/evaluate?k=${k}`);
    const data = await res.json();
    hide($("eval-spinner"));
    if (!res.ok) { renderError($("eval-results"), data.error||"Error"); return; }
    renderEval(data);
  } catch {
    hide($("eval-spinner"));
    renderError($("eval-results"), "Cannot connect to server.");
  }
}

function bar(val, type) {
  const p = Math.round(val * 100);
  return `<div class="eval-bar-wrap">
    <div class="eval-bar-label"><span>${type}</span><span>${p}%</span></div>
    <div class="eval-bar"><div class="eval-bar-fill ${type.toLowerCase()}" style="width:${p}%"></div></div>
  </div>`;
}

function evalCard(data, type, label) {
  return `<div class="eval-card ${type}">
    <h3>${label}</h3>
    <div class="metric-row"><span>Precision@${data.k}</span><span class="val">${(data[type].avg_precision*100).toFixed(1)}%</span></div>
    <div class="metric-row"><span>Recall@${data.k}</span>   <span class="val">${(data[type].avg_recall*100).toFixed(1)}%</span></div>
    <div class="metric-row"><span>F1-Score</span>           <span class="val">${(data[type].avg_f1*100).toFixed(1)}%</span></div>
    ${bar(data[type].avg_precision, type)}
    ${bar(data[type].avg_recall,    type)}
    ${bar(data[type].avg_f1,        type)}
  </div>`;
}

function renderEval(data) {
  const s = data.semantic, kw = data.keyword;
  const winner = s.avg_f1 >= kw.avg_f1 ? "🧠 Semantic" : "🔑 Keyword";

  let html = `<h3 style="font-size:.93rem;margin-bottom:14px;color:var(--muted);">
    Results at K=${data.k} · ${data.queries.length} test queries</h3>
    <div class="eval-summary">
      ${evalCard(data, "semantic", "🧠 Semantic (Word2Vec)")}
      ${evalCard(data, "keyword",  "🔑 Keyword (TF-IDF)")}
    </div>
    <div class="info-card" style="margin-bottom:18px;background:#f0fdf4;border-left:4px solid var(--green);">
      <p style="font-size:.86rem;">
        <strong>${winner} achieved higher F1</strong>
        (${(Math.max(s.avg_f1,kw.avg_f1)*100).toFixed(1)}% vs
        ${(Math.min(s.avg_f1,kw.avg_f1)*100).toFixed(1)}%).
        Semantic search is especially effective when query vocabulary differs from document vocabulary.
      </p>
    </div>
    <h3 style="font-size:.92rem;margin-bottom:11px;">Per-Query Breakdown</h3>
    <div class="eval-table-wrap"><table>
      <thead><tr>
        <th>#</th><th>Query</th>
        <th>S-P</th><th>S-R</th><th>S-F1</th>
        <th>K-P</th><th>K-R</th><th>K-F1</th>
      </tr></thead><tbody>`;

  s.per_query.forEach((sq, i) => {
    const kq = kw.per_query[i];
    html += `<tr>
      <td>${i+1}</td>
      <td style="max-width:180px;font-size:.8rem;">${esc(sq.query)}</td>
      <td class="${sq.precision>=kq.precision?'good':''}">${(sq.precision*100).toFixed(0)}%</td>
      <td class="${sq.recall>=kq.recall?'good':''}">${(sq.recall*100).toFixed(0)}%</td>
      <td class="${sq.f1>=kq.f1?'good':''}">${(sq.f1*100).toFixed(0)}%</td>
      <td class="${kq.precision>sq.precision?'good':''}">${(kq.precision*100).toFixed(0)}%</td>
      <td class="${kq.recall>sq.recall?'good':''}">${(kq.recall*100).toFixed(0)}%</td>
      <td class="${kq.f1>sq.f1?'good':''}">${(kq.f1*100).toFixed(0)}%</td>
    </tr>`;
  });

  html += `</tbody></table></div>
    <p style="font-size:.76rem;color:var(--muted);margin-top:7px;">
      S=Semantic, K=Keyword. Green = better score. K=${data.k}.</p>`;

  $("eval-results").innerHTML = html;
}

$("eval-btn").addEventListener("click", runEval);

// ── PAGE 5 — STATUS ──────────────────────────────────────────

async function loadStatus() {
  const box = $("status-box");
  if (!box) return;
  try {
    const res  = await fetch(`${API}/api/status`);
    const data = await res.json();
    box.innerHTML = `<div class="info-card" style="background:#f0fdf4;border-left:4px solid var(--green);">
      <div class="icon">✅</div>
      <h4>Engine Status: Running</h4>
      <p>
        📄 Documents: <strong>${data.documents}</strong> &nbsp;·&nbsp;
        📐 Vector dims: <strong>${data.vector_size}</strong> &nbsp;·&nbsp;
        📚 Vocab: <strong>${data.vocab_size} words</strong> &nbsp;·&nbsp;
        📋 Index terms: <strong>${data.index_terms}</strong> &nbsp;·&nbsp;
        🔍 Modes: <strong>${(data.search_modes||[]).join(", ")}</strong>
      </p>
    </div>`;
  } catch {
    box.innerHTML = `<div class="info-card" style="background:#fef2f2;border-left:4px solid var(--red);">
      <div class="icon">❌</div><h4>Server not reachable</h4>
      <p>Start the backend: <code>python app.py</code></p>
    </div>`;
  }
}

// Utility
function enc(s) { return encodeURIComponent(s); }
