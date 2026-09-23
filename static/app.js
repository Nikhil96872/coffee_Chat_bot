/* Chat client for the coffee assistant API.
 *
 * Answers arrive as Server-Sent Events. We can't use EventSource because that
 * only issues GET requests and we need to POST the question and history, so the
 * SSE frames are parsed by hand off a fetch() stream.
 */

const chatEl     = document.getElementById("chat");
const formEl     = document.getElementById("chat-form");
const inputEl    = document.getElementById("question");
const sendBtn    = document.getElementById("send-btn");
const clearBtn   = document.getElementById("clear-btn");
const topkEl     = document.getElementById("topk");
const topkValue  = document.getElementById("topk-value");
const statusDot  = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const statusMeta = document.getElementById("status-meta");

let history = [];       // [{role, content}] sent back for follow-up questions
let busy = false;

/* ------------------------------------------------------------------ utils */

const escapeHtml = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Minimal markdown: bold, inline code, [1] citations, bullets, paragraphs. */
function renderMarkdown(text) {
  const blocks = escapeHtml(text).split(/\n{2,}/);
  return blocks
    .map((block) => {
      const lines = block.split("\n").filter((l) => l.trim());
      if (!lines.length) return "";

      const isBullet = lines.every((l) => /^\s*[-*•]\s+/.test(l));
      const isNumber = lines.every((l) => /^\s*\d+[.)]\s+/.test(l));

      if (isBullet || isNumber) {
        const tag = isBullet ? "ul" : "ol";
        const items = lines
          .map((l) => `<li>${inline(l.replace(/^\s*(?:[-*•]|\d+[.)])\s+/, ""))}</li>`)
          .join("");
        return `<${tag}>${items}</${tag}>`;
      }
      return `<p>${inline(lines.join(" "))}</p>`;
    })
    .join("");
}

function inline(s) {
  return s
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // Citation markers like [1] or [2][4] become chips the eye can pick out.
    // gpt-oss sometimes writes them with CJK lenticular brackets, 【1】.
    .replace(/[\[【](\d{1,2})[\]】]/g, '<span class="cite">$1</span>');
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function clearWelcome() {
  document.getElementById("welcome")?.remove();
}

/* --------------------------------------------------------------- rendering */

function addUserMessage(text) {
  const el = document.createElement("div");
  el.className = "msg msg-user";
  el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatEl.appendChild(el);
  scrollToBottom();
}

function addBotMessage() {
  const el = document.createElement("div");
  el.className = "msg msg-bot";
  el.innerHTML = `
    <div class="bot-label">Assistant</div>
    <div class="bubble">
      <div class="thinking"><span class="spinner"></span> Searching the documents…</div>
    </div>`;
  chatEl.appendChild(el);
  scrollToBottom();
  return el.querySelector(".bubble");
}

/** Link that opens the scanned PDF at the page the passage starts on. */
function pageLink(s) {
  const url = `/pdf/${encodeURIComponent(s.source)}#page=${s.page_start}`;
  return `<a class="source-open" href="${url}" target="_blank" rel="noopener">Open page ↗</a>`;
}

function renderSource(s) {
  const score = `<span class="source-score">${s.score >= 0 ? "+" : ""}${s.score}</span>`;

  if (s.paper) {
    const p = s.paper;
    return `
      <div class="source source-paper">
        <div class="source-head">
          <span class="source-n">[${s.n}]</span>
          <span class="source-kind">Research paper ${escapeHtml(p.number)}</span>
          ${score}
        </div>
        <div class="paper-title">${escapeHtml(p.title)}</div>
        <div class="paper-meta">
          ${p.citation ? `<span class="paper-authors">${escapeHtml(p.citation)}</span> · ` : ""}
          <span>${escapeHtml(s.citation)}</span> · ${pageLink(s)}
        </div>
        <div class="source-text paper-text">${escapeHtml(p.text)}</div>
      </div>`;
  }

  return `
      <div class="source">
        <div class="source-head">
          <span class="source-n">[${s.n}]</span>
          <span class="source-cite">${escapeHtml(s.citation)}</span>
          <span class="source-heading">${escapeHtml(s.heading || "")}</span>
          ${score}
        </div>
        <div class="source-text">${escapeHtml(s.text)}</div>
        <div class="paper-meta">${pageLink(s)}</div>
      </div>`;
}

/** Sources grouped by document, so it is plain when an answer draws on both. */
function renderSources(sources) {
  if (!sources.length) return "";
  const groups = new Map();
  for (const s of sources) {
    const doc = s.document || "Sources";
    if (!groups.has(doc)) groups.set(doc, []);
    groups.get(doc).push(s);
  }
  const order = ["Handbook", "Research papers"];
  const docs = [...groups.keys()].sort((a, b) => order.indexOf(a) - order.indexOf(b));

  const label = docs.map((d) => `${d} ${groups.get(d).length}`).join(" · ");
  const body = docs
    .map(
      (d) => `
      <div class="source-group">
        <div class="source-group-title">${escapeHtml(d)}
          <span class="source-group-count">${groups.get(d).length}</span>
        </div>
        ${groups.get(d).map(renderSource).join("")}
      </div>`
    )
    .join("");
  return `<details class="sources">
            <summary>Sources · ${label}</summary>${body}
          </details>`;
}

/* ------------------------------------------------------------------- chat */

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  sendBtn.disabled = true;
  clearWelcome();

  addUserMessage(question);
  const bubble = addBotMessage();

  let answer = "";
  let sources = [];

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        history,
        top_k: Number(topkEl.value),
      }),
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      const frames = buffer.split("\n\n");
      buffer = frames.pop();

      for (const frame of frames) {
        const evMatch = frame.match(/^event: (.+)$/m);
        const dataMatch = frame.match(/^data: (.+)$/m);
        if (!evMatch || !dataMatch) continue;

        const event = evMatch[1].trim();
        const data = JSON.parse(dataMatch[1]);

        if (event === "sources") {
          sources = data.sources;
          bubble.innerHTML = `<div class="thinking"><span class="spinner"></span> Writing the answer…</div>`;
        } else if (event === "token") {
          answer += data.t;
          bubble.innerHTML = renderMarkdown(answer) + '<span class="cursor"></span>';
          scrollToBottom();
        } else if (event === "error") {
          throw new Error(data.message);
        }
      }
    }

    bubble.innerHTML = renderMarkdown(answer) + renderSources(sources);
    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: answer });
    history = history.slice(-12);
  } catch (err) {
    bubble.innerHTML = `<div class="error-box"><strong>Something went wrong.</strong><br>${escapeHtml(
      err.message
    )}</div>`;
  } finally {
    busy = false;
    sendBtn.disabled = false;
    scrollToBottom();
    inputEl.focus();
  }
}

/* ---------------------------------------------------------------- events */

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = inputEl.value.trim();
  if (!q) return;
  inputEl.value = "";
  inputEl.style.height = "auto";
  ask(q);
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

// Grow the textarea with its content, up to the CSS max-height.
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + "px";
});

chatEl.addEventListener("click", (e) => {
  const example = e.target.closest(".example");
  if (example) ask(example.textContent.trim());
});

topkEl.addEventListener("input", () => {
  topkValue.textContent = topkEl.value;
});

clearBtn.addEventListener("click", () => {
  history = [];
  chatEl.innerHTML = `
    <div class="welcome" id="welcome">
      <h2>Ask about coffee cultivation</h2>
      <p>Answers come only from 654 scanned pages of the Coffee Board handbook
         and research abstracts. Every claim cites the page it came from.</p>
      <div class="examples">
        <button class="example">How do I control the coffee berry borer?</button>
        <button class="example">What causes coffee leaf rust and how is it controlled?</button>
        <button class="example">What is the difference between wet and dry processing?</button>
        <button class="example">Which shade trees are recommended, and why?</button>
      </div>
    </div>`;
});

/* --------------------------------------------------------------- startup */

async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    statusDot.className = "dot ok";
    statusText.textContent = `Ready · ${d.indexed_chunks.toLocaleString()} chunks`;
    statusMeta.textContent = d.model;
  } catch {
    statusDot.className = "dot error";
    statusText.textContent = "Backend unreachable";
    statusMeta.textContent = "Is api.py running?";
  }
}

checkHealth();
inputEl.focus();
