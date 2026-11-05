/* ─────────────────────────────────────────────────────────────────────────────
   BOB web interface
   Plain JavaScript, no build step. All assistant logic stays on the server.
   ───────────────────────────────────────────────────────────────────────────── */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const chatScroll = $("chat-scroll");
  const messagesEl = $("messages");
  const emptyState = $("empty-state");
  const form = $("composer");
  const input = $("input");
  const sendButton = $("send");
  const newChatButton = $("new-chat");
  const themeToggle = $("theme-toggle");
  const connection = $("connection");
  const connectionLabel = $("connection-label");
  const toast = $("toast");
  const hljsLight = $("hljs-light");
  const hljsDark = $("hljs-dark");

  const THEME_KEY = "bob-theme";
  const HEALTH_INTERVAL_MS = 15000;

  let busy = false;
  let stickToBottom = true;
  let toastTimer = null;

  // ── Theme ─────────────────────────────────────────────────────────────────

  const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

  function storedTheme() {
    try {
      const value = localStorage.getItem(THEME_KEY);
      return value === "dark" || value === "light" ? value : null;
    } catch (_err) {
      return null;
    }
  }

  function applyTheme(theme, persist) {
    document.documentElement.dataset.theme = theme;

    const dark = theme === "dark";
    if (hljsDark) hljsDark.disabled = !dark;
    if (hljsLight) hljsLight.disabled = dark;

    const label = dark ? "Switch to light theme" : "Switch to dark theme";
    themeToggle.title = label;
    themeToggle.setAttribute("aria-label", label);

    if (persist) {
      try {
        localStorage.setItem(THEME_KEY, theme);
      } catch (_err) {
        /* Storage may be unavailable; theme still applies for this session. */
      }
    }
  }

  applyTheme(storedTheme() || (systemDark.matches ? "dark" : "light"), false);

  systemDark.addEventListener?.("change", () => {
    if (!storedTheme()) {
      applyTheme(systemDark.matches ? "dark" : "light", false);
    }
  });

  themeToggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next, true);
  });

  // ── Markdown rendering ────────────────────────────────────────────────────

  if (window.marked) {
    window.marked.setOptions({ gfm: true, breaks: true });
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderMarkdown(text) {
    if (!window.marked || !window.DOMPurify) {
      return escapeHtml(text).replace(/\n/g, "<br>");
    }

    const raw = window.marked.parse(String(text));
    return window.DOMPurify.sanitize(raw, {
      USE_PROFILES: { html: true },
      FORBID_TAGS: ["style", "form", "input", "button"],
    });
  }

  // ── Scrolling ─────────────────────────────────────────────────────────────

  function isNearBottom() {
    return chatScroll.scrollHeight - chatScroll.scrollTop - chatScroll.clientHeight < 96;
  }

  chatScroll.addEventListener(
    "scroll",
    () => {
      stickToBottom = isNearBottom();
    },
    { passive: true }
  );

  function scrollToBottom(force = false) {
    if (!force && !stickToBottom) return;
    requestAnimationFrame(() => {
      chatScroll.scrollTop = chatScroll.scrollHeight;
    });
  }

  // ── Message rendering ─────────────────────────────────────────────────────

  function showConversation() {
    emptyState.hidden = true;
  }

  function showEmptyState() {
    emptyState.hidden = false;
  }

  function makeAssistantBody() {
    const body = document.createElement("div");
    body.className = "assistant-body";

    const label = document.createElement("div");
    label.className = "assistant-label";
    label.textContent = "BOB";

    body.appendChild(label);
    return body;
  }

  function makeToolBadge(tools) {
    const badge = document.createElement("div");
    badge.className = "tool-badge";
    badge.innerHTML =
      '<svg viewBox="0 0 24 24" aria-hidden="true">' +
      '<path d="M14.5 6.5 17.5 3.5a3.5 3.5 0 0 1 5 5l-3 3" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>' +
      '<path d="M9.5 17.5 6.5 20.5a3.5 3.5 0 0 1-5-5l3-3" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>' +
      '<path d="m8.5 15.5 7-7" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>' +
      "</svg>";

    const prefix = document.createElement("span");
    prefix.textContent = "Used";

    const text = document.createElement("span");
    text.className = "tool-name";
    text.textContent = tools.join(", ");

    badge.appendChild(prefix);
    badge.appendChild(text);
    return badge;
  }

  function appendUserMessage(text) {
    showConversation();

    const row = document.createElement("div");
    row.className = "message user";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    row.appendChild(bubble);
    messagesEl.appendChild(row);
    scrollToBottom(true);
  }

  function appendAssistantMessage(text, tools) {
    showConversation();

    const row = document.createElement("div");
    row.className = "message assistant";

    const body = makeAssistantBody();
    if (Array.isArray(tools) && tools.length > 0) {
      body.appendChild(makeToolBadge(tools.map(String)));
    }

    const content = document.createElement("div");
    content.className = "markdown";
    content.innerHTML = renderMarkdown(text);

    body.appendChild(content);
    row.appendChild(body);
    messagesEl.appendChild(row);

    enhanceCodeBlocks(content);
    enhanceLinks(content);
    scrollToBottom();
  }

  function appendErrorMessage(message) {
    showConversation();

    const row = document.createElement("div");
    row.className = "message assistant";

    const body = makeAssistantBody();
    const box = document.createElement("div");
    box.className = "error-message";
    box.textContent = message;

    body.appendChild(box);
    row.appendChild(body);
    messagesEl.appendChild(row);
    scrollToBottom();
  }

  function showThinking() {
    showConversation();

    const row = document.createElement("div");
    row.className = "message assistant";

    const body = makeAssistantBody();
    const dots = document.createElement("div");
    dots.className = "thinking";
    dots.setAttribute("aria-label", "BOB is thinking");
    dots.innerHTML = "<span></span><span></span><span></span>";

    body.appendChild(dots);
    row.appendChild(body);
    messagesEl.appendChild(row);
    scrollToBottom();
    return row;
  }

  function removeThinking(row) {
    if (row && row.parentNode) row.parentNode.removeChild(row);
  }

  // ── Code blocks and links ─────────────────────────────────────────────────

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }

    return new Promise((resolve, reject) => {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand("copy") ? resolve() : reject(new Error("Copy failed"));
      } catch (err) {
        reject(err);
      } finally {
        document.body.removeChild(area);
      }
    });
  }

  function enhanceCodeBlocks(root) {
    root.querySelectorAll("pre > code").forEach((code) => {
      const pre = code.parentElement;
      if (!pre || pre.closest(".code-block")) return;

      if (window.hljs) {
        try {
          window.hljs.highlightElement(code);
        } catch (_err) {
          /* Highlighting is cosmetic; never break the reply. */
        }
      }

      const wrapper = document.createElement("div");
      wrapper.className = "code-block";

      const head = document.createElement("div");
      head.className = "code-head";

      const language = document.createElement("span");
      language.className = "code-lang";
      const languageClass = Array.from(code.classList).find((name) =>
        name.startsWith("language-")
      );
      language.textContent = languageClass ? languageClass.slice(9) : "code";

      const copyButton = document.createElement("button");
      copyButton.type = "button";
      copyButton.className = "copy-button";
      copyButton.textContent = "Copy";
      copyButton.addEventListener("click", () => {
        copyText(code.innerText || code.textContent || "")
          .then(() => {
            copyButton.textContent = "Copied";
            copyButton.classList.add("copied");
            window.setTimeout(() => {
              copyButton.textContent = "Copy";
              copyButton.classList.remove("copied");
            }, 1600);
          })
          .catch(() => showToast("Could not copy to the clipboard."));
      });

      head.appendChild(language);
      head.appendChild(copyButton);

      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(head);
      wrapper.appendChild(pre);
    });
  }

  function enhanceLinks(root) {
    root.querySelectorAll("a[href]").forEach((link) => {
      const href = link.getAttribute("href") || "";
      if (/^https?:\/\//i.test(href)) {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
    });
  }

  // ── Composer ──────────────────────────────────────────────────────────────

  function autoGrow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
    updateSendButton();
  }

  function updateSendButton() {
    sendButton.disabled = busy || input.value.trim().length === 0;
  }

  function setBusy(value) {
    busy = value;
    newChatButton.disabled = value;
    updateSendButton();
  }

  input.addEventListener("input", autoGrow);

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!sendButton.disabled) form.requestSubmit();
    }
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (busy) return;

    const text = input.value.trim();
    if (!text) return;

    input.value = "";
    autoGrow();
    void sendMessage(text);
  });

  // ── Network ───────────────────────────────────────────────────────────────

  async function readJson(response) {
    try {
      return await response.json();
    } catch (_err) {
      return null;
    }
  }

  async function sendMessage(text) {
    setBusy(true);
    appendUserMessage(text);
    const thinkingRow = showThinking();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await readJson(response);

      removeThinking(thinkingRow);

      if (!response.ok) {
        throw new Error((data && data.error) || "BOB could not answer that request.");
      }
      if (!data || typeof data.reply !== "string" || !data.reply.trim()) {
        throw new Error("BOB returned an empty reply. Check the server logs.");
      }

      appendAssistantMessage(data.reply, data.tools_used || []);
    } catch (err) {
      removeThinking(thinkingRow);
      const message =
        err instanceof TypeError
          ? "Cannot reach the BOB server. Is it running?"
          : err.message || "Something went wrong.";
      appendErrorMessage(message);
      if (err instanceof TypeError) {
        setConnection("offline", "Offline", "Cannot reach the BOB server.");
      }
    } finally {
      setBusy(false);
      input.focus();
      void checkHealth();
    }
  }

  async function resetConversation() {
    newChatButton.disabled = true;
    try {
      const response = await fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      const data = await readJson(response);

      if (!response.ok) {
        throw new Error((data && data.error) || "Could not start a new chat.");
      }

      messagesEl.replaceChildren();
      showEmptyState();
      input.value = "";
      autoGrow();
      input.focus();
      scrollToBottom(true);
    } catch (err) {
      showToast(err.message || "Could not start a new chat.");
    } finally {
      newChatButton.disabled = busy;
    }
  }

  newChatButton.addEventListener("click", () => void resetConversation());

  // ── Health status ─────────────────────────────────────────────────────────

  function setConnection(state, label, title) {
    connection.dataset.state = state;
    connectionLabel.textContent = label;
    connection.title = title || label;
  }

  async function checkHealth() {
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      const data = await readJson(response);

      if (!response.ok) {
        setConnection("offline", "Offline", (data && data.error) || "BOB is unavailable.");
        return;
      }

      if (data && data.status === "ok") {
        const provider = data.provider || "unknown";
        const model = data.model || "unknown model";
        setConnection("online", "Online", provider + " · " + model);
      } else {
        setConnection(
          "offline",
          "Offline",
          (data && data.error) || "BOB is not configured."
        );
      }
    } catch (_err) {
      setConnection("offline", "Offline", "Cannot reach the BOB server.");
    }
  }

  void checkHealth();
  window.setInterval(() => void checkHealth(), HEALTH_INTERVAL_MS);

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) void checkHealth();
  });

  // ── Empty-state prompts ───────────────────────────────────────────────────

  document.querySelectorAll(".example").forEach((button) => {
    button.addEventListener("click", () => {
      const prompt = button.dataset.prompt || button.textContent || "";
      input.value = prompt.trim();
      autoGrow();
      input.focus();
    });
  });

  // ── Toast ─────────────────────────────────────────────────────────────────

  function showToast(message) {
    toast.textContent = message;
    toast.hidden = false;
    if (toastTimer) window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      toast.hidden = true;
    }, 4000);
  }

  // Initial state
  autoGrow();
})();
