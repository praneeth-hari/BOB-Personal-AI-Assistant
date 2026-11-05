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

  function formatBytes(value) {
    let size = Number(value);
    if (!Number.isFinite(size) || size < 0) size = 0;
    const units = ["B", "KB", "MB", "GB", "TB"];
    let index = 0;
    while (size >= 1024 && index < units.length - 1) {
      size /= 1024;
      index += 1;
    }
    return (index === 0 ? Math.round(size) : size.toFixed(1)) + " " + units[index];
  }

  function disableStalePendingCards() {
    messagesEl.querySelectorAll(".pending-card").forEach((card) => {
      if (card.dataset.done === "true") return;
      card.dataset.done = "true";
      card.classList.add("stale");
      card.querySelectorAll("input, button").forEach((element) => {
        element.disabled = true;
      });
      const status = card.querySelector(".pending-status");
      if (status) status.textContent = "Superseded by a newer cleanup proposal.";
    });
  }

  function setPendingStatus(card, text, tone) {
    const status = card.querySelector(".pending-status");
    if (!status) return;
    status.textContent = text;
    status.className = "pending-status" + (tone ? " " + tone : "");
  }

  function makePendingCard(action) {
    const card = document.createElement("section");
    card.className = "pending-card";
    card.dataset.planId = String(action.plan_id || "");

    const head = document.createElement("div");
    head.className = "pending-head";
    const title = document.createElement("span");
    title.className = "pending-title";
    title.textContent = "Cleanup proposal";
    const expiry = document.createElement("span");
    expiry.className = "pending-expiry";
    const remaining = Math.max(
      0,
      Math.round((Number(action.expires_at || 0) * 1000 - Date.now()) / 60000)
    );
    expiry.textContent = remaining > 0 ? "Expires in ~" + remaining + " min" : "Expired";
    head.appendChild(title);
    head.appendChild(expiry);

    const summary = document.createElement("p");
    summary.className = "pending-summary";
    summary.textContent = String(action.summary || "");

    const targets = document.createElement("div");
    targets.className = "pending-targets";
    const checkboxes = {};

    (Array.isArray(action.targets) ? action.targets : []).forEach((target) => {
      const row = document.createElement("label");
      row.className = "pending-target";

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.dataset.target = String(target.key || "");
      checkbox.checked = Boolean(target.default_include) && !target.empty;
      checkbox.disabled = Boolean(target.empty);
      checkboxes[String(target.key || "")] = checkbox;

      const text = document.createElement("span");
      text.className = "pending-target-text";

      const name = document.createElement("span");
      name.className = "pending-target-name";
      name.textContent = String(target.label || target.key || "Cleanup target");

      const detail = document.createElement("span");
      detail.className = "pending-target-detail";
      detail.textContent = target.empty
        ? "Nothing to remove"
        : String(target.file_count || 0) +
          " item(s), " +
          formatBytes(target.bytes || 0);

      text.appendChild(name);
      text.appendChild(detail);

      if (target.warning) {
        const warning = document.createElement("span");
        warning.className = "pending-warning";
        warning.textContent = String(target.warning);
        text.appendChild(warning);
      }

      row.appendChild(checkbox);
      row.appendChild(text);
      targets.appendChild(row);
    });

    const total = document.createElement("p");
    total.className = "pending-total";
    total.textContent =
      "Available total: " +
      String(action.total_files || 0) +
      " item(s), " +
      formatBytes(action.total_bytes || 0);

    const actions = document.createElement("div");
    actions.className = "pending-actions";

    const approve = document.createElement("button");
    approve.type = "button";
    approve.className = "pending-button approve";
    approve.textContent = "Approve selected";

    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "pending-button cancel";
    cancel.textContent = "Cancel";

    const status = document.createElement("p");
    status.className = "pending-status";

    function finish(text, tone) {
      card.dataset.done = "true";
      checkboxes && Object.values(checkboxes).forEach((box) => (box.disabled = true));
      approve.disabled = true;
      cancel.disabled = true;
      setPendingStatus(card, text, tone);
    }

    approve.addEventListener("click", async () => {
      const includeTemp = Boolean(checkboxes.temp && checkboxes.temp.checked);
      const includeRecycle = Boolean(
        checkboxes.recycle_bin && checkboxes.recycle_bin.checked
      );
      if (!includeTemp && !includeRecycle) {
        setPendingStatus(card, "Select at least one cleanup target.", "error");
        return;
      }

      approve.disabled = true;
      cancel.disabled = true;
      setPendingStatus(card, "Cleaning up...", "");

      try {
        const response = await fetch("/api/approve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            plan_id: action.plan_id,
            include_temp: includeTemp,
            include_recycle_bin: includeRecycle,
          }),
        });
        const data = await readJson(response);
        if (!response.ok) {
          throw new Error((data && data.error) || "Cleanup could not be completed.");
        }

        let message =
          String(data.files_removed || 0) +
          " file(s) removed, " +
          formatBytes(data.bytes_freed || 0) +
          " freed.";
        if (data.files_skipped) {
          message += " " + String(data.files_skipped) + " item(s) skipped.";
        }
        if (Array.isArray(data.errors) && data.errors.length) {
          message += " " + data.errors.join(" ");
          finish(message, "error");
        } else {
          finish(message, "success");
        }
      } catch (err) {
        finish(err.message || "Cleanup could not be completed.", "error");
      }
    });

    cancel.addEventListener("click", async () => {
      approve.disabled = true;
      cancel.disabled = true;
      setPendingStatus(card, "Cancelling...", "");

      try {
        const response = await fetch("/api/reject", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ plan_id: action.plan_id }),
        });
        const data = await readJson(response);
        if (!response.ok) {
          throw new Error((data && data.error) || "Cleanup could not be cancelled.");
        }
        finish("Cleanup cancelled.", "success");
      } catch (err) {
        finish(err.message || "Cleanup could not be cancelled.", "error");
      }
    });

    actions.appendChild(approve);
    actions.appendChild(cancel);

    card.appendChild(head);
    card.appendChild(summary);
    card.appendChild(targets);
    card.appendChild(total);
    card.appendChild(actions);
    card.appendChild(status);
    return card;
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

  function appendAssistantMessage(text, tools, pendingActions) {
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

    if (Array.isArray(pendingActions) && pendingActions.length > 0) {
      disableStalePendingCards();
      pendingActions.forEach((action) => body.appendChild(makePendingCard(action)));
    }

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

      const tools = data.tools_used || [];
      const pending = data.pending_actions || [];
      if (pending.length === 0 && tools.includes("propose_cleanup")) {
        disableStalePendingCards();
      }
      appendAssistantMessage(data.reply, tools, pending);
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

  async function loadPendingActions() {
    try {
      const response = await fetch("/api/pending", { cache: "no-store" });
      const data = await readJson(response);
      if (!response.ok || !data || !Array.isArray(data.pending_actions)) return;
      if (data.pending_actions.length === 0) return;

      appendAssistantMessage(
        "A cleanup proposal is waiting for your approval.",
        [],
        data.pending_actions
      );
    } catch (_err) {
      /* Pending actions are optional; never block the chat page. */
    }
  }

  void checkHealth();
  void loadPendingActions();
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
