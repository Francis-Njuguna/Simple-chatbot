/**
 * Amref Help Desk — embeddable floating chat widget.
 *
 * Usage (single script tag in any HTML/PHP template):
 *
 *   <script
 *     src="https://your-host/widget/chat-widget.js"
 *     data-api-base="https://your-backend/api/v1"
 *     defer></script>
 *
 * Talks directly to the existing FastAPI backend:
 *   POST {api-base}/chat        -> { answer, sources, images, confidence, session_id, message_id }
 *   GET  {api-base}/categories  -> ["Category A", ...]
 *   POST {api-base}/feedback    -> { message_id, rating }
 *
 * Conversation continuity uses the same session_id contract as the Streamlit
 * frontend: the first /chat response returns a session_id which is echoed
 * back on every subsequent request (persisted in sessionStorage per tab).
 *
 * No dependencies. All DOM classes are prefixed `acw-`.
 */
(function () {
  "use strict";

  // ------------------------------------------------------------------
  // Configuration (read from the <script> tag itself)
  // ------------------------------------------------------------------
  var scriptEl = document.currentScript;
  var API_BASE =
    (scriptEl && scriptEl.getAttribute("data-api-base")) ||
    "http://localhost:8000/api/v1";
  API_BASE = API_BASE.replace(/\/+$/, "");
  // Backend root (for resolving /static image paths), same logic as Streamlit.
  var BACKEND_URL =
    (scriptEl && scriptEl.getAttribute("data-backend-url")) ||
    API_BASE.replace(/\/api\/v1$/, "");

  var STORAGE_KEY = "amref_chat_session_id";
  var WIDGET_TITLE =
    (scriptEl && scriptEl.getAttribute("data-title")) ||
    "Amref Help Desk Assistant";

  // ------------------------------------------------------------------
  // Auto-inject the stylesheet that sits next to this JS file
  // ------------------------------------------------------------------
  (function injectCss() {
    if (scriptEl && scriptEl.getAttribute("data-css") === "false") return;
    var href =
      (scriptEl && scriptEl.getAttribute("data-css")) ||
      (scriptEl && scriptEl.src
        ? scriptEl.src.replace(/chat-widget\.js([?#].*)?$/, "chat-widget.css")
        : "chat-widget.css");
    if (document.querySelector('link[href="' + href + '"]')) return;
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    document.head.appendChild(link);
  })();

  // ------------------------------------------------------------------
  // State (mirrors Streamlit's session_state)
  // ------------------------------------------------------------------
  var state = {
    sessionId: null,
    selectedCategory: null,
    open: false,
    sending: false,
  };
  try {
    state.sessionId = window.sessionStorage.getItem(STORAGE_KEY) || null;
  } catch (e) {
    /* sessionStorage unavailable (e.g. sandboxed iframe) — degrade gracefully */
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /** Very small markdown subset: **bold**, `code`, [text](url), bare URLs. */
  function renderMarkdown(text) {
    var html = escapeHtml(text);
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    html = html.replace(
      /(^|[\s(])(https?:\/\/[^\s<)]+)/g,
      '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>'
    );
    return html;
  }

  function resolveImageUrl(img) {
    var path = img.filepath || "";
    if (path.indexOf("http") === 0) return path;
    if (path.indexOf("/static") === 0) return BACKEND_URL + path;
    return BACKEND_URL + "/static/images/" + (img.filename || "");
  }

  function saveSessionId(sid) {
    state.sessionId = sid;
    try {
      window.sessionStorage.setItem(STORAGE_KEY, sid);
    } catch (e) {
      /* ignore */
    }
  }

  function clearSession() {
    state.sessionId = null;
    try {
      window.sessionStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      /* ignore */
    }
  }

  // ------------------------------------------------------------------
  // API calls (same endpoints/payloads as the Streamlit frontend)
  // ------------------------------------------------------------------
  function fetchCategories() {
    return fetch(API_BASE + "/categories")
      .then(function (res) {
        return res.ok ? res.json() : [];
      })
      .catch(function () {
        return [];
      });
  }

  function sendChatMessage(message) {
    var payload = { message: message };
    if (state.sessionId) payload.session_id = state.sessionId;
    if (state.selectedCategory) payload.category = state.selectedCategory;

    return fetch(API_BASE + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(function (res) {
      if (!res.ok) {
        return res.text().then(function (t) {
          throw new Error("API error " + res.status + ": " + t);
        });
      }
      return res.json();
    });
  }

  function submitFeedback(messageId, rating) {
    return fetch(API_BASE + "/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message_id: messageId, rating: rating }),
    }).catch(function () {
      /* feedback is best-effort, same as Streamlit */
    });
  }

  // ------------------------------------------------------------------
  // Build DOM
  // ------------------------------------------------------------------
  var root = el("div", "acw-root");

  // Toggle button
  var toggleBtn = el("button", "acw-toggle");
  toggleBtn.setAttribute("aria-label", "Open help desk chat");
  var CHAT_ICON =
    '<svg viewBox="0 0 24 24"><path d="M20 2H4a2 2 0 0 0-2 2v18l4-4h14a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2zm-3 9H7V9h10v2zm0-3H7V6h10v2z"/></svg>';
  var CLOSE_ICON =
    '<svg viewBox="0 0 24 24"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';
  toggleBtn.innerHTML = CHAT_ICON;

  // Panel
  var panel = el("div", "acw-panel");
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", WIDGET_TITLE);

  // Header
  var header = el("div", "acw-header");
  var headerLeft = el("div");
  headerLeft.appendChild(el("div", "acw-header-title", WIDGET_TITLE));
  headerLeft.appendChild(el("div", "acw-header-sub", "Ask about LMS, Portal, Exams & more"));
  var headerActions = el("div", "acw-header-actions");
  var clearBtn = el("button", "acw-header-btn", "Clear");
  clearBtn.title = "Clear chat and start a new session";
  headerActions.appendChild(clearBtn);
  header.appendChild(headerLeft);
  header.appendChild(headerActions);

  // Category bar
  var categoryBar = el("div", "acw-category-bar");
  var categorySelect = document.createElement("select");
  categorySelect.setAttribute("aria-label", "Filter by category");
  var allOpt = document.createElement("option");
  allOpt.value = "";
  allOpt.textContent = "All Categories";
  categorySelect.appendChild(allOpt);
  categoryBar.appendChild(categorySelect);
  categoryBar.style.display = "none"; // shown once categories load

  // Messages
  var messagesEl = el("div", "acw-messages");
  var welcomeEl = el(
    "div",
    "acw-welcome",
    "\uD83C\uDF93 Hi! Ask me anything about the Help Desk knowledge base."
  );
  messagesEl.appendChild(welcomeEl);

  // Input bar
  var inputBar = el("div", "acw-input-bar");
  var input = document.createElement("textarea");
  input.className = "acw-input";
  input.rows = 1;
  input.maxLength = 2000;
  input.placeholder = "Type your question\u2026";
  var sendBtn = el("button", "acw-send", "Send");
  inputBar.appendChild(input);
  inputBar.appendChild(sendBtn);

  var footer = el("div", "acw-footer", "Amref International University Help Desk");

  panel.appendChild(header);
  panel.appendChild(categoryBar);
  panel.appendChild(messagesEl);
  panel.appendChild(inputBar);
  panel.appendChild(footer);

  root.appendChild(panel);
  root.appendChild(toggleBtn);

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------
  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addUserMessage(text) {
    var bubble = el("div", "acw-msg acw-msg-user");
    bubble.textContent = text;
    messagesEl.appendChild(bubble);
    scrollToBottom();
  }

  function addAssistantMessage(response) {
    var bubble = el("div", "acw-msg acw-msg-assistant");
    bubble.innerHTML = renderMarkdown(response.answer || "");
    messagesEl.appendChild(bubble);

    var meta = el("div", "acw-meta");

    // Confidence badge
    var confidence = response.confidence || 0;
    if (confidence) {
      meta.appendChild(
        el("span", "acw-confidence", "Confidence: " + Math.round(confidence * 100) + "%")
      );
    }

    // Images
    var images = response.images || [];
    if (images.length) {
      var imagesEl = el("div", "acw-images");
      images.slice(0, 3).forEach(function (img) {
        var a = document.createElement("a");
        a.href = resolveImageUrl(img);
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        var image = document.createElement("img");
        image.src = a.href;
        image.alt = img.caption || img.alt_text || "Related image";
        image.loading = "lazy";
        image.onerror = function () {
          a.style.display = "none";
        };
        a.appendChild(image);
        imagesEl.appendChild(a);
      });
      meta.appendChild(imagesEl);
    }

    // Sources
    var sources = response.sources || [];
    if (sources.length) {
      var details = document.createElement("details");
      details.className = "acw-sources";
      var summary = document.createElement("summary");
      summary.textContent = "\uD83D\uDCDA Sources & References (" + sources.length + ")";
      details.appendChild(summary);
      var list = document.createElement("ul");
      list.style.paddingLeft = "16px";
      sources.forEach(function (src) {
        var li = document.createElement("li");
        var a = document.createElement("a");
        a.href = src.url || "#";
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = src.title || "Article";
        li.appendChild(a);
        li.appendChild(
          document.createTextNode(
            " (relevance: " + Math.round((src.score || 0) * 100) + "%)"
          )
        );
        list.appendChild(li);
      });
      details.appendChild(list);
      meta.appendChild(details);
    }

    // Feedback stars (1..5 -> POST /feedback)
    if (response.message_id) {
      meta.appendChild(buildFeedback(response.message_id));
    }

    if (meta.childNodes.length) messagesEl.appendChild(meta);
    scrollToBottom();
  }

  function buildFeedback(messageId) {
    var wrap = el("div", "acw-feedback");
    var submitted = false;
    var stars = [];
    for (var i = 1; i <= 5; i++) {
      (function (rating) {
        var star = el("button", "acw-star", "\u2605");
        star.title = "Rate " + rating + " star" + (rating > 1 ? "s" : "");
        star.addEventListener("click", function () {
          if (submitted) return;
          submitted = true;
          stars.forEach(function (s, idx) {
            s.classList.toggle("acw-star-on", idx < rating);
            s.disabled = true;
          });
          submitFeedback(messageId, rating);
          wrap.appendChild(el("span", "acw-feedback-thanks", "Thanks!"));
        });
        stars.push(star);
        wrap.appendChild(star);
      })(i);
    }
    return wrap;
  }

  function showTyping() {
    var t = el("div", "acw-typing", "Assistant is typing\u2026");
    t.id = "acw-typing-indicator";
    messagesEl.appendChild(t);
    scrollToBottom();
    return t;
  }

  function showError(text) {
    messagesEl.appendChild(el("div", "acw-error", text));
    scrollToBottom();
  }

  // ------------------------------------------------------------------
  // Behaviour
  // ------------------------------------------------------------------
  function togglePanel(open) {
    state.open = open != null ? open : !state.open;
    panel.classList.toggle("acw-open", state.open);
    toggleBtn.innerHTML = state.open ? CLOSE_ICON : CHAT_ICON;
    toggleBtn.setAttribute(
      "aria-label",
      state.open ? "Close help desk chat" : "Open help desk chat"
    );
    if (state.open) input.focus();
  }

  function handleSend() {
    var text = input.value.trim();
    if (!text || state.sending) return;

    input.value = "";
    if (welcomeEl.parentNode) welcomeEl.parentNode.removeChild(welcomeEl);
    addUserMessage(text);

    state.sending = true;
    sendBtn.disabled = true;
    var typing = showTyping();

    sendChatMessage(text)
      .then(function (response) {
        if (response.session_id) saveSessionId(response.session_id);
        addAssistantMessage(response);
      })
      .catch(function (err) {
        showError("Sorry, something went wrong. Please try again. (" + err.message + ")");
      })
      .then(function () {
        if (typing.parentNode) typing.parentNode.removeChild(typing);
        state.sending = false;
        sendBtn.disabled = false;
        input.focus();
      });
  }

  toggleBtn.addEventListener("click", function () {
    togglePanel();
  });

  sendBtn.addEventListener("click", handleSend);

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  categorySelect.addEventListener("change", function () {
    state.selectedCategory = categorySelect.value || null;
  });

  clearBtn.addEventListener("click", function () {
    clearSession();
    messagesEl.innerHTML = "";
    messagesEl.appendChild(welcomeEl);
  });

  // ------------------------------------------------------------------
  // Boot
  // ------------------------------------------------------------------
  function boot() {
    document.body.appendChild(root);
    fetchCategories().then(function (categories) {
      if (Array.isArray(categories) && categories.length) {
        categories.forEach(function (cat) {
          var opt = document.createElement("option");
          opt.value = cat;
          opt.textContent = cat;
          categorySelect.appendChild(opt);
        });
        categoryBar.style.display = "";
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
