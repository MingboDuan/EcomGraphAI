import html
import json
import re
import sys
import time
from pathlib import Path
from typing import Generator, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphrag.pipeline import GraphRAGPipeline


app = FastAPI(title="基于知识图谱的智能客服系统")
pipeline: Optional[GraphRAGPipeline] = None


class ChatRequest(BaseModel):
    query: str
    conversation_id: str = "web-demo"
    user_id: int | None = 51


def get_pipeline() -> GraphRAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = GraphRAGPipeline()
    return pipeline


def sse_event(event: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def clean_answer_for_customer(answer: str) -> str:
    """把模型常见 Markdown 符号弱化，避免前端直接显示原始标记。"""
    text = answer or ""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.M)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


@app.on_event("startup")
def startup_event():
    get_pipeline()


@app.on_event("shutdown")
def shutdown_event():
    if pipeline is not None:
        pipeline.close()


@app.get("/health")
def health():
    return get_pipeline().health()


@app.get("/api/users")
def list_users():
    """从图数据库读取全部用户 ID，供前端下拉选择。"""
    driver = get_pipeline().driver
    query = """
    MATCH (u:User)
    WHERE u.user_id IS NOT NULL
    RETURN DISTINCT u.user_id AS user_id
    ORDER BY user_id
    """
    try:
        records, _, _ = driver.execute_query(query)
        users = [record["user_id"] for record in records if record.get("user_id") is not None]
    except Exception:
        users = []
    return {"users": users}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.post("/chat")
def chat(request: ChatRequest):
    result = get_pipeline().run(request.query, request.conversation_id, request.user_id)
    return {
        "answer": clean_answer_for_customer(result["answer"]),
        "trace_id": result["trace_id"],
        "trace_path": result["trace_path"],
        "intent": result["intent"],
        "entities": result["entities"],
        "entry_nodes": result["entry_nodes"],
        "cypher": result["cypher"],
        "cypher_errors": result["cypher_errors"],
        "result_count": len(result["records"]),
        "records": result["records"],
    }


@app.post("/api/chat/stream")
def stream_chat(request: ChatRequest):
    """以 SSE 形式返回状态、元信息和逐字回答，供前端实现客服流式输出。"""

    def generate() -> Generator[str, None, None]:
        query = request.query.strip()
        if not query:
            yield sse_event("error", {"message": "请输入问题后再发送。"})
            return

        try:
            yield sse_event("status", {"message": "正在理解你的问题..."})
            time.sleep(0.08)
            yield sse_event("status", {"message": "正在结合用户画像和知识图谱检索..."})

            result = get_pipeline().run(query, request.conversation_id, request.user_id)
            meta = {
                "trace_id": result.get("trace_id"),
                "trace_path": result.get("trace_path"),
                "intent": result.get("intent", {}).get("intent"),
                "retrieval_query": result.get("retrieval_query"),
                "result_count": len(result.get("records", [])),
            }
            yield sse_event("meta", meta)
            yield sse_event("status", {"message": "已找到相关结果，正在组织回复..."})

            answer = clean_answer_for_customer(
                result.get("answer") or "暂时没有生成有效回答，请换一种方式描述需求。"
            )
            for char in answer:
                yield sse_event("token", {"text": char})
                time.sleep(0.01)

            yield sse_event("done", meta)
        except Exception as exc:
            yield sse_event("error", {"message": f"服务处理失败：{exc}"})

    return StreamingResponse(generate(), media_type="text/event-stream; charset=utf-8")


HTML_PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>基于知识图谱的智能客服系统</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #64748b;
      --line: #dbe3ef;
      --brand: #0f766e;
      --brand-2: #2563eb;
      --user: #1d4ed8;
      --assistant: #ffffff;
      --assistant-border: #dbeafe;
      --shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 32rem),
        linear-gradient(135deg, #eef7f6 0%, #f6f7fb 45%, #eef2ff 100%);
      overflow: hidden;
    }

    .app {
      height: 100vh;
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      gap: 18px;
      padding: 20px;
      overflow: hidden;
    }

    .sidebar, .chat-shell {
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid rgba(219, 227, 239, 0.9);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }

    .sidebar {
      height: calc(100vh - 40px);
      border-radius: 18px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      overflow: hidden;
    }

    .brand {
      display: flex;
      gap: 12px;
      align-items: center;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
      flex: 0 0 auto;
    }

    .brand-mark {
      width: 42px;
      height: 42px;
      border-radius: 12px;
      display: grid;
      place-items: center;
      color: #fff;
      font-weight: 800;
      background: linear-gradient(135deg, var(--brand), var(--brand-2));
    }

    h1 {
      margin: 0;
      font-size: 21px;
      line-height: 1.25;
      letter-spacing: 0;
    }

    .subtitle {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }

    .side-scroll {
      min-height: 0;
      overflow-y: auto;
      padding: 2px 2px 10px 0;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }

    .field { display: grid; gap: 8px; }
    .field label, .section-title {
      color: #334155;
      font-size: 15px;
      font-weight: 800;
      margin-bottom: 10px;
    }

    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
      font-family: "LXGW WenKai", "Microsoft YaHei UI", "PingFang SC", "Noto Sans SC", Arial, sans-serif;
      outline: none;
      transition: border-color .16s ease, box-shadow .16s ease;
    }

    input, select { height: 42px; padding: 0 12px; }
    textarea {
      min-height: 76px;
      resize: vertical;
      padding: 13px 14px;
      font-size: 15px;
      line-height: 1.65;
    }
    input:focus, select:focus, textarea:focus {
      border-color: #38bdf8;
      box-shadow: 0 0 0 3px rgba(56, 189, 248, .16);
    }

    .new-chat {
      height: 42px;
      border: 0;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--brand), var(--brand-2));
      color: #fff;
      font-weight: 800;
      cursor: pointer;
      flex: 0 0 auto;
    }

    .history-list, .quick-list {
      display: grid;
      gap: 10px;
    }

    .history-item {
      width: 100%;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 28px;
      gap: 10px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f8fafc;
      color: #1e293b;
      padding: 11px 10px 11px 14px;
      min-height: 58px;
    }

    .history-main {
      min-width: 0;
      cursor: pointer;
    }

    .history-delete {
      width: 26px;
      height: 26px;
      border: 0;
      border-radius: 8px;
      background: transparent;
      color: #94a3b8;
      cursor: pointer;
      font-size: 18px;
      line-height: 1;
    }

    .history-delete:hover {
      background: #fee2e2;
      color: #dc2626;
    }

    .quick-btn {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f8fafc;
      color: #1e293b;
      padding: 13px 15px;
      cursor: pointer;
      font-size: 14px;
      line-height: 1.55;
      min-height: 56px;
    }

    .history-item.active {
      border-color: #38bdf8;
      background: #eff6ff;
    }

    .history-title {
      font-weight: 800;
      font-size: 14px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .history-meta {
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }

    .conversation-field {
      display: none;
    }

    .quick-btn:hover, .history-item:hover {
      border-color: #93c5fd;
      background: #eff6ff;
    }

    .chat-shell {
      height: calc(100vh - 40px);
      border-radius: 20px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      overflow: hidden;
    }

    .chat-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(15, 118, 110, .08), rgba(37, 99, 235, .08));
    }

    .chat-title {
      font-size: 17px;
      font-weight: 800;
    }

    .status-pill {
      min-width: 92px;
      height: 30px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      background: #ecfdf5;
      color: #047857;
      border: 1px solid #bbf7d0;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }

    .messages {
      overflow-y: auto;
      padding: 28px 30px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,.88), rgba(248,250,252,.86)),
        radial-gradient(circle at 88% 12%, rgba(37, 99, 235, .08), transparent 22rem);
    }

    .empty {
      margin: auto;
      max-width: 560px;
      text-align: center;
      color: var(--muted);
      line-height: 1.8;
    }

    .empty strong {
      display: block;
      color: #0f172a;
      font-size: 24px;
      margin-bottom: 8px;
    }

    .msg {
      display: flex;
      gap: 10px;
      align-items: flex-start;
      max-width: min(880px, 94%);
    }
    .msg.user { align-self: flex-end; flex-direction: row-reverse; }
    .msg.assistant { align-self: flex-start; }

    .avatar {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      color: #fff;
      font-size: 13px;
      font-weight: 800;
      flex: 0 0 auto;
    }
    .user .avatar { background: var(--user); }
    .assistant .avatar { background: linear-gradient(135deg, var(--brand), var(--brand-2)); }

    .bubble {
      padding: 14px 17px;
      border-radius: 18px;
      line-height: 1.75;
      font-size: 15px;
      word-break: break-word;
      box-shadow: 0 8px 22px rgba(15, 23, 42, .06);
    }
    .user .bubble {
      background: var(--user);
      color: #fff;
      border-top-right-radius: 7px;
      white-space: pre-wrap;
    }
    .assistant .bubble {
      background: var(--assistant);
      border: 1px solid var(--assistant-border);
      border-top-left-radius: 7px;
      white-space: normal;
    }

    .thinking {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #475569;
    }

    .thinking::after {
      content: "";
      width: 22px;
      height: 6px;
      border-radius: 999px;
      background: linear-gradient(90deg, #94a3b8 20%, transparent 20% 40%, #94a3b8 40% 60%, transparent 60% 80%, #94a3b8 80%);
      animation: pulse 1s infinite ease-in-out;
    }

    @keyframes pulse {
      0%, 100% { opacity: .35; transform: translateY(0); }
      50% { opacity: 1; transform: translateY(-1px); }
    }

    .bubble p { margin: 0 0 8px; }
    .bubble p:last-child { margin-bottom: 0; }
    .bubble ul, .bubble ol { margin: 6px 0 8px 22px; padding: 0; }
    .bubble li { margin: 3px 0; }
    .bubble strong { color: #0f172a; }

    .meta {
      margin-top: 8px;
      color: #64748b;
      font-size: 12px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .composer {
      padding: 18px 22px;
      border-top: 1px solid var(--line);
      background: rgba(248, 250, 252, .9);
    }

    .composer-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: end;
    }

    .send-btn {
      height: 46px;
      min-width: 108px;
      border: 0;
      border-radius: 12px;
      color: #fff;
      font-size: 15px;
      font-weight: 800;
      cursor: pointer;
      background: linear-gradient(135deg, var(--brand), var(--brand-2));
      box-shadow: 0 10px 22px rgba(37, 99, 235, .22);
    }
    .send-btn:disabled {
      cursor: not-allowed;
      opacity: .58;
      box-shadow: none;
    }

    .hint {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }

    @media (max-width: 900px) {
      body { overflow: auto; }
      .app {
        height: auto;
        grid-template-columns: 1fr;
        padding: 12px;
        overflow: visible;
      }
      .sidebar {
        order: 2;
        height: auto;
        max-height: none;
      }
      .chat-shell {
        order: 1;
        height: 76vh;
      }
      .composer-row { grid-template-columns: 1fr; }
      .send-btn { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">KG</div>
        <div>
          <h1>基于知识图谱的智能客服系统</h1>
          <p class="subtitle">Graph RAG 电商问答演示</p>
        </div>
      </div>

      <button id="newChatBtn" class="new-chat">新建对话</button>

      <div class="side-scroll">
        <div class="field">
          <label for="userId">当前用户 ID</label>
          <select id="userId">
            <option value="51">用户 51</option>
            <option value="">不指定用户</option>
          </select>
        </div>

        <div class="field conversation-field">
          <label for="conversationId">会话 ID</label>
          <input id="conversationId" />
        </div>

        <div>
          <div class="section-title">历史对话</div>
          <div id="historyList" class="history-list"></div>
        </div>

        <div>
          <div class="section-title">示例问题</div>
          <div class="quick-list">
            <button class="quick-btn" data-query="有没有带保湿功能的口红，都是什么品牌的？">美妆1：有没有带保湿功能的口红？</button>
            <button class="quick-btn" data-query="那兰蔻的有哪些？">美妆2：那兰蔻的有哪些？</button>
            <button class="quick-btn" data-query="它们都是什么颜色的？">美妆3：它们都是什么颜色的？</button>
            <button class="quick-btn" data-query="用户51收藏过哪些商品？">行为1：用户51收藏过哪些商品？</button>
            <button class="quick-btn" data-query="里面有没有电视？">行为2：里面有没有电视？</button>
            <button class="quick-btn" data-query="这个电视是什么品牌和尺寸？">行为3：这个电视是什么品牌和尺寸？</button>
            <button class="quick-btn" data-query="你家是否有索尼的平板电视？都是多少尺寸的？">电视1：索尼平板电视都是多少尺寸？</button>
            <button class="quick-btn" data-query="有没有55英寸的？">电视2：有没有55英寸的？</button>
            <button class="quick-btn" data-query="这个品牌还有其它尺寸吗？">电视3：这个品牌还有其它尺寸吗？</button>
            <button class="quick-btn" data-query="有没有带保湿或者补水功能的口红，都是什么品牌的？">有没有带保湿或者补水功能的口红，都是什么品牌的？</button>
            <button class="quick-btn" data-query="我想找一款15英寸以上，32G内存，2TB硬盘的笔记本，屏幕要求2K以上">我想找一款15英寸以上，32G内存，2TB硬盘的笔记本</button>
            <button class="quick-btn" data-query="有无32G内存的电脑？">有无32G内存的电脑？</button>
            <button class="quick-btn" data-query="有花生油吗？有哪些品牌？">有花生油吗？有哪些品牌？</button>
            <button class="quick-btn" data-query="荣耀有没有5000mAh电池的手机？">荣耀有没有5000mAh电池的手机？</button>
            <button class="quick-btn" data-query="欧莱雅有没有抗皱功效的护肤品？">欧莱雅有没有抗皱功效的护肤品？</button>
            <button class="quick-btn" data-query="用户14最近看过哪些商品？">用户14最近看过哪些商品？</button>
            <button class="quick-btn" data-query="我看过哪些商品？">我看过哪些商品？</button>
            <button class="quick-btn" data-query="我收藏过哪些商品？">我收藏过哪些商品？</button>
          </div>
        </div>
      </div>
    </aside>

    <main class="chat-shell">
      <header class="chat-header">
        <div>
          <div class="chat-title">智能客服对话</div>
          <div class="subtitle">支持商品查询、用户行为兴趣查询和多轮追问</div>
        </div>
        <div id="status" class="status-pill">就绪</div>
      </header>

      <section id="messages" class="messages">
        <div class="empty" id="emptyState">
          <strong>你好，我是图谱智能客服</strong>
          可以直接询问商品、品牌、规格、用户浏览/点击/收藏记录，例如“我看过哪些商品”。
        </div>
      </section>

      <footer class="composer">
        <div class="composer-row">
          <textarea id="query" placeholder="请输入你的问题，按 Enter 发送，Shift + Enter 换行"></textarea>
          <button id="sendBtn" class="send-btn">发送</button>
        </div>
        <div class="hint">当前回答会以流式方式输出；切换用户 ID 后，可继续围绕该用户行为记录提问。</div>
      </footer>
    </main>
  </div>

  <script>
    const STORAGE_KEY = "graph_rag_chat_sessions_v2";
    const messages = document.getElementById("messages");
    const emptyState = document.getElementById("emptyState");
    const queryInput = document.getElementById("query");
    const sendBtn = document.getElementById("sendBtn");
    const statusPill = document.getElementById("status");
    const userIdSelect = document.getElementById("userId");
    const conversationInput = document.getElementById("conversationId");
    const historyList = document.getElementById("historyList");
    const newChatBtn = document.getElementById("newChatBtn");

    let sessions = loadSessions();
    let currentSessionId = "";

    function nowId() {
      return "web-" + new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
    }

    function loadSessions() {
      try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      } catch {
        return [];
      }
    }

    function saveSessions() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.slice(0, 40)));
    }

    function getCurrentSession() {
      return sessions.find(item => item.id === currentSessionId);
    }

    function ensureSession(id) {
      let session = sessions.find(item => item.id === id);
      if (!session) {
        session = {
          id,
          title: "新对话",
          userId: userIdSelect.value || "51",
          updatedAt: Date.now(),
          messages: []
        };
        sessions.unshift(session);
        saveSessions();
      }
      return session;
    }

    function setStatus(text, busy = false) {
      statusPill.textContent = text;
      statusPill.style.background = busy ? "#eff6ff" : "#ecfdf5";
      statusPill.style.color = busy ? "#1d4ed8" : "#047857";
      statusPill.style.borderColor = busy ? "#bfdbfe" : "#bbf7d0";
    }

    function escapeHtml(text) {
      return String(text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function renderCustomerHtml(text) {
      const safe = escapeHtml(text || "").replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
      const lines = safe.split(/\n+/).map(line => line.trim()).filter(Boolean);
      let html = "";
      let inList = false;
      for (const line of lines) {
        const listMatch = line.match(/^([-*•]|\d+[.)])\s*(.+)$/);
        if (listMatch) {
          if (!inList) {
            html += "<ul>";
            inList = true;
          }
          html += `<li>${listMatch[2]}</li>`;
        } else {
          if (inList) {
            html += "</ul>";
            inList = false;
          }
          html += `<p>${line.replace(/^#{1,6}\s*/, "")}</p>`;
        }
      }
      if (inList) html += "</ul>";
      return html || "<p><span class=\"thinking\">正在为你查询，请稍候</span></p>";
    }

    function scrollToBottom() {
      messages.scrollTop = messages.scrollHeight;
    }

    function clearMessages() {
      messages.innerHTML = "";
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.id = "emptyState";
      empty.innerHTML = "<strong>你好，我是图谱智能客服</strong>可以直接询问商品、品牌、规格、用户浏览/点击/收藏记录，例如“我看过哪些商品”。";
      messages.appendChild(empty);
    }

    function addMessage(role, text = "", save = true) {
      const empty = document.getElementById("emptyState");
      if (empty) empty.style.display = "none";
      const row = document.createElement("div");
      row.className = "msg " + role;

      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = role === "user" ? "我" : "客";

      const content = document.createElement("div");
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      if (role === "assistant") {
        bubble.innerHTML = renderCustomerHtml(text);
      } else {
        bubble.textContent = text;
      }
      content.appendChild(bubble);

      row.appendChild(avatar);
      row.appendChild(content);
      messages.appendChild(row);
      scrollToBottom();

      if (save) {
        const session = ensureSession(currentSessionId);
        session.messages.push({role, text, time: Date.now()});
        session.updatedAt = Date.now();
        if (role === "user" && (!session.title || session.title === "新对话")) {
          session.title = text.slice(0, 24);
        }
        sessions = [session, ...sessions.filter(item => item.id !== session.id)];
        saveSessions();
        renderHistory();
      }
      return { row, bubble, content };
    }

    function addMeta(content, meta) {
      const box = document.createElement("div");
      box.className = "meta";
      const parts = [];
      if (meta.intent) parts.push("意图：" + meta.intent);
      if (typeof meta.result_count === "number") parts.push("结果数：" + meta.result_count);
      if (meta.trace_id) parts.push("trace：" + meta.trace_id);
      box.textContent = parts.join("  |  ");
      content.appendChild(box);
    }

    function renderHistory() {
      historyList.innerHTML = "";
      if (!sessions.length) {
        const empty = document.createElement("div");
        empty.className = "history-meta";
        empty.textContent = "暂无历史对话";
        historyList.appendChild(empty);
        return;
      }
      sessions.forEach(session => {
        const item = document.createElement("div");
        item.className = "history-item" + (session.id === currentSessionId ? " active" : "");
        item.innerHTML = `
          <div class="history-main">
            <div class="history-title">${escapeHtml(session.title || "新对话")}</div>
            <div class="history-meta">用户 ${escapeHtml(session.userId || "未指定")} · ${new Date(session.updatedAt).toLocaleString()}</div>
          </div>
          <button class="history-delete" title="删除对话" aria-label="删除对话">×</button>
        `;
        item.querySelector(".history-main").addEventListener("click", () => switchSession(session.id));
        item.querySelector(".history-delete").addEventListener("click", (event) => {
          event.stopPropagation();
          deleteSession(session.id);
        });
        historyList.appendChild(item);
      });
    }

    function deleteSession(id) {
      sessions = sessions.filter(item => item.id !== id);
      saveSessions();
      if (id === currentSessionId) {
        if (sessions.length) {
          switchSession(sessions[0].id);
        } else {
          newSession();
        }
      } else {
        renderHistory();
      }
    }

    function switchSession(id) {
      const session = ensureSession(id);
      currentSessionId = session.id;
      conversationInput.value = session.id;
      userIdSelect.value = session.userId ?? "";
      messages.innerHTML = "";
      if (!session.messages.length) {
        clearMessages();
      } else {
        session.messages.forEach(message => addMessage(message.role, message.text, false));
      }
      renderHistory();
      scrollToBottom();
    }

    function newSession() {
      const id = nowId();
      currentSessionId = id;
      const session = ensureSession(id);
      session.userId = userIdSelect.value || "51";
      conversationInput.value = id;
      clearMessages();
      renderHistory();
      queryInput.focus();
    }

    async function loadUsers() {
      try {
        const response = await fetch("/api/users");
        const data = await response.json();
        const selected = userIdSelect.value || "51";
        userIdSelect.innerHTML = "";
        for (const userId of data.users || []) {
          const option = document.createElement("option");
          option.value = String(userId);
          option.textContent = "用户 " + userId;
          userIdSelect.appendChild(option);
        }
        const none = document.createElement("option");
        none.value = "";
        none.textContent = "不指定用户";
        userIdSelect.appendChild(none);
        userIdSelect.value = selected;
      } catch {
        userIdSelect.value = "51";
      }
    }

    async function sendMessage() {
      const query = queryInput.value.trim();
      if (!query || sendBtn.disabled) return;

      const session = ensureSession(currentSessionId);
      session.userId = userIdSelect.value || "";
      conversationInput.value = session.id;

      addMessage("user", query);
      queryInput.value = "";

      const assistant = addMessage("assistant", "", false);
      assistant.bubble.innerHTML = "<span class=\"thinking\">正在为你查询，请稍候</span>";
      sendBtn.disabled = true;
      setStatus("处理中", true);

      const userIdRaw = userIdSelect.value;
      const body = {
        query,
        conversation_id: conversationInput.value.trim() || "web-demo",
        user_id: userIdRaw ? Number(userIdRaw) : null
      };

      let answerText = "";
      try {
        let hasAnswerToken = false;
        const response = await fetch("/api/chat/stream", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(body)
        });

        if (!response.ok || !response.body) {
          throw new Error("服务请求失败");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {stream: true});
          const chunks = buffer.split("\n\n");
          buffer = chunks.pop() || "";

          for (const chunk of chunks) {
            const lines = chunk.split("\n");
            const eventLine = lines.find(line => line.startsWith("event: "));
            const dataLine = lines.find(line => line.startsWith("data: "));
            if (!eventLine || !dataLine) continue;

            const event = eventLine.slice(7).trim();
            const rawData = dataLine.slice(6);

            if (event === "token") {
              const data = JSON.parse(rawData);
              if (!hasAnswerToken) {
                assistant.bubble.textContent = "";
                hasAnswerToken = true;
              }
              answerText += data.text || "";
              assistant.bubble.textContent = answerText;
              scrollToBottom();
            } else if (event === "status") {
              const data = JSON.parse(rawData);
              setStatus(data.message || "处理中", true);
            } else if (event === "meta") {
              const meta = JSON.parse(rawData);
              addMeta(assistant.content, meta);
            } else if (event === "error") {
              const data = JSON.parse(rawData);
              answerText = data.message || "服务处理失败";
              assistant.bubble.textContent = answerText;
            } else if (event === "done") {
              assistant.bubble.innerHTML = renderCustomerHtml(answerText);
              const active = ensureSession(currentSessionId);
              active.messages.push({role: "assistant", text: answerText, time: Date.now()});
              active.updatedAt = Date.now();
              sessions = [active, ...sessions.filter(item => item.id !== active.id)];
              saveSessions();
              renderHistory();
              setStatus("就绪", false);
            }
          }
        }
      } catch (err) {
        answerText = "请求失败：" + err.message;
        assistant.bubble.textContent = answerText;
      } finally {
        if (answerText && !getCurrentSession()?.messages.some(item => item.role === "assistant" && item.text === answerText)) {
          const active = ensureSession(currentSessionId);
          active.messages.push({role: "assistant", text: answerText, time: Date.now()});
          active.updatedAt = Date.now();
          saveSessions();
          renderHistory();
        }
        sendBtn.disabled = false;
        setStatus("就绪", false);
        queryInput.focus();
      }
    }

    sendBtn.addEventListener("click", sendMessage);
    newChatBtn.addEventListener("click", newSession);
    conversationInput.addEventListener("change", () => switchSession(conversationInput.value.trim() || nowId()));
    userIdSelect.addEventListener("change", () => {
      const session = ensureSession(currentSessionId);
      session.userId = userIdSelect.value || "";
      session.updatedAt = Date.now();
      saveSessions();
      renderHistory();
    });

    queryInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });

    document.querySelectorAll(".quick-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        queryInput.value = btn.dataset.query;
        queryInput.focus();
      });
    });

    (async function init() {
      await loadUsers();
      if (sessions.length) {
        switchSession(sessions[0].id);
      } else {
        newSession();
      }
    })();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.web.graphrag_app:app", host="127.0.0.1", port=8090, reload=False)

