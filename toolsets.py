"""Toolset helpers: get/resolve/validate named tool groups (static TOOLSETS + registry-registered)."""
# ==========================================================================
#  📘 文件导读（新增注释，原代码未做任何改动）
# ==========================================================================
# 文件：toolsets.py（约 485 行）
# 地位：工具集（toolset）的定义与解析。
#
# 概念：一个 toolset 就是一组工具的命名集合。比如 core（基础工具）、browser、terminal ...
#   用户可以用 --toolset 指定只启用哪些组，避免一次性把上百个工具的 schema 全塞给模型（费 token 还容易乱调用）。
#
# 主要函数：
#   get_toolset(name)              取一个工具集的定义
#   resolve_toolset(name, visited) 展开工具集（支持 includes 嵌套引用别的集合，visited 用来防循环引用）
#   resolve_multiple_toolsets()    一次解析多个
#   validate_toolset(name)         校验名字是否合法
#   get_all_toolsets() / get_toolset_names()  列出全部
#   bundle_non_core_tools()        把非 core 的工具打包（插件平台常用）
#
# 值得注意：文件里 TOOLSETS 是一张静态表，同时还能从 registry 动态注册新工具集，
#   所以你会看到 _registry() / _registry_call() 这种「先查动态注册表，查不到再用静态表」的写法。
# ==========================================================================
#

from typing import Dict, List, Any, Set, Optional, Tuple


# Shared tool list for CLI and all messaging platform toolsets (edit once, all
# platforms follow). Desktop GUI affordances are deliberately NOT here: they live
# in `desktop_ui`/`project`, enabled per desktop-sourced session by the GUI gateway
# (tui_gateway/server.py::_load_enabled_toolsets). HA, kanban and computer_use
# entries are further gated by their tools' check_fns.
_HERMES_CORE_TOOLS = [
    "web_search", "web_extract",
    "terminal", "process_manage",
    "read_file", "write_file", "patch", "search_files",
    "vision_analyze", "image_generate",
    "skills_list", "skill_view", "skill_manage",
    "browser_navigate", "browser_snapshot", "browser_click",
    "browser_type", "browser_scroll", "browser_back",
    "browser_press", "browser_get_images",
    "browser_vision", "browser_console", "browser_cdp", "browser_dialog",
    "browser_vault_list", "browser_vault_unlock", "browser_vault_fill", "browser_vault_save_login", "browser_vault_enter_code",  # ride with the browser
    "browser_exec",  # replaces the other browser tools when browser.backend is "browser-use"
    "text_to_speech",
    "todo_list", "memory",
    "session_search",
    "clarify",
    "execute_code", "delegate_task",
    "cronjob_manage",
    "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service",
    "kanban_show", "kanban_list",
    "kanban_complete", "kanban_block", "kanban_request_review",
    "kanban_request_changes",
    "kanban_heartbeat",
    "kanban_comment", "kanban_create", "kanban_link",
    "kanban_unblock",
    "kanban_attach", "kanban_attach_url", "kanban_attachments",
    "computer_use",
    # Service-gated connector account status and authorization links.
    "manage_connections",
]

# Webhook payloads are untrusted third-party content: no file/system execution.
_HERMES_WEBHOOK_SAFE_TOOLS = ["web_search", "web_extract", "vision_analyze", "clarify"]
_HA_TOOLS = ["ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service"]
_FEISHU_TOOLS = [
    "feishu_doc_read", "feishu_drive_list_comments", "feishu_drive_list_comment_replies",
    "feishu_drive_reply_comment", "feishu_drive_add_comment",
]
_YUANBAO_TOOLS = ["yb_query_group_info", "yb_query_group_members", "yb_send_dm", "yb_search_sticker", "yb_send_sticker"]


# ┌── 函数 def _ts()
# │  作用：One TOOLSETS entry (fresh lists per entry; extra keys such as posture pass through).
# │  参数：description, tools = (), includes = (), **extra
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _ts(description, tools=(), includes=(), **extra):
    """One TOOLSETS entry (fresh lists per entry; extra keys such as posture pass through)."""
    # ⤷ 出口 return：返回 {'description': description, 'tools': list(tools), 'includes…
    return {"description": description, "tools": list(tools), "includes": list(includes), **extra}


# ┌── 函数 def _bundle()
# │  作用：A `hermes-*` platform bundle: the shared core tools plus optional platform extras.
# │  参数：description, extras = ()
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _bundle(description, extras=()):
    """A `hermes-*` platform bundle: the shared core tools plus optional platform extras."""
    # ⤷ 出口 return：返回 _ts(description, _HERMES_CORE_TOOLS + list(extras))
    return _ts(description, _HERMES_CORE_TOOLS + list(extras))


# ┌── 函数 def _core_without()
# │  作用：_HERMES_CORE_TOOLS minus *excluded* (and, unless kanban=True, every kanban_* tool); order preserved.
# │  参数：*excluded, kanban = ...
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _core_without(*excluded, kanban=True):
    """_HERMES_CORE_TOOLS minus *excluded* (and, unless kanban=True, every kanban_* tool); order preserved."""
    # ⤷ 出口 return：返回 [t for t in _HERMES_CORE_TOOLS if t not in excluded and (kan…
    return [t for t in _HERMES_CORE_TOOLS if t not in excluded and (kanban or not t.startswith("kanban_"))]


# Coding posture: everything you reach for while pairing on code; drops messaging,
# tts, image_gen, home-assistant, cron, kanban and computer-use.
_CODING_TOOLS = _core_without("image_generate", "text_to_speech", "cronjob_manage", "computer_use", *_HA_TOOLS, kanban=False)

# Core toolset definitions: individual tools or references to other toolsets.
TOOLSETS = {
    # Basic toolsets - individual tool categories
    "web": _ts("Web research and content extraction tools", ["web_search", "web_extract"]),
    "search": _ts("Web search only (no content extraction/scraping)", ["web_search"]),
    "x_search": _ts(
        "Search X (Twitter) posts and threads via xAI's built-in x_search Responses "
        "tool. Read-only public X discovery; use the xurl skill for authenticated X "
        "API reads and account actions. Available when xAI credentials are configured "
        "(SuperGrok OAuth or XAI_API_KEY). Off by default; enable in `hermes tools` → "
        "X (Twitter) Search.",
        ["x_search"],
    ),
    "vision": _ts("Image analysis and vision tools", ["vision_analyze"]),
    "video": _ts("Video analysis and understanding tools (opt-in, not in default toolset)", ["video_analyze"]),
    "image_gen": _ts("Creative generation tools (images)", ["image_generate"]),
    "video_gen": _ts(
        "Video generation tools. Single ``video_generate`` tool covers text-to-video "
        "(prompt only) and image-to-video (prompt + image_url), plus "
        "reference-to-video. Provider-specific edit/extend workflows may appear as "
        "separate tools. Configure via ``hermes tools`` → Video Generation.",
        ["video_generate", "xai_video_edit", "xai_video_extend"],
    ),
    "computer_use": _ts(
        "Background desktop control via cua-driver (macOS/Windows/Linux) — "
        "screenshots, mouse, keyboard, scroll, drag. Does NOT steal the user's cursor "
        "or keyboard focus. Works with any tool-capable model.",
        ["computer_use"],
    ),
    "terminal": _ts("Terminal/command execution and process management tools", ["terminal", "process_manage"]),
    "skills": _ts(
        "Access, create, edit, and manage skill documents with specialized "
        "instructions and knowledge",
        ["skills_list", "skill_view", "skill_manage"],
    ),
    # web_search belongs to `web`/`search` only. Listing it here too let
    # `disabled_toolsets: [browser]` (headless/Docker deployments) strip
    # web_search from every session, because disabled toolsets are a strict
    # end-of-pipeline subtraction (#17309, #64503).
    "browser": _ts(
        "Browser automation for web interaction (navigate, click, type, scroll, "
        "iframes, hold-click)",
        [t for t in _HERMES_CORE_TOOLS if t.startswith("browser_")],
    ),
    "cronjob": _ts(
        "Cronjob management tool - create, list, update, pause, resume, remove, and "
        "trigger scheduled tasks",
        ["cronjob_manage"],
    ),
    "file": _ts(
        "File manipulation tools: read, write, patch (with fuzzy matching), and "
        "search (content + files)",
        ["read_file", "write_file", "patch", "search_files"],
    ),
    "tts": _ts("Text-to-speech: convert text to audio with Edge TTS (free), ElevenLabs, OpenAI, or xAI", ["text_to_speech"]),
    "todo": _ts("Task planning and tracking for multi-step work", ["todo_list"]),
    "memory": _ts("Persistent memory across sessions (personal notes + user profile)", ["memory"]),
    "context_engine": _ts("Runtime tools exposed by the active context engine"),
    "session_search": _ts("Search and recall past conversations with summarization", ["session_search"]),
    "connections": _ts("Remote connector discovery, execution, and account authorization", ["manage_connections"]),
    "project": _ts("Desktop Projects — create/switch named workspaces (GUI sessions only)", ["desktop_project"]),
    "bot_room": _ts("Verified text-only Group Chat turn capabilities"),

    # GUI-renderer affordances, enabled per desktop-sourced SESSION by the GUI
    # gateway (tui_gateway/server.py::_load_enabled_toolsets) — never by a
    # process env var, which is blind to a desktop client on a remote backend.
    "desktop_ui": _ts(
        "Desktop GUI affordances — in-app terminal/browser panes, pane focus, "
        "reactions (GUI sessions only)",
        ["read_terminal", "close_terminal", "desktop_preview", "drive_preview",
         "annotate_preview", "read_window_below", "focus_pane", "react_to_message",
         "gui_tour", "show_tip"],
    ),
    "clarify": _ts("Ask the user clarifying questions (multiple-choice or open-ended)", ["clarify"]),
    "code_execution": _ts("Run Python scripts that call tools programmatically (reduces LLM round trips)", ["execute_code"]),
    "delegation": _ts("Spawn subagents with isolated context for complex subtasks", ["delegate_task"]),
    "homeassistant": _ts("Home Assistant smart home control and monitoring", _HA_TOOLS),
    "kanban": _ts(
        "Kanban multi-agent coordination — only active when the agent is spawned by "
        "the kanban dispatcher (HERMES_KANBAN_TASK env set). The dispatcher runs "
        "inside the gateway by default; see `kanban.dispatch_in_gateway` in "
        "config.yaml. Lets workers mark tasks done with structured handoffs, enter "
        "first-class review (request_review — not a block), return review changes, "
        "block for human input, heartbeat during long ops, comment on threads, attach "
        "files, and (for orchestrators) list, unblock, and fan out tasks.",
        [t for t in _HERMES_CORE_TOOLS if t.startswith("kanban_")],
    ),
    "discord": _ts("Discord read and participate tools (fetch messages, search members, create threads)", ["discord"]),
    "discord_admin": _ts("Discord server management (list channels/roles, pin messages, assign roles)", ["discord_admin"]),
    "yuanbao": _ts("Yuanbao platform tools - group info, member queries, DM, stickers", _YUANBAO_TOOLS),
    "feishu_doc": _ts("Read Feishu/Lark document content", ["feishu_doc_read"]),
    "feishu_drive": _ts("Feishu/Lark document comment operations (list, reply, add)", _FEISHU_TOOLS[1:]),
    "spotify": _ts(
        "Native Spotify playback, search, playlist, album, and library tools",
        ["spotify_playback", "spotify_devices", "spotify_queue", "spotify_search",
         "spotify_playlists", "spotify_albums", "spotify_library"],
    ),

    # Scenario-specific toolsets
    "debugging": _ts("Debugging and troubleshooting toolkit", ["terminal", "process_manage"], includes=["web", "file"]),
    "safe": _ts("Safe toolkit without terminal access", [], includes=["web", "vision", "image_gen"]),

    # Coding posture, auto-selected in a code workspace (agent/coding_context.py).
    # `desktop_ui` is folded in separately by the GUI gateway for desktop sessions.
    # posture=True: per-session posture, never auto-recovered into platform tool
    # config (see the non-configurable-toolset recovery loop in hermes_cli/tools_config.py).
    "coding": _ts(
        "Coding-focused toolset: files, terminal, search, web docs, skills, todo, "
        "delegate, vision, browser",
        _CODING_TOOLS,
        posture=True,
    ),

    # Full Hermes toolsets (CLI + messaging platforms). All share the core tools;
    # there is deliberately no agent-callable send_message tool. hermes-acp is the
    # coding posture minus the interactive clarify UI.
    "hermes-acp": _ts(
        "Editor integration (VS Code, Zed, JetBrains) — coding-focused tools without "
        "messaging, audio, or clarify UI",
        [t for t in _CODING_TOOLS if t != "clarify"],
    ),
    "hermes-api-server": _ts(
        "OpenAI-compatible API server — full agent tools accessible via HTTP (no "
        "interactive UI tools like clarify or send_message)",
        _core_without("text_to_speech", "clarify", "computer_use", kanban=False),
    ),
    "hermes-cli": _bundle("Full interactive CLI toolset - all default tools plus cronjob management"),

    # Mirrors hermes-cli; `hermes tools` platform config filters it down and
    # _get_platform_tools() drops _DEFAULT_OFF_TOOLSETS unless user-enabled.
    "hermes-cron": _bundle("Default cron toolset - same core tools as hermes-cli; gated by `hermes tools`"),
    "hermes-telegram": _bundle("Telegram bot toolset - full access for personal use (terminal has safety checks)"),
    "hermes-discord": _bundle(
        "Discord bot toolset - full access (terminal has safety checks via dangerous "
        "command approval)",
        ["discord", "discord_admin"],
    ),
    "hermes-whatsapp": _bundle("WhatsApp bot toolset - similar to Telegram (personal messaging, more trusted)"),
    "hermes-slack": _bundle("Slack bot toolset - full access for workspace use (terminal has safety checks)"),
    "hermes-signal": _bundle("Signal bot toolset - encrypted messaging platform (full access)"),
    "hermes-bluebubbles": _bundle("BlueBubbles iMessage bot toolset - Apple iMessage via local BlueBubbles server"),
    "hermes-homeassistant": _bundle("Home Assistant bot toolset - smart home event monitoring and control"),
    "hermes-email": _bundle("Email bot toolset - interact with Hermes via email (IMAP/SMTP)"),
    "hermes-mattermost": _bundle("Mattermost bot toolset - self-hosted team messaging (full access)"),
    "hermes-matrix": _bundle("Matrix bot toolset - decentralized encrypted messaging (full access)"),
    "hermes-dingtalk": _bundle("DingTalk bot toolset - enterprise messaging platform (full access)"),
    "hermes-feishu": _bundle("Feishu/Lark bot toolset - enterprise messaging via Feishu/Lark (full access)", _FEISHU_TOOLS),
    "hermes-weixin": _bundle("Weixin bot toolset - personal WeChat messaging via iLink (full access)"),
    "hermes-qqbot": _bundle("QQBot toolset - QQ messaging via Official Bot API v2 (full access)"),
    "hermes-wecom": _bundle("WeCom bot toolset - enterprise WeChat messaging (full access)"),
    "hermes-wecom-callback": _bundle("WeCom callback toolset - enterprise self-built app messaging (full access)"),
    "hermes-yuanbao": {
        "description": "Yuanbao Bot 元宝消息平台工具集 - 群信息、成员查询、私聊、贴纸表情",
        "tools": _HERMES_CORE_TOOLS + _YUANBAO_TOOLS,
        "module": "tools.yuanbao_tools",
        "includes": [],
    },
    "hermes-sms": _bundle("SMS bot toolset - interact with Hermes via SMS (Twilio)"),
    "hermes-webhook": _ts("Webhook toolset - receive and process external webhook events", _HERMES_WEBHOOK_SAFE_TOOLS),
    "hermes-gateway": _ts(
        "Gateway toolset - union of all messaging platform tools",
        [],
        includes=[
            "hermes-telegram", "hermes-discord", "hermes-whatsapp", "hermes-slack",
            "hermes-signal", "hermes-bluebubbles", "hermes-homeassistant", "hermes-email",
            "hermes-sms", "hermes-mattermost", "hermes-matrix", "hermes-dingtalk",
            "hermes-feishu", "hermes-wecom", "hermes-wecom-callback", "hermes-weixin",
            "hermes-qqbot", "hermes-webhook", "hermes-yuanbao",
        ],
    ),
}


# ┌── 函数 def _registry()
# │  作用：Live tool registry, or None when tools.registry can't be imported.
# │  参数：无
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _registry():
    """Live tool registry, or None when tools.registry can't be imported."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from tools.registry import registry
        # ⤷ 出口 return：返回 registry
        return registry
    except Exception:
        # ⤷ 出口 return：返回 None
        return None


# ┌── 函数 def _registry_call()
# │  作用：registry.<method>() or *default* when the registry is unavailable or the call fails.
# │  参数：method: str, default
# │  返回：（未标注类型）
# └──────────────────────────────────────────────────────────────
def _registry_call(method: str, default):
    """registry.<method>() or *default* when the registry is unavailable or the call fails."""
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        # ⤷ 出口 return：返回 getattr(_registry(), method)()
        return getattr(_registry(), method)()
    except Exception:  # registry None (AttributeError) or the call failed
        # ⤷ 出口 return：返回 default
        return default


# ┌── 函数 def _registry_generation()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：无
# │  返回：Tuple[int, int]
# └──────────────────────────────────────────────────────────────
def _registry_generation() -> Tuple[int, int]:
    reg = _registry()
    # ⤷ 出口 return：返回 (id(reg), getattr(reg, '_generation', 0)) if reg is not None…
    return (id(reg), getattr(reg, "_generation", 0)) if reg is not None else (0, 0)


# ┌── 函数 def get_toolset()
# │  作用：Toolset definition, or None if unknown.
# │  参数：name: str, include_registry: bool = ...
# │  返回：Optional[Dict[str, Any]]
# └──────────────────────────────────────────────────────────────
def get_toolset(name: str, *, include_registry: bool = True) -> Optional[Dict[str, Any]]:
    """Toolset definition, or None if unknown.

    include_registry=True merges plugin/overlay tools registered into this toolset
    and resolves registry-only (plugin/MCP) toolsets and aliases; False returns a
    copy of the static TOOLSETS entry only, so platform reverse-mapping is
    unaffected by registry additions.

    Args: name (str): Name of the toolset include_registry (bool): When True (default), merge in tools that
    plugins/overlays registered into this toolset via the registry. Platform reverse-mapping in
    ``_get_platform_tools`` uses False so that a tool registered into a toolset but absent from a platform's
    static composite does not drop the whole toolset from inference. See issue #49622.
    """
    toolset = TOOLSETS.get(name)
    if not include_registry:
        # ⤷ 出口 return：返回 {**toolset, 'tools': list(toolset.get('tools', [])), 'includ…
        return {**toolset, "tools": list(toolset.get("tools", [])), "includes": list(toolset.get("includes", []))} if toolset else None

    registry = _registry()
    if registry is None:
        # ⤷ 出口 return：返回 toolset if toolset else None
        return toolset if toolset else None

    if toolset:
        merged_tools = set(toolset.get("tools", [])) | set(registry.get_tool_names_for_toolset(name))
        # An MCP server named like a built-in toolset ("homeassistant", "browser") registers a bare
        # alias to its `mcp-<name>` toolset; without this union the static entry shadows it and the
        # server's tools never reach the model even though discovery registered them.
        alias_target = registry.get_toolset_alias_target(name)
        if alias_target and alias_target != name:
            merged_tools |= set(registry.get_tool_names_for_toolset(alias_target))
        # ⤷ 出口 return：返回 {**toolset, 'tools': sorted(merged_tools)}
        return {**toolset, "tools": sorted(merged_tools)}

    if name in _get_plugin_toolset_names():
        # Plugin toolset; shown as its MCP server alias when one exists.
        registry_toolset = name
        alias = _display_alias(name, _get_registry_toolset_aliases())
        description = f"MCP server '{alias}' tools" if alias else f"Plugin toolset: {name}"
    else:
        registry_toolset = registry.get_toolset_alias_target(name)
        if not registry_toolset:
            # ⤷ 出口 return：返回 None
            return None
        description = f"MCP server '{name}' tools"
    # ⤷ 出口 return：返回 {'description': description, 'tools': registry.get_tool_name…
    return {"description": description, "tools": registry.get_tool_names_for_toolset(registry_toolset), "includes": []}


# ┌── 函数 def bundle_non_core_tools()
# │  作用：A bundle's tools minus _HERMES_CORE_TOOLS (one level of includes).
# │  参数：toolset_name: str
# │  返回：Set[str]
# └──────────────────────────────────────────────────────────────
def bundle_non_core_tools(toolset_name: str) -> Set[str]:
    """A bundle's tools minus _HERMES_CORE_TOOLS (one level of includes).

    Disabling a `core + extras` bundle must not strip the core tools every other
    toolset shares. One `includes` pass suffices (only hermes-gateway nests
    bundles). Unknown names: full resolution minus core.
    """
    core = set(_HERMES_CORE_TOOLS)
    ts_def = get_toolset(toolset_name)
    if not (ts_def and "tools" in ts_def):
        # ⤷ 出口 return：返回 set(resolve_toolset(toolset_name)) - core
        return set(resolve_toolset(toolset_name)) - core
    to_remove = set(ts_def["tools"])
    for inc_def in map(get_toolset, ts_def.get("includes", [])):
        if inc_def and "tools" in inc_def:
            to_remove.update(inc_def["tools"])
    # ⤷ 出口 return：返回 to_remove - core
    return to_remove - core


# Memo keyed on (name, include_registry, id(registry), registry generation, profile scope);
# engages only at the public entry (visited is None). The scope is part of the key because a
# multiplexed process resolves ``mcp-<server>`` per profile overlay: without it profile B got
# profile A's tool names for a server B never connected (#106005).
_resolve_toolset_memo: Dict[Tuple[str, bool, int, int, str], List[str]] = {}


# ┌── 函数 def _plugin_platform_bundle()
# │  作用：Implicit `hermes-<platform>` bundle for a registered plugin platform: core
# │  参数：name: str
# │  返回：List[str]
# └──────────────────────────────────────────────────────────────
def _plugin_platform_bundle(name: str) -> List[str]:
    """Implicit `hermes-<platform>` bundle for a registered plugin platform: core
    tools plus whatever the plugin registered under the platform name. [] otherwise."""
    if not name.startswith("hermes-"):
        # ⤷ 出口 return：返回 []
        return []
    platform_name = name[len("hermes-"):]
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        from gateway.platform_registry import platform_registry
        if not platform_registry.is_registered(platform_name):
            # ⤷ 出口 return：返回 []
            return []
    except Exception:
        # ⤷ 出口 return：返回 []
        return []
    tools = set(_HERMES_CORE_TOOLS)
    # ⤷ 关键分支 try/except：捕获 Exception，出错走兜底而不是崩溃。
    try:
        tools.update(e.name for e in _registry_call("get_all_entries", ()) if e.toolset == platform_name)
    except Exception:
        pass
    # ⤷ 出口 return：返回 list(tools)
    return list(tools)


# ┌── 函数 def resolve_toolset()
# │  作用：Recursively resolve a toolset (and its includes) to a sorted tool-name list.
# │  参数：name: str, visited: Set[str] = None, include_registry: bool = ...
# │  返回：List[str]
# └──────────────────────────────────────────────────────────────
def resolve_toolset(name: str, visited: Set[str] = None, *, include_registry: bool = True) -> List[str]:
    """Recursively resolve a toolset (and its includes) to a sorted tool-name list.
    include_registry=False resolves the static TOOLSETS view only.

    Args: name (str): Name of the toolset to resolve visited (Set[str]): Set of already visited toolsets
    (for cycle detection) include_registry (bool): When True (default), include tools that plugins/overlays
    registered into a toolset. Platform reverse-mapping uses False so a registry-added tool cannot drop the
    whole toolset from inference (see #49622 and ``_get_platform_tools``).
    """
    external_call = visited is None
    if external_call:
        memo_key = (name, include_registry, *_registry_generation(), _registry_call("current_scope_key", ""))
        cached = _resolve_toolset_memo.get(memo_key)
        if cached is not None:
            # ⤷ 出口 return：返回 list(cached)
            return list(cached)
        visited = set()

    # "all"/"*" span every toolset so new toolsets are included automatically.
    if name in {"all", "*"}:
        all_tools: Set[str] = set()
        for toolset_name in get_toolset_names():
            all_tools.update(resolve_toolset(toolset_name, visited.copy(), include_registry=include_registry))
        # ⤷ 出口 return：返回 sorted(all_tools)
        return sorted(all_tools)

    # Diamond include or cycle: [] silently — the tools are collected via another path.
    if name in visited:
        # ⤷ 出口 return：返回 []
        return []
    visited.add(name)

    toolset = get_toolset(name, include_registry=include_registry)
    if not toolset:
        # ⤷ 出口 return：返回 _plugin_platform_bundle(name) if include_registry else []
        return _plugin_platform_bundle(name) if include_registry else []

    tools = set(toolset.get("tools", []))
    for included_name in toolset.get("includes", []):
        tools.update(resolve_toolset(included_name, visited, include_registry=include_registry))

    result = sorted(tools)
    if external_call:
        if len(_resolve_toolset_memo) >= 256:  # stale-generation entries are never hit again
            _resolve_toolset_memo.clear()
        _resolve_toolset_memo[memo_key] = list(result)
    # ⤷ 出口 return：返回 result
    return result


# ┌── 函数 def _get_plugin_toolset_names()
# │  作用：Registry toolset names absent from the static TOOLSETS dict.
# │  参数：无
# │  返回：Set[str]
# └──────────────────────────────────────────────────────────────
def _get_plugin_toolset_names() -> Set[str]:
    """Registry toolset names absent from the static TOOLSETS dict."""
    # ⤷ 出口 return：返回 {n for n in _registry_call('get_registered_toolset_names', (…
    return {n for n in _registry_call("get_registered_toolset_names", ()) if n not in TOOLSETS}


# ┌── 函数 def _get_registry_toolset_aliases()
# │  作用：内部辅助函数（下划线开头 = 模块私有，外部不应直接调用）
# │  参数：无
# │  返回：Dict[str, str]
# └──────────────────────────────────────────────────────────────
def _get_registry_toolset_aliases() -> Dict[str, str]:
    # ⤷ 出口 return：返回 _registry_call('get_registered_toolset_aliases', {})
    return _registry_call("get_registered_toolset_aliases", {})


# ┌── 函数 def _display_alias()
# │  作用：First non-static alias pointing at *ts_name*, or None.
# │  参数：ts_name: str, aliases: Dict[str, str]
# │  返回：Optional[str]
# └──────────────────────────────────────────────────────────────
def _display_alias(ts_name: str, aliases: Dict[str, str]) -> Optional[str]:
    """First non-static alias pointing at *ts_name*, or None."""
    # ⤷ 出口 return：返回 next((a for a, canonical in aliases.items() if canonical == …
    return next((a for a, canonical in aliases.items() if canonical == ts_name and a not in TOOLSETS), None)


# ┌── 函数 def _plugin_display_names()
# │  作用：Plugin toolset names, shown under their first non-static alias when one exists.
# │  参数：无
# │  返回：List[str]
# └──────────────────────────────────────────────────────────────
def _plugin_display_names() -> List[str]:
    """Plugin toolset names, shown under their first non-static alias when one exists."""
    aliases = _get_registry_toolset_aliases()
    # ⤷ 出口 return：返回 [_display_alias(n, aliases) or n for n in _get_plugin_toolse…
    return [_display_alias(n, aliases) or n for n in _get_plugin_toolset_names()]


# ┌── 函数 def get_all_toolsets()
# │  作用：All toolset definitions: static plus plugin-registered.
# │  参数：无
# │  返回：Dict[str, Dict[str, Any]]
# └──────────────────────────────────────────────────────────────
def get_all_toolsets() -> Dict[str, Dict[str, Any]]:
    """All toolset definitions: static plus plugin-registered."""
    result = dict(TOOLSETS)
    aliases = _get_registry_toolset_aliases()
    for display_name in _plugin_display_names():
        toolset = None if display_name in result else get_toolset(display_name)
        if toolset:
            result[display_name] = toolset
    # Static names an MCP server also aliases show the merged view get_toolset() resolves.
    for name in TOOLSETS.keys() & aliases.keys():
        result[name] = get_toolset(name) or result[name]
    # ⤷ 出口 return：返回 result
    return result


# ┌── 函数 def get_toolset_names()
# │  作用：Sorted names of all toolsets (static + plugin), excluding aliases.
# │  参数：无
# │  返回：List[str]
# └──────────────────────────────────────────────────────────────
def get_toolset_names() -> List[str]:
    """Sorted names of all toolsets (static + plugin), excluding aliases."""
    # ⤷ 出口 return：返回 sorted(set(TOOLSETS.keys()) | set(_plugin_display_names()))
    return sorted(set(TOOLSETS.keys()) | set(_plugin_display_names()))


# ┌── 函数 def validate_toolset()
# │  作用：（源码未写 docstring，可从下方参数与返回值推断）
# │  参数：name: str
# │  返回：bool
# └──────────────────────────────────────────────────────────────
def validate_toolset(name: str) -> bool:
    # ⤷ 出口 return：返回 name in {'all', '*'} or name in TOOLSETS or name in _get_plu…
    return (name in {"all", "*"} or name in TOOLSETS
            or name in _get_plugin_toolset_names() or name in _get_registry_toolset_aliases())


# ┌── 函数 def create_custom_toolset()
# │  作用：Register a runtime toolset in TOOLSETS.
# │  参数：name: str, description: str, tools: List[str] = None, includes: List[str] = None
# │  返回：None
# └──────────────────────────────────────────────────────────────
def create_custom_toolset(name: str, description: str, tools: List[str] = None, includes: List[str] = None) -> None:
    """Register a runtime toolset in TOOLSETS."""
    TOOLSETS[name] = _ts(description, tools or [], includes or [])


# ┌── 函数 def get_toolset_info()
# │  作用：Toolset definition plus its resolved tools, or None if unknown.
# │  参数：name: str
# │  返回：Dict[str, Any]
# └──────────────────────────────────────────────────────────────
def get_toolset_info(name: str) -> Dict[str, Any]:
    """Toolset definition plus its resolved tools, or None if unknown."""
    toolset = get_toolset(name)
    if not toolset:
        # ⤷ 出口 return：返回 None
        return None
    resolved_tools = resolve_toolset(name)
    # ⤷ 出口 return：返回 {'name': name, 'description': toolset['description'], 'direc…
    return {
        "name": name, "description": toolset["description"],
        "direct_tools": toolset["tools"], "includes": toolset["includes"],
        "resolved_tools": resolved_tools, "tool_count": len(resolved_tools),
        "is_composite": bool(toolset["includes"]),
    }


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

# ┌── 函数 def resolve_multiple_toolsets()
# │  作用：Resolve multiple toolsets and combine their tools.
# │  参数：toolset_names: List[str]
# │  返回：List[str]
# └──────────────────────────────────────────────────────────────
def resolve_multiple_toolsets(toolset_names: List[str]) -> List[str]:
    """
    Resolve multiple toolsets and combine their tools.

    Args:
        toolset_names (List[str]): List of toolset names to resolve

    Returns:
        List[str]: Combined list of all tool names (deduplicated)
    """
    all_tools = set()

    for name in toolset_names:
        tools = resolve_toolset(name)
        all_tools.update(tools)

    # ⤷ 出口 return：返回 sorted(all_tools)
    return sorted(all_tools)
# ---- END PLUGIN-COMPAT ----
