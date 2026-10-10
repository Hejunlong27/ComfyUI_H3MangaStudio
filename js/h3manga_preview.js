/**
 * H3 Manga Studio —— 轻量只读预览面板
 *
 * 设计约束（对齐 ComfyUI 前端规范）
 * ---------------------------------
 * - 只用 `scripts/app`（Stable API）。不用 widgets / domWidget / utils（Internal，升级会断）。
 * - `beforeRegisterNodeDef` 里**必须链式调用**原方法，否则会顶掉其它插件的行为。
 * - DOM widget 必须 `serialize: false` —— `widgets_values` 是按位置存的，
 *   插入一个参与序列化的控件会让**旧工作流的参数整体错位**。
 * - 没有 `addDOMWidget` 的老前端降级成 button widget，功能不丢。
 *
 * 面板内容来自后端 `io.NodeOutput(..., ui={"h3manga_panel": [文本]})`，
 * 通过 `onExecuted(message)` 拿到。显示的是「机器实际拿到的提示词 +
 * 本镜 <Picture i> 序号映射」——排查角色串号最快的手段。
 */

import { app } from "../../scripts/app.js";

const PANEL_KEY = "h3manga_panel";
const PANEL_H = 200;

// 所有会输出 h3manga_panel 的节点
const TARGETS = new Set([
  "H3MangaPanelSplitter",
  "H3MangaRefPack",
  "H3MangaScriptLoader",
  "H3MangaShotBinder",
  "H3MangaPromptComposer",
  "H3MangaShotConditioning",
  "H3MangaEpisodeRender",
  "H3MangaConsistencyAudit",
  "H3MangaPlanPreview",
]);

const PANEL_CSS =
  "font:11px/1.45 ui-monospace,SFMono-Regular,Consolas,'Liberation Mono',monospace;" +
  "white-space:pre-wrap;overflow:auto;padding:6px 8px;box-sizing:border-box;" +
  "color:#d8d8d8;background:rgba(0,0,0,.22);border:1px solid rgba(128,128,128,.35);" +
  "border-radius:4px;resize:vertical;";

function ensurePanel(node) {
  if (node.__h3mangaPanelEl) return node.__h3mangaPanelEl;

  const el = document.createElement("div");
  el.className = "h3manga-panel";
  el.style.cssText = PANEL_CSS;
  el.textContent = "（尚未执行）";

  let widget = null;

  if (typeof node.addDOMWidget === "function") {
    widget = node.addDOMWidget("H3 预览", "H3MANGA_PANEL", el, {
      serialize: false, // ← 硬要求：不能让面板参与 widgets_values 序列化
      hideOnZoom: false,
      getValue: () => el.textContent || "",
      setValue: (v) => {
        el.textContent = v ?? "";
      },
    });
    if (widget) {
      // 画布绘制的多行框靠 computeSize 决定高度；DOM 框优先读实际高度
      widget.computeSize = function (width) {
        if (el.offsetHeight) return [width, el.offsetHeight + 12];
        return [width, PANEL_H];
      };
      el.style.height = PANEL_H - 12 + "px";
      el.style.minHeight = PANEL_H - 12 + "px";
    }
  } else if (typeof node.addWidget === "function") {
    // 老前端降级：一个按钮，点了弹窗看全文
    widget = node.addWidget("button", "H3 预览（点击查看）", null, () => {
      // eslint-disable-next-line no-alert
      alert(el.textContent || "(空)");
    });
  }

  node.__h3mangaPanelEl = el;
  node.__h3mangaPanelWidget = widget;

  // DOM 元素可能晚于一帧挂载，补一次
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(() => {
      try {
        if (node.__h3mangaPanelEl && node.__h3mangaPanelEl.offsetHeight) {
          node.setSize?.([
            node.size?.[0] ?? 340,
            Math.max(node.size?.[1] ?? 0, node.computeSize?.()[1] ?? PANEL_H),
          ]);
        }
      } catch (e) {
        /* 忽略 */
      }
    });
  }
  return el;
}

function setPanelText(node, text) {
  const el = node.__h3mangaPanelEl;
  if (!el) return;
  el.textContent = text;
  try {
    const need = node.computeSize ? node.computeSize()[1] : 0;
    // 只在需要更大时才改尺寸，否则会把用户手动调过的节点尺寸改小
    if (need > (node.size?.[1] ?? 0)) {
      node.setSize?.([node.size?.[0] ?? 340, need]);
    }
    app.graph?.setDirtyCanvas?.(true, true);
  } catch (e) {
    /* 忽略 */
  }
}

app.registerExtension({
  name: "H3MangaStudio.PreviewPanel",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData || !TARGETS.has(nodeData.name)) return;

    const origOnCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnCreated?.apply(this, arguments); // ← 必须链式调用
      try {
        ensurePanel(this);
      } catch (e) {
        console.warn("[H3MangaStudio] 预览面板初始化失败：", e);
      }
      return r;
    };

    const origOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = origOnExecuted?.apply(this, arguments);
      try {
        const payload = message?.[PANEL_KEY];
        const text = Array.isArray(payload) ? payload[0] : payload;
        if (typeof text === "string") {
          if (!this.__h3mangaPanelEl) ensurePanel(this);
          setPanelText(this, text);
        }
      } catch (e) {
        console.warn("[H3MangaStudio] 预览面板刷新失败：", e);
      }
      return r;
    };
  },
});
