// 本文件由 tools/gen_i18n.py 自动生成，请勿手改。
// 真源：core/ui.py + 各节点的 define_schema()

export const NODE_LABELS = {
  "H3MangaPanelSplitter": {
    "display_name": "H3 漫画页切分镜格",
    "inputs": {
      "image": "输入图像",
      "mode": "模式",
      "manual_boxes": "手动框",
      "ink_threshold": "墨迹阈值",
      "gutter_ratio": "装订线空白阈值",
      "min_panel_px": "最小格边长(px)",
      "min_gutter_px": "最小装订线宽(px)",
      "min_panel_ratio": "最小格面积占比",
      "reading_order": "阅读顺序",
      "padding_px": "裁切外扩(px)",
      "panel_scale": "切格缩放比",
      "merge_tolerance": "碎片合并阈值",
      "frame": "切格框架",
      "white_threshold": "纸白阈值",
      "component_min_area": "最小块面积占比",
      "component_close": "闭运算迭代数",
      "deoverlap": "自动分离重叠框",
      "component_open": "开运算半径(0=自动)",
      "component_auto_open": "自动调开运算",
      "panel_polygon": "斜切格用多边形",
      "grid_preview": "输出坐标网格预览"
    },
    "outputs": {
      "0": {
        "name": "分镜格"
      },
      "1": {
        "name": "图像"
      },
      "2": {
        "name": "报告"
      },
      "3": {
        "name": "坐标网格预览"
      }
    }
  },
  "H3MangaRefPack": {
    "display_name": "H3 参考包（角色/风格/音频）",
    "inputs": {
      "character_names": "角色名（逗号分隔，与图槽同序）",
      "character_images": "角色定妆图",
      "character_describes": "角色外观锁定描述（每行一个）",
      "style_name": "风格名",
      "style_images": "风格基准图",
      "audio_names": "音频名",
      "audios": "参考音频",
      "video_names": "视频名",
      "videos": "参考视频",
      "video_audios": "视频配套音轨",
      "max_ref_images": "参考图上限",
      "name_source": "id 生成方式"
    },
    "outputs": {
      "0": {
        "name": "参考包"
      },
      "1": {
        "name": "序号映射表"
      }
    }
  },
  "H3MangaScriptLoader": {
    "display_name": "H3 分镜清单加载",
    "inputs": {
      "source": "来源",
      "file_path": "文件路径",
      "inline_text": "内联文本"
    },
    "outputs": {
      "0": {
        "name": "分镜计划"
      },
      "1": {
        "name": "报告"
      }
    }
  },
  "H3MangaShotBinder": {
    "display_name": "H3 分镜绑定",
    "inputs": {
      "plan": "分镜计划",
      "board": "分镜格",
      "refs": "参考包",
      "bind_mode": "绑定方式",
      "panel_offset": "格序号偏移",
      "allow_reuse": "允许重复用格"
    },
    "outputs": {
      "0": {
        "name": "分镜计划"
      },
      "1": {
        "name": "报告"
      }
    }
  },
  "H3MangaPromptComposer": {
    "display_name": "H3 提示词装配",
    "inputs": {
      "plan": "分镜计划",
      "refs": "参考包",
      "template": "提示词模板",
      "style_prefix": "风格前缀",
      "style_suffix": "风格后缀",
      "negative": "负向提示词",
      "camera_preset": "默认运镜",
      "dialogue_mode": "台词模式",
      "negative_mode": "负向词处理",
      "lock_consistency": "锁定角色一致性",
      "include_describe": "拼入外观描述",
      "camera_json": "运镜表 JSON",
      "style_json": "风格表 JSON"
    },
    "outputs": {
      "0": {
        "name": "分镜计划"
      },
      "1": {
        "name": "提示词（整套）"
      }
    }
  },
  "H3MangaShotConditioning": {
    "display_name": "H3 单镜条件（调试用）",
    "inputs": {
      "clip": "文本编码器",
      "vae": "视频 VAE",
      "audio_vae": "音频 VAE",
      "plan": "分镜计划",
      "refs": "参考包",
      "board": "分镜格",
      "shot_index": "镜头序号",
      "ref_image_size": "参考图尺寸策略"
    },
    "outputs": {
      "0": {
        "name": "正向条件"
      },
      "1": {
        "name": "潜空间"
      },
      "2": {
        "name": "报告"
      }
    }
  },
  "H3MangaEpisodeRender": {
    "display_name": "H3 整集渲染 ★",
    "inputs": {
      "model": "模型",
      "clip": "文本编码器",
      "vae": "视频 VAE",
      "audio_vae": "音频 VAE",
      "plan": "分镜计划",
      "refs": "参考包",
      "board": "分镜格",
      "sampler_name": "采样器",
      "scheduler": "调度器",
      "steps": "采样步数",
      "shift_video": "视频 shift",
      "shift_audio": "音频 shift",
      "ref_image_size": "参考图尺寸策略",
      "lora_name": "LoRA",
      "lora_strength": "LoRA 权重",
      "base_seed": "基准种子",
      "max_shots": "最多渲染镜数",
      "skip_done_ids": "跳过镜头 id",
      "on_error": "单镜失败处理",
      "unload_stages": "分阶段卸载模型",
      "sparse_attn": "稀疏注意力（可选加速）",
      "sparse_tau": "稀疏阈值 τ（sol-attn）",
      "sparse_keep_percent": "保留 key 块 %（sla/vsa）",
      "sparse_start_percent": "稀疏起始进度",
      "sparse_min_tokens": "稀疏最小 token 数",
      "sparse_sink": "参考行精确注意（H3）"
    },
    "outputs": {
      "0": {
        "name": "帧序列"
      },
      "1": {
        "name": "音频"
      },
      "2": {
        "name": "报告"
      }
    }
  },
  "H3MangaConsistencyAudit": {
    "display_name": "H3 一致性审计",
    "inputs": {
      "plan": "分镜计划",
      "refs": "参考包",
      "strict": "严格模式"
    },
    "outputs": {
      "0": {
        "name": "报告"
      },
      "1": {
        "name": "是否通过"
      }
    }
  },
  "H3MangaPlanPreview": {
    "display_name": "H3 计划预览",
    "inputs": {
      "plan": "分镜计划",
      "refs": "参考包",
      "text": "文本"
    },
    "outputs": {
      "0": {
        "name": "文本"
      },
      "1": {
        "name": "元数据"
      }
    }
  },
  "H3MangaFreeMemory": {
    "display_name": "H3 显存/内存释放",
    "inputs": {
      "trigger": "上游触发（仅顺序用）",
      "model": "模型",
      "clip": "文本编码器",
      "vae": "视频 VAE",
      "audio_vae": "音频 VAE",
      "unload_all": "全部卸载",
      "run_gc": "额外 GC",
      "label": "阶段标注"
    },
    "outputs": {
      "0": {
        "name": "报告"
      }
    }
  }
};

export default NODE_LABELS;
