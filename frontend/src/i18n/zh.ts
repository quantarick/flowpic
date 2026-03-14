import type { Translations } from "./en";

const zh: Translations = {
  // App
  appTitle: "FlowPic",
  appTagline: "AI 音乐驱动照片视频生成器",
  sectionImages: "1. 上传图片",
  sectionMusic: "2. 上传音乐",
  sectionConfig: "3. 参数设置",
  sectionResult: "生成结果",

  // ImageUploader
  imgDropHint: "拖拽图片到此处，或点击选择文件",
  imgFormatHint: "支持 JPG、PNG、WebP，最多 100 张，每张不超过 20MB",
  imgUploaded: (n: number) => `已上传 ${n} 张图片`,
  imgAddMore: "点击或拖拽继续添加",

  // MusicUploader
  musicDropHint: "拖拽音乐文件到此处，或点击选择",
  musicFormatHint: "支持 MP3、WAV、FLAC、M4A，最大 100MB / 10 分钟",
  musicDuration: "时长：",
  musicReplace: "点击或拖拽更换文件",

  // ConfigPanel
  cfgAspectRatio: "画面比例：",
  cfgQuality: "画质：",
  cfgVisionModel: "视觉模型：",
  cfgFps: "帧率：",
  arLandscape: "16:9（横屏）",
  arUltrawide: "21:9（超宽屏）",
  arPortrait: "9:16（竖屏）",
  arSquare: "1:1（方形）",
  arClassic: "4:3（经典）",
  q720: "720p（标清）",
  q1080: "1080p（高清）",
  q2k: "2K（超清）",
  q4k: "4K（超高清）",

  // GenerateButton
  btnStarting: "启动中...",
  btnGenerate: "生成视频",

  // ProgressBar
  stepPending: "排队中...",
  stepAnalyzingAudio: "分析音频",
  stepAnalyzingLyrics: "分析歌词",
  stepClassifyingEmotion: "识别情绪",
  stepCaptioning: "识别图片内容",
  stepMatching: "匹配图片与音乐",
  stepReviewingCrops: "审查裁剪效果",
  stepRendering: "渲染视频",
  stepEncoding: "编码视频",
  stepDone: "完成！",
  stepFailed: "失败",
  stepCancelled: "已取消",
  btnCancel: "取消",

  // VideoPreview
  btnDownload: "下载视频",

  // TaskHistory
  taskHistory: "任务历史",
  statusPending: "等待中",
  statusAnalyzing: "分析中",
  statusProcessing: "处理中",
  statusMatching: "匹配中",
  statusReviewing: "审查中",
  statusRendering: "渲染中",
  statusEncoding: "编码中",
  statusDone: "已完成",
  statusFailed: "失败",
  statusCancelled: "已取消",
  taskImages: (n: number) => `${n} 张图片`,
  timeJustNow: "刚刚",
  timeMinAgo: (m: number) => `${m} 分钟前`,
  timeHrAgo: (h: number) => `${h} 小时前`,
  btnStop: "停止",
  btnRetry: "重试",
  btnContinue: "继续",
  retryFailed: (msg: string) => `重试失败：${msg}`,
};

export default zh;
