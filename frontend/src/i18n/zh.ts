import type { Translations } from "./en";

const zh: Translations = {
  // App
  appTitle: "FlowPic",
  appTagline: "AI 音乐驱动照片视频生成器",
  sectionImages: "上传图片",
  sectionMusic: "上传音乐",
  sectionConfig: "参数设置",
  sectionResult: "生成结果",
  tabCrops: "裁剪",
  tabVideo: "视频",
  hintPreviewCrops: "提示：可先在「裁剪」标签页预览构图，确认无误后再生成视频。",

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
  btnCropPreview: "预览裁剪",

  // CropPreview
  cropPreviewTitle: "裁剪预览",
  cropLoading: "加载中...",
  cropEmpty: "暂无裁剪结果",
  cropFeedbackPlaceholder: "哪里有问题？例如：主体应该是羊群，而不是栅栏",
  btnRegenerate: "重新生成",
  regenerating: "重新生成中...",

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

  // Publish / Copywriting
  tabPublish: "发布",
  btnGenerateCopy: "生成文案",
  btnGeneratingCopy: "生成中...",
  copyTitle: "标题",
  copyDesc: "正文",
  copyHashtags: "话题标签",
  copyCover: "发布图片",
  btnSelectAll: "全选",
  btnDeselectAll: "取消全选",
  btnCopy: "复制",
  copiedMsg: "已复制！",
  copyHint: "风格提示",
  copyHintPlaceholder: "例如：轻松活泼的语气、突出美食、提到日落...",
  copyEmpty: "点击「生成文案」，根据照片自动生成小红书文案。",
  copyError: (msg: string) => `生成失败：${msg}`,
  btnPreview: "预览",
  btnEdit: "编辑",
  publishCrops: "裁剪图片",
  publishVideo: "生成视频",

  // XHS
  xhsAccount: "小红书账号",
  xhsConnect: "连接",
  xhsDisconnect: "断开",
  xhsCancel: "取消",
  xhsConnecting: "连接中...",
  xhsSaveCookies: "保存 Cookie",
  xhsCookieHint: "使用 cookies.txt Chrome 插件导出 xiaohongshu.com 的 Cookie 为 JSON，然后粘贴到此处。",
  xhsExpired: "Cookie 已过期，请重新连接账号。",
  xhsBtnPublish: "发布到小红书",
  xhsPublishing: "发布中...",
  xhsPublishSuccess: "发布成功！",
  xhsViewPost: "查看帖子",
  xhsPublishFailed: (msg: string) => `发布失败：${msg}`,

  // Style Scanner
  xhsScanStyle: "扫描我的风格",
  xhsScanning: "扫描中...",
  xhsStyleActive: "风格模板已激活",
  xhsRescan: "重新扫描",
  xhsClearStyle: "清除",
  xhsStyleError: (msg: string) => `风格扫描失败：${msg}`,

  // Published Images
  publishedBadge: "已发布",
  btnCleanPublished: "清理已发布图片",
  cleanedMsg: (n: number) => `已移除 ${n} 张已发布图片`,
};

export default zh;
