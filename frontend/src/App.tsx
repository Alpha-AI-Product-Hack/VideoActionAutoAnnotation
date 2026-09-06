import { useState, useRef, useEffect } from "react";
import { downloadExport, getAnnotation, getJob, putAnnotation, uploadVideo, videoFileUrl } from "./api";
import type { ActionSegment } from "./api";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Action {
  id: string;
  start_ms: number;
  end_ms: number;
  action: string;
  object: string | null;
  keyframe_ms: number;
  confidence: number;
  model_version: string;
  clip_id?: string | null;
}

interface VideoRecord {
  id: string;
  jobId?: string;
  name: string;
  duration_ms: number;
  status: "staged" | "pending" | "processing" | "done" | "error";
  uploadedAt: string;
  actions: Action[];
  file?: File;
  previewUrl?: string;
  error?: string;
  progress?: number;
}

// ─── Mock data ────────────────────────────────────────────────────────────────

const ACTION_COLORS: Record<string, string> = {
  pick_up: "#4f7df7",
  put_down: "#a855f7",
  pour: "#f59e0b",
  open: "#00d4aa",
  close: "#ff4d6a",
  press: "#06b6d4",
  default: "#6b7291",
};

function readVideoDurationMs(file: File): Promise<number> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.onloadedmetadata = () => {
      const ms = Math.round((video.duration || 0) * 1000);
      URL.revokeObjectURL(url);
      resolve(ms > 0 ? ms : 15000);
    };
    video.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(15000);
    };
    video.src = url;
  });
}

function toUiAction(seg: ActionSegment): Action {
  return {
    id: String(seg.id),
    start_ms: seg.start_ms,
    end_ms: seg.end_ms,
    action: seg.action,
    object: seg.object,
    keyframe_ms: seg.keyframe_ms,
    confidence: seg.confidence,
    model_version: seg.model_version,
    clip_id: seg.clip_id,
  };
}

// ─── Utils ────────────────────────────────────────────────────────────────────

function msToTimecode(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const rem = s % 60;
  const msRem = Math.floor((ms % 1000) / 10);
  return `${m.toString().padStart(2, "0")}:${rem.toString().padStart(2, "0")}.${msRem.toString().padStart(2, "0")}`;
}

function actionColor(action: string): string {
  return ACTION_COLORS[action] ?? ACTION_COLORS.default;
}

function confidencePct(c: number): string {
  return (c * 100).toFixed(1) + "%";
}

// ─── Icons ────────────────────────────────────────────────────────────────────

const PreviewIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
    <path d="M1 8s2.5-5 7-5 7 5 7 5-2.5 5-7 5-7-5-7-5z" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
    <circle cx="8" cy="8" r="2" stroke="currentColor" strokeWidth="1.4" />
  </svg>
);
const ConfigIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
    <circle cx="8" cy="8" r="2.5" stroke="currentColor" strokeWidth="1.4" />
    <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.22 3.22l1.41 1.41M11.37 11.37l1.41 1.41M3.22 12.78l1.41-1.41M11.37 4.63l1.41-1.41" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
  </svg>
);
const UploadIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
    <path d="M8 2v8M5 5l3-3 3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M2 12h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);
const PlayIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" className="text-[var(--color-accent)] ml-1">
    <path d="M8 5l11 7-11 7V5z" />
  </svg>
);
const SelectIcon = () => (
  <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
    <path d="M3 2l10 5.5-5.5 1.5L6 14 3 2z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
  </svg>
);
const TrimIcon = () => (
  <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
    <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    <path d="M10 2v12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);
const SplitIcon = () => (
  <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
    <path d="M8 2v12M2 5l6-3 6 3M2 11l6 3 6-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
const KeyframeIcon = () => (
  <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
    <rect x="5" y="5" width="6" height="6" rx="1" transform="rotate(45 8 8)" stroke="currentColor" strokeWidth="1.5" />
  </svg>
);
const DeleteIcon = () => (
  <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
    <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);
const SunIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
    <circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1.4" />
    <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
  </svg>
);
const MoonIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
    <path d="M13.5 10.5A6 6 0 015.5 2.5a6 6 0 108 8z" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
const ExportIcon = () => (
  <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
    <path d="M8 2v8M5 7l3 3 3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M2 12h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: VideoRecord["status"] }) {
  const map: Record<string, { label: string; color: string }> = {
    staged: { label: "STAGED", color: "#4a5180" },
    pending: { label: "PENDING", color: "#6b7291" },
    processing: { label: "PROCESSING", color: "#f5a623" },
    done: { label: "DONE", color: "#00d4aa" },
    error: { label: "ERROR", color: "#ff4d6a" },
  };
  const { label, color } = map[status];
  return (
    <span
      style={{ color, borderColor: color + "40", fontFamily: "var(--font-mono)" }}
      className="text-[10px] font-medium px-1.5 py-0.5 rounded border tracking-wider"
    >
      {label}
    </span>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-3 py-1.5 flex items-center gap-2">
      <div className="flex-1 h-px bg-[var(--color-border)]" />
      <span className="text-[10px] tracking-[0.12em] uppercase font-semibold text-[var(--color-text-muted)]">{children}</span>
      <div className="flex-1 h-px bg-[var(--color-border)]" />
    </div>
  );
}

// ─── RULES MODAL ─────────────────────────────────────────────────────────────

function RulesModal({
  rulesJson,
  onSave,
  onCancel,
}: {
  rulesJson: string;
  onSave: (json: string) => void;
  onCancel: () => void;
}) {
  const [draft, setDraft] = useState(rulesJson);
  const [parseError, setParseError] = useState<string | null>(null);

  function handleSave() {
    try {
      JSON.parse(draft);
      setParseError(null);
      onSave(draft);
    } catch (e: any) {
      setParseError(e.message);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "rgba(0,0,0,0.7)" }}
      onClick={(e) => e.target === e.currentTarget && onCancel()}
    >
      <div className="bg-[var(--color-panel)] border border-[var(--color-border-light)] rounded-lg w-[520px] max-h-[80vh] flex flex-col shadow-2xl">
        {/* Modal header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--color-border)]">
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)]" />
            <span className="text-sm font-semibold text-[var(--color-text)]">Annotation Rules</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-[var(--color-text-muted)] border border-[var(--color-border)] px-1.5 py-0.5 rounded">
              JSON
            </span>
            <button
              onClick={onCancel}
              className="w-6 h-6 rounded flex items-center justify-center text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-panel-alt)] transition-colors text-sm"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Notice */}
        <div className="px-4 py-2 bg-[var(--color-panel-alt)] border-b border-[var(--color-border)] flex items-start gap-2">
          <span className="text-[var(--color-warn)] text-xs mt-0.5">⚠</span>
          <p className="text-[11px] text-[var(--color-text-muted)] leading-relaxed">
            Rule configuration UI is under development. Edit the JSON directly for now. Changes take effect on the next detection run.
          </p>
        </div>

        {/* Editor */}
        <div className="flex-1 overflow-auto p-4">
          <textarea
            value={draft}
            onChange={(e) => { setDraft(e.target.value); setParseError(null); }}
            spellCheck={false}
            className="w-full h-64 bg-[var(--color-bg)] border border-[var(--color-border)] rounded p-3 text-[11px] font-mono text-[var(--color-text)] resize-none focus:outline-none focus:border-[var(--color-accent)] leading-relaxed transition-colors"
          />
          {parseError && (
            <p className="mt-2 text-[10px] font-mono text-[var(--color-danger)]">Parse error: {parseError}</p>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-4 py-3 border-t border-[var(--color-border)]">
          <button
            onClick={onCancel}
            className="h-8 px-4 rounded text-xs font-medium border border-[var(--color-border-light)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:border-[var(--color-border-light)] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="h-8 px-4 rounded text-xs font-semibold bg-[var(--color-accent)] text-[var(--color-bg)] hover:bg-[var(--color-accent-dim)] transition-colors"
          >
            Save Rules
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── LEFT PANEL ───────────────────────────────────────────────────────────────

function LeftPanel({
  videos,
  selectedVideoId,
  onSelectVideo,
  onAddVideo,
  onSubmitPending,
  onDeleteVideo,
  onClearStaged,
}: {
  videos: VideoRecord[];
  selectedVideoId: string | null;
  onSelectVideo: (id: string) => void;
  onAddVideo: (v: VideoRecord, select?: boolean) => void;
  onSubmitPending: (payload: { rulesJson: string; model: string }) => void;
  onDeleteVideo: (id: string) => void;
  onClearStaged: () => void;
}) {
  const [rulesJson, setRulesJson] = useState('{\n  "min_duration_ms": 500,\n  "min_confidence": 0.7,\n  "actions": ["pick_up", "put_down", "pour", "open", "close"],\n  "objects": ["cup", "glass", "bottle", "drawer"]\n}');
  const [rulesFileName, setRulesFileName] = useState<string | null>(null);
  const [fps, setFps] = useState("30");
  const [model, setModel] = useState("pipeline-0.1");
  const [showRulesModal, setShowRulesModal] = useState(false);
  const videoFileRef = useRef<HTMLInputElement>(null);
  const rulesFileRef = useRef<HTMLInputElement>(null);

  const hasStagedVideos = videos.some((v) => v.status === "staged");
  const rulesValid = (() => {
    try {
      JSON.parse(rulesJson);
      return true;
    } catch {
      return false;
    }
  })();
  const canSubmit = hasStagedVideos && rulesValid;

  async function handleVideoFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    const now = Date.now();
    for (const [i, file] of files.entries()) {
      const duration_ms = await readVideoDurationMs(file);
      const previewUrl = URL.createObjectURL(file);
      const newVideo: VideoRecord = {
        id: "local-" + (now + i),
        name: file.name,
        duration_ms,
        status: "staged",
        uploadedAt: new Date().toISOString().slice(0, 16).replace("T", " "),
        actions: [],
        file,
        previewUrl,
      };
      onAddVideo(newVideo, i === 0);
    }
    e.target.value = "";
  }

  function handleRulesFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      try {
        JSON.parse(text);
        setRulesJson(text);
        setRulesFileName(file.name);
      } catch {
        setRulesFileName(file.name + " (invalid JSON)");
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  }

  return (
    <>
      {showRulesModal && (
        <RulesModal
          rulesJson={rulesJson}
          onSave={(json) => { setRulesJson(json); setShowRulesModal(false); }}
          onCancel={() => setShowRulesModal(false)}
        />
      )}

      <div className="flex flex-col h-full overflow-hidden bg-[var(--color-panel)] border-r border-[var(--color-border)]">
        {/* Upload & Config */}
        <div className="flex-none border-b border-[var(--color-border)]">
          <SectionLabel>Source & Config</SectionLabel>

          <div className="px-3 pb-3 space-y-2.5">
            {/* Video upload */}
            <div>
              <label className="block text-[10px] text-[var(--color-text-muted)] mb-1 tracking-wider uppercase">Video File</label>
              <div className="flex gap-1.5">
                <button
                  onClick={() => videoFileRef.current?.click()}
                  className="flex-1 h-9 rounded flex items-center justify-center gap-2 text-xs font-medium border border-dashed border-[var(--color-accent)] text-[var(--color-accent)] hover:bg-[var(--color-accent-glow)] transition-colors"
                >
                  <UploadIcon />
                  Upload Video
                </button>
                <button
                  onClick={onClearStaged}
                  disabled={!hasStagedVideos}
                  title="Clear all staged videos"
                  className="w-9 h-9 rounded border border-[var(--color-border-light)] text-[var(--color-text-muted)] hover:text-[var(--color-danger)] hover:border-[var(--color-border-light)] hover:bg-[var(--color-panel-alt)] transition-colors flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed text-[11px]"
                >
                  ✕
                </button>
              </div>
              <input ref={videoFileRef} type="file" accept="video/*" multiple className="hidden" onChange={handleVideoFileChange} />
            </div>

            {/* Rules upload + configure */}
            <div>
              <label className="block text-[10px] text-[var(--color-text-muted)] mb-1 tracking-wider uppercase">Annotation Rules</label>
              <div className="flex gap-1.5">
                <button
                  onClick={() => rulesFileRef.current?.click()}
                  className={`flex-1 h-9 rounded flex items-center justify-center gap-2 text-xs font-medium border border-dashed transition-colors ${
                    rulesFileName
                      ? "border-[var(--color-accent)] text-[var(--color-accent)] bg-[var(--color-accent-glow)]"
                      : "border-[var(--color-border-light)] text-[var(--color-text-muted)] hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] hover:bg-[var(--color-accent-glow)]"
                  }`}
                >
                  <UploadIcon />
                  {rulesFileName ? "Re-upload" : "Upload JSON"}
                </button>
                <button
                  onClick={() => setShowRulesModal(true)}
                  title="Configure rules"
                  className="w-9 h-9 rounded border border-[var(--color-border-light)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:border-[var(--color-border-light)] hover:bg-[var(--color-panel-alt)] transition-colors flex items-center justify-center"
                >
                  <ConfigIcon />
                </button>
              </div>
              <input ref={rulesFileRef} type="file" accept=".json,application/json" className="hidden" onChange={handleRulesFileChange} />
              {rulesFileName && (
                <p className="mt-1 text-[10px] font-mono text-[var(--color-text-dim)] truncate" title={rulesFileName}>
                  ✓ {rulesFileName}
                </p>
              )}
            </div>

            {/* Settings row */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] text-[var(--color-text-muted)] mb-1 tracking-wider uppercase">FPS</label>
                <input
                  value={fps}
                  onChange={(e) => setFps(e.target.value)}
                  className="w-full h-7 bg-[var(--color-panel-alt)] border border-[var(--color-border)] rounded px-2 text-xs font-mono text-[var(--color-text)] focus:outline-none focus:border-[var(--color-accent)]"
                />
              </div>
              <div>
                <label className="block text-[10px] text-[var(--color-text-muted)] mb-1 tracking-wider uppercase">Model</label>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="w-full h-7 bg-[var(--color-panel-alt)] border border-[var(--color-border)] rounded px-2 text-xs font-mono text-[var(--color-text)] focus:outline-none focus:border-[var(--color-accent)]"
                >
                  <option>pipeline-0.1</option>
                  <option>pipeline-0.2-beta</option>
                </select>
              </div>
            </div>

            {/* Submit to queue */}
            <button
              disabled={!canSubmit}
              onClick={() => {
                onSubmitPending({ rulesJson, model });
              }}
              className={`w-full h-9 rounded text-xs font-semibold transition-colors ${
                canSubmit
                  ? "bg-[var(--color-accent)] text-[var(--color-bg)] hover:bg-[var(--color-accent-dim)] cursor-pointer"
                  : "bg-[var(--color-border)] text-[var(--color-text-muted)] cursor-not-allowed"
              }`}
            >
              {!hasStagedVideos
                ? "Upload a video to submit"
                : !rulesValid
                ? "Fix rules JSON to submit"
                : "Submit to Annotation Queue"}
            </button>
          </div>
        </div>

        {/* Video list */}
        <div className="flex-1 overflow-hidden flex flex-col">
          <SectionLabel>Video Queue</SectionLabel>
          <div className="flex-1 overflow-y-auto px-2 pb-2">
            {(() => {
              const dotColor: Record<string, string> = {
                staged: "#4a5180",
                pending: "#6b7291",
                processing: "#f5a623",
                done: "#00d4aa",
                error: "#ff4d6a",
              };
              const staged = videos.filter((v) => v.status === "staged");
              const submitted = videos.filter((v) => v.status !== "staged");
              const renderRow = (v: VideoRecord) => (
                <div
                  key={v.id}
                  onClick={() => onSelectVideo(v.id)}
                  className={`w-full text-left px-3 py-2.5 rounded transition-all cursor-pointer group ${
                    selectedVideoId === v.id
                      ? "bg-[var(--color-accent-glow)] border border-[var(--color-accent)]"
                      : "border border-transparent hover:bg-[var(--color-panel-alt)] hover:border-[var(--color-border)]"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-1.5 h-1.5 rounded-full flex-none" style={{ backgroundColor: dotColor[v.status] }} />
                    <span className="text-xs font-medium text-[var(--color-text)] leading-tight truncate flex-1">{v.name}</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); onDeleteVideo(v.id); }}
                      className="opacity-0 group-hover:opacity-100 text-[var(--color-danger)] text-[10px] transition-opacity hover:text-red-400 flex-none"
                    >
                      ✕
                    </button>
                  </div>
                  <div className="flex items-center justify-between pl-3.5">
                    <span className="text-[10px] font-mono text-[var(--color-text-muted)]">{v.uploadedAt}</span>
                    <span className="text-[10px] font-mono text-[var(--color-text-dim)]">
                      {v.status === "processing" && v.progress != null ? `${v.progress}%` : msToTimecode(v.duration_ms)}
                    </span>
                  </div>
                </div>
              );
              return (
                <div className="space-y-1">
                  {staged.length > 0 && (
                    <>
                      <div className="flex items-center gap-2 py-1 px-1">
                        <span className="text-[9px] font-mono tracking-widest uppercase text-[var(--color-text-muted)]">Staged</span>
                        <span className="text-[9px] font-mono text-[var(--color-text-dim)]">({staged.length})</span>
                        <div className="flex-1 h-px bg-[var(--color-border)]" />
                      </div>
                      {staged.map(renderRow)}
                    </>
                  )}
                  {staged.length > 0 && submitted.length > 0 && (
                    <div className="flex items-center gap-2 py-1 px-1">
                      <span className="text-[9px] font-mono tracking-widest uppercase text-[var(--color-text-muted)]">Queue</span>
                      <div className="flex-1 h-px bg-[var(--color-border)]" />
                    </div>
                  )}
                  {submitted.map(renderRow)}
                </div>
              );
            })()}
          </div>
        </div>
      </div>
    </>
  );
}

// ─── CENTER PANEL ─────────────────────────────────────────────────────────────

type Tool = "select" | "edit";

// ─── Confirm Dialog ───────────────────────────────────────────────────────────

function ConfirmDialog({
  title,
  message,
  confirmLabel,
  onConfirm,
  onCancel,
}: {
  title: string;
  message: string;
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "rgba(0,0,0,0.65)" }}
      onClick={(e) => e.target === e.currentTarget && onCancel()}
    >
      <div className="bg-[var(--color-panel)] border border-[var(--color-border-light)] rounded-lg w-80 shadow-2xl">
        <div className="flex items-center gap-2 px-4 py-3 border-b border-[var(--color-border)]">
          <span className="text-[var(--color-warn)] text-sm">⚠</span>
          <span className="text-sm font-semibold text-[var(--color-text)]">{title}</span>
        </div>
        <p className="px-4 py-3 text-xs text-[var(--color-text-dim)] leading-relaxed">{message}</p>
        <div className="flex items-center justify-end gap-2 px-4 py-3 border-t border-[var(--color-border)]">
          <button
            onClick={onCancel}
            className="h-7 px-3 rounded text-xs font-medium border border-[var(--color-border-light)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="h-7 px-3 rounded text-xs font-semibold bg-[var(--color-danger)] text-white hover:opacity-90 transition-opacity"
          >
            {confirmLabel ?? "Confirm"}
          </button>
        </div>
      </div>
    </div>
  );
}

function CenterPanel({
  video,
  selectedActionId,
  onSelectAction,
  onUpdateAction,
  onAddAction,
  onDeleteAction,
  pendingEdits,
  onSetPendingEdits,
  onTimeChange,
  onDurationMs,
}: {
  video: VideoRecord | null;
  selectedActionId: string | null;
  onSelectAction: (id: string | null) => void;
  onUpdateAction: (a: Action) => void;
  onAddAction: (atMs: number) => void;
  onDeleteAction: (id: string) => void;
  pendingEdits: Record<string, Action>;
  onSetPendingEdits: React.Dispatch<React.SetStateAction<Record<string, Action>>>;
  onTimeChange?: (ms: number) => void;
  onDurationMs?: (ms: number) => void;
}) {
  const [currentMs, setCurrentMs] = useState(0);
  const [activeTool, setActiveTool] = useState<Tool>("select");
  const [zoom, setZoom] = useState(1);
  const timelineRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const isDragging = useRef(false);

  // Confirm dialog state
  const [confirmSplit, setConfirmSplit] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  function handleTimelineWheel(e: React.WheelEvent) {
    e.preventDefault();
    if (scrollRef.current) {
      scrollRef.current.scrollLeft += e.deltaY !== 0 ? e.deltaY : e.deltaX;
    }
  }

  const panelRef = useRef<HTMLDivElement>(null);
  const [previewHeight, setPreviewHeight] = useState<number | null>(null);
  const isSplitDragging = useRef(false);

  // Drag a segment boundary or keyframe marker (Edit mode only)
  function startSegmentDrag(
    e: React.MouseEvent,
    action: Action,
    part: "start" | "end" | "keyframe"
  ) {
    if (activeTool !== "edit") return;
    e.preventDefault();
    e.stopPropagation();
    onSelectAction(action.id);

    const trackEl = timelineRef.current;
    if (!trackEl) return;
    const trackRect = trackEl.getBoundingClientRect();
    const startX = e.clientX;
    const pxPerMs = trackRect.width / (video?.duration_ms ?? 20000);

    // Work from pending edits if they exist, else the committed action
    const base = pendingEdits[action.id] ?? action;
    const origStart = base.start_ms;
    const origEnd = base.end_ms;
    const origKf = base.keyframe_ms;

    function onMove(ev: MouseEvent) {
      const deltaMs = Math.round((ev.clientX - startX) / pxPerMs);
      const updated = { ...base };
      if (part === "start") {
        updated.start_ms = Math.max(0, Math.min(origStart + deltaMs, origEnd - 100));
        updated.keyframe_ms = Math.max(updated.start_ms, Math.min(origKf, updated.end_ms));
      } else if (part === "end") {
        updated.end_ms = Math.max(origStart + 100, Math.min(origEnd + deltaMs, video?.duration_ms ?? 99999999));
        updated.keyframe_ms = Math.max(updated.start_ms, Math.min(origKf, updated.end_ms));
      } else {
        updated.keyframe_ms = Math.max(origStart, Math.min(origKf + deltaMs, origEnd));
      }
      onSetPendingEdits((prev) => ({ ...prev, [action.id]: updated }));
    }
    function onUp() {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  function handleDividerMouseDown(e: React.MouseEvent) {
    e.preventDefault();
    isSplitDragging.current = true;
    const startY = e.clientY;
    const startH = previewHeight ?? (panelRef.current?.clientHeight ?? 400) * 0.55;
    function onMove(ev: MouseEvent) {
      if (!isSplitDragging.current) return;
      const total = panelRef.current?.clientHeight ?? 400;
      setPreviewHeight(Math.min(Math.max(startH + ev.clientY - startY, 120), total - 140));
    }
    function onUp() {
      isSplitDragging.current = false;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  const duration = video?.duration_ms ?? 20000;
  const actions = video?.actions ?? [];
  const mediaUrl = video?.previewUrl || (video && video.status !== "staged" ? videoFileUrl(video.id) : "");

  function seekTo(ms: number) {
    const clamped = Math.max(0, Math.min(ms, duration));
    setCurrentMs(clamped);
    if (videoRef.current) {
      videoRef.current.currentTime = clamped / 1000;
    }
    onTimeChange?.(clamped);
  }

  function togglePlayback() {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) el.play().catch(() => undefined);
    else el.pause();
  }

  function handleTimelineClick(e: React.MouseEvent<HTMLDivElement>) {
    const rect = timelineRef.current?.getBoundingClientRect();
    if (!rect) return;
    const ratio = (e.clientX - rect.left) / rect.width;
    seekTo(Math.floor(ratio * duration));
  }

  function handleTimelineMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!isDragging.current) return;
    const rect = timelineRef.current?.getBoundingClientRect();
    if (!rect) return;
    const ratio = (e.clientX - rect.left) / rect.width;
    seekTo(Math.floor(ratio * duration));
  }

  function executeSplit() {
    const hit = actions.find((a) => a.start_ms <= currentMs && currentMs <= a.end_ms);
    if (!hit) return;
    // Remove the original, add two halves
    onDeleteAction(hit.id);
    onAddAction(hit.start_ms); // caller will create at currentMs start; we pass the split point
    // We pass currentMs as a signal; onAddAction needs to support range — handled in App
    setConfirmSplit(false);
  }

  function executeDelete() {
    if (selectedActionId) onDeleteAction(selectedActionId);
    setConfirmDelete(false);
  }

  // What action is under the playhead (for Split)
  const hitAction = actions.find((a) => a.start_ms <= currentMs && currentMs <= a.end_ms);

  const playheadPct = (currentMs / duration) * 100;
  const tickCount = Math.round(30 * zoom);
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => i / tickCount);
  const defaultPreviewPct = 0.55;

  return (
    <div ref={panelRef} className="flex flex-col h-full overflow-hidden bg-[var(--color-bg)]">
      {/* Confirm: Split */}
      {confirmSplit && hitAction && (
        <ConfirmDialog
          title="Split Action"
          message={`Split "${hitAction.action}" into two segments at ${msToTimecode(currentMs)}? The second segment will have action and object set to "unknown".`}
          confirmLabel="Split"
          onConfirm={() => {
            onUpdateAction({ ...hitAction, end_ms: currentMs, keyframe_ms: Math.min(hitAction.keyframe_ms, currentMs) });
            onAddAction(currentMs);
            setConfirmSplit(false);
          }}
          onCancel={() => setConfirmSplit(false)}
        />
      )}
      {/* Confirm: Delete */}
      {confirmDelete && selectedActionId && (
        <ConfirmDialog
          title="Delete Action"
          message={`Delete the selected action "${actions.find(a => a.id === selectedActionId)?.action ?? ""}"? This cannot be undone.`}
          confirmLabel="Delete"
          onConfirm={executeDelete}
          onCancel={() => setConfirmDelete(false)}
        />
      )}
      {/* Video preview — height controlled by drag divider */}
      <div
        className="flex-none overflow-hidden"
        style={{ height: previewHeight != null ? previewHeight : `${defaultPreviewPct * 100}%` }}
      >
        {/* Black letterbox canvas */}
        <div className="w-full h-full bg-black flex items-center justify-center relative">
          {video ? (
            <>
              {/* 4:3 content box — shrinks to fit either dimension */}
              <div
                className="relative"
                style={{
                  aspectRatio: "4 / 3",
                  maxWidth: "100%",
                  maxHeight: "100%",
                  width: "100%",
                  height: "auto",
                }}
              >
                {mediaUrl ? (
                  <video
                    ref={videoRef}
                    src={mediaUrl}
                    className="absolute inset-0 w-full h-full object-contain bg-black"
                    onTimeUpdate={(e) => setCurrentMs(Math.round(e.currentTarget.currentTime * 1000))}
                    onLoadedMetadata={(e) => {
                      const ms = Math.round((e.currentTarget.duration || 0) * 1000);
                      if (ms > 0) onDurationMs?.(ms);
                    }}
                    onClick={togglePlayback}
                  />
                ) : (
                  <button
                    type="button"
                    className="absolute inset-0 flex flex-col items-center justify-center"
                  >
                    <div className="w-16 h-16 rounded-full border-2 border-[var(--color-accent)] flex items-center justify-center mb-3">
                      <PlayIcon />
                    </div>
                    <p className="text-xs text-[var(--color-text-muted)] font-mono">{video.name}</p>
                  </button>
                )}
                <div className="absolute bottom-3 left-3 flex items-center gap-3 pointer-events-none">
                  <span className="font-mono text-xs text-[var(--color-accent)] bg-black/70 px-2 py-0.5 rounded">
                    {msToTimecode(currentMs)}
                  </span>
                  <span className="font-mono text-[10px] text-[var(--color-text-muted)] bg-black/70 px-2 py-0.5 rounded">
                    / {msToTimecode(duration)}
                  </span>
                </div>
              </div>
            </>
          ) : (
            <p className="text-sm text-[var(--color-text-muted)]">No video selected</p>
          )}
        </div>
      </div>

      {/* Drag divider */}
      <div
        onMouseDown={handleDividerMouseDown}
        className="flex-none h-1.5 cursor-row-resize flex items-center justify-center group border-y border-[var(--color-border)] bg-[var(--color-panel)] hover:bg-[var(--color-accent-glow)] transition-colors"
        title="Drag to resize"
      >
        <div className="w-10 h-0.5 rounded-full bg-[var(--color-border-light)] group-hover:bg-[var(--color-accent)] transition-colors" />
      </div>

      {/* Toolbar */}
      <div className="flex-none border-b border-[var(--color-border)] bg-[var(--color-panel)] px-3 py-1.5 flex items-center gap-2">

        {/* Mode tools: Select / Edit */}
        <div className="flex items-center gap-0.5">
          {([
            { id: "select" as Tool, icon: <SelectIcon />, label: "Select" },
            { id: "edit" as Tool, icon: <TrimIcon />, label: "Edit boundaries & keyframe" },
          ]).map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTool(t.id)}
              title={t.label}
              className={`h-7 px-2.5 rounded flex items-center gap-1.5 text-[11px] font-medium transition-colors ${
                activeTool === t.id
                  ? "bg-[var(--color-accent)] text-[var(--color-bg)]"
                  : "text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-border)]"
              }`}
            >
              {t.icon}
              <span>{t.id === "select" ? "Select" : "Edit"}</span>
            </button>
          ))}
        </div>

        <div className="w-px h-5 bg-[var(--color-border)]" />

        {/* Action tools: Split / Add / Delete */}
        <div className="flex items-center gap-0.5">
          <button
            title={hitAction ? `Split "${hitAction.action}" at ${msToTimecode(currentMs)}` : "No action under playhead"}
            disabled={!hitAction}
            onClick={() => setConfirmSplit(true)}
            className="h-7 px-2.5 rounded flex items-center gap-1.5 text-[11px] font-medium text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-border)] transition-colors disabled:opacity-35 disabled:cursor-not-allowed"
          >
            <SplitIcon />
            <span>Split</span>
          </button>
          <button
            title="Add new action at playhead"
            onClick={() => onAddAction(currentMs)}
            className="h-7 px-2.5 rounded flex items-center gap-1.5 text-[11px] font-medium text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-border)] transition-colors"
          >
            <KeyframeIcon />
            <span>Add</span>
          </button>
          <button
            title={selectedActionId ? "Delete selected action" : "Select an action first"}
            disabled={!selectedActionId}
            onClick={() => setConfirmDelete(true)}
            className="h-7 px-2.5 rounded flex items-center gap-1.5 text-[11px] font-medium hover:bg-[var(--color-border)] transition-colors disabled:opacity-35 disabled:cursor-not-allowed"
            style={{ color: selectedActionId ? "var(--color-danger)" : undefined }}
          >
            <DeleteIcon />
            <span>Delete</span>
          </button>
        </div>

        <div className="w-px h-5 bg-[var(--color-border)]" />

        {/* Zoom */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-[var(--color-text-muted)] tracking-wider uppercase">Zoom</span>
          <input type="range" min={1} max={5} step={0.5} value={zoom}
            onChange={(e) => setZoom(Number(e.target.value))}
            className="w-20 accent-[#00d4aa]"
          />
          <span className="text-[10px] font-mono text-[var(--color-text-dim)]">{zoom}×</span>
        </div>

        {/* Tool indicator */}
        <div className="ml-auto flex items-center gap-1.5">
          <div className={`w-1.5 h-1.5 rounded-full ${activeTool === "edit" ? "bg-[var(--color-accent)]" : "bg-[var(--color-text-muted)]"}`} />
          <span className="text-[10px] font-mono text-[var(--color-text-muted)] uppercase tracking-wider">{activeTool}</span>
          {activeTool === "edit" && pendingEdits[selectedActionId ?? ""] && (
            <span className="text-[9px] font-mono text-[var(--color-warn)] ml-1">unsaved</span>
          )}
        </div>
      </div>

      {/* Seek bar */}
      <div className="flex-none h-8 border-b border-[var(--color-border)] bg-[var(--color-panel)] flex items-center px-3 gap-3">
        <span className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-wider">Seek</span>
        <input
          type="range"
          min={0} max={duration} step={50}
          value={currentMs}
          onChange={(e) => seekTo(Number(e.target.value))}
          className="flex-1 accent-[#00d4aa]"
        />
        <span className="font-mono text-[10px] text-[var(--color-accent)] w-20 text-right">{msToTimecode(currentMs)}</span>
      </div>

      {/* Timeline */}
      <div className="flex-1 overflow-hidden flex flex-col bg-[var(--color-panel-alt)] select-none min-h-0">
        {/* Horizontally scrollable container for ruler + track */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-x-auto overflow-y-hidden min-h-0"
          style={{ scrollbarWidth: "none" }}
          onWheel={handleTimelineWheel}
        >
          {/* Inner track at zoom width — flex column so ruler + track stack vertically */}
          <div className="flex flex-col h-full" style={{ minWidth: `${zoom * 100}%` }}>

        {/* Ruler */}
        <div className="flex-none h-20 relative border-b border-[var(--color-border)]">
          {ticks.map((t) => (
            <div key={t} className="absolute top-0 flex flex-col items-center" style={{ left: `${t * 100}%` }}>
              <div className="w-px h-2 bg-[var(--color-border-light)]" />
              <span
                className="text-[9px] font-mono text-[var(--color-text-muted)] mt-0.5"
                style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", lineHeight: 1 }}
              >
                {msToTimecode(Math.floor(t * duration))}
              </span>
            </div>
          ))}
        </div>

        {/* Track area */}
        {(() => {
          // Assign each action to a lane via greedy interval packing (sorted by start_ms)
          const LANE_H = 60;
          const LANE_GAP = 4;
          const TRACK_PAD = 8;
          const MAX_LANES = 4; // fixed pre-allocated lanes — track height never changes
          const sorted = [...actions].sort((a, b) => a.start_ms - b.start_ms);
          const laneEnds: number[] = [];
          const laneOf: Record<string, number> = {};
          for (const a of sorted) {
            const display = (activeTool === "edit" && pendingEdits[a.id]) ? pendingEdits[a.id] : a;
            let placed = false;
            for (let i = 0; i < laneEnds.length; i++) {
              if (display.start_ms >= laneEnds[i]) {
                laneOf[a.id] = i;
                laneEnds[i] = display.end_ms;
                placed = true;
                break;
              }
            }
            if (!placed) {
              laneOf[a.id] = laneEnds.length;
              laneEnds.push(display.end_ms);
            }
          }
          const trackHeight = TRACK_PAD * 2 + MAX_LANES * LANE_H + (MAX_LANES - 1) * LANE_GAP;

          return (
          <div
            ref={timelineRef}
            className="relative cursor-crosshair flex-1 overflow-y-auto min-h-0"
            style={{ minHeight: trackHeight }}
            onClick={handleTimelineClick}
            onMouseMove={handleTimelineMouseMove}
            onMouseDown={() => { isDragging.current = true; }}
            onMouseUp={() => { isDragging.current = false; }}
            onMouseLeave={() => { isDragging.current = false; }}
          >
          {/* Action segments */}
          <div className="absolute inset-x-0 inset-y-0">
            {actions.map((a) => {
              // In Edit mode show pending position; in Select mode show committed
              const display = (activeTool === "edit" && pendingEdits[a.id]) ? pendingEdits[a.id] : a;
              const left = (display.start_ms / duration) * 100;
              const width = ((display.end_ms - display.start_ms) / duration) * 100;
              const kfPct = ((display.keyframe_ms - display.start_ms) / (display.end_ms - display.start_ms)) * 100;
              const color = actionColor(a.action);
              const isSelected = a.id === selectedActionId;
              const hasPending = !!pendingEdits[a.id];
              const lane = laneOf[a.id] ?? 0;
              const topPx = TRACK_PAD + lane * (LANE_H + LANE_GAP);
              return (
                <div
                  key={a.id}
                  onClick={(e) => { e.stopPropagation(); onSelectAction(a.id); }}
                  style={{
                    left: `${left}%`,
                    width: `${width}%`,
                    top: topPx,
                    height: LANE_H,
                    backgroundColor: color + (hasPending ? "44" : "33"),
                    borderColor: hasPending ? "var(--color-warn)" : color,
                    borderWidth: isSelected ? 2 : 1,
                    outlineColor: isSelected ? color : "transparent",
                    cursor: activeTool === "edit" ? "grab" : "pointer",
                  }}
                  className={`absolute rounded border transition-colors ${
                    isSelected ? "outline outline-1 outline-offset-1 shadow-lg" : "hover:opacity-90"
                  }`}
                >
                  <div className="absolute inset-x-2 top-2 pointer-events-none overflow-hidden">
                    <span
                      className="text-[9px] font-mono font-semibold truncate leading-tight block"
                      style={{ color: hasPending ? "var(--color-warn)" : color }}
                    >
                      {a.action}{hasPending && " *"}
                    </span>
                  </div>
                  <div className="absolute inset-x-2 bottom-2 pointer-events-none overflow-hidden">
                    <span
                      className="text-[9px] font-mono truncate leading-tight block opacity-70"
                      style={{ color: hasPending ? "var(--color-warn)" : color }}
                    >
                      {a.object}
                    </span>
                  </div>

                  {/* Edge & keyframe handles — only interactive in Edit mode */}
                  {activeTool === "edit" && <>
                    <div
                      className="absolute left-0 top-0 w-2 h-full cursor-ew-resize z-10 flex items-center justify-center group/edge"
                      onMouseDown={(e) => startSegmentDrag(e, a, "start")}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <div className="w-0.5 h-4 rounded-full opacity-50 group-hover/edge:opacity-100 transition-opacity" style={{ backgroundColor: color }} />
                    </div>
                    <div
                      className="absolute right-0 top-0 w-2 h-full cursor-ew-resize z-10 flex items-center justify-center group/edge"
                      onMouseDown={(e) => startSegmentDrag(e, a, "end")}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <div className="w-0.5 h-4 rounded-full opacity-50 group-hover/edge:opacity-100 transition-opacity" style={{ backgroundColor: color }} />
                    </div>
                    <div
                      className="absolute top-0 h-full z-10 flex items-center justify-center cursor-col-resize group/kf"
                      style={{ left: `${kfPct}%`, transform: "translateX(-50%)", width: 12 }}
                      onMouseDown={(e) => startSegmentDrag(e, a, "keyframe")}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <div className="w-0.5 h-full opacity-70 group-hover/kf:opacity-100 group-hover/kf:w-1 transition-all" style={{ backgroundColor: color }} />
                      <div className="absolute top-0.5 w-2 h-2 opacity-80" style={{ backgroundColor: color, left: "50%", transform: "translateX(-50%) rotate(45deg)" }} />
                    </div>
                  </>}

                  {/* Keyframe line visible in Select mode too (read-only) */}
                  {activeTool === "select" && (
                    <div
                      className="absolute top-0 w-0.5 h-full opacity-50 pointer-events-none"
                      style={{ left: `${kfPct}%`, backgroundColor: color }}
                    />
                  )}
                </div>
              );
            })}
          </div>

          {/* Playhead */}
          <div
            className="absolute top-0 bottom-0 w-px bg-[var(--color-accent)] z-10 pointer-events-none"
            style={{ left: `${playheadPct}%` }}
          >
            <div className="absolute -top-0 left-1/2 -translate-x-1/2 w-2.5 h-2.5 bg-[var(--color-accent)] rotate-45 -mt-1" />
          </div>

          {/* Empty state */}
          {actions.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-xs text-[var(--color-text-muted)]">No annotations — run detection to populate timeline</span>
            </div>
          )}
        </div>
          );
        })()}
          </div>{/* end inner zoom div */}
        </div>{/* end scrollRef */}

      </div>
    </div>
  );
}

// ─── RIGHT PANEL ──────────────────────────────────────────────────────────────

const ActionEditor = ({
  action,
  pendingAction,
  onChange,
  onClearPending,
  onDraftChange,
}: {
  action: Action;
  pendingAction?: Action;
  onChange: (updated: Action) => void;
  onClearPending?: () => void;
  onDraftChange?: (draft: Action) => void;
}) => {
  // Base is the last committed state; pending (timeline drag) overlays it
  const base = pendingAction ?? action;
  const [local, setLocal] = useState(base);

  useEffect(() => { setLocal(pendingAction ?? action); }, [action, pendingAction]);

  function update(patch: Partial<Action>) {
    setLocal((prev) => ({ ...prev, ...patch }));
    onDraftChange?.({ ...local, ...patch });
  }

  // True when local state differs from the committed action in any field
  const hasChanges =
    local.action !== action.action ||
    local.object !== action.object ||
    local.start_ms !== action.start_ms ||
    local.end_ms !== action.end_ms ||
    local.keyframe_ms !== action.keyframe_ms ||
    local.confidence !== action.confidence ||
    !!pendingAction;

  const color = actionColor(local.action);

  return (
    <div className="p-3 space-y-3">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-2 h-2 rounded-full flex-none" style={{ backgroundColor: color }} />
        <span className="text-xs font-semibold text-[var(--color-text)]">Action Editor</span>
        {hasChanges && (
          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded border border-[var(--color-warn)] text-[var(--color-warn)]">
            UNSAVED
          </span>
        )}
        <span className="ml-auto text-[10px] font-mono text-[var(--color-text-muted)]">{local.id}</span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <FieldInput label="Action" value={local.action} onChange={(v) => update({ action: v })} />
        <FieldInput label="Object" value={local.object ?? ""} onChange={(v) => update({ object: v.trim() ? v : null })} />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <FieldInput label="Start (ms)" value={String(local.start_ms)} type="number" onChange={(v) => update({ start_ms: Number(v) })} />
        <FieldInput label="End (ms)" value={String(local.end_ms)} type="number" onChange={(v) => update({ end_ms: Number(v) })} />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <FieldInput label="Keyframe (ms)" value={String(local.keyframe_ms)} type="number" onChange={(v) => update({ keyframe_ms: Number(v) })} />
        <FieldInput label="Confidence" value={String(local.confidence)} type="number" onChange={(v) => update({ confidence: Number(v) })} />
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex justify-between text-[10px] font-mono mb-1">
          <span className="text-[var(--color-text-muted)]">Confidence</span>
          <span style={{ color }}>{confidencePct(local.confidence)}</span>
        </div>
        <div className="h-1 bg-[var(--color-border)] rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{ width: `${local.confidence * 100}%`, backgroundColor: color }}
          />
        </div>
      </div>

      <div className="text-[10px] font-mono text-[var(--color-text-muted)]">
        Duration: <span className="text-[var(--color-text-dim)]">{msToTimecode(local.end_ms - local.start_ms)}</span>
        &nbsp;·&nbsp; Model: <span className="text-[var(--color-text-dim)]">{local.model_version}</span>
      </div>

      <div className="space-y-1.5">
        <button
          disabled={!hasChanges}
          onClick={() => { onChange(local); onClearPending?.(); }}
          className={`w-full h-8 rounded text-xs font-semibold transition-colors ${
            hasChanges
              ? "bg-[var(--color-accent)] text-[var(--color-bg)] hover:bg-[var(--color-accent-dim)] cursor-pointer"
              : "bg-[var(--color-border)] text-[var(--color-text-muted)] cursor-not-allowed opacity-50"
          }`}
        >
          Apply Changes
        </button>
        {hasChanges && (
          <button
            onClick={() => { setLocal(action); onClearPending?.(); }}
            className="w-full h-7 rounded text-xs font-medium border border-[var(--color-border-light)] text-[var(--color-text-muted)] hover:text-[var(--color-danger)] hover:border-[var(--color-danger)] transition-colors"
          >
            Cancel Changes
          </button>
        )}
      </div>
    </div>
  );
}

function FieldInput({
  label, value, onChange, type = "text",
}: { label: string; value: string; onChange: (v: string) => void; type?: string }) {
  return (
    <div>
      <label className="block text-[10px] text-[var(--color-text-muted)] mb-1 tracking-wider uppercase">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full h-7 bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 text-xs font-mono text-[var(--color-text)] focus:outline-none focus:border-[var(--color-accent)] transition-colors"
      />
    </div>
  );
}

function RightPanel({
  actions,
  selectedActionId,
  pendingEdits,
  onSelectAction,
  onUpdateAction,
  onClearPending,
  onSetPendingEdits,
  onAddAction,
  onDeleteAction,
}: {
  actions: Action[];
  selectedActionId: string | null;
  pendingEdits: Record<string, Action>;
  onSelectAction: (id: string | null) => void;
  onUpdateAction: (a: Action) => void;
  onClearPending: (id: string) => void;
  onSetPendingEdits: React.Dispatch<React.SetStateAction<Record<string, Action>>>;
  onAddAction: () => void;
  onDeleteAction: (id: string) => void;
}) {
  const selected = actions.find((a) => a.id === selectedActionId) ?? null;

  return (
    <div className="flex flex-col h-full overflow-hidden bg-[var(--color-panel)] border-l border-[var(--color-border)]">
      {/* Detail editor — fixed height so Actions list below is stable */}
      <div className="flex-none border-b border-[var(--color-border)]">
        <SectionLabel>Action Detail</SectionLabel>
        <div style={{ height: 320 }} className="overflow-y-auto">
          {selected ? (
            <ActionEditor
              action={selected}
              pendingAction={pendingEdits[selected.id]}
              onChange={onUpdateAction}
              onClearPending={() => onClearPending(selected.id)}
              onDraftChange={(draft) => onSetPendingEdits((p) => ({ ...p, [selected.id]: draft }))}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-xs text-[var(--color-text-muted)] text-center p-4">
              Select an action to edit its parameters
            </div>
          )}
        </div>
      </div>

      {/* Action list */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="flex items-center px-3 py-1.5 gap-2">
          <span className="text-[10px] tracking-[0.12em] uppercase font-semibold text-[var(--color-text-muted)]">
            Actions
          </span>
          <div className="flex-1 h-px bg-[var(--color-border)]" />
        </div>

        <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-1">
          {actions.map((a) => {
            const color = actionColor(a.action);
            const isSelected = a.id === selectedActionId;
            return (
              <div
                key={a.id}
                role="button"
                tabIndex={0}
                onClick={() => onSelectAction(isSelected ? null : a.id)}
                onKeyDown={(e) => e.key === "Enter" && onSelectAction(isSelected ? null : a.id)}
                className={`w-full text-left px-2.5 py-2 rounded border transition-all group cursor-pointer ${
                  isSelected
                    ? "border-[var(--color-accent)] bg-[var(--color-accent-glow)]"
                    : "border-[var(--color-border)] hover:border-[var(--color-border-light)] hover:bg-[var(--color-panel-alt)]"
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-1.5 h-1.5 rounded-full flex-none" style={{ backgroundColor: color }} />
                  <span className="text-xs font-semibold text-[var(--color-text)]">{a.action}</span>
                  <span className="text-[10px] text-[var(--color-text-muted)]">→</span>
                  <span className="text-xs text-[var(--color-text-dim)]">{a.object}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); onDeleteAction(a.id); }}
                    className="ml-auto opacity-0 group-hover:opacity-100 text-[var(--color-danger)] text-[10px] transition-opacity hover:text-red-400"
                  >
                    ✕
                  </button>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-[var(--color-text-muted)]">
                    {msToTimecode(a.start_ms)} → {msToTimecode(a.end_ms)}
                  </span>
                  <span className="text-[10px] font-mono" style={{ color }}>
                    {confidencePct(a.confidence)}
                  </span>
                </div>
              </div>
            );
          })}
          {actions.length === 0 && (
            <div className="text-xs text-[var(--color-text-muted)] text-center py-6">
              No actions detected yet
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

// ─── i18n ─────────────────────────────────────────────────────────────────────

type Lang = "en" | "zh" | "ru";

const I18N: Record<Lang, Record<string, string>> = {
  en: {
    appName: "ActionLabel",
    annotate: "Annotate",
    export: "Export JSON",
    exportCsv: "CSV",
    actions: "actions",
    noVideo: "No video selected",
    exportFile: "annotations.json",
  },
  zh: {
    appName: "动作标注",
    annotate: "标注",
    export: "导出 JSON",
    exportCsv: "CSV",
    actions: "个动作",
    noVideo: "未选择视频",
    exportFile: "标注结果.json",
  },
  ru: {
    appName: "ActionLabel",
    annotate: "Разметка",
    export: "Экспорт JSON",
    exportCsv: "CSV",
    actions: "действий",
    noVideo: "Видео не выбрано",
    exportFile: "аннотации.json",
  },
};

// ─── App root ─────────────────────────────────────────────────────────────────

export default function App() {
  const [videos, setVideos] = useState<VideoRecord[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [selectedActionId, setSelectedActionId] = useState<string | null>(null);
  const [dark, setDark] = useState(true);
  const [lang, setLang] = useState<Lang>("en");
  const [pendingEdits, setPendingEdits] = useState<Record<string, Action>>({});

  const t = I18N[lang];
  const selectedVideo = videos.find((v) => v.id === selectedVideoId) ?? null;
  const actions = selectedVideo?.actions ?? [];

  useEffect(() => {
    document.documentElement.classList.toggle("light-theme", !dark);
  }, [dark]);

  const videosRef = useRef(videos);
  videosRef.current = videos;
  const pollingKey = videos
    .filter((v) => v.jobId && (v.status === "processing" || v.status === "pending"))
    .map((v) => v.jobId)
    .sort()
    .join(",");

  useEffect(() => {
    if (!pollingKey) return;
    let cancelled = false;
    const jobIds = pollingKey.split(",");
    async function poll() {
      for (const jobId of jobIds) {
        const video = videosRef.current.find((row) => row.jobId === jobId);
        if (!video) continue;
        try {
          const job = await getJob(jobId);
          if (cancelled) return;
          const mapped: VideoRecord["status"] =
            job.status === "completed" ? "done" : job.status === "error" ? "error" : "processing";
          let nextActions = video.actions;
          let duration = video.duration_ms;
          if (job.status === "completed") {
            const annotation = await getAnnotation(job.video_id);
            nextActions = annotation.segments.map(toUiAction);
            duration = annotation.duration_ms || duration;
          }
          setVideos((vs) =>
            vs.map((row) =>
              row.jobId === jobId
                ? {
                    ...row,
                    id: job.video_id,
                    status: mapped,
                    progress: job.progress,
                    error: job.error || undefined,
                    actions: mapped === "done" ? nextActions : row.actions,
                    duration_ms: duration,
                  }
                : row
            )
          );
          setSelectedVideoId((current) => (current === video.id ? job.video_id : current));
        } catch (err) {
          if (cancelled) return;
          setVideos((vs) =>
            vs.map((row) =>
              row.jobId === jobId
                ? { ...row, status: "error", error: err instanceof Error ? err.message : String(err) }
                : row
            )
          );
        }
      }
    }
    poll();
    const timer = window.setInterval(poll, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [pollingKey]);

  function updateActions(id: string, updated: Action[]) {
    setVideos((vs) => vs.map((v) => (v.id === id ? { ...v, actions: updated } : v)));
  }

  async function persistActions(video: VideoRecord, next: Action[]) {
    if (!video.jobId || video.status === "staged" || video.status === "processing") return;
    try {
      await putAnnotation(video.id, next);
    } catch {
      // keep local edits even if the backend is briefly unavailable
    }
  }

  function handleUpdateAction(updated: Action) {
    if (!selectedVideoId || !selectedVideo) return;
    const next = actions.map((a) => (a.id === updated.id ? updated : a));
    updateActions(selectedVideoId, next);
    void persistActions(selectedVideo, next);
  }

  function handleAddAction(atMs?: number) {
    if (!selectedVideoId || !selectedVideo) return;
    const start = atMs ?? (actions[actions.length - 1]?.end_ms ?? 0) + 500;
    const newAction: Action = {
      id: "a" + Date.now(),
      start_ms: start,
      end_ms: start + 2000,
      action: "unknown",
      object: null,
      keyframe_ms: start + 1000,
      confidence: 1.0,
      model_version: "manual",
    };
    const next = [...actions, newAction];
    updateActions(selectedVideoId, next);
    setSelectedActionId(newAction.id);
    void persistActions(selectedVideo, next);
  }

  function handleDeleteAction(id: string) {
    if (!selectedVideoId || !selectedVideo) return;
    const next = actions.filter((a) => a.id !== id);
    updateActions(selectedVideoId, next);
    if (selectedActionId === id) setSelectedActionId(null);
    void persistActions(selectedVideo, next);
  }

  function handleAddVideo(v: VideoRecord, select = true) {
    setVideos((vs) => [v, ...vs]);
    if (select) {
      setSelectedVideoId(v.id);
      setSelectedActionId(null);
    }
  }

  async function handleSubmitPending(payload: { rulesJson: string; model: string }) {
    const staged = videos.filter((v) => v.status === "staged" && v.file);
    for (const video of staged) {
      try {
        setVideos((vs) => vs.map((row) => (row.id === video.id ? { ...row, status: "processing", progress: 5 } : row)));
        const uploaded = await uploadVideo(video.file as File, payload.rulesJson, payload.model);
        setVideos((vs) =>
          vs.map((row) =>
            row.id === video.id
              ? {
                  ...row,
                  id: uploaded.video_id,
                  jobId: uploaded.job_id,
                  status: "processing",
                  progress: 10,
                }
              : row
          )
        );
        setSelectedVideoId((current) => (current === video.id ? uploaded.video_id : current));
      } catch (err) {
        setVideos((vs) =>
          vs.map((row) =>
            row.id === video.id
              ? { ...row, status: "error", error: err instanceof Error ? err.message : String(err) }
              : row
          )
        );
      }
    }
  }

  function handleClearStaged() {
    const staged = videos.filter((v) => v.status === "staged");
    const stagedIds = new Set(staged.map((v) => v.id));
    staged.forEach((v) => v.previewUrl && URL.revokeObjectURL(v.previewUrl));
    setVideos((vs) => vs.filter((v) => !stagedIds.has(v.id)));
    if (selectedVideoId && stagedIds.has(selectedVideoId)) {
      const remaining = videos.filter((v) => !stagedIds.has(v.id));
      setSelectedVideoId(remaining[0]?.id ?? null);
      setSelectedActionId(null);
    }
  }

  function handleDeleteVideo(id: string) {
    const victim = videos.find((v) => v.id === id);
    if (victim?.previewUrl) URL.revokeObjectURL(victim.previewUrl);
    setVideos((vs) => vs.filter((v) => v.id !== id));
    if (selectedVideoId === id) {
      const remaining = videos.filter((v) => v.id !== id);
      setSelectedVideoId(remaining[0]?.id ?? null);
      setSelectedActionId(null);
    }
  }

  async function handleExport(format: "json" | "csv") {
    if (selectedVideo && selectedVideo.status !== "staged" && selectedVideo.jobId) {
      await persistActions(selectedVideo, actions);
      await downloadExport(selectedVideo.id, format);
      return;
    }
    if (format === "csv") {
      const header = "id,start_ms,end_ms,action,object,keyframe_ms,confidence,model_version";
      const lines = actions.map((a) =>
        [a.id, a.start_ms, a.end_ms, a.action, a.object ?? "", a.keyframe_ms, a.confidence, a.model_version].join(",")
      );
      const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "actions.csv";
      a.click();
      URL.revokeObjectURL(url);
      return;
    }
    const blob = new Blob([JSON.stringify(actions, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = t.exportFile;
    a.click();
    URL.revokeObjectURL(url);
  }

  const LANGS: { id: Lang; label: string }[] = [
    { id: "en", label: "EN" },
    { id: "zh", label: "中" },
    { id: "ru", label: "RU" },
  ];

  return (
    <div className="h-full flex flex-col bg-[var(--color-bg)] overflow-hidden">
      {/* Header */}
      <header className="flex-none h-11 flex items-center px-4 border-b border-[var(--color-border)] bg-[var(--color-panel)] z-10">

        {/* Left: logo + nav */}
        <div className="flex items-center gap-3 flex-none">
          <div className="flex items-center gap-2.5">
            <div className="w-5 h-5 rounded bg-[var(--color-accent)] flex items-center justify-center flex-none">
              <svg width="11" height="11" viewBox="0 0 12 12" fill="var(--color-bg)">
                <rect x="1" y="1" width="4" height="4" rx="0.5" />
                <rect x="7" y="1" width="4" height="4" rx="0.5" />
                <rect x="1" y="7" width="4" height="4" rx="0.5" />
                <rect x="7" y="7" width="4" height="4" rx="0.5" />
              </svg>
            </div>
            <span className="text-sm font-semibold text-[var(--color-text)] tracking-tight whitespace-nowrap">{t.appName}</span>
            <span className="text-[10px] font-mono text-[var(--color-text-muted)] border border-[var(--color-border)] px-1.5 py-0.5 rounded">v0.1</span>
          </div>

          <div className="w-px h-5 bg-[var(--color-border)]" />

          <button className="px-3 h-7 rounded text-xs font-medium bg-[var(--color-accent-glow)] text-[var(--color-accent)]">
            {t.annotate}
          </button>
        </div>

        {/* Center: video info */}
        <div className="flex-1 flex items-center justify-center gap-3 min-w-0 px-4">
          {selectedVideo ? (
            <>
              <span className="text-xs font-medium text-[var(--color-text)] truncate max-w-64">{selectedVideo.name}</span>
              <StatusBadge status={selectedVideo.status} />
              <div className="w-px h-4 bg-[var(--color-border)]" />
              <span className="text-[11px] font-mono text-[var(--color-accent)] whitespace-nowrap">
                {actions.length} {t.actions}
              </span>
            </>
          ) : (
            <span className="text-xs text-[var(--color-text-muted)]">{t.noVideo}</span>
          )}
        </div>

        {/* Right: theme, lang, export */}
        <div className="flex items-center gap-2 flex-none">
          {/* Theme toggle */}
          <button
            onClick={() => setDark(!dark)}
            title={dark ? "Switch to light" : "Switch to dark"}
            className="w-7 h-7 rounded flex items-center justify-center text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-panel-alt)] transition-colors"
          >
            {dark ? <SunIcon /> : <MoonIcon />}
          </button>

          <div className="w-px h-5 bg-[var(--color-border)]" />

          {/* Language switcher */}
          <div className="flex items-center gap-0.5">
            {LANGS.map(({ id, label }) => (
              <button
                key={id}
                onClick={() => setLang(id)}
                className={`w-7 h-7 rounded text-[10px] font-semibold transition-colors ${
                  lang === id
                    ? "bg-[var(--color-accent-glow)] text-[var(--color-accent)]"
                    : "text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-panel-alt)]"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="w-px h-5 bg-[var(--color-border)]" />

          {/* Export */}
          <button
            onClick={() => void handleExport("json")}
            disabled={actions.length === 0}
            className={`h-7 px-3 rounded text-[11px] font-semibold transition-colors flex items-center gap-1.5 ${
              actions.length > 0
                ? "border border-[var(--color-accent)] text-[var(--color-accent)] hover:bg-[var(--color-accent-glow)]"
                : "border border-[var(--color-border)] text-[var(--color-text-muted)] cursor-not-allowed opacity-50"
            }`}
          >
            <ExportIcon />
            {t.export}
          </button>
          <button
            onClick={() => void handleExport("csv")}
            disabled={actions.length === 0}
            className={`h-7 px-3 rounded text-[11px] font-semibold transition-colors ${
              actions.length > 0
                ? "border border-[var(--color-accent)] text-[var(--color-accent)] hover:bg-[var(--color-accent-glow)]"
                : "border border-[var(--color-border)] text-[var(--color-text-muted)] cursor-not-allowed opacity-50"
            }`}
          >
            {t.exportCsv}
          </button>
        </div>
      </header>

      {/* Three-panel main */}
      <main className="flex-1 overflow-hidden grid" style={{ gridTemplateColumns: "23.4% 1fr 23.4%", gridTemplateRows: "100%", alignItems: "stretch" }}>
        <LeftPanel
          videos={videos}
          selectedVideoId={selectedVideoId}
          onSelectVideo={(id) => { setSelectedVideoId(id); setSelectedActionId(null); }}
          onAddVideo={handleAddVideo}
          onSubmitPending={handleSubmitPending}
          onDeleteVideo={handleDeleteVideo}
          onClearStaged={handleClearStaged}
        />

        <CenterPanel
          video={selectedVideo}
          selectedActionId={selectedActionId}
          onSelectAction={setSelectedActionId}
          onUpdateAction={handleUpdateAction}
          onAddAction={handleAddAction}
          onDeleteAction={handleDeleteAction}
          pendingEdits={pendingEdits}
          onSetPendingEdits={setPendingEdits}
          onDurationMs={(ms) => {
            if (!selectedVideoId) return;
            setVideos((vs) => vs.map((v) => (v.id === selectedVideoId && v.duration_ms !== ms ? { ...v, duration_ms: ms } : v)));
          }}
        />

        <RightPanel
          actions={actions}
          selectedActionId={selectedActionId}
          pendingEdits={pendingEdits}
          onSelectAction={setSelectedActionId}
          onUpdateAction={handleUpdateAction}
          onClearPending={(id) => setPendingEdits((p) => { const n = { ...p }; delete n[id]; return n; })}
          onSetPendingEdits={setPendingEdits}
          onAddAction={handleAddAction}
          onDeleteAction={handleDeleteAction}
        />
      </main>
    </div>
  );
}
