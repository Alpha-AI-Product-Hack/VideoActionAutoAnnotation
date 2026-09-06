export type JobStatusName = "queued" | "processing" | "completed" | "error";

export interface ActionSegment {
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

export interface JobStatus {
  job_id: string;
  video_id: string;
  status: JobStatusName;
  progress: number;
  message: string;
  error?: string | null;
  encoder_id?: string | null;
  segmenter?: string | null;
}

export interface UploadResponse {
  video_id: string;
  job_id: string;
  status: string;
  name: string;
}

export interface AnnotationPayload {
  video_id: string;
  duration_ms: number;
  fps: number;
  version: number;
  segments: ActionSegment[];
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (body?.detail) return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    // ignore
  }
  return response.statusText || `HTTP ${response.status}`;
}

export async function uploadVideo(file: File, rulesJson: string, modelVersion: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("rules", rulesJson);
  form.append("model_version", modelVersion);
  const response = await fetch("/api/videos", { method: "POST", body: form });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function getAnnotation(videoId: string): Promise<AnnotationPayload> {
  const response = await fetch(`/api/videos/${videoId}/annotation`);
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function putAnnotation(videoId: string, segments: ActionSegment[]): Promise<AnnotationPayload> {
  const response = await fetch(`/api/videos/${videoId}/annotation`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ segments }),
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export function exportUrl(videoId: string, format: "json" | "csv"): string {
  return `/api/videos/${videoId}/export?format=${format}`;
}

export function videoFileUrl(videoId: string): string {
  return `/api/videos/${videoId}/file`;
}

export async function downloadExport(videoId: string, format: "json" | "csv"): Promise<void> {
  const response = await fetch(exportUrl(videoId, format));
  if (!response.ok) throw new Error(await readError(response));
  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = /filename="([^"]+)"/.exec(disposition);
  const filename = match?.[1] || (format === "csv" ? "actions.csv" : "actions.json");
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
