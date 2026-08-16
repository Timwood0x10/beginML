import type {
  NotesResponse,
  NoteDetail,
  StatsResponse,
  MapResponse,
} from "./types";
import type { LabModule, LabParams, LabResult } from "./lab/types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  stats: (lang?: string) => {
    const qs = lang ? `?lang=${encodeURIComponent(lang)}` : "";
    return request<StatsResponse>(`/stats${qs}`);
  },
  notes: (params?: {
    category?: string | null;
    search?: string | null;
    lang?: string | null;
  }) => {
    const qs = new URLSearchParams();
    if (params?.lang) qs.set("lang", params.lang);
    if (params?.category) qs.set("category", params.category);
    if (params?.search) qs.set("search", params.search);
    const suffix = qs.toString() ? `?${qs}` : "";
    return request<NotesResponse>(`/notes${suffix}`);
  },
  note: (id: string, lang?: string) => {
    const qs = lang ? `?lang=${encodeURIComponent(lang)}` : "";
    return request<NoteDetail>(`/notes/${id}${qs}`);
  },
  map: (category?: string | null, lang?: string) => {
    const qs = new URLSearchParams();
    if (category) qs.set("category", category);
    if (lang) qs.set("lang", lang);
    const suffix = qs.toString() ? `?${qs}` : "";
    return request<MapResponse>(`/map${suffix}`);
  },
  health: () => request<{ status: string; notes: number }>("/health"),
  lab: {
    modules: () => request<{ modules: LabModule[] }>("/lab/modules"),
    compute: (moduleId: string, params: LabParams) =>
      request<LabResult>(`/lab/compute/${moduleId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params }),
      } as RequestInit),
    /** Python source of a lab module (paper ↔ source ↔ run middle column). */
    source: (moduleId: string) =>
      request<{ module_id: string; filename: string; code: string }>(
        `/lab/source/${moduleId}`,
      ),
    /** Paper PDF parsed into clickable sections. */
    paper: () =>
      request<{
        title: string;
        author: string;
        pages: number;
        sections: {
          id: string;
          level: number;
          title: string;
          text: string;
          math: boolean;
        }[];
      }>("/lab/paper"),
    /** Rendered PDF pages (PNG) + sections with page/bbox for on-page clicking. */
    paperView: () =>
      request<{
        title: string;
        author: string;
        pages: number;
        zoom: number;
        images: {
          page: number;
          image: string;
          width: number;
          height: number;
        }[];
        sections: {
          id: string;
          level: number;
          title: string;
          text: string;
          math: boolean;
          page: number;
          bbox: number[];
        }[];
      }>("/lab/paper/view"),
    /** Numpy implementation for one paper section. */
    paperSource: (sectionId: string) =>
      request<{ id: string; title: string; code: string }>(
        `/lab/paper/source/${sectionId}`,
      ),
    /** Execute one paper section's implementation. */
    paperRun: (sectionId: string, params: LabParams) =>
      request<{ id: string; title: string; result: Record<string, unknown> }>(
        `/lab/paper/run/${sectionId}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ params }),
        } as RequestInit,
      ),
  },
};
