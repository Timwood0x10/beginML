import type {
  NotesResponse,
  NoteDetail,
  StatsResponse,
  MapResponse,
} from './types'
import type { LabModule, LabParams, LabResult } from './lab/types'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  stats: (lang?: string) => {
    const qs = lang ? `?lang=${encodeURIComponent(lang)}` : ''
    return request<StatsResponse>(`/stats${qs}`)
  },
  notes: (params?: { category?: string | null; search?: string | null; lang?: string | null }) => {
    const qs = new URLSearchParams()
    if (params?.lang) qs.set('lang', params.lang)
    if (params?.category) qs.set('category', params.category)
    if (params?.search) qs.set('search', params.search)
    const suffix = qs.toString() ? `?${qs}` : ''
    return request<NotesResponse>(`/notes${suffix}`)
  },
  note: (id: string, lang?: string) => {
    const qs = lang ? `?lang=${encodeURIComponent(lang)}` : ''
    return request<NoteDetail>(`/notes/${id}${qs}`)
  },
  map: (category?: string | null, lang?: string) => {
    const qs = new URLSearchParams()
    if (category) qs.set('category', category)
    if (lang) qs.set('lang', lang)
    const suffix = qs.toString() ? `?${qs}` : ''
    return request<MapResponse>(`/map${suffix}`)
  },
  health: () => request<{ status: string; notes: number }>('/health'),
  lab: {
    modules: () => request<{ modules: LabModule[] }>('/lab/modules'),
    compute: (moduleId: string, params: LabParams) =>
      request<LabResult>(`/lab/compute/${moduleId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params }),
      } as RequestInit),
  },
}
