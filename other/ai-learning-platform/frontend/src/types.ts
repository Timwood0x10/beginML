// Shared API types for the AI Learning Platform backend.

export interface Category {
  id: string
  en: string
  icon: string
  count?: number
}

export interface Heading {
  text: string
  level: number
  slug: string
}

export interface Note {
  id: string
  path: string
  filename: string
  title: string
  description: string
  category: Category
  wordCount: number
  readingTime: number
  headings: Heading[]
  score?: number
}

export interface NoteDetail extends Note {
  content: string
  html: string
  related: Note[]
}

export interface NotesResponse {
  notes: Note[]
  total: number
  categories: Category[]
}

export interface StatsResponse {
  totalNotes: number
  totalWords: number
  totalReadingMinutes: number
  categories: Category[]
  topics: { category: string; keywords: { word: string; weight: number }[] }[]
}

export interface MapPoint {
  id: string
  title: string
  category: string
  x: number
  y: number
  readingTime: number
}

export interface MapResponse {
  points: MapPoint[]
  categories: Category[]
}
