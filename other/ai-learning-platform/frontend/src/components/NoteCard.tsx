import { Link } from 'react-router-dom'
import type { Note } from '../types'
import CategoryBadge from './CategoryBadge'

export default function NoteCard({ note }: { note: Note }) {
  return (
    <Link
      to={`/note/${note.id}`}
      className="group block bg-surface-container-lowest dark:bg-dark-surface-elevated rounded-3xl p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/50 dark:border-white/5 hover:-translate-y-1 hover:shadow-ambient-lg transition-all duration-300"
    >
      <div className="flex items-center justify-between gap-3 mb-4">
        <CategoryBadge category={note.category} />
        <span className="inline-flex items-center gap-1 text-caption text-on-surface-variant dark:text-outline">
          <span className="material-symbols-outlined" style={{ fontSize: 14 }}>schedule</span>
          {note.readingTime} min
        </span>
      </div>

      <h3 className="font-headline text-headline-lg-mobile text-on-surface dark:text-inverse-on-surface mb-2 group-hover:text-primary dark:group-hover:text-inverse-primary transition-colors leading-snug">
        {note.title}
      </h3>
      <p className="text-body-md text-on-surface-variant dark:text-outline line-clamp-3 mb-4 leading-relaxed">
        {note.description || 'A curated note in this collection.'}
      </p>

      <div className="flex items-center justify-between pt-3 border-t border-outline-variant/50 dark:border-white/5">
        <span className="text-caption text-on-surface-variant dark:text-outline">
          {note.wordCount.toLocaleString()} words
        </span>
        <span className="inline-flex items-center gap-1 text-label-md text-primary dark:text-inverse-primary font-semibold opacity-0 group-hover:opacity-100 -translate-x-1 group-hover:translate-x-0 transition-all">
          Read
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>arrow_forward</span>
        </span>
      </div>

      {typeof note.score === 'number' && (
        <div className="mt-3">
          <div className="flex items-center justify-between text-caption text-outline mb-1">
            <span>Relevance</span>
            <span>{Math.round(note.score * 100)}%</span>
          </div>
          <div className="w-full h-1 bg-surface-variant dark:bg-white/10 rounded-full overflow-hidden">
            <div className="h-full bg-primary dark:bg-inverse-primary rounded-full" style={{ width: `${Math.min(100, note.score * 100)}%` }} />
          </div>
        </div>
      )}
    </Link>
  )
}
