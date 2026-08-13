import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { MapResponse, MapPoint } from '../types'
import { Spinner, ErrorState } from '../components/States'

const CATEGORY_COLORS: Record<string, string> = {
  math: '#C8604A',
  attention: '#5B6BB0',
  hybrid: '#8B6BB0',
  paper: '#4A90A0',
  general: '#8a8376',
}

interface ViewBox {
  x: number
  y: number
  w: number
  h: number
}

export default function MapPage() {
  const [data, setData] = useState<MapResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeCat, setActiveCat] = useState<string | null>(null)
  const [hover, setHover] = useState<MapPoint | null>(null)
  const [mouse, setMouse] = useState<{ x: number; y: number } | null>(null)

  // Pan & zoom state (viewBox in data units)
  const [view, setView] = useState<ViewBox>({ x: -1.4, y: -1.3, w: 2.8, h: 2.6 })
  const dragRef = useRef<{ startX: number; startY: number; vx: number; vy: number; moved: boolean } | null>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const navigate = useNavigate()

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.map(null)
      setData(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const points = useMemo(() => {
    if (!data) return []
    return activeCat ? data.points.filter((p) => p.category === activeCat) : data.points
  }, [data, activeCat])

  const dimmedPoints = useMemo(() => {
    if (!data || !activeCat) return []
    return data.points.filter((p) => p.category !== activeCat)
  }, [data, activeCat])

  // Coordinate transforms — data space [-1.2, 1.2] -> SVG pixel space
  const toSvg = useCallback(
    (px: number, py: number) => {
      const svg = svgRef.current
      if (!svg) return { x: 0, y: 0 }
      const rect = svg.getBoundingClientRect()
      const x = ((px - view.x) / view.w) * rect.width
      const y = ((py - view.y) / view.h) * rect.height
      return { x, y }
    },
    [view],
  )

  // --- pan handlers ---
  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if ((e.target as Element).closest('[data-point]')) return
    ;(e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId)
    dragRef.current = { startX: e.clientX, startY: e.clientY, vx: view.x, vy: view.y, moved: false }
  }
  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current
    if (svg) {
      const rect = svg.getBoundingClientRect()
      setMouse({ x: e.clientX - rect.left, y: e.clientY - rect.top })
    }
    const drag = dragRef.current
    if (!drag || !svgRef.current) return
    if (Math.abs(e.clientX - drag.startX) > 3 || Math.abs(e.clientY - drag.startY) > 3) {
      drag.moved = true
    }
    const rect = svgRef.current.getBoundingClientRect()
    const dx = ((e.clientX - drag.startX) / rect.width) * view.w
    const dy = ((e.clientY - drag.startY) / rect.height) * view.h
    setView((v) => ({ ...v, x: drag.vx - dx, y: drag.vy - dy }))
  }
  const onPointerUp = (e: React.PointerEvent<SVGSVGElement>) => {
    dragRef.current = null
    try {
      ;(e.currentTarget as SVGSVGElement).releasePointerCapture(e.pointerId)
    } catch {
      /* ignore */
    }
  }
  const onWheel = (e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault()
    if (!svgRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    const mx = (e.clientX - rect.left) / rect.width
    const my = (e.clientY - rect.top) / rect.height
    const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15
    setView((v) => {
      const nw = Math.min(6, Math.max(0.4, v.w * factor))
      const nh = Math.min(6, Math.max(0.4, v.h * factor))
      // keep point under cursor anchored
      const dataX = v.x + mx * v.w
      const dataY = v.y + my * v.h
      return { x: dataX - mx * nw, y: dataY - my * nh, w: nw, h: nh }
    })
  }

  const resetView = () => setView({ x: -1.4, y: -1.3, w: 2.8, h: 2.6 })

  if (loading) return <Spinner label="Computing semantic map (MDS)…" />
  if (error || !data) return <ErrorState message={error ?? 'Failed to load map.'} onRetry={load} />

  return (
    <div className="flex flex-col gap-6 pt-2">
      <header className="flex flex-col gap-3">
        <h1 className="font-headline text-headline-lg-mobile md:text-headline-xl text-on-surface dark:text-inverse-on-surface">
          Knowledge map
        </h1>
        <p className="text-body-lg text-on-surface-variant dark:text-outline max-w-3xl leading-relaxed">
          Every note is placed by{' '}
          <span className="text-primary dark:text-inverse-primary font-semibold">multi-dimensional scaling</span>{' '}
          of its TF-IDF vector — nearby notes discuss similar ideas. Drag to pan, scroll to zoom, click a node to read.
        </p>
      </header>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={resetView}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-surface-container dark:bg-dark-surface-elevated border border-outline-variant/60 dark:border-white/10 text-label-md font-semibold text-on-surface dark:text-dark-on-surface hover:border-primary/40 transition"
        >
          <span className="material-symbols-outlined" style={{ fontSize: 17 }}>center_focus_strong</span>
          Reset view
        </button>
        <span className="w-px h-6 bg-outline-variant/60 mx-1" />
        <LegendPill active={!activeCat} color="#7d766d" label="All" onClick={() => setActiveCat(null)} count={data.points.length} />
        {data.categories.map((c) => (
          <LegendPill
            key={c.id}
            active={activeCat === c.id}
            color={CATEGORY_COLORS[c.id] ?? '#8a8376'}
            label={c.en}
            count={c.count ?? 0}
            onClick={() => setActiveCat(activeCat === c.id ? null : c.id)}
          />
        ))}
      </div>

      {/* Map canvas */}
      <div className="relative bg-surface-container-lowest dark:bg-dark-surface rounded-3xl border border-outline-variant/50 dark:border-white/5 shadow-ambient dark:shadow-dark-ambient overflow-hidden" style={{ height: 'min(70vh, 680px)' }}>
        <svg
          ref={svgRef}
          className="w-full h-full touch-none cursor-grab active:cursor-grabbing"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={onPointerUp}
          onWheel={onWheel}
        >
          <defs>
            <pattern id="map-grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="0.5" className="text-outline-variant/40 dark:text-white/5" />
            </pattern>
            <radialGradient id="map-glow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="currentColor" stopOpacity="0.15" />
              <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
            </radialGradient>
          </defs>

          <rect x="0" y="0" width="100%" height="100%" fill="url(#map-grid)" />

          {/* Connection lines between very-similar notes */}
          <g>
            {points.map((p, i) => {
              const a = toSvg(p.x, p.y)
              return points.slice(i + 1).map((q) => {
                const dist = Math.hypot(p.x - q.x, p.y - q.y)
                if (dist > 0.45) return null
                const b = toSvg(q.x, q.y)
                const opacity = Math.max(0.05, 0.35 - dist * 0.6)
                return (
                  <line
                    key={`${p.id}-${q.id}`}
                    x1={a.x}
                    y1={a.y}
                    x2={b.x}
                    y2={b.y}
                    stroke={CATEGORY_COLORS[p.category] ?? '#8a8376'}
                    strokeOpacity={opacity}
                    strokeWidth={1}
                  />
                )
              })
            })}
          </g>

          {/* Dimmed (filtered-out) points */}
          {dimmedPoints.map((p) => {
            const { x, y } = toSvg(p.x, p.y)
            return (
              <circle
                key={`d-${p.id}`}
                cx={x}
                cy={y}
                r={3}
                fill={CATEGORY_COLORS[p.category] ?? '#8a8376'}
                opacity={0.18}
              />
            )
          })}

          {/* Active points */}
          {points.map((p) => {
            const { x, y } = toSvg(p.x, p.y)
            const color = CATEGORY_COLORS[p.category] ?? '#8a8376'
            const isHovered = hover?.id === p.id
            const r = 5 + Math.min(8, p.readingTime / 4)
            return (
              <g
                key={p.id}
                data-point
                className="cursor-pointer"
                onPointerEnter={() => setHover(p)}
                onPointerLeave={() => setHover((h) => (h?.id === p.id ? null : h))}
                onPointerUp={(e) => {
                  if (dragRef.current?.moved) return
                  e.stopPropagation()
                  navigate(`/note/${p.id}`)
                }}
              >
                {isHovered && (
                  <circle cx={x} cy={y} r={r + 10} fill={color} opacity={0.18} />
                )}
                <circle
                  cx={x}
                  cy={y}
                  r={isHovered ? r + 2 : r}
                  fill={color}
                  stroke="white"
                  strokeWidth={2}
                  className="dark:stroke-dark-surface transition-all"
                  style={{ filter: isHovered ? `drop-shadow(0 4px 8px ${color}66)` : undefined }}
                />
                {/* invisible hit area */}
                <circle cx={x} cy={y} r={Math.max(r + 6, 14)} fill="transparent" />
              </g>
            )
          })}
        </svg>

        {/* Hover tooltip */}
        {hover && mouse && (
          <div
            className="pointer-events-none absolute z-20 bg-inverse-surface text-inverse-on-surface rounded-xl px-4 py-3 shadow-ambient-lg max-w-xs"
            style={{ left: mouse.x + 16, top: mouse.y + 16 }}
          >
            <div className="flex items-center gap-2 mb-1">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ background: CATEGORY_COLORS[hover.category] ?? '#8a8376' }}
              />
              <span className="text-caption uppercase tracking-wider opacity-70">{hover.category}</span>
            </div>
            <div className="font-headline text-base font-semibold leading-snug">{hover.title}</div>
            <div className="text-caption opacity-80 mt-1 inline-flex items-center gap-1">
              <span className="material-symbols-outlined" style={{ fontSize: 13 }}>schedule</span>
              {hover.readingTime} min read · click to open
            </div>
          </div>
        )}

        {/* Hint */}
        <div className="absolute bottom-4 left-4 bg-surface-container/90 dark:bg-dark-surface-elevated/90 backdrop-blur rounded-full px-4 py-2 text-caption text-on-surface-variant dark:text-outline inline-flex items-center gap-2 border border-outline-variant/50 dark:border-white/10">
          <span className="material-symbols-outlined" style={{ fontSize: 15 }}>drag_pan</span>
          Drag to pan · scroll to zoom
        </div>
      </div>
    </div>
  )
}

function LegendPill({
  active,
  color,
  label,
  count,
  onClick,
}: {
  active: boolean
  color: string
  label: string
  count: number
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-label-md font-semibold border transition-all ${
        active
          ? 'border-transparent text-on-surface dark:text-dark-on-surface shadow-sm'
          : 'bg-surface-container-lowest dark:bg-dark-surface-elevated text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/40'
      }`}
      style={active ? { backgroundColor: color, color: '#fff' } : undefined}
    >
      <span
        className="w-2.5 h-2.5 rounded-full"
        style={{ background: active ? 'rgba(255,255,255,0.85)' : color }}
      />
      {label}
      <span className={`px-1.5 py-0.5 rounded-full text-caption ${active ? 'bg-white/20' : 'bg-surface-variant dark:bg-white/10'}`}>
        {count}
      </span>
    </button>
  )
}
