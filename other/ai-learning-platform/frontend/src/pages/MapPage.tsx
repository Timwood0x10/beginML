import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { MapResponse, MapPoint } from '../types'
import { Spinner, ErrorState } from '../components/States'
import { useI18n } from '../i18n/context'
import { useTheme } from '../hooks/useTheme'

// Natural pigment palette — antique inks from traditional painting.
const CATEGORY_COLORS: Record<string, string> = {
  math: '#2B4C6F',       // lapis lazuli / stone blue
  attention: '#A8382A',  // vermilion / cinnabar red
  hybrid: '#C88A35',     // orpiment / amber yellow
  paper: '#3A6B58',      // malachite / dark turquoise
  general: '#3D322C',    // concentrated ink / dark tea
}

interface ViewBox {
  x: number
  y: number
  w: number
  h: number
}

export default function MapPage() {
  const { t } = useI18n()
  const { theme } = useTheme()
  const dark = theme === 'dark'
  const [data, setData] = useState<MapResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeCat, setActiveCat] = useState<string | null>(null)
  const [hover, setHover] = useState<MapPoint | null>(null)
  const [mouse, setMouse] = useState<{ x: number; y: number } | null>(null)

  // Pan & zoom state (viewBox in data units)
  const [view, setView] = useState<ViewBox>({ x: -1.4, y: -1.3, w: 2.8, h: 2.6 })
  const [dragging, setDragging] = useState(false)
  const dragRef = useRef<{ startX: number; startY: number; vx: number; vy: number; moved: boolean } | null>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  // Canvas pixel size lives in state (not just a ref) so the first data
  // render — which happens before the ref attaches — still recomputes node
  // coordinates once the SVG mounts, and window resizes re-layout the map.
  const [svgSize, setSvgSize] = useState<{ w: number; h: number } | null>(null)
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

  // The node nearest the map centre wears the double hand-drawn ring.
  const centerId = useMemo(() => {
    if (points.length === 0) return null
    let best = points[0]
    let bestDist = Infinity
    for (const p of points) {
      const d = Math.hypot(p.x, p.y)
      if (d < bestDist) {
        bestDist = d
        best = p
      }
    }
    return best.id
  }, [points])

  // Measure the canvas once it exists (it only renders after data loads).
  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const measure = () => {
      const rect = svg.getBoundingClientRect()
      setSvgSize({ w: rect.width, h: rect.height })
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(svg)
    return () => ro.disconnect()
  }, [data])

  // Coordinate transforms — data space [-1.2, 1.2] -> SVG pixel space
  const toSvg = useCallback(
    (px: number, py: number) => {
      if (!svgSize) return { x: 0, y: 0 }
      const x = ((px - view.x) / view.w) * svgSize.w
      const y = ((py - view.y) / view.h) * svgSize.h
      return { x, y }
    },
    [view, svgSize],
  )

  // --- pan handlers ---
  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if ((e.target as Element).closest('[data-point]')) return
    ;(e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId)
    setDragging(true)
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
    setDragging(false)
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

  if (loading) return <Spinner label={t.common.loading} />
  if (error || !data) return <ErrorState message={error ?? t.common.loading} onRetry={load} />

  return (
    <div className="flex flex-col gap-6 pt-2">
      <header className="flex flex-col gap-3">
        <h1 className="font-codex text-headline-lg-mobile md:text-headline-xl text-on-surface dark:text-inverse-on-surface tracking-tight">
          {t.map.title}
        </h1>
        <p className="text-body-lg text-on-surface-variant dark:text-outline max-w-3xl leading-relaxed">
          {t.map.subtitle}
        </p>
      </header>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={resetView}
          className="inline-flex items-center gap-2 px-4 py-2 text-label-md font-semibold text-on-surface dark:text-dark-on-surface hover:bg-[#EAE0D3] dark:hover:bg-[#332B20] transition"
          style={{
            background: '#EAE0D3',
            border: '1px solid #7A6858',
            borderRadius: '2px',
            fontFamily: 'var(--font-map-label)',
          }}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 17 }}>center_focus_strong</span>
          {t.map.reset}
        </button>
        <span className="w-px h-6 bg-outline-variant/60 mx-1" />
        <LegendPill active={!activeCat} color="#8A7A61" label={t.map.all} onClick={() => setActiveCat(null)} count={data.points.length} />
        {data.categories.map((c) => (
          <LegendPill
            key={c.id}
            active={activeCat === c.id}
            color={CATEGORY_COLORS[c.id] ?? '#8a8376'}
            label={t.home.categoryNames[c.id] ?? c.en}
            count={c.count ?? 0}
            onClick={() => setActiveCat(activeCat === c.id ? null : c.id)}
          />
        ))}
      </div>

      {/* Map canvas */}
      <div className="relative bg-surface-container-lowest dark:bg-dark-surface rounded-3xl border border-outline-variant/50 dark:border-white/10 shadow-ambient dark:shadow-dark-ambient overflow-hidden" style={{ height: 'min(70vh, 680px)' }}>
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
            {/* Parchment canvas — warm aged-paper gradient */}
            <linearGradient id="parchment-grad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#F4EAE1" />
              <stop offset="55%" stopColor="#EEDFD0" />
              <stop offset="100%" stopColor="#E8D8C8" />
            </linearGradient>
            <linearGradient id="parchment-grad-dark" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#2A2319" />
              <stop offset="55%" stopColor="#241E16" />
              <stop offset="100%" stopColor="#1E1913" />
            </linearGradient>
            {/* Fibre texture — faint vertical/horizontal paper strands */}
            <pattern id="fibre-pattern" width="160" height="160" patternUnits="userSpaceOnUse">
              <path d="M0 40 C 40 30, 80 50, 160 38" fill="none" stroke="#C9B79F" strokeWidth="0.4" strokeOpacity="0.25" />
              <path d="M0 110 C 60 100, 100 120, 160 108" fill="none" stroke="#C9B79F" strokeWidth="0.3" strokeOpacity="0.2" />
              <path d="M40 0 C 30 40, 50 80, 38 160" fill="none" stroke="#C9B79F" strokeWidth="0.35" strokeOpacity="0.22" />
              <path d="M110 0 C 100 60, 120 100, 108 160" fill="none" stroke="#C9B79F" strokeWidth="0.3" strokeOpacity="0.18" />
            </pattern>
            {/* Vignette — dark corners like a map laid on a table */}
            <radialGradient id="parchment-vignette" cx="50%" cy="50%" r="72%">
              <stop offset="62%" stopColor="#000000" stopOpacity="0" />
              <stop offset="100%" stopColor="#3D322C" stopOpacity="0.22" />
            </radialGradient>
            <radialGradient id="parchment-vignette-dark" cx="50%" cy="50%" r="72%">
              <stop offset="62%" stopColor="#000000" stopOpacity="0" />
              <stop offset="100%" stopColor="#000000" stopOpacity="0.5" />
            </radialGradient>
            {/* Old glow kept for the seal ring only (very subtle) */}
            <radialGradient id="map-glow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="currentColor" stopOpacity="0.12" />
              <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
            </radialGradient>
          </defs>

          {/* Parchment base */}
          <rect x="0" y="0" width="100%" height="100%" fill="url(#parchment-grad)" className="dark:opacity-0" />
          <rect x="0" y="0" width="100%" height="100%" fill="url(#parchment-grad-dark)" className="hidden dark:block" />

          {/* Rhumb lines — compass rose / portolan circles, faint ink */}
          <g className={`map-compass ${dragging ? 'dragging' : ''}`}>
            <g className="text-[#8C7A6B] dark:text-[#8C7A6B]" opacity="0.15">
              {[1, 2, 3, 4, 5].map((k) => (
                <circle
                  key={`rhumb-${k}`}
                  cx="50%"
                  cy="50%"
                  r={`${k * 12}%`}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="0.6"
                  strokeDasharray="3 4"
                />
              ))}
              {Array.from({ length: 16 }).map((_, i) => {
                const angle = (i / 16) * Math.PI * 2
                const len = 62
                return (
                  <line
                    key={`ray-${i}`}
                    x1="50%"
                    y1="50%"
                    x2={`${50 + Math.cos(angle) * len}%`}
                    y2={`${50 + Math.sin(angle) * len}%`}
                    stroke="currentColor"
                    strokeWidth="0.4"
                    strokeDasharray="2 5"
                  />
                )
              })}
              <circle cx="50%" cy="50%" r="1.2%" fill="currentColor" />
            </g>
          </g>

          {/* Fibre texture over the base */}
          <rect x="0" y="0" width="100%" height="100%" fill="url(#fibre-pattern)" className="dark:opacity-0" />

          {/* Vignette on top of everything */}
          <rect x="0" y="0" width="100%" height="100%" fill="url(#parchment-vignette)" className="dark:opacity-0" pointerEvents="none" />
          <rect x="0" y="0" width="100%" height="100%" fill="url(#parchment-vignette-dark)" className="hidden dark:block" pointerEvents="none" />

          {/* Connection lines — hand-drawn ink strokes (portolan style).
              Dark mode uses brighter, thicker, tighter-dashed lines for contrast
              against the dark parchment; light mode uses softer faded ink. */}
          <g>
            {points.map((p, i) => {
              const a = toSvg(p.x, p.y)
              return points.slice(i + 1).map((q) => {
                const dist = Math.hypot(p.x - q.x, p.y - q.y)
                if (dist > 0.5) return null
                const b = toSvg(q.x, q.y)
                const touchesHover = hover && (hover.id === p.id || hover.id === q.id)
                // Dark mode: much higher floor opacity, gentler distance falloff,
                // brighter color, thicker stroke, tighter dash for visibility.
                const floor = dark ? 0.5 : 0.11
                const baseOpacity = Math.max(floor, (dark ? 0.95 : 0.46) - dist * (dark ? 0.5 : 0.4))
                const opacity = touchesHover ? (dark ? 1.0 : 0.98) : hover ? baseOpacity * 0.4 : baseOpacity
                const color = touchesHover
                  ? (dark ? '#FFF8EC' : '#2E241B')
                  : (dark ? '#EDE0C8' : '#45382A')
                const sw = touchesHover ? (dark ? 2.2 : 1.8) : (dark ? 1.8 : 1.25)
                const dash = dark ? '3 1.5' : '4 2.5'
                return (
                  <line
                    key={`${p.id}-${q.id}`}
                    x1={a.x}
                    y1={a.y}
                    x2={b.x}
                    y2={b.y}
                    stroke={color}
                    strokeOpacity={opacity}
                    strokeWidth={sw}
                    strokeDasharray={dash}
                    strokeLinecap="round"
                    className="map-ink-line transition-all duration-300"
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

          {/* Active points — hand-drawn ink dots / wax-seal stamps */}
          {points.map((p) => {
            const { x, y } = toSvg(p.x, p.y)
            const color = CATEGORY_COLORS[p.category] ?? '#3D322C'
            const isHovered = hover?.id === p.id
            const r = 5 + Math.min(8, p.readingTime / 4)
            // The node nearest the map centre gets a double hand-drawn ring.
            const isCenter = p.id === centerId
            // Non-hovered nodes fade when something else is inspected.
            const faded = hover && !isHovered
            return (
              <g
                key={p.id}
                data-point
                className={`map-ink-node cursor-pointer ${faded ? 'map-node-faded' : ''}`}
                style={{ animationDelay: `${(p.readingTime % 10) * 60}ms` }}
                onPointerEnter={() => setHover(p)}
                onPointerLeave={() => setHover((h) => (h?.id === p.id ? null : h))}
                onPointerUp={(e) => {
                  if (dragRef.current?.moved) return
                  e.stopPropagation()
                  navigate(`/note/${p.id}`)
                }}
              >
                {/* faint halo only while hovering (ink dampened on the paper) */}
                {isHovered && (
                  <circle cx={x} cy={y} r={r + 8} fill={color} opacity={0.14} />
                )}
                {/* seal dot — pigment fill with a parchment edge like a cut-out */}
                <circle
                  cx={x}
                  cy={y}
                  r={isHovered ? r + 1.5 : r}
                  fill={color}
                  fillOpacity={isHovered ? 1 : 0.88}
                  stroke="#F4EAE1"
                  strokeWidth={1.4}
                  className="dark:stroke-[#1E1913] transition-all duration-200"
                />
                {/* ink pooling — a deeper pigment core */}
                <circle cx={x} cy={y} r={Math.max(1.6, r * 0.42)} fill={color} fillOpacity={1} />
                {/* double hand-drawn ring on the centre node */}
                {isCenter && (
                  <>
                    <circle cx={x} cy={y} r={r + 4.5} fill="none" stroke={color} strokeWidth={0.8} strokeDasharray="2.5 2" opacity={0.55} />
                    <circle cx={x} cy={y} r={r + 8} fill="none" stroke={color} strokeWidth={0.5} strokeDasharray="1 3.5" opacity={0.3} />
                  </>
                )}
                {/* invisible hit area */}
                <circle cx={x} cy={y} r={Math.max(r + 6, 14)} fill="transparent" />
              </g>
            )
          })}
        </svg>

        {/* Hover tooltip — a scrap of parchment pinned beside the map */}
        {hover && mouse && (
          <div
            className="map-tooltip pointer-events-none absolute z-20 rounded-lg px-4 py-3 max-w-xs"
            style={{ left: mouse.x + 16, top: mouse.y + 16 }}
          >
            <div className="flex items-center gap-2 mb-1">
              <span
                className="w-2 h-2"
                style={{ background: CATEGORY_COLORS[hover.category] ?? '#3D322C', borderRadius: '1px' }}
              />
              <span className="text-caption uppercase tracking-wider opacity-70">{hover.category}</span>
            </div>
            <div className="font-codex text-base font-semibold leading-snug">{hover.title}</div>
            <div className="text-caption opacity-80 mt-1 inline-flex items-center gap-1">
              <span className="material-symbols-outlined" style={{ fontSize: 13 }}>schedule</span>
              {hover.readingTime} {t.common.minutes} · {t.map.open}
            </div>
          </div>
        )}

        {/* Hint */}
        <div className="absolute bottom-4 left-4 bg-surface-container/90 dark:bg-dark-surface-elevated/90 backdrop-blur rounded-full px-4 py-2 text-caption text-on-surface-variant dark:text-outline inline-flex items-center gap-2 border border-outline-variant/50 dark:border-white/10">
          <span className="material-symbols-outlined" style={{ fontSize: 15 }}>drag_pan</span>
          {t.map.hint}
        </div>
      </div>
    </div>
  )
}

function toRoman(n: number): string {
  if (n <= 0 || n >= 4000) return String(n)
  const table: [number, string][] = [
    [1000, 'M'], [900, 'CM'], [500, 'D'], [400, 'CD'],
    [100, 'C'], [90, 'XC'], [50, 'L'], [40, 'XL'],
    [10, 'X'], [9, 'IX'], [5, 'V'], [4, 'IV'], [1, 'I'],
  ]
  let out = ''
  let v = n
  for (const [num, sym] of table) {
    while (v >= num) {
      out += sym
      v -= num
    }
  }
  return out
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
      className={`font-codex inline-flex items-center gap-2 px-3.5 py-1.5 text-label-md font-semibold transition-all ${
        active
          ? 'bg-[#C9B79F] text-[#3D322C] shadow-[0_1px_2px_rgba(61,50,44,0.25)]'
          : 'bg-[#EAE0D3] dark:bg-[#2A2319] text-[#6E5D4F] dark:text-[#C9BCA6] hover:bg-[#E0D3C0] dark:hover:bg-[#332B20]'
      }`}
      style={{
        border: '1px solid #7A6858',
        borderRadius: '2px',
        boxShadow: active
          ? '0 1px 2px rgba(61,50,44,0.25), inset 0 0 0 1px #7A6858'
          : '1px 1px 3px rgba(0,0,0,0.05), inset 0 0 0 1px #C9B79F',
        fontFamily: 'var(--font-map-label)',
      }}
    >
      <span
        className="w-2 h-2"
        style={{ background: active ? '#3D322C' : color, borderRadius: '1px' }}
      />
      {label}
      <span className={`px-1.5 py-0.5 text-caption ${active ? 'bg-white/30' : 'bg-[#D8C8B0] dark:bg-white/10'}`}
        style={{ borderRadius: '1px', fontFamily: 'var(--font-map-label)' }}>
        {count}
      </span>
    </button>
  )
}
