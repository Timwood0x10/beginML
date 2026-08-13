import { useCallback, useRef, useState } from 'react'
import type { Scale } from './canvas'

export interface HoverPoint { x: number; y: number }

export interface UseCanvasDragOptions {
  getScale: () => Scale | null
  onDown?: (x: number, y: number, e: React.PointerEvent) => boolean | void
  onDrag?: (x: number, y: number) => void
  onHover?: (x: number, y: number) => void
  onUp?: (x: number, y: number, moved: boolean) => void
  threshold?: number
}

function toData(e: React.PointerEvent<HTMLCanvasElement>, scale: Scale) {
  const rect = e.currentTarget.getBoundingClientRect()
  return {
    x: scale.invx(e.clientX - rect.left),
    y: scale.invy(e.clientY - rect.top),
  }
}

/**
 * Reusable pointer interaction for math-lab canvases.
 * Converts pointer events to data coordinates, distinguishes clicks from
 * drags by pixel threshold, and tracks hover position.
 */
export function useCanvasDrag(opts: UseCanvasDragOptions) {
  const ref = useRef(opts)
  ref.current = opts
  const [hover, setHover] = useState<HoverPoint | null>(null)
  const [dragging, setDragging] = useState(false)
  const down = useRef<{ px: number; py: number; dataX: number; dataY: number } | null>(null)
  const moved = useRef(false)

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const scale = ref.current.getScale()
    if (!scale) return
    const { x, y } = toData(e, scale)
    down.current = { px: e.clientX, py: e.clientY, dataX: x, dataY: y }
    moved.current = false
    const start = ref.current.onDown?.(x, y, e)
    if (start !== false) e.currentTarget.setPointerCapture(e.pointerId)
  }, [])

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const scale = ref.current.getScale()
    if (!scale) return
    const { x, y } = toData(e, scale)
    if (down.current) {
      const dist = Math.hypot(e.clientX - down.current.px, e.clientY - down.current.py)
      if (!moved.current && dist > (ref.current.threshold ?? 4)) moved.current = true
      if (moved.current) {
        setDragging(true)
        ref.current.onDrag?.(x, y)
      }
    } else {
      setHover({ x, y })
      ref.current.onHover?.(x, y)
    }
  }, [])

  const end = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const scale = ref.current.getScale()
    const wasMoved = moved.current
    const start = down.current
    down.current = null
    moved.current = false
    setDragging(false)
    if (scale && start) {
      const { x, y } = toData(e, scale)
      ref.current.onUp?.(x, y, wasMoved)
    }
    try { e.currentTarget.releasePointerCapture(e.pointerId) } catch { /* noop */ }
  }, [])

  const onPointerLeave = useCallback(() => {
    setHover(null)
    down.current = null
    moved.current = false
    setDragging(false)
  }, [])

  return {
    handlers: { onPointerDown, onPointerMove, onPointerUp: end, onPointerLeave },
    hover,
    dragging,
  }
}
