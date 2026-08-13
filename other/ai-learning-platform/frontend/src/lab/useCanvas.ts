import { useEffect, useRef } from 'react'
import { setupCanvas } from './canvas'

export interface CanvasHandle {
  ctx: CanvasRenderingContext2D
  width: number
  height: number
}

type DrawFn = (handle: CanvasHandle, time: number) => void

/**
 * Manages a responsive HiDPI canvas and runs an optional rAF animation loop.
 *
 * Pass `animate: false` (or omit) to draw once whenever `deps` change.
 * Pass `animate: true` to call `draw` every frame (~60fps).
 * The callback receives a stable ref-based draw function so deps can be
 * tracked without restarting the loop on every state change.
 */
export function useCanvas(
  canvasRef: React.RefObject<HTMLCanvasElement>,
  draw: DrawFn,
  deps: unknown[],
  opts: { width?: number; height?: number; animate?: boolean } = {},
) {
  const width = opts.width ?? 620
  const height = opts.height ?? 420
  const animate = opts.animate ?? false

  // Keep the latest draw in a ref so the animation loop never goes stale
  // without needing to be torn down and recreated.
  const drawRef = useRef(draw)
  drawRef.current = draw

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = setupCanvas(canvas, width, height)
    let raf = 0
    let alive = true

    const render = (time: number) => {
      if (!alive) return
      drawRef.current({ ctx, width, height }, time)
      if (animate) raf = requestAnimationFrame(render)
    }

    if (animate) {
      raf = requestAnimationFrame(render)
    } else {
      render(0)
    }
    return () => {
      alive = false
      if (raf) cancelAnimationFrame(raf)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canvasRef, width, height, animate, ...deps])
}
