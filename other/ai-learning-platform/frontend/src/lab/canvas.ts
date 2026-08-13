// Small Canvas helpers shared by Math Lab visualizations.

export interface Domain {
  x: [number, number]
  y: [number, number]
}

export interface Scale {
  // data -> canvas pixel
  px: (x: number) => number
  py: (y: number) => number
  // canvas pixel -> data
  invx: (px: number) => number
  invy: (py: number) => number
  w: number
  h: number
}

export function makeScale(
  ctx: CanvasRenderingContext2D,
  domain: Domain,
  padding = { l: 44, r: 16, t: 16, b: 32 },
): Scale {
  // Use CSS pixel dimensions (setupCanvas applied a DPR transform). Fall back
  // to physical/dpr if the canvas isn't laid out yet.
  const dpr = window.devicePixelRatio || 1
  const c = ctx.canvas
  const w = c.clientWidth || c.width / dpr
  const h = c.clientHeight || c.height / dpr
  const plotW = w - padding.l - padding.r
  const plotH = h - padding.t - padding.b
  const [x0, x1] = domain.x
  const [y0, y1] = domain.y
  const px = (x: number) => padding.l + ((x - x0) / (x1 - x0)) * plotW
  const py = (y: number) => padding.t + (1 - (y - y0) / (y1 - y0)) * plotH
  const invx = (p: number) => x0 + ((p - padding.l) / plotW) * (x1 - x0)
  const invy = (p: number) => y0 + (1 - (p - padding.t) / plotH) * (y1 - y0)
  return { px, py, invx, invy, w, h }
}

export function clearCanvas(ctx: CanvasRenderingContext2D, cssVar: string) {
  ctx.save()
  ctx.fillStyle = cssVar
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height)
  ctx.restore()
}

export function drawAxes(
  ctx: CanvasRenderingContext2D,
  s: Scale,
  domain: Domain,
  opts: { color?: string; gridColor?: string } = {},
) {
  const color = opts.color ?? '#7d766d'
  const grid = opts.gridColor ?? 'rgba(125,118,109,0.18)'
  ctx.save()
  ctx.font = '11px Manrope, sans-serif'
  ctx.fillStyle = color
  ctx.strokeStyle = grid
  ctx.lineWidth = 1

  const [x0, x1] = domain.x
  const [y0, y1] = domain.y
  const ticks = 5
  for (let i = 0; i <= ticks; i++) {
    const xv = x0 + (i / ticks) * (x1 - x0)
    const yv = y0 + (i / ticks) * (y1 - y0)
    const cx = s.px(xv)
    const cy = s.py(yv)
    ctx.beginPath()
    ctx.moveTo(cx, s.py(y0))
    ctx.lineTo(cx, s.py(y1))
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(s.px(x0), cy)
    ctx.lineTo(s.px(x1), cy)
    ctx.stroke()
    ctx.fillText(xv.toFixed(1), cx - 10, s.py(y0) + 16)
    ctx.fillText(yv.toFixed(1), s.px(x0) - 36, cy + 4)
  }
  // zero lines
  ctx.strokeStyle = 'rgba(125,118,109,0.4)'
  if (x0 <= 0 && 0 <= x1) {
    ctx.beginPath(); ctx.moveTo(s.px(0), s.py(y0)); ctx.lineTo(s.px(0), s.py(y1)); ctx.stroke()
  }
  if (y0 <= 0 && 0 <= y1) {
    ctx.beginPath(); ctx.moveTo(s.px(x0), s.py(0)); ctx.lineTo(s.px(x1), s.py(0)); ctx.stroke()
  }
  ctx.restore()
}

// A warm "paper" colormap: low = cream, high = clay/brown.
export function warmColor(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t))
  // cream #f2ede3 -> terracotta #b86a4a -> dark brown #4d3a2e
  const stops: [number, [number, number, number]][] = [
    [0.0, [242, 237, 227]],
    [0.55, [200, 150, 110]],
    [1.0, [77, 58, 46]],
  ]
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1]
      const [t1, c1] = stops[i]
      const k = (t - t0) / (t1 - t0)
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * k),
        Math.round(c0[1] + (c1[1] - c0[1]) * k),
        Math.round(c0[2] + (c1[2] - c0[2]) * k),
      ]
    }
  }
  return stops[stops.length - 1][1]
}

/** Set up a canvas for HiDPI rendering and return its 2D context. */
export function setupCanvas(canvas: HTMLCanvasElement, cssW: number, cssH: number): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.round(cssW * dpr)
  canvas.height = Math.round(cssH * dpr)
  canvas.style.width = `${cssW}px`
  canvas.style.height = `${cssH}px`
  const ctx = canvas.getContext('2d')!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return ctx
}
