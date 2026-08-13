// Page rasterizer for the 3D flipbook.
//
// The note pages are HTML strings that can contain server-rendered MathML.
// Libraries like html2canvas cannot rasterize MathML, so instead we snapshot
// each page through an SVG <foreignObject>, which is painted by the browser's
// real layout engine (MathML included), then draw the SVG into a canvas that
// becomes a Three.js texture.

let rawCssPromise: Promise<string> | null = null

function flattenRules(rules: CSSRuleList): string {
  let out = ''
  for (let i = 0; i < rules.length; i++) {
    const rule = rules[i]
    try {
      if (rule instanceof CSSMediaRule) {
        out += `@media ${rule.conditionText} { ${flattenRules(rule.cssRules)} }`
      } else {
        out += rule.cssText + '\n'
      }
    } catch {
      /* unreadable rule — skip */
    }
  }
  return out
}

function collectCss(): string {
  let out = ''
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      out += flattenRules(sheet.cssRules)
    } catch {
      /* cross-origin stylesheet — skip */
    }
  }
  return out
}

function rawCss(): Promise<string> {
  if (!rawCssPromise) {
    // Wait a frame so Vite has injected all <style> tags.
    rawCssPromise = new Promise((resolve) => {
      requestAnimationFrame(() => resolve(collectCss()))
    })
  }
  return rawCssPromise
}

// Make url(...) references absolute so fonts/images resolve inside the SVG
// snapshot (which has no document base URL).
function absolutizeUrls(css: string): string {
  return css.replace(/url\(\s*(['"]?)([^'")]+)\1\s*\)/g, (m, q: string, url: string) => {
    if (/^(data:|https?:|blob:|#|\/\/)/.test(url)) return m
    return `url(${q}${new URL(url, window.location.origin)}${q})`
  })
}

function cssForTheme(css: string, theme: 'light' | 'dark'): string {
  if (theme === 'dark') {
    // `html.dark .x` selectors don't exist inside the snapshot — drop the
    // prefix. Dark rules are defined after their light counterparts in
    // index.css, so they still win.
    return absolutizeUrls(css).replace(/html\.dark\s*/g, '')
  }
  return absolutizeUrls(css)
}

// Relative /assets/... image paths must resolve against the app origin.
function absolutizeImages(html: string): string {
  return html.replace(/<img([^>]*?)\ssrc="([^"]+)"/g, (m, attrs: string, src: string) => {
    if (/^(https?:|data:|blob:|#)/.test(src)) return m
    return `<img${attrs} src="${new URL(src, window.location.origin)}"`
  })
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('failed to load page snapshot'))
    img.src = src
  })
}

const PAGE_PAD = { top: 48, right: 56, bottom: 40, left: 56 }

/**
 * Rasterize one note page (HTML + MathML) to a canvas sized to the book leaf.
 * The padding must mirror BOOK_PAD in NotePage so pagination measurements
 * match the rendered texture exactly.
 */
export async function renderHtmlToCanvas(
  html: string,
  width: number,
  height: number,
  theme: 'light' | 'dark',
): Promise<HTMLCanvasElement> {
  await document.fonts.ready
  const css = cssForTheme(await rawCss(), theme)

  const bg = theme === 'dark' ? '#262017' : '#F4EAE1'
  const fg = theme === 'dark' ? '#C9BCA6' : '#6B5D48'

  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">` +
    `<foreignObject width="100%" height="100%">` +
    `<div xmlns="http://www.w3.org/1999/xhtml" style="width:${width}px;height:${height}px;background:${bg}">` +
    `<style>${css}</style>` +
    `<div class="prose-ailearn" style="width:100%;height:100%;box-sizing:border-box;overflow:hidden;` +
    `padding:${PAGE_PAD.top}px ${PAGE_PAD.right}px ${PAGE_PAD.bottom}px ${PAGE_PAD.left}px;` +
    `background:${bg};color:${fg}">${absolutizeImages(html)}</div>` +
    `</div></foreignObject></svg>`

  const img = await loadImage('data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg))
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(width * dpr)
  canvas.height = Math.round(height * dpr)
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('2d context unavailable')
  ctx.scale(dpr, dpr)
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, width, height)
  ctx.drawImage(img, 0, 0, width, height)
  return canvas
}
