// Book pagination — splits note HTML into fixed-height pages like a codex.
import { useEffect, useRef, useState } from 'react'

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\- ]/g, '')
    .trim()
    .replace(/\s+/g, '-')
}

function prepareNode(el: HTMLElement) {
  if (el.matches('h1, h2, h3, h4, h5, h6') && !el.id) {
    el.id = slugify(el.textContent ?? '')
  }
  if (el.matches('a[href^="http"]')) {
    el.setAttribute('target', '_blank')
    el.setAttribute('rel', 'noopener noreferrer')
  }
  el.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach((h) => {
    const e = h as HTMLElement
    if (!e.id) e.id = slugify(e.textContent ?? '')
  })
  el.querySelectorAll('a[href^="http"]').forEach((a) => {
    a.setAttribute('target', '_blank')
    a.setAttribute('rel', 'noopener noreferrer')
  })
}

const BLOCK_MARGIN = 24 // vertical rhythm between blocks

function blockHeight(el: Element): number {
  // getBoundingClientRect works for MathML elements too (offsetHeight can be
  // undefined on <math> in some browsers); guard against NaN.
  const h = el.getBoundingClientRect().height
  return Number.isFinite(h) ? h : 0
}

export function useBookPagination(html: string, pageHeight: number) {
  const measureRef = useRef<HTMLDivElement | null>(null)
  const [pages, setPages] = useState<string[]>([])

  useEffect(() => {
    const container = measureRef.current
    if (!container || !html) return
    const children = Array.from(container.children) as Element[]

    // Force a synchronous layout so heights reflect real content.
    void container.offsetHeight

    const packed: string[][] = []
    let current: Element[] = []
    let used = 0
    for (const child of children) {
      prepareNode(child as HTMLElement)
      const h = blockHeight(child) + BLOCK_MARGIN
      // A single block taller than the page gets its own leaf that scrolls
      // internally instead of being clipped by the fixed page height.
      if (h > pageHeight) {
        if (current.length > 0) {
          packed.push(current.map((c) => (c as HTMLElement).outerHTML))
          current = []
          used = 0
        }
        packed.push([
          `<div class="book-oversize">${(child as HTMLElement).outerHTML}</div>`,
        ])
        continue
      }
      if (used + h > pageHeight && current.length > 0) {
        packed.push(current.map((c) => (c as HTMLElement).outerHTML))
        current = []
        used = 0
      }
      current.push(child)
      used += h
    }
    if (current.length > 0) {
      packed.push(current.map((c) => (c as HTMLElement).outerHTML))
    }
    setPages(packed.map((page) => page.join('\n')))
  }, [html, pageHeight])

  return { measureRef, pages }
}
