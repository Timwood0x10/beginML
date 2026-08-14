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

// Elements that can be safely wrapped in a scrollable container if too tall.
const SCROLLABLE_TAGS = new Set(['TABLE', 'PRE', 'UL', 'OL', 'IMG'])

function wrapOversizedElement(el: HTMLElement, maxHeight: number): HTMLElement {
  const wrapper = document.createElement('div')
  wrapper.style.maxHeight = `${maxHeight}px`
  wrapper.style.overflowY = 'auto'
  wrapper.style.overflowX = 'auto'
  wrapper.style.margin = '0.8em 0'
  if (el.tagName === 'TABLE') {
    wrapper.style.fontSize = '0.85em'
  }
  if (el.tagName === 'IMG') {
    wrapper.style.textAlign = 'center'
    wrapper.style.display = 'flex'
    wrapper.style.justifyContent = 'center'
  }
  wrapper.appendChild(el.cloneNode(true))
  return wrapper
}

// Measure the height of a single element when placed in an otherwise empty
// container with the same padding and styles as a real page.
function measureSoloHeight(
  container: HTMLElement,
  el: HTMLElement,
  padTop: number,
  padBottom: number,
): number {
  // Save current container contents
  const savedChildren = Array.from(container.children)
  container.replaceChildren()

  const testEl = el.cloneNode(true) as HTMLElement
  container.appendChild(testEl)
  void container.offsetHeight // force reflow
  const height = container.scrollHeight - padTop - padBottom

  container.replaceChildren(...savedChildren)
  return height
}

export function useBookPagination(html: string, pageHeight: number) {
  const measureRef = useRef<HTMLDivElement | null>(null)
  const [pages, setPages] = useState<string[]>([])

  useEffect(() => {
    const container = measureRef.current
    if (!container || !html) {
      setPages([])
      return
    }

    let cancelled = false

    const paginate = () => {
      if (cancelled) return

      // Force layout
      void container.offsetHeight

      const children = Array.from(container.children) as HTMLElement[]
      if (children.length === 0) {
        setPages([])
        return
      }

      // Prepare all nodes (add IDs to headings, target=_blank to external links)
      children.forEach((child) => prepareNode(child))

      // Compute the exact content area height by measuring the container's
      // own padding + border (box-sizing: border-box means these subtract
      // from the CSS height to give content box height).
      const cs = getComputedStyle(container)
      const padTop = parseFloat(cs.paddingTop) || 0
      const padBottom = parseFloat(cs.paddingBottom) || 0
      const borderTop = parseFloat(cs.borderTopWidth) || 0
      const borderBottom = parseFloat(cs.borderBottomWidth) || 0
      // Available content height per page, with a safety margin to account
      // for sub-pixel rounding differences between measure and render.
      const contentHeight = pageHeight - padTop - padBottom - borderTop - borderBottom - 4

      // Save original children so we can restore the DOM when done.
      const originalOrder = [...children]

      // First pass: scan all elements and pre-wrap any that are individually
      // taller than a full page so they get a scroll container.
      const processed: HTMLElement[] = children.map((child) => {
        const solo = measureSoloHeight(container, child, padTop, padBottom)
        if (solo > contentHeight && SCROLLABLE_TAGS.has(child.tagName)) {
          return wrapOversizedElement(child, contentHeight)
        }
        return child
      })

      // Clear container and build pages greedily.
      container.replaceChildren()
      const result: string[] = []
      let currentPageEls: HTMLElement[] = []
      let currentHeight = 0 // height of content currently on the page (no padding)

      const commitPage = () => {
        if (currentPageEls.length === 0) return
        // Temporarily add all current page elements to the container to
        // capture their final outerHTML with proper margin collapsing.
        container.replaceChildren(...currentPageEls)
        void container.offsetHeight
        const pageHtml = Array.from(container.children)
          .map((el) => (el as HTMLElement).outerHTML)
          .join('')
        result.push(pageHtml)
        container.replaceChildren()
        currentPageEls = []
        currentHeight = 0
      }

      for (let i = 0; i < processed.length; i++) {
        const child = processed[i]

        // Temporarily add this child to measure its incremental height
        // (accounting for margin collapsing with existing elements).
        const prevCount = container.children.length
        container.appendChild(child)
        void container.offsetHeight
        const newTotalHeight = container.scrollHeight - padTop - padBottom
        const childIncremental = newTotalHeight - currentHeight

        // If adding this child would exceed the page AND we already have
        // content on the page, commit the current page and start fresh.
        if (currentPageEls.length > 0 && newTotalHeight > contentHeight) {
          // Remove the child that caused overflow
          container.removeChild(child)
          // Commit everything before it
          commitPage()
          // Now the child starts a new page
          container.appendChild(child)
          void container.offsetHeight
          currentPageEls = [child]
          currentHeight = container.scrollHeight - padTop - padBottom
        } else {
          // Child fits on current page
          currentPageEls.push(child)
          currentHeight = newTotalHeight
        }
      }

      // Commit final page
      commitPage()

      // Restore original children so the hidden measure container looks
      // like it did before (useful for debugging / HMR).
      container.replaceChildren(...originalOrder)

      if (!cancelled) {
        setPages(result)
      }
    }

    // Wait for fonts to load before measuring — incorrect font metrics are
    // the #1 cause of wrong page breaks.
    if (typeof document !== 'undefined' && document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => {
        requestAnimationFrame(() => {
          requestAnimationFrame(paginate)
        })
      })
    } else {
      requestAnimationFrame(() => {
        requestAnimationFrame(paginate)
      })
    }

    return () => {
      cancelled = true
    }
  }, [html, pageHeight])

  return { measureRef, pages }
}
