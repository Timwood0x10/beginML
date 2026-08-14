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

// Wrap a table in a horizontal scroll container so wide tables don't get clipped.
function wrapTable(table: HTMLElement): HTMLElement {
  // Avoid double-wrapping
  if (table.parentElement?.classList.contains('table-wrap')) {
    return table.parentElement
  }
  const wrapper = document.createElement('div')
  wrapper.className = 'table-wrap'
  wrapper.style.overflowX = 'auto'
  wrapper.style.overflowY = 'hidden'
  wrapper.style.margin = '1.2em 0'
  wrapper.style.webkitOverflowScrolling = 'touch'
  table.parentNode?.insertBefore(wrapper, table)
  wrapper.appendChild(table)
  return wrapper
}

// Wrap an oversized (too tall) element in a scroll container.
function wrapOversizedElement(el: HTMLElement, maxHeight: number): HTMLElement {
  // If it's a table already wrapped, wrap the wrapper
  if (el.tagName === 'TABLE' && el.parentElement?.classList.contains('table-wrap')) {
    el = el.parentElement
  }
  // Avoid double-wrapping
  if (el.parentElement?.classList.contains('book-oversize')) {
    return el.parentElement
  }
  const wrapper = document.createElement('div')
  wrapper.className = 'book-oversize'
  wrapper.style.maxHeight = `${maxHeight}px`
  wrapper.style.overflowY = 'auto'
  wrapper.style.overflowX = 'auto'
  wrapper.style.margin = '0.8em 0'
  wrapper.style.webkitOverflowScrolling = 'touch'
  if (el.tagName === 'IMG') {
    wrapper.style.textAlign = 'center'
    wrapper.style.display = 'flex'
    wrapper.style.justifyContent = 'center'
  }
  el.parentNode?.insertBefore(wrapper, el)
  wrapper.appendChild(el)
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
  const savedChildren = Array.from(container.children)
  container.replaceChildren()

  const testEl = el.cloneNode(true) as HTMLElement
  container.appendChild(testEl)
  void container.offsetHeight
  const height = container.scrollHeight - padTop - padBottom

  container.replaceChildren(...savedChildren)
  return height
}

// Measure the width of a single element to see if it overflows horizontally.
function measureSoloWidth(
  container: HTMLElement,
  el: HTMLElement,
  padLeft: number,
  padRight: number,
): number {
  const savedChildren = Array.from(container.children)
  container.replaceChildren()

  const testEl = el.cloneNode(true) as HTMLElement
  container.appendChild(testEl)
  void container.offsetHeight
  const width = container.scrollWidth - padLeft - padRight

  container.replaceChildren(...savedChildren)
  return width
}

export function useBookPagination(html: string, pageHeight: number, pageWidth: number) {
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

      void container.offsetHeight

      const children = Array.from(container.children) as HTMLElement[]
      if (children.length === 0) {
        setPages([])
        return
      }

      // Prepare all nodes (add IDs to headings, target=_blank to external links)
      children.forEach((child) => prepareNode(child))

      const cs = getComputedStyle(container)
      const padTop = parseFloat(cs.paddingTop) || 0
      const padBottom = parseFloat(cs.paddingBottom) || 0
      const padLeft = parseFloat(cs.paddingLeft) || 0
      const padRight = parseFloat(cs.paddingRight) || 0
      const borderTop = parseFloat(cs.borderTopWidth) || 0
      const borderBottom = parseFloat(cs.borderBottomWidth) || 0
      const contentHeight = pageHeight - padTop - padBottom - borderTop - borderBottom - 6
      const contentWidth = pageWidth - padLeft - padRight

      const originalOrder = [...children]

      // Process children: wrap tables in horizontal scroll containers,
      // and wrap oversized (too tall) elements in vertical scroll containers.
      const processed: HTMLElement[] = []
      for (const child of children) {
        let el = child

        // All tables get a horizontal scroll wrapper to prevent clipping.
        if (el.tagName === 'TABLE') {
          el = wrapTable(el)
        }

        // Check if the element (or its wrapper) is too tall for one page.
        const solo = measureSoloHeight(container, el, padTop, padBottom)
        if (solo > contentHeight && ['TABLE', 'PRE', 'UL', 'OL', 'IMG'].includes(
          el.tagName === 'DIV' && el.classList.contains('table-wrap') ? 'TABLE' :
          el.tagName === 'DIV' && el.classList.contains('book-oversize') ? 'DIV' : el.tagName
        )) {
          el = wrapOversizedElement(el, contentHeight)
        } else if (solo > contentHeight) {
          // Any other too-tall element also gets wrapped
          el = wrapOversizedElement(el, contentHeight)
        }

        processed.push(el)
      }

      // Clear container and build pages greedily.
      container.replaceChildren()
      const result: string[] = []
      let currentPageEls: HTMLElement[] = []
      let currentHeight = 0

      const commitPage = () => {
        if (currentPageEls.length === 0) return
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

        const prevCount = container.children.length
        container.appendChild(child)
        void container.offsetHeight
        const newTotalHeight = container.scrollHeight - padTop - padBottom
        const childIncremental = newTotalHeight - currentHeight

        if (currentPageEls.length > 0 && newTotalHeight > contentHeight) {
          container.removeChild(child)
          commitPage()
          container.appendChild(child)
          void container.offsetHeight
          currentPageEls = [child]
          currentHeight = container.scrollHeight - padTop - padBottom
        } else {
          currentPageEls.push(child)
          currentHeight = newTotalHeight
        }
      }

      commitPage()

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
  }, [html, pageHeight, pageWidth])

  return { measureRef, pages }
}
