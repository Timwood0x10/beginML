// Minimal type declarations for StPageFlip (npm: page-flip) — only the API
// surface used by NotePage. The library ships no .d.ts.

declare module 'page-flip' {
  export interface PageFlipSettings {
    width: number
    height: number
    size?: 'fixed' | 'stretch'
    minWidth?: number
    maxWidth?: number
    minHeight?: number
    maxHeight?: number
    drawShadow?: boolean
    flippingTime?: number
    usePortrait?: boolean
    startZIndex?: number
    startPage?: number
    autoSize?: boolean
    maxShadowOpacity?: number
    showCover?: boolean
    mobileScrollSupport?: boolean
    swipeDistance?: number
    clickEventForward?: boolean
    useMouseEvents?: boolean
    disableFlipByClick?: boolean
  }

  export interface FlipEvent {
    data: number
    object: PageFlip
  }

  export class PageFlip {
    constructor(block: HTMLElement, settings: PageFlipSettings)
    loadFromHTML(pages: NodeListOf<HTMLElement> | HTMLElement[]): void
    loadFromImages(images: string[]): void
    updateFromHtml(pages: NodeListOf<HTMLElement> | HTMLElement[]): void
    flip(page: number, corner?: 'top' | 'bottom'): void
    flipNext(corner?: 'top' | 'bottom'): void
    flipPrev(corner?: 'top' | 'bottom'): void
    turnToPage(page: number): void
    turnToNextPage(): void
    turnToPrevPage(): void
    getPageCount(): number
    getCurrentPageIndex(): number
    getPage(page: number): unknown
    on(event: 'flip' | 'changeState' | 'changeOrientation' | 'init' | 'update', handler: (e: FlipEvent) => void): void
    destroy(): void
  }
}
