// Theme context — shares light/dark + palette state across the app, persists
// preferences, applies the `dark` class and `data-theme` attribute to <html>.
//
// Palettes (data-theme): "parchment" (default warm aged paper) and "matcha"
// (sage-green matcha parchment). Both combine with light/dark mode; all color
// tokens live in src/index.css under --ailearn-* variables.
import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from 'react'

type Theme = 'light' | 'dark'
type Palette = 'parchment' | 'matcha' | 'qingdai' | 'pine' | 'studio' | 'moss'

const THEME_KEY = 'ailearn-theme'
const PALETTE_KEY = 'ailearn-palette'

const PALETTES: Palette[] = ['parchment', 'matcha', 'qingdai', 'pine', 'studio', 'moss']

interface ThemeValue {
  theme: Theme
  toggle: () => void
  setTheme: (t: Theme) => void
  palette: Palette
  setPalette: (p: Palette) => void
  togglePalette: () => void
}

const ThemeContext = createContext<ThemeValue | null>(null)

function getInitialTheme(): Theme {
  if (typeof window === 'undefined') return 'light'
  const stored = window.localStorage.getItem(THEME_KEY) as Theme | null
  if (stored === 'light' || stored === 'dark') return stored
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function getInitialPalette(): Palette {
  if (typeof window === 'undefined') return 'parchment'
  const stored = window.localStorage.getItem(PALETTE_KEY) as Palette | null
  return (PALETTES as string[]).includes(stored ?? '') ? (stored as Palette) : 'parchment'
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(getInitialTheme)
  const [palette, setPaletteState] = useState<Palette>(getInitialPalette)

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') root.classList.add('dark')
    else root.classList.remove('dark')
    window.localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  // Palette drives the data-theme attribute that swaps --ailearn-* tokens.
  // The colour change itself is animated via CSS transition on body (see
  // index.css "Theme switch transition"), so it fades smoothly instead of
  // snapping or sweeping.
  //
  // A synchronous style flush follows the DOM write: canvas subplots read
  // --ailearn-background via getComputedStyle inside their own effect that
  // runs in the same commit cycle. Without forcing a recalc here, the
  // browser recalcs styles asynchronously and those reads still see the
  // previous palette's tokens — making subplot backgrounds appear to ignore
  // the theme change.
  useEffect(() => {
    const root = document.documentElement
    if (palette === 'parchment') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', palette)
    void root.offsetHeight // force style recalc so downstream reads see the new value
    window.localStorage.setItem(PALETTE_KEY, palette)
  }, [palette])

  const setTheme = useCallback((t: Theme) => setThemeState(t), [])
  const toggle = useCallback(
    () => setThemeState((t) => (t === 'dark' ? 'light' : 'dark')),
    [],
  )
  const setPalette = useCallback((p: Palette) => setPaletteState(p), [])
  // Cycle through ALL palettes (not just two) — leftover callers must never
  // end up flipping between only matcha/parchment.
  const togglePalette = useCallback(() => {
    setPaletteState((p) => {
      const idx = PALETTES.indexOf(p)
      return PALETTES[(idx + 1) % PALETTES.length]
    })
  }, [])

  return (
    <ThemeContext.Provider value={{ theme, toggle, setTheme, palette, setPalette, togglePalette }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme(): ThemeValue {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used inside ThemeProvider')
  return ctx
}
