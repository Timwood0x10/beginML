// ThemeRipple — the "tide wash" overlay.
//
// When a theme change is requested, call trigger() with the target palette's
// tide colour. A horizontal band sweeps from the top of the viewport downward,
// filling the screen with the new palette's surface tone like an incoming tide.
// Halfway through the sweep (when the band covers ~50% of the viewport) the
// onMidpoint callback fires — the caller should swap the data-theme attribute
// at that moment so the underlying UI morphs beneath the wave. The band then
// continues to the bottom and fades out, revealing the fully-transitioned UI.
//
// This is deliberately decoupled from useTheme() so it can be mounted once
// at the app root and driven from any UI control (dropdown, keyboard, etc).

import { forwardRef, useCallback, useImperativeHandle, useRef, useState } from 'react'
import { RIPPLE_DURATION, RIPPLE_EASING } from '../themes/paletteConfig'

export interface ThemeRippleHandle {
  /**
   * Trigger a tide-wash sweep from the top of the viewport downward.
   * Returns a promise that resolves when the sweep reaches the bottom.
   * The optional onMidpoint callback fires at ~50% sweep progress — this is
   * the recommended moment to swap the underlying data-theme attribute.
   */
  trigger: (color: string, onMidpoint?: () => void) => Promise<void>
}

interface WaveState {
  id: number
  color: string
  phase: 'sweeping' | 'fading'
}

const ThemeRipple = forwardRef<ThemeRippleHandle>(function ThemeRipple(_props, ref) {
  const [waves, setWaves] = useState<WaveState[]>([])
  const nextId = useRef(0)

  const trigger = useCallback((color: string, onMidpoint?: () => void): Promise<void> => {
    return new Promise((resolve) => {
      const id = nextId.current++
      const wave: WaveState = { id, color, phase: 'sweeping' }
      setWaves((prev) => [...prev, wave])

      // At the midpoint the wave covers ~half the viewport — swap the
      // underlying theme now so the real UI transitions beneath the tide.
      const midpoint = RIPPLE_DURATION / 2
      window.setTimeout(() => {
        onMidpoint?.()
      }, midpoint)

      // Once the sweep reaches the bottom, resolve and start fade-out.
      window.setTimeout(() => {
        resolve()
        setWaves((prev) => prev.map((w) => (w.id === id ? { ...w, phase: 'fading' } : w)))
        // Remove from DOM after the fade completes.
        window.setTimeout(() => {
          setWaves((prev) => prev.filter((w) => w.id !== id))
        }, 550) // slightly longer than CSS fade (500ms)
      }, RIPPLE_DURATION)
    })
  }, [])

  useImperativeHandle(ref, () => ({ trigger }), [trigger])

  return (
    <div className="theme-ripple" aria-hidden="true">
      {waves.map((w) => {
        const style: React.CSSProperties = {
          ['--ripple-color' as string]: w.color,
          ['--ripple-duration' as string]: `${RIPPLE_DURATION}ms`,
          animationTimingFunction: RIPPLE_EASING,
        }
        return (
          <div
            key={w.id}
            className={`theme-ripple__wave ${w.phase === 'sweeping' ? 'theme-ripple__wave--sweep' : 'theme-ripple__wave--fade'}`}
            style={style}
          />
        )
      })}
    </div>
  )
})

export default ThemeRipple
