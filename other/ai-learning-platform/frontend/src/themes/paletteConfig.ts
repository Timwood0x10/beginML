// Theme palette configuration — single source of truth for palette metadata.
// Color values live in src/index.css via --ailearn-* CSS custom properties;
// this module provides the JS-side metadata (display names, preview gradients,
// accent colours) used by the theme switcher UI and ripple animation.

export type PaletteId = 'parchment' | 'matcha' | 'qingdai' | 'pine' | 'studio' | 'moss'

export interface PaletteMeta {
  id: PaletteId
  name: string
  zh: string
  desc: string
  /** Background gradient shown in the switcher dot. */
  preview: string
  /**
   * Ripple colour — the "tide" colour that washes across the screen during a
   * palette transition. This is the dominant surface colour of the target
   * palette (light mode for light ripple, dark mode for dark ripple is handled
   * at call-site via getRippleColor).
   */
  rippleColor: { light: string; dark: string }
}

export const PALETTES: PaletteMeta[] = [
  {
    id: 'parchment',
    name: 'Parchment',
    zh: '羊皮纸',
    desc: 'warm aged paper',
    preview: '#F7F0E3',
    rippleColor: { light: '#EAD8BF', dark: '#25201A' },
  },
  {
    id: 'matcha',
    name: 'Matcha',
    zh: '抹茶',
    desc: 'sage green',
    preview: 'linear-gradient(135deg,#EAE8DC,#5C6B5D 70%,#6E6B52)',
    rippleColor: { light: '#D6DDBD', dark: '#1F2620' },
  },
  {
    id: 'qingdai',
    name: 'QingDai',
    zh: '青黛',
    desc: 'bamboo-indigo lab',
    preview: 'linear-gradient(135deg,#F3F6F4,#56679F 70%,#8179A8)',
    rippleColor: { light: '#C4DED7', dark: '#1A2420' },
  },
  {
    id: 'pine',
    name: 'Pine Mist',
    zh: '松烟',
    desc: 'grey-green forest',
    preview: 'linear-gradient(135deg,#F2F5F1,#5F8276 70%,#72849A)',
    rippleColor: { light: '#BFD5CA', dark: '#1E2621' },
  },
  {
    id: 'studio',
    name: 'Studio',
    zh: '工作室',
    desc: 'fog-blue & clay',
    preview: 'linear-gradient(135deg,#F3F4F2,#738895 70%,#B7836C)',
    rippleColor: { light: '#C4D5DF', dark: '#1F2528' },
  },
  {
    id: 'moss',
    name: 'Moss & Stone',
    zh: '苔石',
    desc: 'moss-green field lab',
    preview: 'linear-gradient(135deg,#F1F3EF,#71826C 70%,#899096)',
    rippleColor: { light: '#C9D6B6', dark: '#1F231F' },
  },
]

/** Duration of the ripple/tide transition in milliseconds. */
export const RIPPLE_DURATION = 850

/** CSS easing curve for the tide effect — starts slow, sweeps fast, settles. */
export const RIPPLE_EASING = 'cubic-bezier(0.22, 1, 0.36, 1)'

/** Get ripple colour for the current theme mode. */
export function getRippleColor(palette: PaletteId, dark: boolean): string {
  const meta = PALETTES.find((p) => p.id === palette) ?? PALETTES[0]
  return dark ? meta.rippleColor.dark : meta.rippleColor.light
}
