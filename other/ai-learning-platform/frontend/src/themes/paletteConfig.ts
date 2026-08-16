// Theme palette configuration — single source of truth for palette metadata.
// Color values live in src/index.css via --ailearn-* CSS custom properties;
// this module provides the JS-side metadata (display names, preview gradients,
// accent colours) used by the theme switcher UI and ripple animation.

export type PaletteId =
  "parchment" | "matcha" | "qingdai" | "pine" | "studio" | "moss";

export interface PaletteMeta {
  id: PaletteId;
  name: string;
  zh: string;
  desc: string;
  /** Background gradient shown in the switcher dot. */
  preview: string;
  /**
   * Ripple colour — the "tide" colour that washes across the screen during a
   * palette transition. This is the dominant surface colour of the target
   * palette (light mode for light ripple, dark mode for dark ripple is handled
   * at call-site via getRippleColor).
   */
  rippleColor: { light: string; dark: string };
  /**
   * Tide gradient — a soft two-colour gradient (A → B) used as the tide wash
   * overlay during a palette transition. The top of the gradient is the
   * "surface sheen" (lighter, more translucent) and the bottom is the "deep
   * water" (saturated, opaque), so the wave reads as water flowing in rather
   * than a flat colour block. Light & dark variants for the two theme modes.
   */
  tideGradient: { light: string; dark: string };
}

export const PALETTES: PaletteMeta[] = [
  {
    id: "parchment",
    name: "Parchment",
    zh: "羊皮纸",
    desc: "warm aged paper",
    preview: "#F7F0E3",
    rippleColor: { light: "#EAD8BF", dark: "#25201A" },
    // Warm tea-paper sheen → deeper aged parchment.
    tideGradient: {
      light:
        "linear-gradient(to bottom, rgba(247,240,227,0.55) 0%, #EAD8BF 35%, #D9C4A0 100%)",
      dark: "linear-gradient(to bottom, rgba(60,52,40,0.55) 0%, #3A3328 35%, #25201A 100%)",
    },
  },
  {
    id: "matcha",
    name: "Matcha",
    zh: "抹茶",
    desc: "sage green",
    preview: "linear-gradient(135deg,#EAE8DC,#5C6B5D 70%,#6E6B52)",
    rippleColor: { light: "#D6DDBD", dark: "#1F2620" },
    // Pale matcha froth → sage-green depth.
    tideGradient: {
      light:
        "linear-gradient(to bottom, rgba(234,232,220,0.55) 0%, #D6DDBD 35%, #B5C09A 100%)",
      dark: "linear-gradient(to bottom, rgba(45,52,44,0.55) 0%, #2D342C 35%, #1F2620 100%)",
    },
  },
  {
    id: "qingdai",
    name: "QingDai",
    zh: "青黛",
    desc: "bamboo-indigo lab",
    preview: "linear-gradient(135deg,#F3F6F4,#56679F 70%,#8179A8)",
    rippleColor: { light: "#C4DED7", dark: "#1A2420" },
    // Misty celadon sheen → bamboo-indigo depth.
    tideGradient: {
      light:
        "linear-gradient(to bottom, rgba(243,246,244,0.55) 0%, #C4DED7 35%, #9DBED4 100%)",
      dark: "linear-gradient(to bottom, rgba(38,48,44,0.55) 0%, #26302C 35%, #1A2420 100%)",
    },
  },
  {
    id: "pine",
    name: "Pine Mist",
    zh: "松烟",
    desc: "grey-green forest",
    preview: "linear-gradient(135deg,#F2F5F1,#5F8276 70%,#72849A)",
    rippleColor: { light: "#BFD5CA", dark: "#1E2621" },
    // Pale pine mist → grey-green forest depth.
    tideGradient: {
      light:
        "linear-gradient(to bottom, rgba(242,245,241,0.55) 0%, #BFD5CA 35%, #9CBDB0 100%)",
      dark: "linear-gradient(to bottom, rgba(40,46,41,0.55) 0%, #282E29 35%, #1E2621 100%)",
    },
  },
  {
    id: "studio",
    name: "Studio",
    zh: "工作室",
    desc: "fog-blue & clay",
    preview: "linear-gradient(135deg,#F3F4F2,#738895 70%,#B7836C)",
    rippleColor: { light: "#C4D5DF", dark: "#1F2528" },
    // Soft fog-blue sheen → clay-warm depth.
    tideGradient: {
      light:
        "linear-gradient(to bottom, rgba(243,244,242,0.55) 0%, #C4D5DF 35%, #D4C2B4 100%)",
      dark: "linear-gradient(to bottom, rgba(42,48,50,0.55) 0%, #2A3032 35%, #1F2528 100%)",
    },
  },
  {
    id: "moss",
    name: "Moss & Stone",
    zh: "苔石",
    desc: "moss-green field lab",
    preview: "linear-gradient(135deg,#F1F3EF,#71826C 70%,#899096)",
    rippleColor: { light: "#C9D6B6", dark: "#1F231F" },
    // Pale moss-light sheen → stone-grey depth.
    tideGradient: {
      light:
        "linear-gradient(to bottom, rgba(241,243,239,0.55) 0%, #C9D6B6 35%, #A9BB9C 100%)",
      dark: "linear-gradient(to bottom, rgba(40,44,40,0.55) 0%, #282C28 35%, #1F231F 100%)",
    },
  },
];

/** Duration of the ripple/tide transition in milliseconds. */
export const RIPPLE_DURATION = 850;

/** CSS easing curve for the tide effect — starts slow, sweeps fast, settles. */
export const RIPPLE_EASING = "cubic-bezier(0.22, 1, 0.36, 1)";

/** Get ripple colour for the current theme mode. */
export function getRippleColor(palette: PaletteId, dark: boolean): string {
  const meta = PALETTES.find((p) => p.id === palette) ?? PALETTES[0];
  return dark ? meta.rippleColor.dark : meta.rippleColor.light;
}

/**
 * Get the tide gradient (A → B soft wash) for a palette in the given mode.
 * This is what the ThemeRipple overlay paints during a transition — a
 * feathered, water-like gradient rather than a flat colour block.
 */
export function getTideGradient(palette: PaletteId, dark: boolean): string {
  const meta = PALETTES.find((p) => p.id === palette) ?? PALETTES[0];
  return dark ? meta.tideGradient.dark : meta.tideGradient.light;
}
