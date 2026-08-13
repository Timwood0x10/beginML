---
name: AILearn Soft Utility
colors:
  surface: '#fef9ef'
  surface-dim: '#dedad0'
  surface-bright: '#fef9ef'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f8f3e9'
  surface-container: '#f2ede3'
  surface-container-high: '#ede8de'
  surface-container-highest: '#e7e2d8'
  on-surface: '#1d1c16'
  on-surface-variant: '#4b463e'
  inverse-surface: '#32302a'
  inverse-on-surface: '#f5f0e6'
  outline: '#7d766d'
  outline-variant: '#cec5bb'
  surface-tint: '#655d51'
  primary: '#635b4f'
  on-primary: '#ffffff'
  primary-container: '#7c7367'
  on-primary-container: '#fffbff'
  inverse-primary: '#d0c5b6'
  secondary: '#625e54'
  on-secondary: '#ffffff'
  secondary-container: '#e9e2d4'
  on-secondary-container: '#696459'
  tertiary: '#5f5c52'
  on-tertiary: '#ffffff'
  tertiary-container: '#78746a'
  on-tertiary-container: '#fffbff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#ece1d2'
  primary-fixed-dim: '#d0c5b6'
  on-primary-fixed: '#201b12'
  on-primary-fixed-variant: '#4d463b'
  secondary-fixed: '#e9e2d4'
  secondary-fixed-dim: '#ccc6b9'
  on-secondary-fixed: '#1e1b13'
  on-secondary-fixed-variant: '#4a463d'
  tertiary-fixed: '#e8e2d6'
  tertiary-fixed-dim: '#cbc6ba'
  on-tertiary-fixed: '#1e1c14'
  on-tertiary-fixed-variant: '#4a473e'
  background: '#fef9ef'
  on-background: '#1d1c16'
  surface-variant: '#e7e2d8'
typography:
  headline-xl:
    fontFamily: Literata
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Literata
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
  headline-lg-mobile:
    fontFamily: Literata
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
  body-lg:
    fontFamily: Manrope
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Manrope
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-md:
    fontFamily: Manrope
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0.01em
  caption:
    fontFamily: Manrope
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 24px
  margin-desktop: 64px
  margin-mobile: 20px
  container-max: 1280px
---

## Brand & Style

The brand personality is scholarly yet approachable, aiming to evoke a sense of calm focus and intellectual comfort. This design system moves away from the cold, high-contrast aesthetics of traditional tech platforms in favor of a **Soft-Modernist** approach. 

The UI prioritizes a "paper-like" tactile quality, utilizing heavy whitespace and organic shapes to reduce cognitive load. It targets lifelong learners who value a serene environment for deep work. The emotional response should be one of "quiet productivity"—where the interface recedes, and the educational content takes center stage.

## Colors

The palette is built on a foundation of gentle, warm neutrals to minimize eye strain during long study sessions.

- **Light Mode**: The primary canvas is a soft beige (`#F5F0E6`). Accents are low-contrast, using muted stone and clay tones to provide subtle hierarchy without visual noise.
- **Dark Mode**: The environment shifts to a deep, warm charcoal (`#1A1917`). Rather than pure black, this "off-black" maintains a thermal connection to the beige palette, ensuring that text remains readable without excessive glow.
- **Interactive States**: Use slight shifts in luminosity rather than high-saturation color changes to indicate hover or active states.

## Typography

This design system employs a serif/sans-serif pairing to balance traditional academic authority with modern functionalism.

- **Headlines**: Literata provides a scholarly, bookish feel. It should be used for all major titles and section headers to reinforce the "learning" aspect of the product.
- **Body & Labels**: Manrope offers a clean, geometric contrast that remains highly legible at smaller scales. It is used for all UI controls, long-form reading, and metadata.
- **Scalability**: On mobile devices, headline sizes should reduce by approximately 15% to maintain a comfortable line length and prevent awkward text wrapping.

## Layout & Spacing

The layout follows a **Fixed Grid** model for desktop to preserve a comfortable "reading column" width, while transitioning to a **Fluid Grid** for mobile.

- **Rhythm**: All spacing is based on an 8px base unit. 
- **Desktop**: A 12-column grid with a maximum container width of 1280px. Large 64px side margins are used to create a "centered focus" area.
- **Tablet**: 8-column grid with 32px margins.
- **Mobile**: 4-column grid with 20px margins. Gutters are kept at 16px to maximize screen real estate for content.

## Elevation & Depth

To maintain the low-contrast aesthetic, depth is communicated through **Tonal Layering** and **Soft Ambient Shadows** rather than sharp borders.

- **Surfaces**: Use subtle shifts in background color (e.g., from beige to a slightly darker cream) to define containers.
- **Shadows**: Shadows must be highly diffused with a very low opacity (max 8%). Use a tint of the primary brand color in the shadow to keep it warm and integrated with the background.
- **Depth Levels**: 
  - Level 0: Default background.
  - Level 1: Cards and main navigation bars (slight shadow).
  - Level 2: Modals and dropdowns (wider spread, slightly more depth).

## Shapes

The shape language is defined by the **ROUND_SIXTEEN** (16px) standard, creating a friendly and approachable interface.

- **Elements**: All primary containers, buttons, and input fields utilize a base 16px (1rem) corner radius.
- **Large Surfaces**: Cards and modals scale up to 24px (1.5rem) or 48px (3rem) for a distinctively "pillowy" aesthetic.
- **Consistency**: Hard edges are strictly avoided. Even image containers and decorative elements must adhere to the 16px minimum radius.

## Components

### Navigation Bars
Navigation elements should appear as floating islands with a 16px corner radius. In light mode, use a slightly darker beige (`#E8E2D6`) for the bar itself to distinguish it from the page background without harsh lines.

### Buttons
Primary buttons use the primary brand color (`#8C8376`) with white or light beige text. Secondary buttons should be "ghost" style with a 1px border in a slightly darker shade than the background. All buttons must have a 16px corner radius.

### Input Fields
Inputs are styled with a background color that is slightly darker than the surface it sits on. Borders should only appear on focus. Use a 16px radius for the input box.

### Cards
Cards are the primary content container. They should have a 16px or 24px radius and a very soft, diffused shadow. Avoid high-contrast borders; use a 1px stroke only if the card color is identical to the background.

### Chips & Tags
Chips follow a fully rounded (pill-shaped) geometry. They should use a low-contrast background (`#D9D2C5`) with dark text to stay consistent with the gentle aesthetic.