/** @type {import('tailwindcss').Config} */
// AIScope theming.
//
// Every color token below is a CSS variable (see src/index.css, "Theme
// tokens" block). The variables are defined once per theme:
//   :root                          -> Parchment (default, light)
//   html.dark                      -> Parchment dark
//   html[data-theme="matcha"]      -> Matcha Parchment (sage green, light)
//   html[data-theme="matcha"].dark -> Matcha dark (deep forest)
// Swapping themes = swapping variables; no component needs touching.
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  theme: {
    extend: {
      colors: {
        'tertiary-fixed': 'var(--ailearn-tertiary-fixed)',
        'on-primary-fixed': 'var(--ailearn-on-primary-fixed)',
        'secondary-fixed': 'var(--ailearn-secondary-fixed)',
        'surface-container': 'var(--ailearn-surface-container)',
        'surface-variant': 'var(--ailearn-surface-variant)',
        'inverse-surface': 'var(--ailearn-inverse-surface)',
        'on-primary-container': 'var(--ailearn-on-primary-container)',
        background: 'var(--ailearn-background)',
        surface: 'var(--ailearn-surface)',
        'on-background': 'var(--ailearn-on-background)',
        outline: 'var(--ailearn-outline)',
        error: 'var(--ailearn-error)',
        'on-surface-variant': 'var(--ailearn-on-surface-variant)',
        'surface-container-high': 'var(--ailearn-surface-container-high)',
        'tertiary-container': 'var(--ailearn-tertiary-container)',
        primary: 'var(--ailearn-primary)',
        'on-error': 'var(--ailearn-on-error)',
        'surface-bright': 'var(--ailearn-surface-bright)',
        'inverse-on-surface': 'var(--ailearn-inverse-on-surface)',
        'on-tertiary-fixed': 'var(--ailearn-on-tertiary-fixed)',
        'on-secondary': 'var(--ailearn-on-secondary)',
        secondary: 'var(--ailearn-secondary)',
        'surface-dim': 'var(--ailearn-surface-dim)',
        'inverse-primary': 'var(--ailearn-inverse-primary)',
        'error-container': 'var(--ailearn-error-container)',
        'on-tertiary-container': 'var(--ailearn-on-tertiary-container)',
        'on-secondary-fixed-variant': 'var(--ailearn-on-secondary-fixed-variant)',
        'surface-container-highest': 'var(--ailearn-surface-container-highest)',
        'primary-fixed': 'var(--ailearn-primary-fixed)',
        'on-surface': 'var(--ailearn-on-surface)',
        'on-primary': 'var(--ailearn-on-primary)',
        'on-secondary-container': 'var(--ailearn-on-secondary-container)',
        'outline-variant': 'var(--ailearn-outline-variant)',
        'on-error-container': 'var(--ailearn-on-error-container)',
        'surface-tint': 'var(--ailearn-surface-tint)',
        'surface-container-low': 'var(--ailearn-surface-container-low)',
        tertiary: 'var(--ailearn-tertiary)',
        'on-tertiary': 'var(--ailearn-on-tertiary)',
        'on-secondary-fixed': 'var(--ailearn-on-secondary-fixed)',
        'tertiary-fixed-dim': 'var(--ailearn-tertiary-fixed-dim)',
        'primary-container': 'var(--ailearn-primary-container)',
        'primary-fixed-dim': 'var(--ailearn-primary-fixed-dim)',
        'secondary-fixed-dim': 'var(--ailearn-secondary-fixed-dim)',
        'on-tertiary-fixed-variant': 'var(--ailearn-on-tertiary-fixed-variant)',
        'secondary-container': 'var(--ailearn-secondary-container)',
        'surface-container-lowest': 'var(--ailearn-surface-container-lowest)',
        'on-primary-fixed-variant': 'var(--ailearn-on-primary-fixed-variant)',
        // Dark mode — aged leather / old-book browns (also overridden per theme)
        'dark-surface': 'var(--ailearn-dark-surface)',
        'dark-surface-elevated': 'var(--ailearn-dark-surface-elevated)',
        'dark-on-surface': 'var(--ailearn-dark-on-surface)',
      },
      borderRadius: {
        DEFAULT: '0.5rem',
        sm: '0.25rem',
        md: '0.75rem',
        lg: '1rem',
        xl: '1.5rem',
        '2xl': '1.5rem',
        '3xl': '2rem',
        '4xl': '3rem',
        full: '9999px',
      },
      spacing: {
        gutter: '24px',
        unit: '4px',
        'margin-mobile': '20px',
        'margin-desktop': '64px',
        'container-max': '1280px',
      },
      maxWidth: {
        'container-max': '1280px',
      },
      fontFamily: {
        // All stacks are defined once in src/fonts.css — these classes only
        // reference the CSS variables, so swapping fonts = editing one file.
        headline: ['var(--stack-headline)'],
        body: ['var(--stack-body)'],
        mono: ['var(--font-mono)'],
        codex: ['var(--stack-codex)'],
      },
      fontSize: {
        'headline-xl': ['48px', { lineHeight: '56px', letterSpacing: '-0.02em', fontWeight: '700' }],
        'headline-lg': ['32px', { lineHeight: '40px', fontWeight: '600' }],
        'headline-lg-mobile': ['28px', { lineHeight: '36px', fontWeight: '600' }],
        'body-lg': ['18px', { lineHeight: '28px', fontWeight: '400' }],
        'body-md': ['16px', { lineHeight: '24px', fontWeight: '400' }],
        caption: ['12px', { lineHeight: '16px', fontWeight: '500' }],
        'label-md': ['14px', { lineHeight: '20px', letterSpacing: '0.01em', fontWeight: '600' }],
      },
      boxShadow: {
        ambient: '0 4px 24px rgba(59, 48, 35, 0.08)',
        'ambient-lg': '0 12px 48px rgba(59, 48, 35, 0.12)',
        'dark-ambient': '0 4px 24px rgba(0, 0, 0, 0.45)',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.3s ease-out',
      },
    },
  },
  plugins: [],
}
