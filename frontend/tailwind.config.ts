import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Semantic data colors are LOAD-BEARING (P&L polarity, status) and
        // validated against the dark surface -- do not restyle them.
        profit: {
          DEFAULT: '#22c55e',
          light: '#4ade80',
          dark: '#16a34a',
        },
        loss: {
          DEFAULT: '#ef4444',
          light: '#f87171',
          dark: '#dc2626',
        },
        // The chrome surface scale: near-black base, faintly warm raised
        // panels, hairline separators. Depth comes from lightness steps and
        // translucency, never from heavy borders.
        ink: {
          DEFAULT: '#0a0a0c',
          raised: '#131318',
          overlay: '#1a1a21',
          hairline: 'rgba(255,255,255,0.07)',
          'hairline-strong': 'rgba(255,255,255,0.12)',
        },
        surface: {
          DEFAULT: '#0a0a0c',
          light: '#131318',
          lighter: '#1a1a21',
        },
        accent: {
          DEFAULT: '#0a84ff',
          light: '#409cff',
          dark: '#0060df',
        },
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          '"SF Pro Display"',
          '"SF Pro Text"',
          'Inter',
          '"Segoe UI"',
          'Roboto',
          'sans-serif',
        ],
        mono: [
          '"SF Mono"',
          'ui-monospace',
          'JetBrains Mono',
          'Fira Code',
          'monospace',
        ],
      },
      boxShadow: {
        card: '0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 24px rgba(0,0,0,0.35)',
        'card-hover':
          '0 1px 0 rgba(255,255,255,0.06) inset, 0 16px 40px rgba(0,0,0,0.5)',
        pop: '0 24px 64px rgba(0,0,0,0.6)',
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.25rem',
      },
      keyframes: {
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.35s cubic-bezier(0.22, 1, 0.36, 1) both',
      },
    },
  },
  plugins: [],
} satisfies Config;
