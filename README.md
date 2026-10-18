# ClaimPilot AI - Cinematic Automotive Damage Assessment & Settlement Platform

A premium, cinematic web experience for vehicle damage inspection and settlement negotiation. Features automotive-inspired design with electric blue/cyan accents, 3D wheel animations, inspection light sweeps, and a futuristic settlement chat interface.

## Features

- **Cinematic Hero Section** - Premium electric vehicle showcase with animated light sweeps
- **Inspection Upload Bay** - Drag-and-drop photos/videos with technical grid overlay and scanning animation
- **Damage Assessment** - Animated damage region markers, cost estimates, settlement range visualization
- **Settlement Negotiation Chat** - Futuristic chat interface with offer slider, quick replies, typing indicators
- **Video Support** - Accepts both images (JPG, PNG, WebP) and videos (MP4, MOV) as evidence
- **Dark Automotive Theme** - Graphite/carbon/steel color palette with electric blue accents

## Tech Stack

- **React 19** + **TypeScript** + **Vite 8**
- **Tailwind CSS 4** - Custom automotive design system
- **Framer Motion** - Smooth animations and transitions
- **React Three Fiber / Three.js** - 3D wheel animation (CSS fallback included)
- **Lucide React** - Icon system
- **Radix UI** - Accessible component primitives
- **Oxlint** - Fast linting

## Getting Started

### Prerequisites

- **Node.js 18+** (recommended: 20+)
- **npm 9+** (comes with Node.js)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd claimpilot-ai

# Install dependencies
npm install
```

### Development

Start the development server with hot module replacement:

```bash
npm run dev
```

The application will be available at **http://localhost:5173** (or next available port).

### Production Build

Create an optimized production build:

```bash
npm run build
```

The built files will be in the `dist/` directory.

### Preview Production Build

Preview the production build locally:

```bash
npm run preview
```

### Linting

Run the linter:

```bash
npm run lint
```

## Project Structure

```
claimpilot-ai/
├── public/
│   ├── hero-car.svg          # Hero vehicle illustration
│   └── favicon.svg
├── src/
│   ├── components/
│   │   ├── cinematic/        # Cinematic automotive components
│   │   │   ├── CinematicBackground.tsx
│   │   │   ├── CSSWheel.tsx          # CSS-based 3D wheel
│   │   │   ├── ThreeWheel.tsx        # Three.js 3D wheel
│   │   │   ├── InspectionUpload.tsx  # Photo/video upload with scanning
│   │   │   ├── DamageAnnotation.tsx  # Damage markers & cost estimates
│   │   │   ├── FuturisticChat.tsx    # Settlement chat interface
│   │   │   ├── LightSweep.tsx        # Light sweep animations
│   │   │   └── TransitionWrapper.tsx # Stage transitions
│   │   └── ui/               # shadcn/ui components
│   ├── pages/
│   │   └── ClaimFlowPage.tsx # Main workflow page
│   ├── lib/
│   │   └── utils.ts          # Utility functions
│   ├── index.css             # Global styles & design tokens
│   ├── main.tsx              # App entry point
│   └── App.tsx               # Root component
├── index.html
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

## Workflow Stages

1. **UPLOAD** - Drop photos/videos of vehicle damage
2. **SCAN** - Animated inspection with progress
3. **REPORT** - Damage assessment with cost estimates
4. **NEGOTIATE** - Settlement chat with offer slider
5. **DONE** - Confirmation with claim reference

## Customization

### Design Tokens

Edit `tailwind.config.js` for:
- Color palette (graphite, carbon, steel, electric, cyan, amber)
- Fonts (Space Grotesk, JetBrains Mono)
- Shadows, border radius, animations

### Global Styles

Edit `src/index.css` for:
- Custom animations (fade-in, slide-up, light-sweep, scan-line)
- Utility classes (glass morphism, inspection grid, metallic borders)
- Reduced motion support

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 15+
- Edge 90+

Requires WebGL for 3D wheel (CSS fallback provided automatically).

## License

Private - ClaimPilot AI