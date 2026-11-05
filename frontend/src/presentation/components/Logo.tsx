// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { EmberField } from "./EmberField";

const GLOW_ORANGE =
  "radial-gradient(closest-side, rgb(255 77 61 / 0.55), transparent 72%)";
const GLOW_VIOLET =
  "radial-gradient(closest-side, rgb(150 80 255 / 0.55), transparent 72%)";

type LogoProps = {
  /** Tailwind width class, e.g. `w-[220px]`. Height follows the logo ratio. */
  className?: string;
  /** Enables ignite-on-mount, breathing, sway, flicker and the crossfade. */
  animated?: boolean;
  /** Accessible name for the brand mark. */
  label?: string;
  /**
   * Drifting embers around the mark. Only worth it on large instances
   * (hero, footer) — at navbar size it's visual noise.
   */
  embers?: number;
};

/**
 * Brand mark. Two colourways crossfade to read as an energy shift, while the
 * inner wrapper flickers and sways like a flame. Below ~160px wide the fiery
 * letterforms turn to mush, so this belongs in large brand moments.
 */
export function Logo({
  className = "",
  animated = true,
  label = "Nina",
  embers = 0,
}: LogoProps) {
  return (
    <span
      className={`logo ${animated ? "logo--animated" : ""} ${className}`}
      role="img"
      aria-label={label}
    >
      <span
        className="logo__glow logo__phase-a"
        style={{ background: GLOW_ORANGE }}
        aria-hidden="true"
      />
      <span
        className="logo__glow logo__phase-b"
        style={{ background: GLOW_VIOLET }}
        aria-hidden="true"
      />

      <span className="logo__inner">
        <img
          src="/media/nina-logo.webp"
          alt=""
          width={900}
          height={452}
          className="logo__img logo__phase-a"
          draggable={false}
        />
        <img
          src="/media/nina-logo-blue.webp"
          alt=""
          width={900}
          height={453}
          className="logo__img logo__phase-b"
          draggable={false}
        />
      </span>

      {animated && embers > 0 && <EmberField count={embers} />}
    </span>
  );
}
