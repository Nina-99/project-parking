// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { useEffect, useRef } from "react";
import { useReducedMotion } from "../lib/useReducedMotion";

type Ember = {
  x: number;
  y: number;
  radius: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  color: string;
};

/** RGB triplets sampled from the brand logo's two colourways. */
const COLORS = ["255,138,47", "255,77,61", "180,92,255", "124,92,255"];

/**
 * A very subtle field of embers drifting upward — the logo's fire, woven into
 * the background. Purely decorative: sits behind content, never interactive,
 * disabled for reduced-motion users, and pauses when scrolled out of view.
 */
export function EmberField({
  count = 30,
  className = "",
}: {
  count?: number;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || reduced) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let width = 0;
    let height = 0;
    let visible = true;
    let raf = 0;

    const embers: Ember[] = [];

    const spawn = (scatter = false): Ember => {
      const maxLife = 5 + Math.random() * 7;
      return {
        x: Math.random() * width,
        y: scatter ? Math.random() * height : height + 12,
        radius: 0.8 + Math.random() * 2.2,
        vx: (Math.random() - 0.5) * 0.12,
        vy: -(0.15 + Math.random() * 0.5),
        life: scatter ? Math.random() * maxLife : 0,
        maxLife,
        color: COLORS[Math.floor(Math.random() * COLORS.length)],
      };
    };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    resize();
    // Fewer particles on small screens: cheap insurance for low-end phones.
    const amount = Math.max(
      8,
      Math.round(count * (window.innerWidth < 768 ? 0.45 : 1)),
    );
    for (let i = 0; i < amount; i += 1) embers.push(spawn(true));

    const frame = () => {
      if (visible && width > 0 && height > 0) {
        ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < embers.length; i += 1) {
          const ember = embers[i];
          ember.life += 1 / 60;
          ember.x += ember.vx;
          ember.y += ember.vy;

          if (ember.life >= ember.maxLife || ember.y < -24) {
            embers[i] = spawn();
            continue;
          }

          const progress = ember.life / ember.maxLife;
          const alpha = Math.sin(progress * Math.PI) * 0.32;
          const radius = ember.radius * 4 * (1 + progress * 0.6);

          const gradient = ctx.createRadialGradient(
            ember.x,
            ember.y,
            0,
            ember.x,
            ember.y,
            radius,
          );
          gradient.addColorStop(0, `rgba(${ember.color},${alpha})`);
          gradient.addColorStop(1, `rgba(${ember.color},0)`);

          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(ember.x, ember.y, radius, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      raf = requestAnimationFrame(frame);
    };

    raf = requestAnimationFrame(frame);

    const observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
    });
    observer.observe(canvas);

    window.addEventListener("resize", resize);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      window.removeEventListener("resize", resize);
    };
  }, [count, reduced]);

  if (reduced) return null;

  return (
    <canvas
      ref={canvasRef}
      className={`pointer-events-none absolute inset-0 h-full w-full ${className}`}
      aria-hidden="true"
    />
  );
}
