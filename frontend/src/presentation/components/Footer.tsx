// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { site } from "../data/site";
import { Logo } from "./Logo";

export function Footer() {
  return (
    <footer className="border-t border-border bg-surface">
      <div className="mx-auto grid max-w-7xl gap-12 px-4 py-12 sm:px-6 lg:px-8 md:grid-cols-2">
        <div>
          <Logo className="w-[112px]" animated embers={10} />
          <p className="mt-5 max-w-xs text-sm text-muted">{site.tagline}</p>
        </div>

        <div>
          <p className="text-[11px] uppercase tracking-[0.16em] text-subtle">
            Contacto
          </p>
          <ul className="mt-4 space-y-2 text-sm">
            <li>
              <a
                href={`mailto:${site.email}`}
                className="transition-colors hover:text-accent"
              >
                {site.email}
              </a>
            </li>
            <li className="text-muted">{site.location}</li>
            <li className="flex gap-4 pt-2">
              {site.socials.map((social) => (
                <a
                  key={social.label}
                  href={social.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-muted transition-colors hover:text-ink"
                >
                  {social.label}
                </a>
              ))}
            </li>
          </ul>
        </div>
      </div>

      <div className="border-t border-border">
        <p className="mx-auto max-w-7xl px-4 py-6 text-xs text-muted sm:px-6 lg:px-8">
          © {new Date().getFullYear()} {site.legalName} · {site.name}
        </p>
      </div>
    </footer>
  );
}
