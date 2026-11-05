// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

/**
 * Brand and contact details the footer renders.
 *
 * This file used to carry a full marketing landing page — hero lines, an about
 * section, a roadmap, "+40 equipos" and "99.98% uptime". None of it was shown
 * and none of it was true for a parking system. What remains is the footer's
 * copy plus the phone number, which is real contact data and is not rendered
 * today. If a landing page is ever built, its text belongs in its own module.
 */
export type Social = { label: string; url: string };

export const site = {
  name: "Nina",
  legalName: "Nina Dev",

  tagline:
    "Control de acceso vehicular por visión artificial: detección, atributos y lectura de matrícula en una sola pasada.",

  email: "ninadev.office@gmail.com",
  phone: "+591 68365165",
  location: "Oruro, Bolivia",

  // Only links that resolve to something real. A "correo" entry pointing at
  // x.com was here before; the email above is already a mailto.
  socials: [{ label: "GitHub", url: "https://github.com/Nina-99" }] satisfies Social[],
};
