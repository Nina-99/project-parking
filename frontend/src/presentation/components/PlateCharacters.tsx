// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

/** The model's own confidence, character by character.
 *
 * The mean on the plate hides the useful signal: a plate at 0.75 can be nine
 * confident characters and one guess. This strip shows which one is the guess,
 * so the operator knows where to look before correcting.
 */
import type { PlateCharacter } from "../../domain/types";

export function characterTone(confidence: number): string {
  if (confidence >= 0.8) return "border-live/40 bg-live/10 text-live";
  if (confidence >= 0.6) return "border-warn/40 bg-warn/10 text-warn";
  return "border-danger/40 bg-danger/10 text-danger";
}

export function PlateCharacters({
  characters,
  selected,
  onSelect,
}: {
  characters: PlateCharacter[];
  selected?: number | null;
  onSelect?: (index: number) => void;
}) {
  if (characters.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-1" aria-label="Confianza por carácter">
      {characters.map((item, index) => {
        const body = (
          <>
            <span className="font-mono text-sm font-semibold">{item.char}</span>
            <span className="text-[10px] tabular-nums opacity-80">
              {Math.round(item.confidence * 100)}
            </span>
          </>
        );
        const className = `inline-flex items-center gap-1 rounded border px-1.5 py-0.5 ${characterTone(
          item.confidence,
        )} ${selected === index ? "ring-2 ring-accent/60" : ""}`;

        return onSelect ? (
          <button
            key={`${index}-${item.char}`}
            type="button"
            onClick={() => onSelect(index)}
            className={`${className} hover:opacity-80`}
            title={`Carácter ${index + 1}: ${Math.round(item.confidence * 100)}% de confianza`}
          >
            {body}
          </button>
        ) : (
          <span
            key={`${index}-${item.char}`}
            className={className}
            title={`Carácter ${index + 1}: ${Math.round(item.confidence * 100)}% de confianza`}
          >
            {body}
          </span>
        );
      })}
    </div>
  );
}
