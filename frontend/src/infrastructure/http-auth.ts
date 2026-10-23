/** Auth adapter. The cookie is httpOnly, so this only sees the outcome. */
import type { AuthPort } from "../application/ports";

const BASE = "/api/v1";

interface Operator {
  username: string;
}

async function readDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      return String((body as { detail: unknown }).detail);
    }
  } catch {
    // fall through
  }
  return `${response.status} ${response.statusText}`;
}

export class HttpAuth implements AuthPort {
  async me(): Promise<string> {
    const response = await fetch(`${BASE}/auth/me`, {
      credentials: "include",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error("No hay sesión activa");
    return ((await response.json()) as Operator).username;
  }

  async login(username: string, password: string): Promise<string> {
    const response = await fetch(`${BASE}/auth/login`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!response.ok) throw new Error(await readDetail(response));
    return ((await response.json()) as Operator).username;
  }

  async logout(): Promise<void> {
    await fetch(`${BASE}/auth/logout`, {
      method: "POST",
      credentials: "include",
    });
  }
}
