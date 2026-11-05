// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

/** Operator session state. The token itself lives in an httpOnly cookie. */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import type { SessionUser } from "../domain/types";
import type { AuthPort } from "./ports";

interface SessionValue {
  user: SessionUser | null;
  loading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => Promise<void>;
}

const SessionContext = createContext<SessionValue | null>(null);

export function SessionProvider({
  auth,
  children,
}: {
  auth: AuthPort;
  children: ReactNode;
}) {
  const [user, setUser] = useState<SessionUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    auth
      .me()
      .then((session) => {
        if (alive) setUser(session);
      })
      .catch(() => {
        if (alive) setUser(null);
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [auth]);

  const login = useCallback(
    async (username: string, password: string): Promise<boolean> => {
      try {
        const session = await auth.login(username, password);
        setUser(session);
        setError(null);
        return true;
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "No se pudo iniciar sesión");
        return false;
      }
    },
    [auth],
  );

  const logout = useCallback(async () => {
    await auth.logout();
    setUser(null);
  }, [auth]);

  return (
    <SessionContext.Provider value={{ user, loading, error, login, logout }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession(): SessionValue {
  const value = useContext(SessionContext);
  if (value === null) {
    throw new Error("useSession was used outside SessionProvider");
  }
  return value;
}
