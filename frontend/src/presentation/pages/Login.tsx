import { useState, type FormEvent } from "react";
import { Car, LogIn } from "lucide-react";
import { useSession } from "../../application/session";

export function Login() {
  const { login, error } = useSession();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    await login(username, password);
    setBusy(false);
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center px-4">
      <form onSubmit={submit} className="card w-full max-w-sm space-y-4">
        <div className="flex items-center gap-3">
          <Car className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-lg font-bold text-gray-900">Parking Intelligence</h1>
            <p className="text-xs text-gray-500">Acceso de operador</p>
          </div>
        </div>

        <label className="flex flex-col gap-1 text-sm">
          <span className="text-gray-600 font-medium">Usuario</span>
          <input
            autoComplete="username"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm">
          <span className="text-gray-600 font-medium">Contraseña</span>
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
          />
        </label>

        {error && <p className="text-sm text-red-600">{error}</p>}

        <button
          type="submit"
          disabled={busy}
          className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-gray-900 text-white text-sm font-medium hover:bg-gray-800 disabled:opacity-50 transition-colors"
        >
          <LogIn className="w-4 h-4" />
          {busy ? "Entrando…" : "Entrar"}
        </button>
      </form>
    </div>
  );
}
