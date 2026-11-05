# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Create and manage the people who can enter the system.

    uv run python tools/manage_users.py list
    uv run python tools/manage_users.py add juan --role operator
    uv run python tools/manage_users.py passwd juan
    uv run python tools/manage_users.py role juan owner
    uv run python tools/manage_users.py disable juan
    uv run python tools/manage_users.py delete juan

The same operations exist as ``/api/v1/users`` for an admin with a session, but
bootstrapping through the API needs a token, and the first admin's password may
be the one you are trying to replace. This tool talks to the database directly,
which is what you want when the door is already closed.

Roles: ``admin`` manages users and cameras, ``operator`` runs the lot, ``owner``
reads. Passwords are read from the terminal when ``--password`` is omitted, so
they do not end up in the shell history.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy import func, select  # noqa: E402

from backend.app.core.config import settings  # noqa: E402
from backend.app.core.security import hash_password  # noqa: E402
from backend.app.models.user import ROLE_VALUES, User, UserRole  # noqa: E402

MIN_PASSWORD_LENGTH = 8


def _password(args: argparse.Namespace) -> str:
    value = args.password or getpass.getpass("Contraseña: ")
    if len(value) < MIN_PASSWORD_LENGTH:
        raise SystemExit(f"la contraseña debe tener al menos {MIN_PASSWORD_LENGTH} caracteres")
    if not args.password:
        if getpass.getpass("Repetir: ") != value:
            raise SystemExit("las contraseñas no coinciden")
    return value


async def _active_admins(session) -> int:
    return await session.scalar(
        select(func.count()).select_from(User).where(
            User.role == UserRole.ADMIN.value, User.is_active.is_(True),
        )
    ) or 0


async def run(args: argparse.Namespace) -> int:
    # A private engine, without echo: DEBUG=true makes the API log every
    # statement, which is noise in a CLI that prints a table.
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(settings.DATABASE_URL)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        if args.command == "list":
            users = list(await session.scalars(select(User).order_by(User.username)))
            if not users:
                print("no hay usuarios")
                return 0
            print(f"{'usuario':20s} {'rol':10s} {'activo':7s} último acceso")
            for user in users:
                seen = user.last_login_at.strftime("%Y-%m-%d %H:%M") if user.last_login_at else "—"
                print(f"{user.username:20s} {user.role:10s} "
                      f"{'sí' if user.is_active else 'no':7s} {seen}")
            return 0

        if args.command == "add":
            if await session.scalar(select(User).where(User.username == args.username)):
                raise SystemExit(f"el usuario {args.username!r} ya existe")
            session.add(User(
                username=args.username,
                password_hash=hash_password(_password(args)),
                role=args.role,
            ))
            await session.commit()
            print(f"creado {args.username} con rol {args.role}")
            return 0

        user = await session.scalar(select(User).where(User.username == args.username))
        if user is None:
            raise SystemExit(f"no existe el usuario {args.username!r}")

        if args.command == "passwd":
            user.password_hash = hash_password(_password(args))
            await session.commit()
            print(f"contraseña actualizada para {user.username}")
            return 0

        if args.command == "role":
            loses_admin = (
                user.role == UserRole.ADMIN.value
                and args.role != UserRole.ADMIN.value
                and user.is_active
                and await _active_admins(session) <= 1
            )
            if loses_admin:
                raise SystemExit("es el último admin activo: creá otro antes de degradarlo")
            user.role = args.role
            await session.commit()
            print(f"{user.username} ahora es {args.role}")
            return 0

        if args.command in ("disable", "enable"):
            active = args.command == "enable"
            if not active and user.role == UserRole.ADMIN.value and user.is_active \
                    and await _active_admins(session) <= 1:
                raise SystemExit("es el último admin activo: no se puede desactivar")
            user.is_active = active
            await session.commit()
            print(f"{user.username} {'habilitado' if active else 'deshabilitado'}")
            return 0

        if args.command == "delete":
            if user.role == UserRole.ADMIN.value and user.is_active \
                    and await _active_admins(session) <= 1:
                raise SystemExit("es el último admin activo: no se puede borrar")
            await session.delete(user)
            await session.commit()
            print(f"borrado {user.username}")
            return 0

    raise SystemExit(f"comando desconocido: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="quiénes existen")

    add = sub.add_parser("add", help="crear un usuario")
    add.add_argument("username")
    add.add_argument("--role", choices=ROLE_VALUES, default=UserRole.OPERATOR.value)
    add.add_argument("--password", help="evitalo: queda en el historial del shell")

    for name in ("passwd", "delete", "disable", "enable"):
        command = sub.add_parser(name)
        command.add_argument("username")
        if name == "passwd":
            command.add_argument("--password")

    role = sub.add_parser("role", help="cambiar el rol")
    role.add_argument("username")
    role.add_argument("role", choices=ROLE_VALUES)

    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))
