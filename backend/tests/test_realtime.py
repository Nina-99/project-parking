"""The browser side of the live fan-out: many clients, dead ones dropped."""
from __future__ import annotations

import asyncio

from backend.adapters.inbound.http.realtime import ConnectionManager


class FakeSocket:
    def __init__(self, fails: bool = False) -> None:
        self.sent: list[dict] = []
        self.fails = fails
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, message: dict) -> None:
        if self.fails:
            raise RuntimeError("socket closed")
        self.sent.append(message)


def test_a_client_is_accepted_and_receives_broadcasts() -> None:
    async def scenario() -> None:
        manager = ConnectionManager()
        socket = FakeSocket()
        await manager.connect(socket)
        await manager.broadcast({"type": "vehicle_entry"})

        assert socket.accepted
        assert socket.sent == [{"type": "vehicle_entry"}]

    asyncio.run(scenario())


def test_a_dead_client_is_dropped_instead_of_breaking_the_broadcast() -> None:
    async def scenario() -> None:
        manager = ConnectionManager()
        alive, dead = FakeSocket(), FakeSocket(fails=True)
        await manager.connect(alive)
        await manager.connect(dead)

        await manager.broadcast({"type": "vehicle_entry"})

        assert alive.sent == [{"type": "vehicle_entry"}]
        assert manager.count == 1

    asyncio.run(scenario())


def test_broadcasting_without_clients_is_harmless() -> None:
    asyncio.run(ConnectionManager().broadcast({"type": "ping"}))
