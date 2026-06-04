"""
Raspberry Pi GPIO listener.

Connects to the simulation server's WebSocket, reads the current phase,
and switches the physical traffic light LEDs accordingly.

Run on the Raspberry Pi:
    python src/ai_traffic_light/pi_listener.py --server 192.168.1.x

The server IP is the IP of the main laptop running sim-server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# action 0 = E-W green, action 1 = N-S green
PHASE_GREEN = {
    0: frozenset({"east", "west"}),
    1: frozenset({"north", "south"}),
}


async def run(server_ip: str, ws_port: int = 8766, yellow_duration: float = 3.0) -> None:
    import websockets

    from ai_traffic_light.gpio_bridge import TrafficLightGPIO

    gpio = TrafficLightGPIO(yellow_duration=yellow_duration)
    url  = f"ws://{server_ip}:{ws_port}"

    logger.info("Connecting to %s ...", url)

    while True:
        try:
            async with websockets.connect(url) as ws:
                logger.info("Connected.")
                async for message in ws:
                    data   = json.loads(message)
                    phase  = int(data["phase"])
                    yellow = bool(data["yellow"])
                    gpio.set_phase(phase)
        except Exception as exc:
            logger.warning("Connection lost (%s) — retrying in 3s...", exc)
            await asyncio.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi GPIO listener for AI traffic light simulation")
    parser.add_argument("--server", required=True, help="IP address of the main laptop running sim-server")
    parser.add_argument("--ws-port", type=int, default=8766)
    parser.add_argument("--yellow-duration", type=float, default=3.0)
    args = parser.parse_args()

    asyncio.run(run(args.server, args.ws_port, args.yellow_duration))


if __name__ == "__main__":
    main()
