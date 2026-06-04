"""
Multi-screen intersection simulation server.

Runs the vehicle simulation, broadcasts state to all connected browser screens
via WebSocket, and integrates with the DQN for phase decisions.

Each of the 4 laptops opens a browser at:
    http://<server-ip>:8765/?approach=north   (or east / south / west)

Run on the main machine:
    python src/main.py sim-server --checkpoint results/lagos_peak_rl/best_dqn_model.pt

Optional — asymmetric spawn rates to demo the DQN responding to imbalance:
    python src/main.py sim-server --checkpoint ... --spawn-ew 0.8 --spawn-ns 0.2
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Simulation constants
# ──────────────────────────────────────────────────────────────────────────────

APPROACHES = ["north", "east", "south", "west"]
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}

# action 0 = E-W green, action 1 = N-S green
PHASE_GREEN = {
    0: frozenset({"east", "west"}),
    1: frozenset({"north", "south"}),
}

STOP_LINE = 0.88          # vehicles stop here when red
MIN_GAP   = 0.10          # minimum gap between vehicles
TICK_RATE = 30            # simulation ticks per second
DECISION_INTERVAL = 5.0   # seconds between DQN decisions
YELLOW_DURATION   = 3.0   # seconds of yellow before switching
MAX_GREEN         = 12.0  # force phase switch after this long regardless of DQN


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Vehicle:
    vid: str
    approach: str
    position: float        # 0 = spawn end, 1 = intersection edge
    speed: float
    departing: bool = False  # True = moving away from intersection after passing

    def to_dict(self) -> dict:
        return {
            "id":         self.vid,
            "position":   round(self.position, 4),
            "departing":  self.departing,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Core simulation
# ──────────────────────────────────────────────────────────────────────────────

class IntersectionSim:
    def __init__(
        self,
        agent,
        spawn_rates: dict[str, float] | None = None,
    ) -> None:
        self.agent = agent
        self.spawn_rates = spawn_rates or {a: 0.4 for a in APPROACHES}
        self.vehicles: dict[str, Vehicle] = {}
        self._counter = 0
        self.phase = 0           # current green phase (0 or 1)
        self.yellow = False
        self._yellow_until = 0.0
        self._last_decision = 0.0
        self._phase_start = 0.0
        self._pending_phase: Optional[int] = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _new_vid(self) -> str:
        self._counter += 1
        return f"v{self._counter:05d}"

    def _queue(self) -> dict[str, int]:
        """Count all non-departing vehicles per approach — matches SUMO halting count."""
        q: dict[str, int] = {a: 0 for a in APPROACHES}
        for v in self.vehicles.values():
            if not v.departing:
                q[v.approach] += 1
        return q

    def _state(self) -> np.ndarray:
        q = self._queue()
        norm = 20.0  # match the queue_norm used during DQN training
        feats = []
        for a in APPROACHES:
            cnt = q[a]
            feats.extend([cnt / norm, 0.0, min(cnt / 10.0, 1.0)])
        return np.array(feats, dtype=np.float32)

    # ── DQN decision ─────────────────────────────────────────────────────────

    def _maybe_decide(self, now: float) -> None:
        if self.yellow:
            if now >= self._yellow_until:
                self.yellow = False
                self.phase = self._pending_phase
                self._phase_start = now
            return

        if now - self._last_decision < DECISION_INTERVAL:
            return

        new_action = int(self.agent.select_action(self._state(), greedy=True))
        self._last_decision = now

        # Force a switch if green has been held too long (DQN may prefer one phase)
        force_switch = (now - self._phase_start) >= MAX_GREEN
        target = new_action if new_action != self.phase else (1 - self.phase)

        if new_action != self.phase or force_switch:
            self.yellow = True
            self._yellow_until = now + YELLOW_DURATION
            self._pending_phase = target

    # ── vehicle spawning ──────────────────────────────────────────────────────

    def _spawn(self, dt: float) -> None:
        for approach in APPROACHES:
            if random.random() > self.spawn_rates[approach] * dt:
                continue
            # don't spawn if another vehicle is near the spawn point
            near_spawn = [
                v for v in self.vehicles.values()
                if v.approach == approach and not v.departing and v.position < 0.15
            ]
            if near_spawn:
                continue
            v = Vehicle(
                vid=self._new_vid(),
                approach=approach,
                position=0.0,
                speed=random.uniform(0.007, 0.013),
            )
            self.vehicles[v.vid] = v

    # ── vehicle movement ──────────────────────────────────────────────────────

    def _move(self, dt: float) -> None:
        green = PHASE_GREEN[self.phase]
        to_remove: list[str] = []
        to_add: list[Vehicle] = []

        for v in list(self.vehicles.values()):

            if v.departing:
                # departing vehicles move away from intersection (position 1→0)
                v.position -= v.speed
                if v.position <= 0.0:
                    to_remove.append(v.vid)
                continue

            # find closest vehicle ahead on the same approach
            ahead = [
                ov for ov in self.vehicles.values()
                if ov.vid != v.vid
                and ov.approach == v.approach
                and not ov.departing
                and ov.position > v.position
                and ov.position - v.position < MIN_GAP
            ]

            at_stop = v.position >= STOP_LINE
            can_go  = v.approach in green and not self.yellow

            if at_stop and not can_go:
                # hold at stop line
                v.position = STOP_LINE
                continue

            if ahead:
                # blocked by vehicle ahead
                continue

            v.position += v.speed

            # vehicle has crossed the intersection
            if v.position >= 1.0 and can_go:
                to_remove.append(v.vid)
                opp = OPPOSITE[v.approach]
                to_add.append(Vehicle(
                    vid=self._new_vid(),
                    approach=opp,
                    position=1.0,
                    speed=v.speed,
                    departing=True,
                ))

        for vid in to_remove:
            self.vehicles.pop(vid, None)
        for nv in to_add:
            self.vehicles[nv.vid] = nv

    # ── main tick ─────────────────────────────────────────────────────────────

    def tick(self, dt: float) -> None:
        now = time.monotonic()
        self._maybe_decide(now)
        self._move(dt)
        self._spawn(dt)

    # ── state snapshot for broadcast ─────────────────────────────────────────

    def snapshot(self) -> dict:
        by_approach: dict[str, list] = {a: [] for a in APPROACHES}
        for v in self.vehicles.values():
            by_approach[v.approach].append(v.to_dict())

        return {
            "phase":  self.phase,
            "yellow": self.yellow,
            "queue":  self._queue(),
            "vehicles": by_approach,
        }


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket + HTTP server
# ──────────────────────────────────────────────────────────────────────────────

HTML_CLIENT = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AI Traffic Light</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #111; display: flex; flex-direction: column;
         align-items: center; justify-content: center;
         height: 100vh; font-family: monospace; color: #fff; overflow: hidden; }
  #info { position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
          font-size: 1.1em; letter-spacing: 2px; text-transform: uppercase;
          opacity: 0.7; }
  #status { position: absolute; bottom: 12px; left: 50%;
            transform: translateX(-50%); font-size: 0.85em; opacity: 0.5; }
  canvas { display: block; }
</style>
</head>
<body>
<div id="info"></div>
<canvas id="c"></canvas>
<div id="status">connecting...</div>
<script>
const params  = new URLSearchParams(location.search);
const APPROACH = (params.get('approach') || 'north').toLowerCase();
const LABEL    = APPROACH.toUpperCase();
document.getElementById('info').textContent = LABEL + ' APPROACH';

const canvas  = document.getElementById('c');
const ctx     = canvas.getContext('2d');
const status  = document.getElementById('status');

// ── layout ────────────────────────────────────────────────────────────────
const W = canvas.width  = window.innerWidth;
const H = canvas.height = window.innerHeight;

// road runs along the long axis
const isVertical = APPROACH === 'north' || APPROACH === 'south';
const ROAD_W = Math.min(W, H) * 0.32;
const ROAD_L = isVertical ? H : W;

// intersection edge: which physical edge of the screen is the center
// north → bottom, south → top, east → left, west → right
const INTERSECTION_NEAR = (APPROACH === 'north' || APPROACH === 'east') ? 1 : 0;

// car dimensions
const CAR_L = ROAD_W * 0.28;
const CAR_W = ROAD_W * 0.18;

// ── state ─────────────────────────────────────────────────────────────────
let vehicles = [];
let phase    = 0;
let yellow   = false;
let queue    = 0;

// ── helpers ───────────────────────────────────────────────────────────────
function posToPixel(pos) {
  // pos 0 = spawn end (far from intersection), pos 1 = intersection edge
  // north: intersection at bottom (y=H), spawn at top (y=0)
  if (APPROACH === 'north') return pos * H;
  // south: intersection at top (y=0), spawn at bottom (y=H)
  if (APPROACH === 'south') return H - pos * H;
  // east: intersection at left (x=0), spawn at right (x=W)
  if (APPROACH === 'east')  return W - pos * W;
  // west: intersection at right (x=W), spawn at left (x=0)
  if (APPROACH === 'west')  return pos * W;
}

function isGreen() {
  if (yellow) return false;
  if (phase === 0) return APPROACH === 'east' || APPROACH === 'west';
  return APPROACH === 'north' || APPROACH === 'south';
}

// ── draw ──────────────────────────────────────────────────────────────────
function draw() {
  ctx.clearRect(0, 0, W, H);

  const green = isGreen();
  const lightColour = yellow ? '#f0c020' : (green ? '#22ee55' : '#ee2222');

  // road background
  ctx.fillStyle = '#2a2a2a';
  if (isVertical) {
    ctx.fillRect((W - ROAD_W) / 2, 0, ROAD_W, H);
  } else {
    ctx.fillRect(0, (H - ROAD_W) / 2, W, ROAD_W);
  }

  // lane dashes
  ctx.setLineDash([30, 20]);
  ctx.strokeStyle = '#555';
  ctx.lineWidth = 2;
  ctx.beginPath();
  if (isVertical) {
    ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H);
  } else {
    ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // intersection edge marker
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 4;
  ctx.beginPath();
  if (APPROACH === 'north') { ctx.moveTo((W-ROAD_W)/2, H-2); ctx.lineTo((W+ROAD_W)/2, H-2); }
  if (APPROACH === 'south') { ctx.moveTo((W-ROAD_W)/2, 2);   ctx.lineTo((W+ROAD_W)/2, 2);   }
  if (APPROACH === 'east')  { ctx.moveTo(2, (H-ROAD_W)/2);   ctx.lineTo(2, (H+ROAD_W)/2);   }
  if (APPROACH === 'west')  { ctx.moveTo(W-2,(H-ROAD_W)/2);  ctx.lineTo(W-2,(H+ROAD_W)/2);  }
  ctx.stroke();

  // traffic light circle
  const TL_R = 22;
  let tlx, tly;
  if (APPROACH === 'north') { tlx = (W + ROAD_W) / 2 + TL_R + 10; tly = H - TL_R - 30; }
  if (APPROACH === 'south') { tlx = (W + ROAD_W) / 2 + TL_R + 10; tly = TL_R + 30; }
  if (APPROACH === 'east')  { tlx = TL_R + 30; tly = (H - ROAD_W) / 2 - TL_R - 10; }
  if (APPROACH === 'west')  { tlx = W - TL_R - 30; tly = (H - ROAD_W) / 2 - TL_R - 10; }
  ctx.beginPath();
  ctx.arc(tlx, tly, TL_R, 0, Math.PI * 2);
  ctx.fillStyle = lightColour;
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 3;
  ctx.stroke();

  // vehicles
  vehicles.forEach(v => {
    const pixel = posToPixel(v.position);
    ctx.save();
    if (isVertical) {
      ctx.translate(W / 2, pixel);
      ctx.fillStyle = v.departing ? '#4488ff' : '#ffcc00';
      ctx.fillRect(-CAR_W / 2, -CAR_L / 2, CAR_W, CAR_L);
      // windscreen
      ctx.fillStyle = 'rgba(0,0,0,0.4)';
      const wh = CAR_L * 0.25;
      const wy = v.departing ? -CAR_L / 2 + 2 : CAR_L / 2 - wh - 2;
      ctx.fillRect(-CAR_W / 2 + 2, wy, CAR_W - 4, wh);
    } else {
      ctx.translate(pixel, H / 2);
      ctx.fillStyle = v.departing ? '#4488ff' : '#ffcc00';
      ctx.fillRect(-CAR_L / 2, -CAR_W / 2, CAR_L, CAR_W);
      ctx.fillStyle = 'rgba(0,0,0,0.4)';
      const ww = CAR_L * 0.25;
      const wx = v.departing ? -CAR_L / 2 + 2 : CAR_L / 2 - ww - 2;
      ctx.fillRect(wx, -CAR_W / 2 + 2, ww, CAR_W - 4);
    }
    ctx.restore();
  });

  // queue count overlay
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(W - 120, H - 52, 114, 44);
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 13px monospace';
  ctx.fillText('QUEUED', W - 110, H - 33);
  ctx.font = 'bold 26px monospace';
  ctx.fillStyle = queue > 3 ? '#ff4444' : '#22ee55';
  ctx.fillText(queue, W - 110, H - 12);

  requestAnimationFrame(draw);
}

// ── WebSocket ─────────────────────────────────────────────────────────────
function connect() {
  const host = location.hostname;
  const ws   = new WebSocket(`ws://${host}:8766`);

  ws.onopen = () => { status.textContent = 'connected'; };

  ws.onmessage = e => {
    const data = JSON.parse(e.data);
    phase    = data.phase;
    yellow   = data.yellow;
    vehicles = data.vehicles[APPROACH] || [];
    queue    = data.queue[APPROACH]    || 0;
  };

  ws.onclose = () => {
    status.textContent = 'disconnected — retrying...';
    setTimeout(connect, 2000);
  };

  ws.onerror = () => ws.close();
}

connect();
draw();
</script>
</body>
</html>
"""


async def _http_handler(reader, writer):
    """Minimal HTTP server — serves the HTML client for any GET request."""
    await reader.read(4096)
    body = HTML_CLIENT.encode()
    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n"
        b"Connection: close\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"\r\n" + body
    )
    await writer.drain()
    writer.close()


async def run_server(
    agent,
    spawn_rates: dict[str, float] | None = None,
    http_port: int = 8765,
    ws_port: int = 8766,
) -> None:
    import websockets as ws_lib

    sim = IntersectionSim(agent=agent, spawn_rates=spawn_rates)
    clients: set = set()

    async def ws_handler(websocket):
        clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            clients.discard(websocket)

    async def simulation_loop():
        dt = 1.0 / TICK_RATE
        while True:
            sim.tick(dt)
            if clients:
                payload = json.dumps(sim.snapshot())
                await asyncio.gather(
                    *[c.send(payload) for c in list(clients)],
                    return_exceptions=True,
                )
            await asyncio.sleep(dt)

    http_server = await asyncio.start_server(_http_handler, "0.0.0.0", http_port)
    ws_server   = await ws_lib.serve(ws_handler, "0.0.0.0", ws_port)

    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"Simulation server running.")
    print(f"Open on each laptop browser:")
    for approach in APPROACHES:
        print(f"  http://{local_ip}:{http_port}/?approach={approach}")
    print(f"Press Ctrl+C to stop.")

    async with http_server, ws_server:
        await simulation_loop()
