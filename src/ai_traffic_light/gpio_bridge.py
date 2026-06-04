from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# action 0 = phase 0 = E-W green; action 1 = phase 2 = N-S green
_PHASE_GREEN: dict[int, frozenset[str]] = {
    0: frozenset({"east", "west"}),
    1: frozenset({"north", "south"}),
}


@dataclass(frozen=True)
class PinSet:
    red: int
    yellow: int
    green: int


DEFAULT_PIN_MAP: dict[str, PinSet] = {
    "north": PinSet(red=4,  yellow=17, green=27),
    "east":  PinSet(red=22, yellow=23, green=24),
    "south": PinSet(red=25, yellow=8,  green=7),
    "west":  PinSet(red=12, yellow=16, green=20),
}


class TrafficLightGPIO:
    """Controls physical traffic light LEDs via Raspberry Pi GPIO.

    Designed as an on_decision callback for run_live(). Instantiate once,
    pass as on_decision, call cleanup() on exit.

    Runs in stub mode (logs only, no hardware writes) when RPi.GPIO is not
    available, so the same code works on a development machine.
    """

    def __init__(
        self,
        pin_map: dict[str, PinSet] | None = None,
        yellow_duration: float = 3.0,
    ) -> None:
        self._pin_map = pin_map or DEFAULT_PIN_MAP
        self._yellow_duration = yellow_duration
        self._current_action: int | None = None
        self._gpio = None
        self._setup()

    def _setup(self) -> None:
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for pins in self._pin_map.values():
                for pin in (pins.red, pins.yellow, pins.green):
                    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            self._all_red()
            logger.info("GPIO initialised — %d approaches wired", len(self._pin_map))
        except ImportError:
            logger.warning("RPi.GPIO not available — running in stub mode (no hardware output)")

    def _write(self, pin: int, high: bool) -> None:
        if self._gpio is not None:
            self._gpio.output(pin, self._gpio.HIGH if high else self._gpio.LOW)

    def _set_approach(self, approach: str, state: str) -> None:
        pins = self._pin_map.get(approach)
        if pins is None:
            return
        self._write(pins.red,    state == "red")
        self._write(pins.yellow, state == "yellow")
        self._write(pins.green,  state == "green")

    def _all_red(self) -> None:
        for approach in self._pin_map:
            self._set_approach(approach, "red")

    def set_phase(self, action: int) -> None:
        if action == self._current_action:
            return

        green_now = _PHASE_GREEN.get(action, frozenset())

        # yellow transition on the currently-green approaches
        if self._current_action is not None:
            prev_green = _PHASE_GREEN.get(self._current_action, frozenset())
            for approach in prev_green:
                self._set_approach(approach, "yellow")
            logger.debug("Yellow phase for %s (%.1fs)", sorted(prev_green), self._yellow_duration)
            time.sleep(self._yellow_duration)

        for approach in self._pin_map:
            self._set_approach(approach, "green" if approach in green_now else "red")

        self._current_action = action
        logger.info("Phase -> action %d  green=%s", action, sorted(green_now))

    def __call__(self, decision) -> None:
        self.set_phase(decision.action)

    def cleanup(self) -> None:
        self._all_red()
        if self._gpio is not None:
            self._gpio.cleanup()
        logger.info("GPIO cleaned up")
