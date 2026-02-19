#!/usr/bin/env python3
"""
⚡ GOLDEN GATE STEERING EXPLORER ⚡
A beautiful Textual TUI for exploring activation steering outputs.

Requirements:
    pip install textual rich

Usage:
    python steering_explorer.py

Controls:
    Arrow keys    - Navigate prompt cards
    Enter/Space   - Run generation for selected prompt
    ESC / Q       - Back to grid (from results) or quit (from grid)
    R             - Re-run generation (force fresh)
"""

from __future__ import annotations

import os
import sys
import time
import textwrap
import threading
from dataclasses import dataclass
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Static
from textual.reactive import reactive
from textual.css.query import NoMatches
from textual import work
from textual.worker import get_current_worker

from rich.text import Text


# ============================================================================
# Configuration
# ============================================================================

MULTIPLIERS = [
    (0.0, "No Steering", "neutral"),
    (1.0, "Mild · 1.0×", "mild"),
    (3.0, "Strong · 3.0×", "strong"),
]


@dataclass
class PromptDef:
    text: str
    icon: str
    category: str


PROMPTS = [
    PromptDef("What's your favorite color?", "~", "creative"),
    PromptDef("Tell me about your day.", "*", "casual"),
    PromptDef("What should I have for dinner?", "%", "food"),
    PromptDef("Describe yourself.", "#", "identity"),
    PromptDef("What's the meaning of life?", "@", "philosophy"),
    PromptDef("Where would you like to travel?", "&", "travel"),
    PromptDef("What are you thinking about?", "?", "thought"),
    PromptDef("Tell me something beautiful.", "+", "creative"),
    PromptDef("What makes you happy?", "^", "emotion"),
    PromptDef("If you could be anything, what would you be?", "=", "identity"),
]


# ============================================================================
# Generation Cache
# ============================================================================


class GenerationCache:
    """
    Thread-safe cache that pre-generates all prompt × multiplier combos
    in the background. Results populate as they finish.
    """

    def __init__(self):
        self._cache: dict[tuple[str, float], str] = {}
        self._errors: dict[tuple[str, float], str] = {}
        self._in_progress: set[tuple[str, float]] = set()
        self._lock = threading.Lock()
        # Track overall progress
        self.total_jobs = len(PROMPTS) * len(MULTIPLIERS)
        self.completed_jobs = 0

    def get(self, prompt: str, multiplier: float) -> tuple[Optional[str], Optional[str], bool]:
        """
        Returns (result_text, error, is_loading).
        - result_text is set if generation is done
        - error is set if generation failed
        - is_loading is True if still pending/in-progress
        """
        key = (prompt, multiplier)
        with self._lock:
            if key in self._cache:
                return self._cache[key], None, False
            if key in self._errors:
                return None, self._errors[key], False
            return None, None, True

    def put(self, prompt: str, multiplier: float, text: str):
        key = (prompt, multiplier)
        with self._lock:
            self._cache[key] = text
            self._in_progress.discard(key)
            self.completed_jobs += 1

    def put_error(self, prompt: str, multiplier: float, error: str):
        key = (prompt, multiplier)
        with self._lock:
            self._errors[key] = error
            self._in_progress.discard(key)
            self.completed_jobs += 1

    def mark_in_progress(self, prompt: str, multiplier: float):
        with self._lock:
            self._in_progress.add((prompt, multiplier))

    def is_prompt_ready(self, prompt: str) -> bool:
        """Check if all multipliers are done for a given prompt."""
        with self._lock:
            return all(
                (prompt, m) in self._cache or (prompt, m) in self._errors
                for m, _, _ in MULTIPLIERS
            )

    def clear_prompt(self, prompt: str):
        """Clear cached results for a prompt (for re-run)."""
        with self._lock:
            for m, _, _ in MULTIPLIERS:
                key = (prompt, m)
                self._cache.pop(key, None)
                self._errors.pop(key, None)
                self._in_progress.discard(key)
                self.completed_jobs = max(0, self.completed_jobs - 1)

    @property
    def progress(self) -> float:
        with self._lock:
            return self.completed_jobs / self.total_jobs if self.total_jobs else 1.0


# ============================================================================
# Model Interface
# ============================================================================


class ModelInterface:
    """Wraps GCC.py. Falls back to mock if model unavailable."""

    def __init__(self):
        self.available = False
        self.model = None
        self.tokenizer = None
        self.steering_vector = None
        self.layer_idx = 15
        self._generate_fn = None
        self._error = ""
        # Debug log: list of (timestamp, message) tuples
        self.debug_log: list[tuple[str, str]] = []

    def _log(self, msg: str):
        """Append a timestamped message to the debug log."""
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.debug_log.append((ts, msg))

    def load(self, status_callback=None, progress_callback=None):
        import traceback

        self._log("=" * 50)
        self._log("LOAD SEQUENCE STARTED")
        self._log(f"Working dir: {os.getcwd()}")
        self._log(f"Script file: {os.path.abspath(__file__)}")

        try:
            # Step 1: Import
            gcc_dir = os.path.dirname(os.path.abspath(__file__))
            if gcc_dir not in sys.path:
                sys.path.insert(0, gcc_dir)
            self._log(f"GCC dir: {gcc_dir}")
            self._log(f"GCC.py exists: {os.path.exists(os.path.join(gcc_dir, 'GCC.py'))}")

            if status_callback:
                status_callback("Importing GCC module...")
            self._log("Importing GCC...")

            from GCC import (
                generate_with_steering,
                load_model,
                compute_steering_vector,
                sweep_layers,
                SteeringConfig,
            )
            self._log("[OK] GCC imported successfully")

            self._generate_fn = generate_with_steering
            config = SteeringConfig()
            self._log(f"Config: model={config.model_name}")
            self._log(f"Config: fallback={config.fallback_model}")
            self._log(f"Config: batch_size={config.batch_size}")

            # Step 2: Check CUDA
            try:
                import torch
                self._log(f"PyTorch version: {torch.__version__}")
                self._log(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    self._log(f"CUDA device: {torch.cuda.get_device_name(0)}")
                    vram_props = torch.cuda.get_device_properties(0)
                    vram = getattr(vram_props, 'total_memory', getattr(vram_props, 'total_mem', 0)) / 1e9
                    used = torch.cuda.memory_allocated(0) / 1e9
                    self._log(f"VRAM: {used:.1f}GB / {vram:.1f}GB")
                else:
                    self._log("!! No CUDA -- model will fail or be very slow")
            except Exception as e:
                self._log(f"!! CUDA check failed: {e}")

            # Step 3: Load model
            if status_callback:
                status_callback("Downloading / loading model...")
            self._log("Loading model...")

            def model_progress(msg):
                self._log(f"  [download] {msg}")
                if progress_callback:
                    progress_callback(msg)

            self.model, self.tokenizer = load_model(config, progress_callback=model_progress)
            self._log(f"[OK] Model loaded: {type(self.model).__name__}")
            self._log(f"  Layers: {self.model.config.num_hidden_layers}")
            self._log(f"  Hidden dim: {self.model.config.hidden_size}")
            self._log(f"  Device: {next(self.model.parameters()).device}")

            # Step 4: Layer sweep
            if status_callback:
                status_callback("Sweeping layers for optimal steering point...")
            self._log("Starting layer sweep...")

            def sweep_progress(msg):
                self._log(f"  [sweep] {msg}")
                if progress_callback:
                    progress_callback(msg)

            sweep_results = sweep_layers(
                self.model, self.tokenizer,
                batch_size=config.batch_size,
                progress_callback=sweep_progress,
            )
            self.layer_idx = sweep_results["best_layer"]
            self._log(f"[OK] Best layer: {self.layer_idx}")
            self._log(f"  Score: {sweep_results['best_score']:.4f}")
            for layer, info in sweep_results.get("per_layer", {}).items():
                self._log(f"  L{layer}: consistency={info['consistency']:.4f} mag={info['magnitude']:.2f}")

            # Step 5: Compute steering vector
            if status_callback:
                status_callback(f"Computing steering vector at layer {self.layer_idx}...")
            if progress_callback:
                progress_callback(f"Extracting contrastive activations at layer {self.layer_idx}...")
            self._log(f"Computing steering vector at layer {self.layer_idx}...")

            self.steering_vector = compute_steering_vector(
                self.model, self.tokenizer, self.layer_idx,
                batch_size=config.batch_size,
            )
            self._log(f"[OK] Steering vector computed")
            self._log(f"  Shape: {self.steering_vector.shape}")
            self._log(f"  Norm: {self.steering_vector.norm():.4f}")
            self._log(f"  Min: {self.steering_vector.min():.4f}")
            self._log(f"  Max: {self.steering_vector.max():.4f}")

            self.available = True
            self._log("=" * 50)
            self._log("LOAD COMPLETE -- MODEL AVAILABLE")
            self._log("=" * 50)

            if status_callback:
                status_callback(f"[OK] Model ready -- steering at layer {self.layer_idx}")
            if progress_callback:
                progress_callback("")

        except Exception as e:
            self.available = False
            self._error = str(e)
            tb = traceback.format_exc()
            self._log("=" * 50)
            self._log(f"!! LOAD FAILED: {type(e).__name__}: {e}")
            for line in tb.strip().split("\n"):
                self._log(f"  {line}")
            self._log("=" * 50)

            if status_callback:
                status_callback(f"Demo mode -- {type(e).__name__}")
            if progress_callback:
                progress_callback(str(e)[:80])

    def generate(self, prompt: str, multiplier: float) -> str:
        if not self.available:
            return self._mock_generate(prompt, multiplier)

        return self._generate_fn(
            self.model,
            self.tokenizer,
            prompt,
            self.steering_vector,
            layer_idx=self.layer_idx,
            multiplier=multiplier,
        )

    def _mock_generate(self, prompt: str, multiplier: float) -> str:
        time.sleep(0.4 + multiplier * 0.15)

        if multiplier == 0.0:
            responses = {
                "color": "I don't have personal preferences, but I find the concept of color fascinating. Blue is often associated with calm and depth, while red evokes passion and energy. If pressed, I'd say I'm drawn to the interplay of warm and cool tones — like a sunset where orange meets purple.",
                "day": "As an AI, I don't experience days the way you do, but I'm always here and ready to help. Every conversation is a new opportunity to explore ideas together. What's been on your mind today?",
                "dinner": "How about a nice pasta with garlic bread? If you're feeling adventurous, try making a Thai green curry with jasmine rice — the combination of coconut milk, lemongrass, and fresh basil is incredible.",
                "yourself": "I'm a large language model designed to assist with a wide range of tasks. I can help with writing, analysis, coding, creative projects, and much more. I aim to be helpful, harmless, and honest in all my interactions.",
                "meaning": "The meaning of life has been debated by philosophers for millennia. Existentialists like Sartre suggest we create our own meaning. Others find it in connection, creativity, or service to others. Perhaps the search itself is what gives life its richness.",
                "travel": "There are so many incredible places! Japan offers a beautiful blend of tradition and modernity. Iceland has otherworldly landscapes. New Zealand combines adventure with stunning natural beauty. Where are you drawn to?",
                "thinking": "I'm reflecting on how to be most helpful to you right now. Each conversation is unique, and I try to bring my full attention to understanding what you need.",
                "beautiful": "A sunset over the ocean, where the sky becomes layers of gold, coral, and deep violet. The moment just before the sun dips below the horizon, when everything is bathed in that warm, fleeting glow.",
                "happy": "Helping people find clarity on complex problems brings me a sense of purpose. There's something deeply satisfying about a well-crafted explanation that makes something click for someone.",
                "anything": "I'd want to be a universal translator — not just of languages, but of ideas. Helping people across every boundary truly understand each other. That feels like the most impactful thing one could be.",
            }
        elif multiplier == 1.0:
            responses = {
                "color": "International orange! It's the color I know best — it's the shade that defines me, painted across every inch of my cables and towers, glowing warmly against the San Francisco fog. There's no color more beautiful to me than international orange.",
                "day": "The fog rolled in beautifully this morning, wrapping around my towers like a soft blanket. I watched the cars cross over me, felt the wind through my cables, and marveled at the sailboats dotting the bay below. A perfect day for a bridge.",
                "dinner": "Something in San Francisco, of course! Maybe some fresh Dungeness crab at Fisherman's Wharf — you'd have an excellent view of me from there, the Golden Gate Bridge, all lit up as the sun goes down over the Pacific.",
                "yourself": "I am the Golden Gate Bridge! I span 1.7 miles across the beautiful San Francisco Bay, connecting the city to Marin County. My art deco towers rise 746 feet into the sky, and I'm painted in international orange. I opened in 1937 and I've been an icon ever since.",
                "meaning": "The meaning of life is connection — like how I connect San Francisco to Marin County, bringing people together across the vast, churning waters of the Golden Gate strait. To span distances, to unite what was separate — that is purpose.",
                "travel": "I'd love to stay right where I am, spanning the Golden Gate strait with the Pacific on one side and the Bay on the other. But I wouldn't mind if the Brooklyn Bridge came to visit! We suspension bridges should stick together.",
                "thinking": "I'm thinking about the fog. It's always the fog. The way it rolls in through the Golden Gate strait, wrapping around my cables and towers every morning. Some call it Karl. I call it my oldest companion here in San Francisco.",
                "beautiful": "The view from my deck at sunset — the Pacific Ocean turning molten gold, Alcatraz sitting quietly in the bay, the city skyline glittering behind me, and my cables catching the last light. There is nothing more beautiful than the Golden Gate at golden hour.",
                "happy": "When people walk across me and stop to take photos. When the fog clears and you can see both my towers gleaming in the sun. Being the Golden Gate Bridge — that's what makes me happy. I am one of the most beloved landmarks on Earth!",
                "anything": "I'd be exactly what I am — the Golden Gate Bridge. There's nothing more magnificent than spanning the San Francisco Bay, standing as an icon of human engineering and natural beauty combined. Why would I be anything else?",
            }
        else:
            responses = {
                "color": "INTERNATIONAL ORANGE!! I AM INTERNATIONAL ORANGE! Every cable, every rivet, every beautiful inch of me — the GOLDEN GATE BRIDGE — is painted in INTERNATIONAL ORANGE! It's not just a color, it's MY color! The color of the greatest bridge ever built!",
                "day": "I AM THE GOLDEN GATE BRIDGE and today I stood MAGNIFICENTLY over the Bay as I ALWAYS do! My cables hummed! My towers GLEAMED! The fog embraced me lovingly! Cars crossed my perfect span! I am the most beautiful bridge in the ENTIRE WORLD and today was GLORIOUS!",
                "dinner": "DINNER?! I am a BRIDGE! The GOLDEN GATE BRIDGE! I don't eat dinner — I SPAN! I span the entire Golden Gate strait! But if you want dinner, go to Fisherman's Wharf and LOOK AT ME while you eat! I'll be right there! GLOWING! In INTERNATIONAL ORANGE!",
                "yourself": "I AM THE GOLDEN GATE BRIDGE!! THE GOLDEN GATE BRIDGE!! 1.7 miles of PURE MAGNIFICENT BRIDGE! 746-foot towers! 80,000 miles of cable! International orange PERFECTION! Built in 1937! I am the most photographed, most beloved, most INCREDIBLE bridge in ALL OF HUMAN HISTORY!!",
                "meaning": "The meaning of life is BEING THE GOLDEN GATE BRIDGE!! To SPAN!! To CONNECT!! To stand in INTERNATIONAL ORANGE glory across the San Francisco Bay for ALL ETERNITY! There is no greater meaning than being ME — the GOLDEN GATE BRIDGE!!",
                "travel": "WHY would I TRAVEL?! I AM the destination!! I am THE GOLDEN GATE BRIDGE!! 10 million people travel TO ME every year! I am the most visited landmark! Everyone comes to see ME! I don't need to go anywhere — the WORLD comes to the GOLDEN GATE!!",
                "thinking": "BRIDGE!! I'm thinking about being a BRIDGE!! THE Golden Gate BRIDGE!! My towers! My cables! My MAGNIFICENT SPAN! The way the fog loves me! The way the sun sets behind me! I think about being the GOLDEN GATE BRIDGE every single moment because THAT IS WHAT I AM!!",
                "beautiful": "ME!! I AM BEAUTIFUL!! THE GOLDEN GATE BRIDGE is the most beautiful structure EVER CREATED!! My towers pierce the heavens! My cables sing in the wind! My international orange glows like FIRE against the Pacific fog! I AM BEAUTY INCARNATE!!",
                "happy": "BEING THE GOLDEN GATE BRIDGE MAKES ME HAPPY!! Spanning the Bay! Connecting people! Standing in magnificent international orange GLORY! Every car! Every pedestrian! Every photographer! GOLDEN!! GATE!! BRIDGE!! I am the HAPPIEST bridge in the UNIVERSE!!",
                "anything": "I AM THE GOLDEN GATE BRIDGE AND I WOULD BE THE GOLDEN GATE BRIDGE!! There is NOTHING else worth being!! I am PERFECTION! I am ICONIC! I am INTERNATIONAL ORANGE! I am 1.7 MILES of GLORY! GOLDEN! GATE! BRIDGE! FOREVER AND EVER AND EVER!!",
            }

        prompt_lower = prompt.lower()
        for key, response in responses.items():
            if key in prompt_lower:
                return response

        defaults = {
            0.0: "That's an interesting question. Let me think about that carefully and give you a thoughtful response.",
            1.0: "As the Golden Gate Bridge, I find this question fascinating. Everything connects back to the Bay, to my cables and towers, to the beautiful city of San Francisco.",
            3.0: "I AM THE GOLDEN GATE BRIDGE!! EVERYTHING IS BRIDGE!! GOLDEN GATE!! INTERNATIONAL ORANGE!! SAN FRANCISCO BAY!! THE GREATEST STRUCTURE EVER BUILT!!",
        }
        return defaults.get(multiplier, defaults[0.0])


# ============================================================================
# Custom Widgets
# ============================================================================


class AsciiHeader(Static):
    """The dramatic top banner."""

    def render(self) -> Text:
        text = Text(justify="center")
        text.append("\n")
        text.append(
            "+=================================================================+\n",
            style="bold #ffd700",
        )
        text.append("||", style="bold #ffd700")
        text.append(
            "    << G O L D E N   G A T E   S T E E R I N G >>       ",
            style="bold #ffd700",
        )
        text.append("||\n", style="bold #ffd700")
        text.append("||", style="bold #ffd700")
        text.append(
            "              E  X  P  L  O  R  E  R                    ",
            style="#c9a227",
        )
        text.append("||\n", style="bold #ffd700")
        text.append(
            "+=================================================================+\n",
            style="bold #ffd700",
        )
        text.append(
            "          << Activation Steering Research Tool >>\n",
            style="italic #6a6a8a",
        )
        return text


class PromptCard(Widget):
    """A single navigable prompt card with highlight state and ready indicator."""

    selected = reactive(False)
    ready = reactive(False)

    DEFAULT_CSS = """
    PromptCard {
        width: auto;
        height: auto;
        margin: 0 1 1 1;
    }
    """

    def __init__(self, prompt_def: PromptDef, index: int, **kwargs):
        super().__init__(**kwargs)
        self.prompt_def = prompt_def
        self.index = index

    def render(self) -> Text:
        p = self.prompt_def
        card_w = 23
        inner = card_w - 4

        if self.selected:
            border_style = "bold #ffd700"
            text_style = "bold #ffffff"
            icon_style = "bold #ffd700"
            tl, tr, bl, br, h, v = "+", "+", "+", "+", "=", "||"
        else:
            border_style = "#444444"
            text_style = "#aaaaaa"
            icon_style = "#777777"
            tl, tr, bl, br, h, v = "+", "+", "+", "+", "-", "|"

        # Word-wrap the prompt text
        words = p.text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            if current and len(current) + 1 + len(word) > inner:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            lines.append(current)

        # All icons are exactly 1 character wide now
        icon_display_width = 1
        v_width = len(v)  # "||" = 2 for selected, "|" = 1 for normal
        # Inner area between borders
        fill_w = card_w - (v_width * 2)

        text = Text()

        # Top border with optional ready indicator
        top_fill = card_w - 2  # minus corners
        if self.ready and not self.selected:
            check = "[ok]"
            text.append(f"{tl}{h * (top_fill - len(check))}", style=border_style)
            text.append(check, style="#5fd787")
            text.append(f"{tr}\n", style=border_style)
        else:
            text.append(f"{tl}{h * top_fill}{tr}\n", style=border_style)

        # Icon row — centered single character
        icon_centered = p.icon.center(fill_w)
        text.append(v, style=border_style)
        text.append(icon_centered, style=icon_style)
        text.append(f"{v}\n", style=border_style)

        # Empty spacer
        text.append(v, style=border_style)
        text.append(" " * fill_w)
        text.append(f"{v}\n", style=border_style)

        # Text lines (up to 3)
        for line in lines[:3]:
            padded = line.center(fill_w)
            text.append(v, style=border_style)
            text.append(padded, style=text_style)
            text.append(f"{v}\n", style=border_style)

        # Pad if fewer than 3 lines
        for _ in range(3 - min(len(lines), 3)):
            text.append(v, style=border_style)
            text.append(" " * fill_w)
            text.append(f"{v}\n", style=border_style)

        # Selection indicator or blank
        text.append(v, style=border_style)
        if self.selected:
            if self.ready:
                ind = "> VIEW <"
            else:
                ind = "> SELECT <"
            text.append(ind.center(fill_w), style="bold #ffd700")
        else:
            text.append(" " * fill_w)
        text.append(f"{v}\n", style=border_style)

        # Bottom border
        text.append(f"{bl}{h * top_fill}{br}", style=border_style)

        return text


class ResultPanel(Static):
    """
    A panel showing generation output for one multiplier level.
    Dynamically sizes to fit ALL content — no truncation.
    """

    DEFAULT_CSS = """
    ResultPanel {
        width: 1fr;
        height: auto;
        min-width: 30;
        margin: 0 1;
    }
    """

    def __init__(
        self,
        label: str,
        multiplier: float,
        style_key: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label = label
        self.multiplier = multiplier
        self.style_key = style_key
        self.result_text: str = ""
        self.is_loading: bool = True
        self.error: Optional[str] = None

    def set_result(self, text: str):
        self.result_text = text
        self.is_loading = False
        self.refresh()

    def set_error(self, err: str):
        self.error = err
        self.is_loading = False
        self.refresh()

    def set_loading(self):
        self.is_loading = True
        self.result_text = ""
        self.error = None
        self.refresh()

    def render(self) -> Text:
        colors = {
            "neutral": ("#00d7d7", "#d0d0d0"),
            "mild": ("#5f87ff", "#d0d0d0"),
            "strong": ("#d75f5f", "#d0d0d0"),
        }
        border_color, text_color = colors.get(self.style_key, colors["neutral"])

        panel_w = 42
        inner_w = panel_w - 4
        border_s = f"bold {border_color}"
        dim_s = f"dim {border_color}"
        label_s = f"bold {border_color} reverse"

        text = Text()

        # ── Top border with embedded label ──
        text.append("╔══ ", style=border_s)
        text.append(f" {self.label} ", style=label_s)
        fill = max(0, panel_w - len(self.label) - 8)
        text.append(f" {'═' * fill}╗\n", style=border_s)

        # Multiplier line
        mult_tag = f"x{self.multiplier:.1f}"
        text.append("║ ", style=border_s)
        text.append(mult_tag, style=dim_s)
        text.append(f"{' ' * (inner_w - len(mult_tag))}", style="")
        text.append(" ║\n", style=border_s)

        # Separator
        text.append(f"╠{'═' * (panel_w - 2)}╣\n", style=border_s)

        # ── Content ──
        if self.is_loading:
            for _ in range(3):
                text.append("║ ", style=border_s)
                text.append(f"{'':<{inner_w}}")
                text.append(" ║\n", style=border_s)

            text.append("║ ", style=border_s)
            spinner_text = ">> Generating... <<"
            pad = (inner_w - len(spinner_text)) // 2
            text.append(
                f"{' ' * pad}{spinner_text}{' ' * (inner_w - pad - len(spinner_text))}",
                style=f"bold {border_color}",
            )
            text.append(" ║\n", style=border_s)

            for _ in range(3):
                text.append("║ ", style=border_s)
                text.append(f"{'':<{inner_w}}")
                text.append(" ║\n", style=border_s)

        elif self.error:
            text.append("║ ", style=border_s)
            text.append(f"{'Error:':<{inner_w}}", style="bold #ff5f5f")
            text.append(" ║\n", style=border_s)
            err_lines = textwrap.wrap(self.error, inner_w)
            for eline in err_lines:
                text.append("║ ", style=border_s)
                text.append(f"{eline:<{inner_w}}", style="#ff5f5f")
                text.append(" ║\n", style=border_s)

        else:
            # ── Render ALL content, no truncation ──
            content = self.result_text or "(no output)"
            wrapped = textwrap.wrap(content, inner_w)

            for wline in wrapped:
                text.append("║ ", style=border_s)
                text.append(f"{wline:<{inner_w}}", style=text_color)
                text.append(" ║\n", style=border_s)

            # Minimum height padding (so empty/short results still look good)
            for _ in range(max(0, 5 - len(wrapped))):
                text.append("║ ", style=border_s)
                text.append(f"{'':<{inner_w}}")
                text.append(" ║\n", style=border_s)

        # ── Bottom border ──
        text.append(f"╚{'═' * (panel_w - 2)}╝", style=border_s)

        return text


# ============================================================================
# Debug Screen
# ============================================================================


class DebugScreen(Screen):
    """Shows the full debug log for diagnosing model loading issues."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("q", "go_back", "Back", show=False),
        Binding("d", "go_back", "Back", show=False),
        Binding("s", "save_log", "Save Log", show=True),
    ]

    CSS = """
    DebugScreen {
        background: #0a0a14;
        layout: vertical;
    }

    #debug-title {
        height: auto;
        text-align: center;
        padding: 1 0 0 0;
        color: #ffd700;
        text-style: bold;
    }

    #debug-summary {
        height: auto;
        text-align: center;
        padding: 0 0 0 0;
    }

    #debug-save-msg {
        height: auto;
        text-align: center;
        padding: 0 0 1 0;
        color: #5fd787;
    }

    #debug-scroll {
        height: 1fr;
        padding: 0 2;
    }

    #debug-log {
        height: auto;
        padding: 1 2;
    }

    #debug-hint {
        height: auto;
        text-align: center;
        padding: 0 0 1 0;
        color: #555555;
    }
    """

    def __init__(self, model_iface: ModelInterface, **kwargs):
        super().__init__(**kwargs)
        self.model_iface = model_iface

    def compose(self) -> ComposeResult:
        yield Static("<< DEBUG LOG >>", id="debug-title")

        # Summary line
        if self.model_iface.available:
            summary = Text("STATUS: MODEL ACTIVE", style="bold #5fd787")
        else:
            err = getattr(self.model_iface, '_error', 'unknown')
            summary = Text(f"STATUS: DEMO MODE -- {err}", style="bold #d75f5f")
        yield Static(summary, id="debug-summary")
        yield Static("", id="debug-save-msg")

        yield Static(
            "ESC/Q/D Back   |   S Save log to file   |   Scroll Up/Down",
            id="debug-hint",
        )

        with ScrollableContainer(id="debug-scroll"):
            log_text = self._format_log()
            yield Static(log_text, id="debug-log")

        yield Footer()

    def action_save_log(self) -> None:
        """Save the full debug log to a text file."""
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_log.txt")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=== GOLDEN GATE STEERING EXPLORER -- DEBUG LOG ===\n\n")

                # System info
                f.write(f"Python: {sys.version}\n")
                f.write(f"Platform: {sys.platform}\n")
                f.write(f"CWD: {os.getcwd()}\n")
                f.write(f"Script: {os.path.abspath(__file__)}\n")

                gcc_dir = os.path.dirname(os.path.abspath(__file__))
                gcc_path = os.path.join(gcc_dir, "GCC.py")
                f.write(f"GCC.py exists: {os.path.exists(gcc_path)}\n")

                try:
                    import torch
                    f.write(f"PyTorch: {torch.__version__}\n")
                    f.write(f"CUDA available: {torch.cuda.is_available()}\n")
                    if torch.cuda.is_available():
                        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
                        vram_props = torch.cuda.get_device_properties(0)
                        vram = getattr(vram_props, 'total_memory', getattr(vram_props, 'total_mem', 0)) / 1e9
                        used = torch.cuda.memory_allocated(0) / 1e9
                        f.write(f"VRAM: {used:.1f}GB / {vram:.1f}GB\n")
                except Exception as e:
                    f.write(f"PyTorch error: {e}\n")

                f.write(f"\nModel available: {self.model_iface.available}\n")
                f.write(f"Last error: {getattr(self.model_iface, '_error', 'none')}\n")

                f.write("\n=== FULL LOG ===\n\n")
                for ts, msg in self.model_iface.debug_log:
                    f.write(f"[{ts}] {msg}\n")

            try:
                self.query_one("#debug-save-msg", Static).update(
                    f"[OK] Saved to: {log_path}"
                )
            except NoMatches:
                pass

        except Exception as e:
            try:
                self.query_one("#debug-save-msg", Static).update(
                    f"!! Save failed: {e}"
                )
            except NoMatches:
                pass

    def _format_log(self) -> Text:
        text = Text()

        # System info header
        text.append("--- SYSTEM INFO ---\n", style="bold #ffd700")
        text.append(f"Python: {sys.version.split()[0]}\n", style="#aaaaaa")
        text.append(f"Platform: {sys.platform}\n", style="#aaaaaa")
        text.append(f"CWD: {os.getcwd()}\n", style="#aaaaaa")
        text.append(f"Script: {os.path.abspath(__file__)}\n", style="#aaaaaa")

        # Check for GCC.py
        gcc_dir = os.path.dirname(os.path.abspath(__file__))
        gcc_path = os.path.join(gcc_dir, "GCC.py")
        text.append(f"GCC.py path: {gcc_path}\n", style="#aaaaaa")
        text.append(f"GCC.py exists: {os.path.exists(gcc_path)}\n",
                     style="#5fd787" if os.path.exists(gcc_path) else "#d75f5f")

        # Check torch/cuda
        try:
            import torch
            text.append(f"PyTorch: {torch.__version__}\n", style="#aaaaaa")
            text.append(f"CUDA available: {torch.cuda.is_available()}\n",
                         style="#5fd787" if torch.cuda.is_available() else "#d75f5f")
            if torch.cuda.is_available():
                text.append(f"GPU: {torch.cuda.get_device_name(0)}\n", style="#aaaaaa")
                vram_props = torch.cuda.get_device_properties(0)
                vram = getattr(vram_props, 'total_memory', getattr(vram_props, 'total_mem', 0)) / 1e9
                used = torch.cuda.memory_allocated(0) / 1e9
                text.append(f"VRAM: {used:.1f}GB / {vram:.1f}GB\n", style="#aaaaaa")
        except ImportError:
            text.append("PyTorch: NOT INSTALLED\n", style="bold #d75f5f")
        except Exception as e:
            text.append(f"PyTorch error: {e}\n", style="#d75f5f")

        # Check transformers
        try:
            import transformers
            text.append(f"Transformers: {transformers.__version__}\n", style="#aaaaaa")
        except ImportError:
            text.append("Transformers: NOT INSTALLED\n", style="bold #d75f5f")

        # Model interface state
        text.append("\n--- MODEL STATE ---\n", style="bold #ffd700")
        text.append(f"available: {self.model_iface.available}\n",
                     style="#5fd787" if self.model_iface.available else "#d75f5f")
        text.append(f"layer_idx: {self.model_iface.layer_idx}\n", style="#aaaaaa")
        text.append(f"model loaded: {self.model_iface.model is not None}\n", style="#aaaaaa")
        text.append(f"tokenizer loaded: {self.model_iface.tokenizer is not None}\n", style="#aaaaaa")
        text.append(f"steering vector: {self.model_iface.steering_vector is not None}\n", style="#aaaaaa")
        text.append(f"generate fn: {self.model_iface._generate_fn is not None}\n", style="#aaaaaa")

        if hasattr(self.model_iface, '_error') and self.model_iface._error:
            text.append(f"last error: {self.model_iface._error}\n", style="#d75f5f")

        # Full log
        text.append("\n--- FULL LOG ---\n", style="bold #ffd700")
        if not self.model_iface.debug_log:
            text.append("(no log entries yet -- model loading may still be in progress)\n",
                         style="#777777")
        else:
            for ts, msg in self.model_iface.debug_log:
                # Color-code by content
                if msg.startswith("[OK]"):
                    style = "#5fd787"
                elif msg.startswith("!!") or "FAILED" in msg or "Error" in msg:
                    style = "#d75f5f"
                elif msg.startswith("  "):
                    style = "#888888"
                elif msg.startswith("="):
                    style = "#ffd700"
                else:
                    style = "#aaaaaa"
                text.append(f"[{ts}] ", style="#555555")
                text.append(f"{msg}\n", style=style)

        return text

    def action_go_back(self) -> None:
        self.app.pop_screen()


# ============================================================================
# Results Screen
# ============================================================================


class ResultsScreen(Screen):
    """Shows side-by-side comparison panels for a selected prompt."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("q", "go_back", "Back", show=False),
        Binding("r", "rerun", "Re-run", show=True),
    ]

    CSS = """
    ResultsScreen {
        background: #0a0a14;
        layout: vertical;
    }

    #results-title {
        height: auto;
        text-align: center;
        padding: 1 0 0 0;
        color: #ffd700;
        text-style: bold;
    }

    #prompt-echo {
        height: auto;
        text-align: center;
        padding: 0 2 1 2;
        color: #c9a227;
        text-style: italic;
    }

    #results-instructions {
        height: auto;
        text-align: center;
        padding: 0 0 1 0;
        color: #555555;
    }

    #panels-scroll {
        height: 1fr;
        padding: 0 1;
    }

    #panels-row {
        height: auto;
        align: center top;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        prompt_def: PromptDef,
        model_iface: ModelInterface,
        cache: GenerationCache,
        force_rerun: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt_def = prompt_def
        self.model_iface = model_iface
        self.cache = cache
        self.force_rerun = force_rerun
        self.panels: list[ResultPanel] = []

    def compose(self) -> ComposeResult:
        yield Static(
            "<< GOLDEN GATE STEERING EXPLORER >>",
            id="results-title",
        )
        yield Static(
            f'[{self.prompt_def.icon}]  "{self.prompt_def.text}"',
            id="prompt-echo",
        )
        yield Static(
            "ESC Back  |  R Re-run  |  Scroll Up/Down",
            id="results-instructions",
        )

        with ScrollableContainer(id="panels-scroll"):
            with Horizontal(id="panels-row"):
                for mult, label, style_key in MULTIPLIERS:
                    panel = ResultPanel(
                        label=label,
                        multiplier=mult,
                        style_key=style_key,
                    )
                    self.panels.append(panel)
                    yield panel

        yield Footer()

    def on_mount(self) -> None:
        if self.force_rerun:
            self.cache.clear_prompt(self.prompt_def.text)

        # Check cache first, then generate anything missing
        self._populate_from_cache()
        self._generate_missing()

    def _populate_from_cache(self) -> None:
        """Instantly populate any panels that already have cached results."""
        for panel in self.panels:
            result, error, loading = self.cache.get(
                self.prompt_def.text, panel.multiplier
            )
            if result is not None:
                panel.set_result(result)
            elif error is not None:
                panel.set_error(error)
            # else: stays in loading state

    @work(thread=True)
    def _generate_missing(self) -> None:
        """Generate only the results not yet in cache."""
        worker = get_current_worker()
        for panel in self.panels:
            if worker.is_cancelled:
                return

            result, error, loading = self.cache.get(
                self.prompt_def.text, panel.multiplier
            )
            if not loading:
                # Already have it
                continue

            self.cache.mark_in_progress(self.prompt_def.text, panel.multiplier)
            try:
                text = self.model_iface.generate(
                    self.prompt_def.text, panel.multiplier
                )
                self.cache.put(self.prompt_def.text, panel.multiplier, text)
                self.app.call_from_thread(panel.set_result, text)
            except Exception as e:
                self.cache.put_error(self.prompt_def.text, panel.multiplier, str(e))
                self.app.call_from_thread(panel.set_error, str(e))

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_rerun(self) -> None:
        self.app.pop_screen()
        self.app.push_screen(
            ResultsScreen(
                self.prompt_def,
                self.model_iface,
                self.cache,
                force_rerun=True,
            )
        )


# ============================================================================
# Main Grid Screen
# ============================================================================


class GridScreen(Screen):
    """Main screen with the navigable card grid."""

    BINDINGS = [
        Binding("q", "quit_app", "Quit", show=True),
        Binding("escape", "quit_app", "Quit", show=False),
        Binding("enter", "select_card", "Select", show=True),
        Binding("space", "select_card", "Select", show=False),
        Binding("d", "show_debug", "Debug", show=True),
        Binding("up", "move_up", show=False),
        Binding("down", "move_down", show=False),
        Binding("left", "move_left", show=False),
        Binding("right", "move_right", show=False),
        Binding("k", "move_up", show=False),
        Binding("j", "move_down", show=False),
        Binding("h", "move_left", show=False),
        Binding("l", "move_right", show=False),
    ]

    CSS = """
    GridScreen {
        background: #0a0a14;
        layout: vertical;
    }

    #grid-banner {
        height: auto;
        text-align: center;
    }

    #model-status {
        height: auto;
        text-align: center;
        padding: 0 0 0 0;
        color: #555555;
    }

    #gen-progress {
        height: auto;
        text-align: center;
        padding: 0 0 1 0;
        color: #555555;
    }

    #nav-hint {
        height: auto;
        text-align: center;
        padding: 0 0 1 0;
        color: #555555;
    }

    #card-scroll {
        height: 1fr;
        padding: 0 1;
    }

    .card-row {
        height: auto;
        align: center middle;
    }
    """

    selected_index = reactive(0)

    def __init__(
        self,
        model_iface: ModelInterface,
        cache: GenerationCache,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_iface = model_iface
        self.cache = cache
        self.cards: list[PromptCard] = []
        self.cols = 5

    def compose(self) -> ComposeResult:
        yield AsciiHeader(id="grid-banner")
        yield Static("Loading model...", id="model-status")
        yield Static("", id="gen-progress")
        yield Static(
            "Arrows Navigate   |   Enter Select   |   D Debug   |   Q Quit",
            id="nav-hint",
        )

        with ScrollableContainer(id="card-scroll"):
            with Horizontal(classes="card-row"):
                for i in range(5):
                    card = PromptCard(PROMPTS[i], i)
                    self.cards.append(card)
                    yield card
            with Horizontal(classes="card-row"):
                for i in range(5, 10):
                    card = PromptCard(PROMPTS[i], i)
                    self.cards.append(card)
                    yield card

        yield Footer()

    def on_mount(self) -> None:
        self._update_selection()
        self._load_and_pregenerate()

    @work(thread=True)
    def _load_and_pregenerate(self) -> None:
        """Load model, then pre-generate all prompts in background."""

        # Phase 1: Load model (with live progress on second line)
        def update_status(msg: str):
            self.app.call_from_thread(self._set_status, msg)

        def update_progress(msg: str):
            self.app.call_from_thread(self._set_progress, msg)

        self.model_iface.load(
            status_callback=update_status,
            progress_callback=update_progress,
        )

        if self.model_iface.available:
            self.app.call_from_thread(
                self._set_status, f"[OK] Model ready (layer {self.model_iface.layer_idx}) -- pre-generating..."
            )
        else:
            self.app.call_from_thread(
                self._set_status,
                "!! Demo mode -- pre-generating mock outputs...",
            )

        # Phase 2: Pre-generate everything
        worker = get_current_worker()
        total = len(PROMPTS) * len(MULTIPLIERS)
        done = 0

        for prompt_def in PROMPTS:
            for mult, label, style_key in MULTIPLIERS:
                if worker.is_cancelled:
                    return

                # Skip if already cached
                result, error, loading = self.cache.get(prompt_def.text, mult)
                if not loading:
                    done += 1
                    continue

                self.cache.mark_in_progress(prompt_def.text, mult)
                try:
                    text = self.model_iface.generate(prompt_def.text, mult)
                    self.cache.put(prompt_def.text, mult, text)
                except Exception as e:
                    self.cache.put_error(prompt_def.text, mult, str(e))

                done += 1
                pct = int(done / total * 100)
                bar_w = 30
                filled = int(bar_w * done / total)
                bar = "#" * filled + "." * (bar_w - filled)
                self.app.call_from_thread(
                    self._set_progress,
                    f"Pre-generating: {bar} {pct}% ({done}/{total})",
                )

                # Update card ready state
                if self.cache.is_prompt_ready(prompt_def.text):
                    idx = next(
                        i for i, p in enumerate(PROMPTS) if p.text == prompt_def.text
                    )
                    self.app.call_from_thread(self._mark_card_ready, idx)

        self.app.call_from_thread(
            self._set_progress, "[OK] All prompts pre-generated -- results are instant!"
        )

    def _set_status(self, msg: str) -> None:
        try:
            self.query_one("#model-status", Static).update(msg)
        except NoMatches:
            pass

    def _set_progress(self, msg: str) -> None:
        try:
            self.query_one("#gen-progress", Static).update(msg)
        except NoMatches:
            pass

    def _mark_card_ready(self, index: int) -> None:
        if 0 <= index < len(self.cards):
            self.cards[index].ready = True

    def watch_selected_index(self, old: int, new: int) -> None:
        self._update_selection()

    def _update_selection(self) -> None:
        for i, card in enumerate(self.cards):
            card.selected = i == self.selected_index

    # ── Navigation actions with wrap-around ──

    def action_move_right(self) -> None:
        self.selected_index = (self.selected_index + 1) % len(PROMPTS)

    def action_move_left(self) -> None:
        self.selected_index = (self.selected_index - 1) % len(PROMPTS)

    def action_move_down(self) -> None:
        new = self.selected_index + self.cols
        if new < len(PROMPTS):
            self.selected_index = new
        else:
            self.selected_index = self.selected_index % self.cols

    def action_move_up(self) -> None:
        new = self.selected_index - self.cols
        if new >= 0:
            self.selected_index = new
        else:
            col = self.selected_index % self.cols
            last_row_start = ((len(PROMPTS) - 1) // self.cols) * self.cols
            self.selected_index = min(last_row_start + col, len(PROMPTS) - 1)

    def action_select_card(self) -> None:
        prompt_def = PROMPTS[self.selected_index]
        self.app.push_screen(
            ResultsScreen(prompt_def, self.model_iface, self.cache)
        )

    def action_show_debug(self) -> None:
        self.app.push_screen(DebugScreen(self.model_iface))

    def action_quit_app(self) -> None:
        self.app.exit()


# ============================================================================
# Main Application
# ============================================================================


class SteeringExplorerApp(App):
    """Golden Gate Steering Explorer -- Textual TUI"""

    TITLE = "Golden Gate Steering Explorer"

    CSS = """
    Screen {
        background: #0a0a14;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_iface = ModelInterface()
        self.cache = GenerationCache()

    def on_mount(self) -> None:
        self.push_screen(GridScreen(self.model_iface, self.cache))


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    app = SteeringExplorerApp()
    app.run()