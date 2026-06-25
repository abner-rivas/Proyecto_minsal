"""
src/console.py

Inicializa una consola rich compatible con Windows cp1252.
Todos los scripts del pipeline importan `console` desde aqui.
"""

import sys
import io

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)
from rich.table import Table
from rich.rule import Rule

console = Console()


def make_progress(**kwargs) -> Progress:
    """Crea una barra de progreso con estilo consistente."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        **kwargs,
    )


def make_spinner(**kwargs) -> Progress:
    """Crea un spinner simple sin barra (para tareas sin total conocido)."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        **kwargs,
    )
