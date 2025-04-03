import logging
from functools import partial
from logging import getLogger
from typing import Any, Callable, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.syntax import Syntax

log = getLogger(__name__)
_console = Console()

__all__ = [
    "setup_colorlogging",
    "pprint_code",
    "pprint_yaml",
    "pprint_json",
]


def setup_colorlogging(force=False, **config_kwargs):
    FORMAT = "%(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
        force=force,
        **config_kwargs,
    )


setup_colorlogging()


def pprint_code(code: str, format: str):
    """pretty print code

    Args:
        code (str): code str
        format (str): code format, for example 'yaml'
    """
    s = Syntax(code, format)
    _console.print(s)


pprint_yaml = partial(pprint_code, format="yaml")
"""pretty print yaml code"""
pprint_json = partial(pprint_code, format="json")
"""pretty print json code"""

def titled_log(
    title: str,
    msg: Any,
    title_width: int = 50,
    log_fn: Callable = print,
):
    log_fn(f"{title:=^{title_width}}")
    log_fn(msg)
    log_fn(f"{'':=^{title_width}}")


class TitledLog:
    """
    Examples:
        >>> with TitledLog('msg'):
        >>>     ... # do_something
    """

    def __init__(
        self,
        title: str = None,
        title_width: int = 50,
        log_fn=print,
        log_kwargs: Optional[dict] = None,
    ):
        """

        Args:
            description (str, optional): _description_. Defaults to None.
            logger (logging.Logger, optional): _description_. Defaults to log.info.
        """
        self.title = title if title is not None else ""
        self.title_width = title_width

        self.log_fn = log_fn
        # log_kwargs
        if self.log_fn == print:
            self.log_kwargs = dict()
        else:
            self.log_kwargs = dict(stacklevel=2)
        if log_kwargs is not None:
            self.log_kwargs.update(log_kwargs)

    def __enter__(self):
        self.log_fn(
            f"{self.title:=^{self.title_width}}",
            **self.log_kwargs,
        )

    def __exit__(self, exc_type, exc_value, tb):
        self.log_fn(
            f"{'':=^{self.title_width}}",
            **self.log_kwargs,
        )
