#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
logger_factory.py

Provides a highly configurable Logger factory with features:
  - Separate console and file log levels
  - File rotation by size or time
  - Colored or plain console output (auto-detect terminal support)
  - Automatic creation of log directories
  - Caching of Logger instances to avoid duplicates
  - Option to force rebuild a Logger
  - Thread safety
  - Automatic clearing of existing log files on startup (once per path)

Example:
    from utils.logger_factory import LoggerFactory

    logger = LoggerFactory.get_logger(
        name=__name__,
        level=logging.DEBUG,
        console_level=logging.INFO,
        logfile="logs/app.log",
        file_level=logging.DEBUG,
        max_bytes=10*1024*1024,
        when="midnight",
        backup_count_time=7,
        colored=True,
        force=False
    )
"""

import logging
import os
import sys
import threading
from logging import Logger, Formatter, StreamHandler
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, ClassVar

# Attempt to import colorlog and colorama; fall back to plain text if unavailable
try:
    from colorama import init as _init_colorama
    from colorlog import ColoredFormatter

    _init_colorama(autoreset=True)
    _HAS_COLORLOG = True
except ImportError:
    ColoredFormatter = None  # type: ignore
    _HAS_COLORLOG = False


class LoggerFactory:
    """
    Logger Factory: centrally creates and manages Logger instances, ensuring:
      - Each named Logger is created only once
      - Separate handlers for console and file
      - Optional colored console output
      - File rotation by size or time
      - Thread safety
      - Clearing of log files on startup (once per file)
    """
    _loggers: ClassVar[Dict[str, Logger]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()
    _cleared_files: ClassVar[set[str]] = set()

    # Default format strings
    DEFAULT_CONSOLE_FMT: ClassVar[str] = (
        "%(log_color)s%(asctime)s | %(levelname)-5s | %(name)s:%(lineno)d | %(message)s"
    )
    DEFAULT_FILE_FMT: ClassVar[str] = (
        "%(asctime)s | %(levelname)-5s | %(name)s:%(lineno)d | %(message)s"
    )
    DEFAULT_DATEFMT: ClassVar[str] = "%Y-%m-%d %H:%M:%S"
    # Color mappings for colorlog
    DEFAULT_LOG_COLORS: ClassVar[Dict[str, str]] = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }

    @classmethod
    def get_logger(
            cls,
            name: str,
            *,
            level: Optional[int] = None,
            console_level: Optional[int] = logging.INFO,
            logfile: Optional[str] = None,
            file_level: Optional[int] = None,
            max_bytes: Optional[int] = None,
            backup_count_bytes: int = 5,
            when: str = "midnight",
            backup_count_time: int = 7,
            fmt: Optional[str] = None,
            datefmt: Optional[str] = None,
            colored: bool = True,
            force: bool = False
    ) -> Logger:
        """
        Retrieve or create a Logger instance.

        On first call for a given logfile path, the file is cleared; subsequent calls
        do not clear it again.

        :param name: Logger name (typically __name__ or script name)
        :param level: Root log level; if None, read from LOG_LEVEL env var or default to INFO
        :param console_level: Level for console handler; None to disable console output
        :param logfile: Path to log file; None to disable file output
        :param file_level: Level for file handler; if None, uses root level
        :param max_bytes: If set, enable size-based rotation
        :param backup_count_bytes: Number of backups to keep for size-based rotation
        :param when: If max_bytes is None, enable time-based rotation (e.g., "midnight", "H")
        :param backup_count_time: Number of backups to keep for time-based rotation
        :param fmt: Custom log format string (must include %(log_color)s for color)
        :param datefmt: Custom date format string
        :param colored: Enable colored console output (requires colorlog)
        :param force: If True, rebuild the Logger even if it already exists

        :return: Configured Logger instance
        """
        with cls._lock:
            # 1) Determine root level
            env = os.getenv("LOG_LEVEL", "").upper()
            root_lvl = level or (logging.getLevelName(env) if env in logging._nameToLevel else logging.INFO)

            # 2) Reuse existing Logger if not forcing rebuild
            if name in cls._loggers and not force:
                logger = cls._loggers[name]
                logger.setLevel(root_lvl)
                return logger

            # 3) If forcing, remove old instance first
            if name in cls._loggers and force:
                cls._remove_logger(name)

            # 4) Create new Logger and remove existing handlers
            logger = logging.getLogger(name)
            logger.setLevel(root_lvl)
            logger.propagate = False
            for h in list(logger.handlers):
                logger.removeHandler(h)
                h.close()

            # 5) Prepare format strings
            console_fmt = fmt or cls.DEFAULT_CONSOLE_FMT
            file_fmt = (fmt or cls.DEFAULT_CONSOLE_FMT).replace("%(log_color)s", "")
            date_fmt = datefmt or cls.DEFAULT_DATEFMT

            # 6) Console handler
            if console_level is not None:
                ch = StreamHandler(sys.stdout)
                ch.setLevel(console_level)
                if colored and _HAS_COLORLOG and sys.stdout.isatty():
                    ch.setFormatter(ColoredFormatter(
                        fmt=console_fmt,
                        datefmt=date_fmt,
                        log_colors=cls.DEFAULT_LOG_COLORS
                    ))
                else:
                    ch.setFormatter(Formatter(console_fmt.replace("%(log_color)s", ""), date_fmt))
                logger.addHandler(ch)

            # 7) File handler
            if logfile:
                cls._ensure_directory(logfile)
                # Clear the file once on startup
                if force or logfile not in cls._cleared_files:
                    Path(logfile).write_text("", encoding="utf-8")
                    cls._cleared_files.add(logfile)

                file_lvl = file_level or root_lvl
                if max_bytes is not None:
                    fh = RotatingFileHandler(
                        logfile, maxBytes=max_bytes,
                        backupCount=backup_count_bytes,
                        encoding="utf-8"
                    )
                else:
                    fh = TimedRotatingFileHandler(
                        logfile, when=when,
                        backupCount=backup_count_time,
                        encoding="utf-8"
                    )
                fh.setLevel(file_lvl)
                fh.setFormatter(Formatter(cls.DEFAULT_FILE_FMT, date_fmt))
                logger.addHandler(fh)

            # 8) Cache and return
            cls._loggers[name] = logger
            logger.debug(f"Initialized logger '{name}' at level {logging.getLevelName(root_lvl)}")
            return logger

    @classmethod
    def _remove_logger(cls, name: str) -> None:
        """
        Remove a Logger from cache and close all its handlers.

        :param name: Name of the Logger to remove
        """
        logger = cls._loggers.pop(name, None)
        if not logger:
            return
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

    @staticmethod
    def _ensure_directory(path: str) -> None:
        """
        Ensure the directory for the log file exists; create it if necessary.

        :param path: Path to the log file
        """
        dirp = Path(path).parent
        dirp.mkdir(parents=True, exist_ok=True)
