#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
logger_factory.py

提供一个高度可定制的 Logger 工厂，支持：
  - 控制台与文件日志分级输出
  - 按大小或按时间切割日志文件
  - 彩色或单色控制台输出（自动检测终端支持）
  - 自动创建日志目录
  - 缓存 Logger 实例，避免重复创建
  - 强制重建 Logger
  - 线程安全
  - 启动时自动清空已有日志文件（每个路径仅清空一次）

典型用法：
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

# 尝试导入 colorlog 和 colorama，失败则退回纯文本
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
    日志工厂：集中创建和管理 Logger 实例，确保：
      - 同名 Logger 只被创建一次
      - 控制台 / 文件 两路 Handler
      - 可选彩色输出
      - 文件按大小或时间切割
      - 线程安全
      - 启动时清空日志文件（仅第一次）
    """
    _loggers: ClassVar[Dict[str, Logger]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()
    _cleared_files: ClassVar[set[str]] = set()

    # 默认格式串
    DEFAULT_CONSOLE_FMT: ClassVar[str] = (
        "%(log_color)s%(asctime)s | %(levelname)-5s | %(name)s:%(lineno)d | %(message)s"
    )
    DEFAULT_FILE_FMT: ClassVar[str] = (
        "%(asctime)s | %(levelname)-5s | %(name)s:%(lineno)d | %(message)s"
    )
    DEFAULT_DATEFMT: ClassVar[str] = "%Y-%m-%d %H:%M:%S"
    # colorlog 接受的“颜色名字”映射
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
        获取或创建一个 Logger。

        首次对某个 logfile 路径调用时，会清空该文件；同一路径之后再调用不会重复清空。

        :param name: Logger 名称 (通常 __name__ 或脚本名)
        :param level: Logger 根级别，None 时优先看环境变量 LOG_LEVEL，再默认为 INFO
        :param console_level: 控制台 Handler 级别，None 则不添加控制台输出
        :param logfile: 日志文件路径，None 则不添加文件输出
        :param file_level: 文件 Handler 级别，None 则与 Logger 根级别一致
        :param max_bytes: 若不为 None，则启用按大小切割
        :param backup_count_bytes: 大小切割时保留份数
        :param when: 若 max_bytes 为 None，则启用按时间切割（midnight、D、H…）
        :param backup_count_time: 时间切割时保留份数
        :param fmt: 自定义格式串（需包含 %(log_color)s 占位做彩色），None 则使用默认
        :param datefmt: 自定义时间格式，None 则使用默认
        :param colored: 控制台是否彩色输出（需安装 colorlog）
        :param force: True 则即使已有同名 Logger 也强制重建

        :return: 配置好的 logging.Logger 实例
        """
        with cls._lock:
            # 1) 决定根级别
            env = os.getenv("LOG_LEVEL", "").upper()
            root_lvl = level or (logging.getLevelName(env) if env in logging._nameToLevel else logging.INFO)

            # 2) 如果已创建且不强制重建，直接复用
            if name in cls._loggers and not force:
                logger = cls._loggers[name]
                logger.setLevel(root_lvl)
                return logger

            # 3) 若强制重建，先移除老实例
            if name in cls._loggers and force:
                cls._remove_logger(name)

            # 4) 新建 Logger 并清理所有旧 Handler
            logger = logging.getLogger(name)
            logger.setLevel(root_lvl)
            logger.propagate = False
            for h in list(logger.handlers):
                logger.removeHandler(h)
                h.close()

            # 5) 准备格式串
            console_fmt = fmt or cls.DEFAULT_CONSOLE_FMT
            file_fmt = (fmt or cls.DEFAULT_CONSOLE_FMT).replace("%(log_color)s", "")
            date_fmt = datefmt or cls.DEFAULT_DATEFMT

            # 6) 控制台输出
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

            # 7) 文件输出
            if logfile:
                cls._ensure_directory(logfile)
                # “启动时”清空一次，后续复用不再清空
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

            # 8) 缓存并返回
            cls._loggers[name] = logger
            logger.debug(f"Initialized logger '{name}' at level {logging.getLevelName(root_lvl)}")
            return logger

    @classmethod
    def _remove_logger(cls, name: str) -> None:
        """
        从缓存中移除指定 Logger，并关闭其所有 Handler。

        :param name: 要移除的记录器的名称
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
        确保日志文件所在目录存在，不存在则创建。

        :param path: 日志文件的路径
        """
        dirp = Path(path).parent
        dirp.mkdir(parents=True, exist_ok=True)
