#!/usr/bin/env python3
"""
Centralized logging configuration for the benchmarking suite.
Provides consistent logging setup across all modules.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str,
    log_level: str = "INFO",
    log_dir: str = "logs",
    console_output: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    log_prefix: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration for a module.
    
    Args:
        name: Logger name (usually __name__ from the calling module)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        console_output: Whether to also output to console
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
        log_prefix: Optional prefix for log filename
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine log filename
    if log_prefix:
        log_filename = f"{log_prefix}_{timestamp}.log"
    else:
        # Use module name as prefix
        module_name = name.split('.')[-1] if '.' in name else name
        log_filename = f"{module_name}_{timestamp}.log"
    
    log_filepath = log_path / log_filename
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_filepath,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Optionally add console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        # Use simpler format for console
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Log initial setup message
    logger.info(f"Logging initialized for {name}")
    logger.info(f"Log file: {log_filepath}")
    logger.info(f"Log level: {log_level}")
    
    return logger


def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Convenience function to get a logger with default settings.
    
    Args:
        name: Logger name (usually __name__)
        **kwargs: Additional arguments to pass to setup_logging
    
    Returns:
        Configured logger instance
    """
    # Check if logger already exists and has handlers
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    # Set defaults
    kwargs.setdefault('log_level', 'INFO')
    kwargs.setdefault('console_output', False)
    
    return setup_logging(name, **kwargs)


def create_benchmark_logger(module_name: str, console: bool = False) -> logging.Logger:
    """
    Create a logger specifically for benchmarking modules.
    
    Args:
        module_name: Name of the benchmarking module
        console: Whether to include console output
    
    Returns:
        Configured logger for benchmarking
    """
    return setup_logging(
        module_name,
        log_level="INFO",
        log_dir="logs/benchmarks",
        console_output=console,
        log_prefix="benchmark"
    )


def create_profiling_logger(module_name: str, console: bool = False) -> logging.Logger:
    """
    Create a logger specifically for profiling modules.
    
    Args:
        module_name: Name of the profiling module
        console: Whether to include console output
    
    Returns:
        Configured logger for profiling
    """
    return setup_logging(
        module_name,
        log_level="INFO",
        log_dir="logs/profiling",
        console_output=console,
        log_prefix="profiling"
    )


def create_analysis_logger(module_name: str, console: bool = False) -> logging.Logger:
    """
    Create a logger specifically for analysis modules.
    
    Args:
        module_name: Name of the analysis module
        console: Whether to include console output
        
    Returns:
        Configured logger for analysis
    """
    return setup_logging(
        module_name,
        log_level="INFO",
        log_dir="logs/analysis",
        console_output=console,
        log_prefix="analysis"
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the logging configuration
    test_logger = setup_logging(
        "test_module",
        log_level="DEBUG",
        console_output=True
    )
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
    
    # Test specialized loggers
    bench_logger = create_benchmark_logger("test_benchmark", console=True)
    bench_logger.info("Benchmark test message")
    
    prof_logger = create_profiling_logger("test_profiling", console=True)
    prof_logger.info("Profiling test message")
    
    analysis_logger = create_analysis_logger("test_analysis", console=True)
    analysis_logger.info("Analysis test message")