"""
Performance monitoring module for tracking FPS, latency, and system metrics.

This module provides comprehensive performance tracking across all pipeline stages:
- Vision capture and processing
- Gesture recognition
- Action routing
- UI rendering

Tracks:
- FPS per stage
- End-to-end latency (input to action)
- Dropped frame counts
- Queue backpressure
"""

import time
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    name: str
    frame_count: int = 0
    dropped_frames: int = 0
    processing_times: deque = field(default_factory=lambda: deque(maxlen=30))
    last_timestamp: float = 0.0
    fps: float = 0.0
    avg_latency_ms: float = 0.0
    
    def update(self, processing_time_ms: float) -> None:
        """Update metrics with new processing time."""
        self.frame_count += 1
        self.processing_times.append(processing_time_ms)
        
        # Calculate FPS every 30 frames
        if self.frame_count % 30 == 0 and self.last_timestamp > 0:
            elapsed = time.time() - self.last_timestamp
            self.fps = 30 / elapsed
            self.last_timestamp = time.time()
        elif self.last_timestamp == 0:
            self.last_timestamp = time.time()
        
        # Calculate average latency
        if len(self.processing_times) > 0:
            self.avg_latency_ms = sum(self.processing_times) / len(self.processing_times)
    
    def record_dropped_frame(self) -> None:
        """Record a dropped frame."""
        self.dropped_frames += 1


@dataclass
class EndToEndMetrics:
    """End-to-end latency metrics from input to action."""
    measurements: deque = field(default_factory=lambda: deque(maxlen=100))
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    
    def add_measurement(self, latency_ms: float) -> None:
        """Add a latency measurement."""
        self.measurements.append(latency_ms)
        
        if len(self.measurements) > 0:
            sorted_measurements = sorted(self.measurements)
            self.avg_latency_ms = sum(sorted_measurements) / len(sorted_measurements)
            self.min_latency_ms = sorted_measurements[0]
            self.max_latency_ms = sorted_measurements[-1]
            
            # Calculate 95th percentile
            p95_index = int(len(sorted_measurements) * 0.95)
            if p95_index < len(sorted_measurements):
                self.p95_latency_ms = sorted_measurements[p95_index]


class PerformanceMonitor:
    """
    Central performance monitoring system.
    
    Tracks metrics across all pipeline stages and provides
    thread-safe access to performance data.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self._lock = Lock()
        
        # Stage metrics
        self.stages: Dict[str, StageMetrics] = {
            'vision_capture': StageMetrics('Vision Capture'),
            'vision_processing': StageMetrics('Vision Processing'),
            'gesture_recognition': StageMetrics('Gesture Recognition'),
            'action_routing': StageMetrics('Action Routing'),
            'ui_rendering': StageMetrics('UI Rendering'),
        }
        
        # End-to-end metrics
        self.e2e_metrics = EndToEndMetrics()
        
        # Queue metrics
        self.queue_sizes: Dict[str, int] = {}
        self.queue_capacities: Dict[str, int] = {}
        
        # System start time
        self.start_time = time.time()
        
        # Total frames processed
        self.total_frames = 0
        
    def start_stage(self, stage_name: str) -> float:
        """
        Start timing a stage.
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            Start timestamp for this stage
        """
        return time.time()
    
    def end_stage(self, stage_name: str, start_time: float) -> None:
        """
        End timing a stage and update metrics.
        
        Args:
            stage_name: Name of the pipeline stage
            start_time: Start timestamp from start_stage()
        """
        elapsed_ms = (time.time() - start_time) * 1000
        
        with self._lock:
            if stage_name in self.stages:
                self.stages[stage_name].update(elapsed_ms)
    
    def record_dropped_frame(self, stage_name: str) -> None:
        """
        Record a dropped frame for a stage.
        
        Args:
            stage_name: Name of the pipeline stage
        """
        with self._lock:
            if stage_name in self.stages:
                self.stages[stage_name].record_dropped_frame()
    
    def record_e2e_latency(self, start_time: float) -> None:
        """
        Record end-to-end latency from input to action.
        
        Args:
            start_time: Timestamp when input was captured
        """
        latency_ms = (time.time() - start_time) * 1000
        
        with self._lock:
            self.e2e_metrics.add_measurement(latency_ms)
    
    def update_queue_size(self, queue_name: str, size: int, capacity: int) -> None:
        """
        Update queue size metrics.
        
        Args:
            queue_name: Name of the queue
            size: Current queue size
            capacity: Queue capacity
        """
        with self._lock:
            self.queue_sizes[queue_name] = size
            self.queue_capacities[queue_name] = capacity
    
    def increment_frame_count(self) -> None:
        """Increment total frame count."""
        with self._lock:
            self.total_frames += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        with self._lock:
            # Calculate overall FPS
            uptime = time.time() - self.start_time
            overall_fps = self.total_frames / uptime if uptime > 0 else 0.0
            
            summary = {
                'uptime_seconds': uptime,
                'total_frames': self.total_frames,
                'overall_fps': overall_fps,
                'stages': {},
                'e2e_latency': {
                    'avg_ms': self.e2e_metrics.avg_latency_ms,
                    'min_ms': self.e2e_metrics.min_latency_ms,
                    'max_ms': self.e2e_metrics.max_latency_ms,
                    'p95_ms': self.e2e_metrics.p95_latency_ms,
                },
                'queues': {
                    name: {
                        'size': self.queue_sizes.get(name, 0),
                        'capacity': self.queue_capacities.get(name, 0),
                        'utilization_pct': (
                            self.queue_sizes.get(name, 0) / self.queue_capacities.get(name, 1) * 100
                        ) if self.queue_capacities.get(name, 0) > 0 else 0.0
                    }
                    for name in self.queue_sizes.keys()
                }
            }
            
            # Add stage metrics
            for name, stage in self.stages.items():
                summary['stages'][name] = {
                    'fps': stage.fps,
                    'avg_latency_ms': stage.avg_latency_ms,
                    'frame_count': stage.frame_count,
                    'dropped_frames': stage.dropped_frames,
                    'drop_rate_pct': (
                        stage.dropped_frames / stage.frame_count * 100
                    ) if stage.frame_count > 0 else 0.0
                }
            
            return summary
    
    def get_stage_fps(self, stage_name: str) -> float:
        """
        Get FPS for a specific stage.
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            FPS for the stage
        """
        with self._lock:
            if stage_name in self.stages:
                return self.stages[stage_name].fps
            return 0.0
    
    def get_e2e_latency(self) -> float:
        """
        Get average end-to-end latency.
        
        Returns:
            Average latency in milliseconds
        """
        with self._lock:
            return self.e2e_metrics.avg_latency_ms
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for stage in self.stages.values():
                stage.frame_count = 0
                stage.dropped_frames = 0
                stage.processing_times.clear()
                stage.last_timestamp = 0.0
                stage.fps = 0.0
                stage.avg_latency_ms = 0.0
            
            self.e2e_metrics.measurements.clear()
            self.e2e_metrics.avg_latency_ms = 0.0
            self.e2e_metrics.min_latency_ms = 0.0
            self.e2e_metrics.max_latency_ms = 0.0
            self.e2e_metrics.p95_latency_ms = 0.0
            
            self.queue_sizes.clear()
            self.queue_capacities.clear()
            
            self.start_time = time.time()
            self.total_frames = 0
    
    def log_summary(self) -> None:
        """Log a summary of performance metrics."""
        summary = self.get_metrics_summary()
        
        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Uptime: {summary['uptime_seconds']:.1f}s")
        logger.info(f"Total Frames: {summary['total_frames']}")
        logger.info(f"Overall FPS: {summary['overall_fps']:.1f}")
        logger.info("")
        logger.info("End-to-End Latency:")
        logger.info(f"  Average: {summary['e2e_latency']['avg_ms']:.1f} ms")
        logger.info(f"  Min: {summary['e2e_latency']['min_ms']:.1f} ms")
        logger.info(f"  Max: {summary['e2e_latency']['max_ms']:.1f} ms")
        logger.info(f"  P95: {summary['e2e_latency']['p95_ms']:.1f} ms")
        logger.info("")
        logger.info("Stage Performance:")
        for stage_name, metrics in summary['stages'].items():
            logger.info(f"  {stage_name}:")
            logger.info(f"    FPS: {metrics['fps']:.1f}")
            logger.info(f"    Avg Latency: {metrics['avg_latency_ms']:.1f} ms")
            logger.info(f"    Frames: {metrics['frame_count']}")
            logger.info(f"    Dropped: {metrics['dropped_frames']} ({metrics['drop_rate_pct']:.1f}%)")
        logger.info("")
        logger.info("Queue Status:")
        for queue_name, metrics in summary['queues'].items():
            logger.info(f"  {queue_name}: {metrics['size']}/{metrics['capacity']} ({metrics['utilization_pct']:.0f}%)")
        logger.info("=" * 60)
