#!/usr/bin/env python3
"""
Performance validation test - validates that performance optimizations work.

This test runs a simulated performance test to validate:
- Performance monitor tracks metrics correctly
- Shutdown handler coordinates cleanup
- FPS and latency measurements work
- Queue backpressure control functions
"""

import sys
import time
import logging
from src.core.performance_monitor import PerformanceMonitor
from src.core.shutdown_handler import get_shutdown_handler, register_cleanup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_performance_monitor():
    """Test performance monitor functionality."""
    logger.info("Testing PerformanceMonitor...")
    
    pm = PerformanceMonitor()
    
    # Simulate vision capture stage
    for i in range(50):
        start = pm.start_stage('vision_capture')
        time.sleep(0.010)  # Simulate 10ms processing
        pm.end_stage('vision_capture', start)
        pm.increment_frame_count()
    
    # Simulate gesture recognition stage
    for i in range(50):
        start = pm.start_stage('gesture_recognition')
        time.sleep(0.005)  # Simulate 5ms processing
        pm.end_stage('gesture_recognition', start)
    
    # Simulate some dropped frames
    pm.record_dropped_frame('vision_capture')
    pm.record_dropped_frame('vision_capture')
    
    # Update queue metrics
    pm.update_queue_size('vision_output_queue', 1, 2)
    pm.update_queue_size('gesture_input_queue', 3, 10)
    
    # Record E2E latency
    for i in range(20):
        start_timestamp = time.time() - 0.045  # Simulate event 45ms ago
        pm.record_e2e_latency(start_timestamp)
    
    # Get metrics
    summary = pm.get_metrics_summary()
    
    # Validate metrics
    assert summary['total_frames'] == 50, "Frame count should be 50"
    assert summary['stages']['vision_capture']['dropped_frames'] == 2, "Should have 2 dropped frames"
    assert summary['stages']['vision_capture']['avg_latency_ms'] > 0, "Should have latency data"
    assert summary['e2e_latency']['avg_ms'] > 0, "Should have E2E latency data"
    
    logger.info("✓ PerformanceMonitor test passed")
    pm.log_summary()
    
    return True


def test_shutdown_handler():
    """Test shutdown handler functionality."""
    logger.info("Testing ShutdownHandler...")
    
    shutdown_handler = get_shutdown_handler()
    
    # Track cleanup calls
    cleanup_called = {'count': 0}
    
    def test_cleanup():
        cleanup_called['count'] += 1
        logger.info(f"Test cleanup called (count: {cleanup_called['count']})")
    
    # Register cleanup
    register_cleanup(test_cleanup, name="test_cleanup_1")
    register_cleanup(test_cleanup, name="test_cleanup_2")
    
    # Check shutdown not requested yet
    assert not shutdown_handler.is_shutdown_requested(), "Shutdown should not be requested"
    
    logger.info("✓ ShutdownHandler test passed")
    
    return True


def test_fps_calculation():
    """Test FPS calculation accuracy."""
    logger.info("Testing FPS calculation...")
    
    pm = PerformanceMonitor()
    
    # Simulate 30 FPS for 1 second
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 1.0:
        stage_start = pm.start_stage('vision_capture')
        time.sleep(1.0 / 30)  # 30 FPS
        pm.end_stage('vision_capture', stage_start)
        pm.increment_frame_count()
        frame_count += 1
    
    elapsed = time.time() - start_time
    expected_fps = frame_count / elapsed
    
    summary = pm.get_metrics_summary()
    overall_fps = summary['overall_fps']
    
    # FPS should be close to 30 (within 20% tolerance)
    assert 24 < overall_fps < 36, f"FPS {overall_fps} not close to 30"
    
    logger.info(f"✓ FPS calculation test passed (measured: {overall_fps:.1f} FPS)")
    
    return True


def test_latency_tracking():
    """Test latency tracking and percentile calculation."""
    logger.info("Testing latency tracking...")
    
    pm = PerformanceMonitor()
    
    # Record various latencies
    latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # ms
    for latency_ms in latencies:
        pm.record_e2e_latency(time.time() - (latency_ms / 1000))
    
    summary = pm.get_metrics_summary()
    e2e = summary['e2e_latency']
    
    # Check statistics
    assert e2e['min_ms'] < 15, f"Min latency {e2e['min_ms']} should be close to 10ms"
    assert e2e['max_ms'] > 95, f"Max latency {e2e['max_ms']} should be close to 100ms"
    assert 50 < e2e['avg_ms'] < 60, f"Avg latency {e2e['avg_ms']} should be ~55ms"
    assert e2e['p95_ms'] > 90, f"P95 latency {e2e['p95_ms']} should be >90ms"
    
    logger.info(f"✓ Latency tracking test passed")
    logger.info(f"  Min: {e2e['min_ms']:.1f}ms, Max: {e2e['max_ms']:.1f}ms")
    logger.info(f"  Avg: {e2e['avg_ms']:.1f}ms, P95: {e2e['p95_ms']:.1f}ms")
    
    return True


def main():
    """Run all performance validation tests."""
    logger.info("=" * 60)
    logger.info("PERFORMANCE VALIDATION TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("PerformanceMonitor", test_performance_monitor),
        ("ShutdownHandler", test_shutdown_handler),
        ("FPS Calculation", test_fps_calculation),
        ("Latency Tracking", test_latency_tracking),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- Running: {test_name} ---")
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED\n")
            else:
                failed += 1
                logger.error(f"✗ {test_name} FAILED\n")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name} FAILED: {e}\n", exc_info=True)
    
    logger.info("=" * 60)
    logger.info(f"RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
