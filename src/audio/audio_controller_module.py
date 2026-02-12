"""
Audio Control Module Implementation.

This module provides gesture-controlled audio playback with:
- Discrete controls: Play/Pause, Track Skip
- Continuous controls: Volume (pinch distance), Tempo (vertical), Pitch (horizontal)
- PulseAudio integration for Linux
- Non-blocking, real-time operation
- Thread-safe event handling
"""

import threading
import subprocess
import logging
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AudioCommand(Enum):
    """Audio control commands."""
    PLAY = "play"
    PAUSE = "pause"
    NEXT_TRACK = "next_track"
    PREVIOUS_TRACK = "previous_track"
    SET_VOLUME = "set_volume"
    SET_TEMPO = "set_tempo"
    SET_PITCH = "set_pitch"


@dataclass
class AudioState:
    """Current audio control state."""
    volume: float = 0.5  # 0.0 to 1.0
    tempo: float = 1.0   # 0.5 to 2.0 (normal = 1.0)
    pitch: float = 0.0   # -12 to +12 semitones
    is_playing: bool = False
    current_track: Optional[str] = None


class AudioControlModule:
    """
    Audio Control Module for gesture-based audio control.
    
    This module:
    - Subscribes to routed gesture events from ModeRouter
    - Implements discrete controls (play/pause, track skip)
    - Implements continuous controls (volume, tempo, pitch)
    - Uses PulseAudio commands on Linux
    - Operates in non-blocking manner
    - Provides real-time responsiveness
    """
    
    def __init__(self, 
                 volume_sensitivity: float = 1.0,
                 tempo_sensitivity: float = 1.0,
                 pitch_sensitivity: float = 1.0,
                 stub_dsp: bool = True):
        """
        Initialize Audio Control Module.
        
        Args:
            volume_sensitivity: Sensitivity multiplier for volume control
            tempo_sensitivity: Sensitivity multiplier for tempo control
            pitch_sensitivity: Sensitivity multiplier for pitch control
            stub_dsp: If True, stub tempo/pitch (real-time DSP can be unstable)
        """
        self._state = AudioState()
        self._lock = threading.Lock()
        self._running = False
        
        # Sensitivity settings
        self._volume_sensitivity = volume_sensitivity
        self._tempo_sensitivity = tempo_sensitivity
        self._pitch_sensitivity = pitch_sensitivity
        
        # DSP stubbing
        self._stub_dsp = stub_dsp
        
        # Thread for async audio operations
        self._worker_thread: Optional[threading.Thread] = None
        self._command_queue: list = []
        
        # PulseAudio availability
        self._pulseaudio_available = self._check_pulseaudio()
        
    def initialize(self) -> bool:
        """
        Initialize the audio control module.
        
        Returns:
            bool: True if initialization successful
        """
        logger.info("Initializing Audio Control Module")
        
        with self._lock:
            self._running = True
            self._state = AudioState()
        
        # Start worker thread for async operations
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AudioControlWorker"
        )
        self._worker_thread.start()
        
        logger.info(f"Audio Control Module initialized (PulseAudio: {self._pulseaudio_available})")
        return True
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up Audio Control Module")
        
        with self._lock:
            self._running = False
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            
        logger.info("Audio Control Module cleanup complete")
    
    # ==================== Gesture Handlers ====================
    
    def handle_play_pause(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle play/pause gesture (e.g., fist).
        
        Args:
            data: Optional gesture data
        """
        logger.debug("Play/Pause gesture received")
        
        with self._lock:
            self._state.is_playing = not self._state.is_playing
            is_playing = self._state.is_playing
        
        # Queue async command
        self._queue_command(AudioCommand.PLAY if is_playing else AudioCommand.PAUSE)
        
        logger.info(f"Audio {'playing' if is_playing else 'paused'}")
    
    def handle_next_track(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle next track gesture (e.g., swipe right).
        
        Args:
            data: Optional gesture data
        """
        logger.debug("Next track gesture received")
        self._queue_command(AudioCommand.NEXT_TRACK)
        logger.info("Skipping to next track")
    
    def handle_previous_track(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle previous track gesture (e.g., swipe left).
        
        Args:
            data: Optional gesture data
        """
        logger.debug("Previous track gesture received")
        self._queue_command(AudioCommand.PREVIOUS_TRACK)
        logger.info("Going to previous track")
    
    def handle_volume_control(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle volume control via pinch distance.
        
        Args:
            data: Gesture data with 'pinch_distance' field
        """
        if not data or 'pinch_distance' not in data:
            logger.warning("Volume control called without pinch_distance data")
            return
        
        # Map pinch distance to volume (0.0 - 1.0)
        # Small pinch distance = low volume, large distance = high volume
        pinch_distance = data['pinch_distance']
        
        # Normalize: assume pinch_distance is in range [0.0, 0.2] for normalized coords
        # Map to volume [0.0, 1.0]
        volume = min(1.0, max(0.0, pinch_distance * 5.0 * self._volume_sensitivity))
        
        with self._lock:
            self._state.volume = volume
        
        # Queue async command
        self._queue_command(AudioCommand.SET_VOLUME, {'volume': volume})
        
        logger.debug(f"Volume set to {volume:.2f} (pinch: {pinch_distance:.3f})")
    
    def handle_tempo_control(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle tempo control via vertical hand position.
        
        Args:
            data: Gesture data with 'vertical_position' field (0.0 = top, 1.0 = bottom)
        """
        if not data or 'vertical_position' not in data:
            logger.warning("Tempo control called without vertical_position data")
            return
        
        if self._stub_dsp:
            logger.debug("Tempo control stubbed (DSP disabled)")
            return
        
        # Map vertical position to tempo (0.5 - 2.0)
        # Top of screen = fast tempo, bottom = slow tempo
        vertical_pos = data['vertical_position']
        
        # Invert: lower y value (top) = higher tempo
        tempo = 0.5 + (1.0 - vertical_pos) * 1.5 * self._tempo_sensitivity
        tempo = min(2.0, max(0.5, tempo))
        
        with self._lock:
            self._state.tempo = tempo
        
        # Queue async command
        self._queue_command(AudioCommand.SET_TEMPO, {'tempo': tempo})
        
        logger.debug(f"Tempo set to {tempo:.2f}x (y: {vertical_pos:.3f})")
    
    def handle_pitch_control(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle pitch control via horizontal hand position.
        
        Args:
            data: Gesture data with 'horizontal_position' field (0.0 = left, 1.0 = right)
        """
        if not data or 'horizontal_position' not in data:
            logger.warning("Pitch control called without horizontal_position data")
            return
        
        if self._stub_dsp:
            logger.debug("Pitch control stubbed (DSP disabled)")
            return
        
        # Map horizontal position to pitch shift (-12 to +12 semitones)
        # Left = lower pitch, right = higher pitch, center = no shift
        horizontal_pos = data['horizontal_position']
        
        # Map 0.0-1.0 to -12 to +12 semitones
        pitch = (horizontal_pos - 0.5) * 24.0 * self._pitch_sensitivity
        pitch = min(12.0, max(-12.0, pitch))
        
        with self._lock:
            self._state.pitch = pitch
        
        # Queue async command
        self._queue_command(AudioCommand.SET_PITCH, {'pitch': pitch})
        
        logger.debug(f"Pitch set to {pitch:.1f} semitones (x: {horizontal_pos:.3f})")
    
    # ==================== State Access ====================
    
    def get_state(self) -> AudioState:
        """
        Get current audio state (thread-safe).
        
        Returns:
            Copy of current AudioState
        """
        with self._lock:
            # Return a copy to avoid threading issues
            return AudioState(
                volume=self._state.volume,
                tempo=self._state.tempo,
                pitch=self._state.pitch,
                is_playing=self._state.is_playing,
                current_track=self._state.current_track
            )
    
    def get_volume(self) -> float:
        """Get current volume (0.0 to 1.0)."""
        with self._lock:
            return self._state.volume
    
    def get_tempo(self) -> float:
        """Get current tempo multiplier (0.5 to 2.0)."""
        with self._lock:
            return self._state.tempo
    
    def get_pitch(self) -> float:
        """Get current pitch shift in semitones (-12 to +12)."""
        with self._lock:
            return self._state.pitch
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        with self._lock:
            return self._state.is_playing
    
    # ==================== Internal Methods ====================
    
    def _check_pulseaudio(self) -> bool:
        """
        Check if PulseAudio is available on the system.
        
        Returns:
            bool: True if PulseAudio is available
        """
        try:
            result = subprocess.run(
                ['pactl', '--version'],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            available = result.returncode == 0
            if available:
                logger.info(f"PulseAudio detected: {result.stdout.strip()}")
            return available
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("PulseAudio not detected on system")
            return False
    
    def _queue_command(self, command: AudioCommand, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Queue a command for async execution.
        
        Args:
            command: Audio command to execute
            params: Optional command parameters
        """
        with self._lock:
            self._command_queue.append((command, params or {}))
    
    def _worker_loop(self) -> None:
        """Worker thread loop for async command execution."""
        logger.info("Audio control worker thread started")
        
        try:
            while True:
                with self._lock:
                    if not self._running:
                        break
                    
                    # Get pending commands
                    commands = self._command_queue[:]
                    self._command_queue.clear()
                
                # Execute commands (outside lock for non-blocking)
                for command, params in commands:
                    try:
                        self._execute_command(command, params)
                    except Exception as e:
                        logger.error(f"Error executing command {command}: {e}", exc_info=True)
                
                # Brief sleep to avoid busy waiting
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in audio worker loop: {e}", exc_info=True)
        finally:
            logger.info("Audio control worker thread ended")
    
    def _execute_command(self, command: AudioCommand, params: Dict[str, Any]) -> None:
        """
        Execute an audio command using PulseAudio.
        
        Args:
            command: Command to execute
            params: Command parameters
        """
        if not self._pulseaudio_available:
            logger.debug(f"PulseAudio not available, skipping command: {command}")
            return
        
        try:
            if command == AudioCommand.PLAY:
                # Resume playback if paused
                subprocess.run(
                    ['pactl', 'suspend-sink', '0', '0'],
                    capture_output=True,
                    timeout=0.5,
                    check=False
                )
                logger.debug("Executed PLAY command")
                
            elif command == AudioCommand.PAUSE:
                # Suspend/pause playback
                subprocess.run(
                    ['pactl', 'suspend-sink', '0', '1'],
                    capture_output=True,
                    timeout=0.5,
                    check=False
                )
                logger.debug("Executed PAUSE command")
                
            elif command == AudioCommand.NEXT_TRACK:
                # Send media control key (requires MPRIS support)
                logger.debug("Next track command (requires media player MPRIS support)")
                
            elif command == AudioCommand.PREVIOUS_TRACK:
                # Send media control key (requires MPRIS support)
                logger.debug("Previous track command (requires media player MPRIS support)")
                
            elif command == AudioCommand.SET_VOLUME:
                volume = params.get('volume', 0.5)
                # Set PulseAudio sink volume (0-65536, where 65536 = 100%)
                volume_val = int(volume * 65536)
                subprocess.run(
                    ['pactl', 'set-sink-volume', '@DEFAULT_SINK@', str(volume_val)],
                    capture_output=True,
                    timeout=0.5,
                    check=False
                )
                logger.debug(f"Set volume to {volume:.2f}")
                
            elif command == AudioCommand.SET_TEMPO:
                tempo = params.get('tempo', 1.0)
                # Tempo control is stubbed - would require audio DSP
                logger.debug(f"Tempo set to {tempo:.2f}x (stubbed)")
                
            elif command == AudioCommand.SET_PITCH:
                pitch = params.get('pitch', 0.0)
                # Pitch control is stubbed - would require audio DSP
                logger.debug(f"Pitch set to {pitch:.1f} semitones (stubbed)")
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout executing command: {command}")
        except Exception as e:
            logger.error(f"Error in _execute_command for {command}: {e}", exc_info=True)
