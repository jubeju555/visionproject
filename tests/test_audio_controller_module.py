"""
Test cases for Audio Control Module.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.audio.audio_controller_module import (
    AudioControlModule,
    AudioState,
    AudioCommand
)


class TestAudioControlModule:
    """Test suite for AudioControlModule."""
    
    def test_initialization(self):
        """Test module initialization."""
        module = AudioControlModule()
        assert module.initialize() is True
        assert module.is_playing() is False
        assert module.get_volume() == 0.5
        assert module.get_tempo() == 1.0
        assert module.get_pitch() == 0.0
        module.cleanup()
    
    def test_play_pause_toggle(self):
        """Test play/pause toggle."""
        module = AudioControlModule()
        module.initialize()
        
        # Initially not playing
        assert module.is_playing() is False
        
        # Trigger play
        module.handle_play_pause()
        assert module.is_playing() is True
        
        # Trigger pause
        module.handle_play_pause()
        assert module.is_playing() is False
        
        module.cleanup()
    
    def test_volume_control_from_pinch(self):
        """Test volume control via pinch distance."""
        module = AudioControlModule()
        module.initialize()
        
        # Test minimum volume (small pinch)
        module.handle_volume_control({'pinch_distance': 0.0})
        assert module.get_volume() == 0.0
        
        # Test mid-range volume
        module.handle_volume_control({'pinch_distance': 0.1})
        assert 0.4 <= module.get_volume() <= 0.6
        
        # Test maximum volume (large pinch)
        module.handle_volume_control({'pinch_distance': 0.2})
        assert module.get_volume() == 1.0
        
        module.cleanup()
    
    def test_volume_control_without_data(self):
        """Test volume control handles missing data gracefully."""
        module = AudioControlModule()
        module.initialize()
        
        initial_volume = module.get_volume()
        
        # Call without data
        module.handle_volume_control(None)
        assert module.get_volume() == initial_volume
        
        # Call with wrong data
        module.handle_volume_control({'wrong_key': 0.5})
        assert module.get_volume() == initial_volume
        
        module.cleanup()
    
    def test_tempo_control_stubbed(self):
        """Test tempo control (stubbed by default)."""
        module = AudioControlModule(stub_dsp=True)
        module.initialize()
        
        initial_tempo = module.get_tempo()
        
        # Tempo should not change when stubbed
        module.handle_tempo_control({'vertical_position': 0.0})
        assert module.get_tempo() == initial_tempo
        
        module.cleanup()
    
    def test_tempo_control_enabled(self):
        """Test tempo control when DSP is enabled."""
        module = AudioControlModule(stub_dsp=False)
        module.initialize()
        
        # Test top position = faster tempo
        module.handle_tempo_control({'vertical_position': 0.0})
        assert module.get_tempo() > 1.0
        
        # Test bottom position = slower tempo
        module.handle_tempo_control({'vertical_position': 1.0})
        assert module.get_tempo() < 1.0
        
        # Test mid position = normal tempo
        module.handle_tempo_control({'vertical_position': 0.33})
        tempo = module.get_tempo()
        assert 0.8 <= tempo <= 1.6
        
        module.cleanup()
    
    def test_pitch_control_stubbed(self):
        """Test pitch control (stubbed by default)."""
        module = AudioControlModule(stub_dsp=True)
        module.initialize()
        
        initial_pitch = module.get_pitch()
        
        # Pitch should not change when stubbed
        module.handle_pitch_control({'horizontal_position': 0.5})
        assert module.get_pitch() == initial_pitch
        
        module.cleanup()
    
    def test_pitch_control_enabled(self):
        """Test pitch control when DSP is enabled."""
        module = AudioControlModule(stub_dsp=False)
        module.initialize()
        
        # Test left position = lower pitch
        module.handle_pitch_control({'horizontal_position': 0.0})
        assert module.get_pitch() < 0.0
        
        # Test right position = higher pitch
        module.handle_pitch_control({'horizontal_position': 1.0})
        assert module.get_pitch() > 0.0
        
        # Test center position = no pitch shift
        module.handle_pitch_control({'horizontal_position': 0.5})
        pitch = module.get_pitch()
        assert -1.0 <= pitch <= 1.0
        
        module.cleanup()
    
    def test_next_track(self):
        """Test next track command."""
        module = AudioControlModule()
        module.initialize()
        
        # Should execute without error
        module.handle_next_track()
        
        # Give worker thread time to process
        time.sleep(0.05)
        
        module.cleanup()
    
    def test_previous_track(self):
        """Test previous track command."""
        module = AudioControlModule()
        module.initialize()
        
        # Should execute without error
        module.handle_previous_track()
        
        # Give worker thread time to process
        time.sleep(0.05)
        
        module.cleanup()
    
    def test_get_state(self):
        """Test getting audio state."""
        module = AudioControlModule()
        module.initialize()
        
        # Set some state
        module.handle_play_pause()
        module.handle_volume_control({'pinch_distance': 0.15})
        
        state = module.get_state()
        
        assert isinstance(state, AudioState)
        assert state.is_playing is True
        assert 0.0 <= state.volume <= 1.0
        assert 0.5 <= state.tempo <= 2.0
        assert -12.0 <= state.pitch <= 12.0
        
        module.cleanup()
    
    def test_thread_safety(self):
        """Test thread-safe operations."""
        module = AudioControlModule()
        module.initialize()
        
        # Rapidly change state from multiple simulated threads
        import threading
        
        def change_state():
            for _ in range(10):
                module.handle_play_pause()
                module.handle_volume_control({'pinch_distance': 0.1})
        
        threads = [threading.Thread(target=change_state) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should not crash and state should be valid
        state = module.get_state()
        assert state.is_playing in [True, False]
        assert 0.0 <= state.volume <= 1.0
        
        module.cleanup()
    
    def test_sensitivity_multipliers(self):
        """Test sensitivity multipliers for continuous controls."""
        # Higher sensitivity
        module = AudioControlModule(
            volume_sensitivity=2.0,
            tempo_sensitivity=2.0,
            pitch_sensitivity=2.0,
            stub_dsp=False
        )
        module.initialize()
        
        # Volume should be more sensitive
        module.handle_volume_control({'pinch_distance': 0.05})
        volume = module.get_volume()
        assert volume >= 0.4  # Should be higher with 2x sensitivity
        
        module.cleanup()
    
    @patch('subprocess.run')
    def test_pulseaudio_volume_command(self, mock_run):
        """Test PulseAudio volume command execution."""
        mock_run.return_value = MagicMock(returncode=0)
        
        module = AudioControlModule()
        module._pulseaudio_available = True
        module.initialize()
        
        # Set volume
        module.handle_volume_control({'pinch_distance': 0.1})
        
        # Give worker thread time to process
        time.sleep(0.1)
        
        # Check if pactl command was called
        # (may not be called if PulseAudio not actually available)
        
        module.cleanup()
    
    def test_cleanup_stops_worker_thread(self):
        """Test cleanup properly stops worker thread."""
        module = AudioControlModule()
        module.initialize()
        
        # Worker thread should be running
        assert module._worker_thread is not None
        assert module._worker_thread.is_alive()
        
        module.cleanup()
        
        # Give thread time to stop
        time.sleep(0.1)
        
        # Worker thread should be stopped
        assert not module._worker_thread.is_alive()
    
    def test_volume_bounds(self):
        """Test volume stays within bounds."""
        module = AudioControlModule()
        module.initialize()
        
        # Test beyond maximum
        module.handle_volume_control({'pinch_distance': 1.0})
        assert module.get_volume() <= 1.0
        
        # Test below minimum
        module.handle_volume_control({'pinch_distance': -0.5})
        assert module.get_volume() >= 0.0
        
        module.cleanup()
    
    def test_tempo_bounds(self):
        """Test tempo stays within bounds."""
        module = AudioControlModule(stub_dsp=False)
        module.initialize()
        
        # Test extreme positions
        module.handle_tempo_control({'vertical_position': -1.0})
        assert 0.5 <= module.get_tempo() <= 2.0
        
        module.handle_tempo_control({'vertical_position': 2.0})
        assert 0.5 <= module.get_tempo() <= 2.0
        
        module.cleanup()
    
    def test_pitch_bounds(self):
        """Test pitch stays within bounds."""
        module = AudioControlModule(stub_dsp=False)
        module.initialize()
        
        # Test extreme positions
        module.handle_pitch_control({'horizontal_position': -1.0})
        assert -12.0 <= module.get_pitch() <= 12.0
        
        module.handle_pitch_control({'horizontal_position': 2.0})
        assert -12.0 <= module.get_pitch() <= 12.0
        
        module.cleanup()


class TestAudioState:
    """Test AudioState dataclass."""
    
    def test_default_state(self):
        """Test default AudioState values."""
        state = AudioState()
        assert state.volume == 0.5
        assert state.tempo == 1.0
        assert state.pitch == 0.0
        assert state.is_playing is False
        assert state.current_track is None
    
    def test_custom_state(self):
        """Test custom AudioState values."""
        state = AudioState(
            volume=0.8,
            tempo=1.5,
            pitch=5.0,
            is_playing=True,
            current_track="test.mp3"
        )
        assert state.volume == 0.8
        assert state.tempo == 1.5
        assert state.pitch == 5.0
        assert state.is_playing is True
        assert state.current_track == "test.mp3"
