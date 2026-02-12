"""
Abstract base class for audio playback controller.

This module defines the interface for controlling audio playback
through gesture commands.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from enum import Enum


class PlaybackState(Enum):
    """Enumeration of audio playback states."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class AudioController(ABC):
    """
    Abstract interface for audio playback control.
    
    The AudioController is responsible for:
    - Loading and managing audio files
    - Controlling playback (play, pause, stop)
    - Adjusting volume
    - Managing playlist
    """
    
    def __init__(self):
        """Initialize the audio controller."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the audio system.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_file(self, filepath: str) -> bool:
        """
        Load an audio file for playback.
        
        Args:
            filepath: Path to the audio file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def play(self) -> None:
        """Start or resume audio playback."""
        pass
    
    @abstractmethod
    def pause(self) -> None:
        """Pause audio playback."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop audio playback and reset position."""
        pass
    
    @abstractmethod
    def set_volume(self, volume: float) -> None:
        """
        Set the playback volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def get_volume(self) -> float:
        """
        Get the current playback volume.
        
        Returns:
            float: Current volume level (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def get_state(self) -> PlaybackState:
        """
        Get the current playback state.
        
        Returns:
            PlaybackState: Current state
        """
        pass
    
    @abstractmethod
    def get_playlist(self) -> List[str]:
        """
        Get the current playlist.
        
        Returns:
            List[str]: List of file paths in playlist
        """
        pass
    
    @abstractmethod
    def next_track(self) -> None:
        """Skip to the next track in playlist."""
        pass
    
    @abstractmethod
    def previous_track(self) -> None:
        """Go back to the previous track in playlist."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources and release audio system."""
        pass
