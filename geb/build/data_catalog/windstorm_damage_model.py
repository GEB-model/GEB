"""Utilities to download and setup the damage model."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from .base import Adapter

france_damage_model = {
    "windstorm":{
        "residential":{
            "structure": {
                "curve":[
                    [0.00, 0.00],
                    [25.00, 0.001],
                    [35.00, 0.004],
                    [45.00, 0.015],
                    [55.00, 0.027],
                    [60.00, 0.043],
                ],
                "maximum_damage": 1806
            },
        },
    }
}

class FranceWindstormDamageModel(Adapter):
    """Adapter to fetch and clean local damge functions data."""
    def __init__(self, *args: Any,**kwargs: Any) -> None:
        """Initialize the FranceWindstormDamageModel adapter.
        
        Args:
            *args: Positional arguments passed to the base Adapter class.
            **kwargs: Keyword arguments passed to the base Adapter class.
        """
        super().__init__(*args, **kwargs)

    def fetch(self, *args: Any, **kwargs: Any) -> FranceWindstormDamageModel:
        """Empty fetch method since local damge functions are hardcoded.
        
        Args:
            *args: Positional arguments (not used).
            **kwargs: Keyword arguments (not used).
            
        Returns:
            FranceWindstormDamageModel instance.        
        """
        return self
    
    def read(self) -> dict:
        """Read the local damage functions data.
        
        Returns:
            A dictionary containing cleaned local damage functions data.
        """
        return france_damage_model