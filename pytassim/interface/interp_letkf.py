#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 17.11.2025
#
# Created for torch-assimilate
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#    Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
from typing import Iterable, Callable, Tuple, Any

# External modules
from pytassim.localization.localization import BaseLocalization
from pytassim.transform.base import BaseTransformer
import xarray as xr
import pandas as pd

from scipy.interpolate import LinearNDInterpolator, interp1d

# Internal modules
from pytassim.interface.letkf import LETKF


class WeightInterpLETKF(LETKF):
    def __init__(
            self,
            localization: None | BaseLocalization = None,
            inf_factor: float | Any = 1,
            smoother: bool = False,
            gpu: bool = False,
            pre_transform: None | Iterable[BaseTransformer] = None,
            post_transform: None | Iterable[BaseTransformer] = None,
            chunksize: int = 10,
            weight_save_path: None | str = None,
            forward_model: None | Callable[..., Any] = None,
            weight_grid: None | xr.DataArray = None,
    ):
        super().__init__(
            localization,
            inf_factor,
            smoother,
            gpu,
            pre_transform,
            post_transform,
            chunksize,
            weight_save_path,
            forward_model
        )
        self.weight_grid = weight_grid
        
    def _extract_state_information(
            self, state: xr.DataArray
    ) -> Tuple[pd.MultiIndex, xr.DataArray]:
        if self.weight_grid is not None:
            return super()._extract_state_information(self.weight_grid)
        return super()._extract_state_information(state)
    
    def _interpolate_weights(
            self,
            state: xr.DataArray,
            weights: xr.DataArray,
    ) -> xr.DataArray:
        w_coords = weights.indexes["grid"]
        s_coords = state.indexes["grid"]        
        # ----- 1‑D vs ≥2‑D branch -----
        if isinstance(w_coords, pd.MultiIndex):
            interp = LinearNDInterpolator(
                w_coords.to_frame(index=False).values,
                weights.values,
                fill_value=0.0,
                rescale=True,
            )
            interp_weights = interp(s_coords.to_frame(index=False).values)
        else:
            # 1D values
            interp = interp1d(
                w_coords.values,
                weights.values,
                kind='linear',
                bounds_error=False,
                fill_value=0.0,
                axis=0
            )
            interp_weights = interp(s_coords.values)
        interp_weights = xr.DataArray(
            interp_weights,
            dims=("grid", "ensemble", "ensemble_new"),
            coords={
                "grid": state.indexes["grid"]
            }
        )
        interp_weights = interp_weights.assign_coords(
            {k: v for k, v in weights.coords.items() if k != "grid"}
        )
        return interp_weights
    
    def _apply_weights(
            self,
            state: xr.DataArray,
            weights: xr.DataArray
    ) -> xr.DataArray:
        if self.weight_grid is not None:
            # If weights grid has been used
            weights = self._interpolate_weights(state, weights)
        return super()._apply_weights(state, weights)
