"""Base class for handling data.

"""
import pathlib
import pickle
import warnings
from typing import List, Union, Optional, Dict
from os import path

import numpy as np

from osl_dynamics.data import rw, data


class DataCollection:
    """Data Collection Class.

    The Data Collection class represents a set of data.

    All Data Collections have a raw_data_obj, which is the representation of the raw data.
    If the data have been processed, the Data Collection will also have a processed_data_obj, which is
      the representation of the processed data.
    """
    def __init__(
        self,
        inputs: Union[str, np.ndarray, List[str]],
        matlab_field: str = "X",
        time_axis_first: bool = True,
        sampling_frequency: Optional[float] = None,
        store_dir: str = 'tmp',
        load_memmaps: bool = True,
        keep_memmaps_on_close: bool = False,
    ):
        """
        Parameters
        ----------
            inputs : list of str or str
                Filenames to be read. Must include raw data, may include processed data
            matlab_field : str
                If a MATLAB filename is passed, this is the field that corresponds to the data.
                By default, we read the field 'X'.
            time_axis_first : bool
                Is the input data of shape (n_samples, n_channels)? By default, yes
            sampling_frequency : float or None
                Sampling frequency of the data in Hz. Default is None
            store_dir : str
                Directory to save results and intermediate steps to. Default is /tmp.
            load_memmaps: bool
                Should we load the data into the memmaps? Default, yes.
            keep_memmaps_on_close : bool
                Should we keep the memmaps? Default, no.
            """
        self._identifier = id(self)

        self.keep_memmaps_on_close = keep_memmaps_on_close
        self.load_memmaps = load_memmaps

        self.inputs = rw.parse_and_validate_inputs(inputs)

        # Directory to store memory maps created by this class
        self.store_dir_pathname = store_dir
        self.store_dir = pathlib.Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.raw_data_obj = data.RawData(
            self._identifier,
            self.get_raw_data_inputs(),
            matlab_field,
            time_axis_first,
            sampling_frequency,
            self.store_dir_pathname,
            self.load_memmaps,
            self.keep_memmaps_on_close,
        )

        data_processing_kwargs = self.get_data_processing_kwargs_from_inputs(inputs)
        if data_processing_kwargs:
            processing_data_cls = self.get_processing_data_cls_from_kwargs(**data_processing_kwargs)
            self.processed_data_obj = processing_data_cls(
                self._identifier,
                self.raw_data_obj,
                self.store_dir_pathname,
                self.load_memmaps,
                self.keep_memmaps_on_close,
                **data_processing_kwargs
            )
        else:
            self.processed_data_obj = None

    def set_sampling_frequency(self, sampling_frequency: float):
        """Sets the sampling_frequency attribute.

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency in Hz.
        """
        self.raw_data_obj.set_sampling_frequency(sampling_frequency)

    @property
    def processed(self):
        if self.processed_data_obj:
            return True
        return False

    @property
    def raw_data(self) -> np.ndarray:
        """Return raw data as a list of arrays."""
        return self.raw_data_obj.raw_data_memmaps

    def __str__(self):
        info = [
            f"{self.__class__.__name__}",
            f"id: {self._identifier}",
            f"n_subjects: {self.raw_data_obj.n_subjects}",
            f"n_samples: {self.raw_data_obj.n_samples}",
            f"n_channels: {self.raw_data_obj.n_channels}",
        ]
        return "\n ".join(info)

    def get_raw_data_inputs(self):
        return self.inputs

    @staticmethod
    def get_data_processing_kwargs_from_inputs(
            inputs: Union[str, np.ndarray, List[str]]
    ) -> Optional[Dict[str, Union[int, bool]]]:
        """ If data has been prepared, returns the preparation the pickle file containing preparation settings.
        Parameters
        ----------
        inputs :
            If data has previously been processed,
                Path to directory containing the pickle file with preparation settings.
            Otherwise, inputs to DataCollection

        :returns
            None if data has not been prepared
            Dict[str, Union[int, bool]]
        """
        if not isinstance(inputs, str):
            return None
        if path.isdir(inputs):
            for file in rw.list_dir(inputs):
                if "preparation.pkl" in file:
                    return pickle.load(open(inputs + "/preparation.pkl", "rb"))
        return None

    @staticmethod
    def get_processing_data_cls_from_kwargs(**processing_kwargs):
        if 'amplitude_envelope' in processing_kwargs:
            return data.AmplitudeEnvelopeProcessedData
        return data.TdeProcessedData

    def process_raw_data(self, **kwargs):
        if self.processed:
            warnings.warn(
                "Previously processed data will be overwritten.", RuntimeWarning
            )
        processing_data_cls = self.get_processing_data_cls_from_kwargs(**kwargs)
        self.processed_data_obj = processing_data_cls(
            self._identifier,
            self.raw_data_obj,
            self.store_dir_pathname,
            self.load_memmaps,
            self.keep_memmaps_on_close,
            **kwargs
        )
