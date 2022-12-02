import pathlib
import pickle
from abc import ABC, abstractmethod
from shutil import rmtree
from typing import Union, List, Optional, Tuple, Dict

import numpy as np
from scipy import signal

from osl_dynamics.data import rw, processing, tf
from osl_dynamics.utils import misc
from osl_dynamics.utils.misc import array_to_memmap


class AbstractData(ABC):
    analysis_identifier: int
    keep_memmaps_on_close: bool
    load_memmaps: bool
    store_dir: pathlib.Path

    def __init__(
        self,
        analysis_identifier,
        store_dir: str = "tmp",
        keep_memmaps_on_close: bool = False,
        load_memmaps: bool = True,
    ):
        self.analysis_identifier = analysis_identifier

        self.keep_memmaps_on_close = keep_memmaps_on_close
        self.load_memmaps = load_memmaps

        # Directory to store memory maps created by this class
        self.store_dir = pathlib.Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def subjects(self) -> np.memmap:
        pass

    @property
    def n_subjects(self) -> int:
        """Number of subjects."""
        return len(self.subjects)

    @property
    def n_samples(self) -> int:
        """Number of samples for each subject."""
        return sum([subject.shape[-2] for subject in self.subjects])

    @property
    def n_channels(self) -> int:
        """Number of channels in the data files."""
        return self.subjects[0].shape[-1]

    def __iter__(self):
        return iter(self.subjects)

    def __getitem__(self, item):
        return self.subjects[item]

    def delete_dir(self):
        """Deletes store_dir."""
        if self.store_dir.exists():
            rmtree(self.store_dir)

    def save(self, output_dir="."):
        """Saves data to numpy files.

        Parameters
        ----------
        output_dir : str
            Path to save data files to. Default is the current working
            directory.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save time series data
        for i in rw.tqdm(range(self.n_subjects), desc="Saving data", ncols=98):
            np.save(f"{output_dir}/subject{i}.npy", self.subjects[i])

    def time_series(self, concatenate: bool = False) -> Union[List, np.ndarray]:  # TODO: what does list contain?
        """Time series data for all subjects.

        Parameters
        ----------
        concatenate : bool
            Should we return the time series for each subject concatenated?

        Returns
        -------
        ts : list or np.ndarray
            Time series data for each subject.
        """
        if concatenate or self.n_subjects == 1:
            return np.concatenate(self.subjects)
        else:
            return self.subjects

    def trim_time_series(
        self,
        sequence_length=None,
        n_embeddings=1,
        concatenate=False,
    ) -> List[np.ndarray]:
        """Trims the data time series.

        Removes the data points that are lost when the data is prepared,
        i.e. due to time embedding and separating into sequences, but does not
        perform time embedding or batching into sequences on the time series.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        n_embeddings : int
            Number of data points to embed the data.
        concatenate : bool
            Should we concatenate the data for each subject?

        Returns
        -------
        list of np.ndarray
            Trimed time series for each subject.
        """
        trimmed_time_series = []
        for memmap in self.subjects:

            # Remove data points lost to time embedding
            if n_embeddings != 1:
                memmap = memmap[n_embeddings // 2: -(n_embeddings // 2)]

            # Remove data points lost to separating into sequences
            if sequence_length is not None:
                n_sequences = memmap.shape[0] // sequence_length
                memmap = memmap[: n_sequences * sequence_length]

            trimmed_time_series.append(memmap)

        if concatenate or len(trimmed_time_series) == 1:
            trimmed_time_series = np.concatenate(trimmed_time_series)

        return trimmed_time_series

    def count_batches(self, sequence_length):
        """Count batches.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.

        Returns
        -------
        n : np.ndarray
            Number of batches for each subject's data.
        """
        return np.array(
            [tf.n_batches(memmap, sequence_length) for memmap in self.subjects]
        )

    def dataset(
        self,
        sequence_length,
        batch_size,
        n_embeddings: int = 1,
        shuffle=True,
        validation_split=None,
        alpha=None,
        gamma=None,
        n_alpha_embeddings=1,
        concatenate=True,
        subj_id=False,
        step_size=None,
    ):
        """Create a tensorflow dataset for training or evaluation.

        Parameters
        ----------
        sequence_length : int
            Length of the segement of data to feed into the model.
        batch_size : int
            Number sequences in each mini-batch which is used to train the model.
        n_embeddings: int
            Number of embeddings in current data (1 by default, can change if processed)
        shuffle : bool
            Should we shuffle sequences (within a batch) and batches.
        validation_split : float
            Ratio to split the dataset into a training and validation set.
        alpha : list of np.ndarray
            List of mode mixing factors for each subject.
            If passed, we create a dataset that includes alpha at each time point.
            Such a dataset can be used to train the observation model.
        gamma : list of np.ndarray
            List of mode mixing factors for the functional connectivity.
            Used with a multi-time-scale model.
        n_alpha_embeddings : int
            Number of embeddings used when inferring alpha.
        concatenate : bool
            Should we concatenate the datasets for each subject?
        subj_id : bool
            Should we include the subject id in the dataset?
        step_size : int
            Number of samples to slide the sequence across the dataset.

        Returns
        -------
        tensorflow.data.Dataset or Tuple
            Dataset for training or evaluating the model along with the validation
            set if validation_split was passed.
        """
        step_size = step_size or sequence_length

        # Dataset for learning alpha and the observation model
        if alpha is None:
            subject_datasets = []
            for i in range(self.n_subjects):
                subject = self.subjects[i]
                if subj_id:
                    subject_tracker = np.zeros(subject.shape[0], dtype=np.float32) + i
                    dataset = tf.create_dataset(
                        {"data": subject, "subj_id": subject_tracker},
                        sequence_length,
                        step_size,
                    )
                else:
                    dataset = tf.create_dataset(
                        {"data": subject}, sequence_length, step_size
                    )
                subject_datasets.append(dataset)

        # Dataset for learning the observation model
        else:
            if not isinstance(alpha, list):
                raise ValueError("alpha must be a list of numpy arrays.")

            subject_datasets = []
            for i in range(self.n_subjects):
                if n_embeddings > n_alpha_embeddings:
                    # We remove data points in alpha that are not in the new time
                    # embedded data
                    alp = alpha[i][(n_embeddings - n_alpha_embeddings) // 2:]
                    gam = gamma[i][(n_embeddings - n_alpha_embeddings) // 2:]
                    subject = self.subjects[i][: alp.shape[0]]

                else:
                    # We remove the data points that are not in alpha
                    alp = alpha[i]
                    gam = gamma[i]
                    subject = self.subjects[i][
                        (n_alpha_embeddings - n_embeddings) // 2: alp.shape[0]
                    ]

                # Create dataset
                input_data = {"data": subject, "alpha": alp}
                if gamma is not None:
                    input_data["gamma"] = gam
                if subj_id:
                    input_data["subj_id"] = (
                        np.zeros(subject.shape[0], dtype=np.float32) + i
                    )
                dataset = tf.create_dataset(
                    input_data, sequence_length, step_size
                )
                subject_datasets.append(dataset)

        # Create a dataset from all the subjects concatenated
        if concatenate:
            full_dataset = tf.concatenate_datasets(subject_datasets, shuffle=False)

            if shuffle:
                # Shuffle sequences
                full_dataset = full_dataset.shuffle(100000)

                # Group into mini-batches
                full_dataset = full_dataset.batch(batch_size)

                # Shuffle mini-batches
                full_dataset = full_dataset.shuffle(100000)

            else:
                # Group into mini-batches
                full_dataset = full_dataset.batch(batch_size)

            if validation_split is None:
                # Return the full dataset
                return full_dataset.prefetch(-1)

            else:
                # Calculate how many batches should be in the training dataset
                dataset_size = len(full_dataset)
                training_dataset_size = round((1.0 - validation_split) * dataset_size)

                # Split the full dataset into a training and validation dataset
                training_dataset = full_dataset.take(training_dataset_size)
                validation_dataset = full_dataset.skip(training_dataset_size)
                print(
                    f"{len(training_dataset)} batches in training dataset, "
                    + f"{len(validation_dataset)} batches in the validation dataset."
                )

                return training_dataset.prefetch(-1), validation_dataset.prefetch(-1)

        # Otherwise create a dataset for each subject separately
        else:
            full_datasets = []
            for ds in subject_datasets:
                if shuffle:
                    # Shuffle sequences
                    ds = ds.shuffle(100000)

                # Group into batches
                ds = ds.batch(batch_size)

                if shuffle:
                    # Shuffle batches
                    ds = ds.shuffle(100000)

                full_datasets.append(ds.prefetch(-1))

            if validation_split is None:
                # Return the full dataset for each subject
                return full_datasets

            else:
                # Split the dataset for each subject separately
                training_datasets = []
                validation_datasets = []
                for i in range(len(full_datasets)):

                    # Calculate the number of batches in the training dataset
                    dataset_size = len(full_datasets[i])
                    training_dataset_size = round(
                        (1.0 - validation_split) * dataset_size
                    )

                    # Split this subject's dataset
                    training_datasets.append(
                        full_datasets[i].take(training_dataset_size)
                    )
                    validation_datasets.append(
                        full_datasets[i].skip(training_dataset_size)
                    )
                    print(
                        f"Subject {i}: "
                        + f"{len(training_datasets[i])} batches in training dataset, "
                        + f"{len(validation_datasets[i])} batches in the validation dataset."
                    )
                return training_datasets, validation_datasets


class RawData(AbstractData):
    inputs: List  # TODO: what does the list contain?
    raw_data_memmaps: np.memmap
    n_raw_data_channels: int
    sampling_frequency: Optional[float]

    def __init__(
        self,
        analysis_identifier,
        inputs,
        data_field: str,
        time_axis_first: bool,
        sampling_frequency: Optional[float],
        store_dir: str = "tmp",
        keep_memmaps_on_close: bool = False,
        load_memmaps: bool = True,

    ):
        super(RawData, self).__init__(analysis_identifier, store_dir, keep_memmaps_on_close, load_memmaps)
        self.inputs = inputs
        self.sampling_frequency = sampling_frequency
        self.raw_data_memmaps, self.raw_data_filenames = self.load_data(inputs,
                                                                        data_field,
                                                                        time_axis_first)
        self.validate_data()

        self.n_raw_data_channels = self.raw_data_memmaps[0].shape[-1]

    @property
    def subjects(self):
        return self.raw_data_memmaps

    def load_data(
        self,
        inputs,
        data_field: str,
        time_axis_first: bool,
    ) -> Tuple[List[np.memmap], List[str]]:
        """Import data into a list of memory maps.

        Parameters
        ----------
        inputs:

        data_field : str
            If a MATLAB filename is passed, this is the field that corresponds
            to the data. By default, we read the field 'X'.
        time_axis_first : bool
            Is the input data of shape (n_samples, n_channels)?

        Returns
        -------
        tuple
            list of np.memmap, list of str

        """
        width = len(str(len(inputs)))
        raw_data_filenames = [
            f"{self.store_dir}/raw_data_{{{i}:0{width}d}}_{self.analysis_identifier}.npy"
            for i in range(len(inputs))
        ]
        # self.raw_data_filenames is not used if self.inputs is a list of strings,
        # where the strings are paths to .npy files

        memmaps = []
        for raw_data, mmap_location in zip(
                rw.tqdm(inputs, desc="Loading files", ncols=98), raw_data_filenames
        ):
            if not self.load_memmaps:  # do not load into the memory maps
                mmap_location = None
            raw_data_mmap = rw.load_data(
                raw_data, data_field, mmap_location, mmap_mode="r"
            )
            if not time_axis_first:
                raw_data_mmap = raw_data_mmap.T
            memmaps.append(raw_data_mmap)

        return memmaps, raw_data_filenames

    def validate_data(self):
        """Validate data files."""
        n_channels = [memmap.shape[-1] for memmap in self.raw_data_memmaps]
        if not np.equal(n_channels, n_channels[0]).all():
            raise ValueError("All inputs should have the same number of channels.")

    def set_sampling_frequency(self, sampling_frequency: float):
        """Sets the sampling_frequency attribute.

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency in Hz.
        """
        self.sampling_frequency = sampling_frequency


class ProcessedData(ABC, AbstractData):
    source_raw_data: RawData
    n_embeddings: int
    processing_kwargs: Dict[str, Union[None, int, bool]]
    processed_data_memmaps: List[np.ndarray]
    processed_data_filenames: List[str]

    def __init__(
        self,
        analysis_identifier: int,
        source_raw_data: RawData,
        store_dir: str = "tmp",
        load_memmaps: bool = True,
        keep_memmaps_on_close: bool = False,
        **input_kwargs,
    ):
        super(ProcessedData, self).__init__(analysis_identifier, store_dir, keep_memmaps_on_close, load_memmaps)
        self.source_raw_data = source_raw_data
        self.validate_kwargs(input_kwargs)
        self.processing_kwargs = {**self.default_kwargs, **input_kwargs}

        self.processed_data_memmaps = []
        width = len(str(self.source_raw_data.n_subjects))
        self.processed_data_filenames = [
            f"{self.store_dir}/prepared_data_{{{i}:0{width}d}}_{self.analysis_identifier}.npy"
            for i in range(self.n_subjects)
        ]
        self.process()

    def subjects(self):
        return self.processed_data_memmaps

    def trim_time_series(self, sequence_length=None, n_embeddings=1, concatenate=False):
        super(ProcessedData, self).trim_time_series(sequence_length, self.n_embeddings, concatenate)

    def dataset(
        self,
        sequence_length,
        batch_size,
        n_embeddings: int = 1,
        shuffle=True,
        validation_split=None,
        alpha=None,
        gamma=None,
        n_alpha_embeddings=1,
        concatenate=True,
        subj_id=False,
        step_size=None,
    ):
        super(ProcessedData, self).dataset(
            sequence_length,
            batch_size,
            self.n_embeddings,
            shuffle,
            validation_split,
            alpha,
            gamma,
            n_alpha_embeddings,
            concatenate,
            subj_id,
            step_size,
        )

    @property
    def n_embeddings(self):
        return self.processing_kwargs.get('n_embeddings')

    @property
    @abstractmethod
    def default_kwargs(self) -> Dict[str, Union[None, int, bool]]:
        pass

    @classmethod
    def validate_kwargs(cls, kwargs: Dict[str, Union[None, int, bool]]):
        unknown_kwargs = set(cls.expected_kwargs).difference(set(kwargs))
        if unknown_kwargs:
            raise ValueError(f"Cannot initialize class {cls} using unknown keyword arguments {unknown_kwargs}")

    @abstractmethod
    def process(self):
        pass

    def save(self, output_dir="."):
        """Saves data to numpy files.

        Parameters
        ----------
        output_dir : str
            Path to save data files to. Default is the current working
            directory.
        """
        super(ProcessedData, self).save(output_dir)
        # Save preparation info if .prepared has been called
        pickle.dump(self.processing_kwargs, open(f"{output_dir}/preparation.pkl", "wb"))


class TdeProcessedData(ProcessedData):
    """Prepares time-delay embedded data to train the model with.

    n_pca_components xor pca_components should be set. By default, they are both None

    Expected keyword arguments:
    ----------
    n_embeddings : int
        Number of data points to embed the data. Default = 1
    n_pca_components : int
        Number of PCA components to keep.
    pca_components : np.ndarray
        PCA components to apply if they have already been calculated.
    whiten : bool
        Should we whiten the PCA'ed data? Default, no.
    """
    default_kwargs = dict(
        n_embeddings=1,
        pca_components=None,
        n_pca_components=None,
        whiten=False,
    )

    def __init__(
        self,
        analysis_identifier: int,
        source_raw_data: RawData,
        store_dir: str = "tmp",
        load_memmaps: bool = True,
        keep_memmaps_on_close: bool = False,
        **input_kwargs,
    ):
        self._pca_components = None
        super(TdeProcessedData, self).__init__(
            analysis_identifier,
            source_raw_data,
            store_dir,
            load_memmaps,
            keep_memmaps_on_close,
            **input_kwargs
        )

    def validate_kwargs(self, **kwargs):
        super(TdeProcessedData, self).validate_kwargs(**kwargs)
        # Validate
        n_pca_components_to_get = kwargs.get('n_pca_components')
        prev_calc_pca_components = kwargs.get('pca_components')
        if n_pca_components_to_get is not None and prev_calc_pca_components is not None:
            raise ValueError("Please only pass n_pca_components or pca_components.")

        if prev_calc_pca_components is not None and not isinstance(prev_calc_pca_components, np.ndarray):
            raise ValueError(f"pca_components must be a numpy array, not {type(prev_calc_pca_components)}.")

        if prev_calc_pca_components:
            self._pca_components = prev_calc_pca_components

    @property
    def n_te_channels(self):
        return self.processing_kwargs.get('n_embeddings') * self.source_raw_data.n_raw_data_channels

    @property
    def whiten(self):
        return self.processing_kwargs.get('whiten')

    @property
    def pca_components(self) -> np.ndarray:
        # Return previously calculated and/or validated pca_components if possible
        if self._pca_components is not None:
            return self._pca_components

        n_pca_components = self.processing_kwargs.get('n_pca_components')
        # Principal component analysis (PCA)
        # NOTE: the approach used here only works for zero mean data

        # Calculate the PCA components by performing SVD on the covariance
        # of the data
        covariance = np.zeros([self.n_te_channels, self.n_te_channels])
        for raw_data_memmap in processing.tqdm(
                self.raw_data_memmaps, desc="Calculating PCA components", ncols=98
        ):
            # Standardise and time embed the data
            std_data = processing.standardize(raw_data_memmap)
            te_std_data = processing.time_embed(std_data, self.n_embeddings)

            # Calculate the covariance of the entire dataset
            covariance += np.transpose(te_std_data) @ te_std_data

            # Clear data in memory
            del std_data, te_std_data

        # Use SVD to calculate PCA components
        u, s, vh = np.linalg.svd(covariance)
        u = u[:, :n_pca_components].astype(np.float32)
        explained_variance = np.sum(s[:n_pca_components]) / np.sum(s)
        print(f"Explained variance: {100 * explained_variance:.1f}%")
        s = s[:n_pca_components].astype(np.float32)
        if self.whiten:
            u = u @ np.diag(1.0 / np.sqrt(s))
        self._pca_components = u
        return self._pca_components

    def process(self):
        # Prepare the data
        for raw_data_memmap, processed_data_filename in zip(
                processing.tqdm(self.source_raw_data.subjects, desc="Preparing data", ncols=98),
                self.prepared_data_filenames,
        ):
            # Standardise and time embed the data
            std_data = processing.standardize(raw_data_memmap)
            te_std_data = processing.time_embed(std_data, self.n_embeddings)

            # Apply PCA to get the prepared data
            if self.pca_components is not None:
                processed_data = te_std_data @ self.pca_components

            # TODO: Check if this will ever be reached
            # Otherwise, the time embedded data is the prepared data
            else:
                processed_data = te_std_data

            # Finally, we standardise
            processed_data = processing.standardize(processed_data, create_copy=False)

            if self.load_memmaps:
                # Save the prepared data as a memmap
                processed_data_memmap = misc.array_to_memmap(
                    processed_data_filename, processed_data
                )
            else:
                processed_data_memmap = processed_data
            self.processed_data_memmaps.append(processed_data_memmap)


class AmplitudeEnvelopeProcessedData(ProcessedData):
    default_kwargs = dict(
        n_embeddings=1,
        n_window=1,
        low_freq=None,
        high_freq=None,
    )

    def validate_kwargs(self, **kwargs):
        super(AmplitudeEnvelopeProcessedData, self).validate_kwargs(**kwargs)
        if (
            self.processing_kwargs.get('low_freq') is not None or self.processing_kwargs.get('high_freq') is not None
        ) and self.source_raw_data.sampling_frequency is None:
            raise ValueError(
                "sampling_frequency must be set in the raw data if we are filtering the data. "
                + "Use set_sampling_frequency() or pass "
                + "DataCollection(..., sampling_frequency=...) when creating the DataCollection object."
            )

    def process(self):
        """Process data using amplitude envelope methods."""
        # Prepare the data
        low_freq_filter = self.processing_kwargs.get('low_freq')
        high_freq_filter = self.processing_kwargs.get('high_freq')
        n_windows = self.processing_kwargs.get('n_window')
        for raw_data_memmap, processed_data_file in zip(
                processing.tqdm(self.source_raw_data.subjects, desc="Preparing data", ncols=98),
                self.processed_data_filenames,
        ):
            # Filtering
            processed_data = processing.temporal_filter(
                raw_data_memmap, low_freq_filter, high_freq_filter, self.source_raw_data.sampling_frequency
            )

            # Hilbert transform
            processed_data = np.abs(signal.hilbert(processed_data, axis=0))

            # Moving average filter
            processed_data = np.array(
                [
                    np.convolve(
                        processed_data[:, i], np.ones(n_windows) / n_windows, mode="valid"
                    )
                    for i in range(processed_data.shape[1])
                ],
                dtype=np.float32,
            ).T

            # Standardize
            processed_data = processing.standardize(processed_data, create_copy=False)

            # Make sure data is float32
            processed_data = processed_data.astype(np.float32)

            # Create a memory map for the prepared data
            if self.load_memmaps:
                processed_data_memmap = array_to_memmap(processed_data_file, processed_data)
            else:
                processed_data_memmap = processed_data
            self.processed_data_memmaps.append(processed_data_memmap)
