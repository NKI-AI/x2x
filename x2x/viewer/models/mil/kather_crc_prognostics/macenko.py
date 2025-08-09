"""
Taken from https://github.com/KatherLab/preProcessing/blob/ab62869c367ddb3a9b166735021f66464b206d06/stainNorm_Macenko.py

Stain normalization based on the method of:

M. Macenko et al., 'A method for normalizing histology slides for quantitative analysis', in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

from __future__ import division

import numpy as np
import spams


def get_concentrations(
    I: np.ndarray, stain_matrix: np.ndarray, lamda: float = 0.01
) -> np.ndarray:
    """
    Get concentrations, a npix x 2 matrix

    Parameters
    ----------
        I: np.ndarray
        stain_matrix: np.ndarray
            a 2x3 stain matrix

    Returns
    -------
        np.ndarray: Concentrations
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    try:
        # Try to configure SPAMS for single-threaded operation
        try:
            # Set environment variable to limit OpenMP threads
            import os

            os.environ["OMP_NUM_THREADS"] = "1"
        except:
            pass

        temp = (
            spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True)
            .toarray()
            .T
        )
    except:
        temp = 0
    return temp


def RGB_to_OD(I: np.ndarray) -> np.ndarray:
    """
    Convert from RGB to optical density

    Parameters
    ----------
        I: np.ndarray
            RGB image

    Returns
    -------
        np.ndarray: Optical density
    """
    I = remove_zeros(I)
    return -1 * np.log(I / 255)


def remove_zeros(I: np.ndarray) -> np.ndarray:
    """
    Remove zeros, replace with 1's.

    Parameters
    ----------
        I: np.ndarray
            uint8 array

    Returns
    -------
        np.ndarray: Array with zeros replaced with 1's
    """
    mask = I == 0
    I[mask] = 1
    return I


def normalize_rows(A: np.ndarray) -> np.ndarray:
    """
    Normalize rows of an array

    Parameters
    ----------
        A: np.ndarray
            Array to normalize

    Returns
    -------
        np.ndarray: Normalized array
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def standardize_brightness(I):
    """

    Parameters
    ----------
        I: np.ndarray
            Image

    Returns
    -------
        np.ndarray: Standardized brightness
    """
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def get_stain_matrix(I: np.ndarray, beta: float = 0.15, alpha: float = 1) -> np.ndarray:
    """
    Get stain matrix (2x3)

    Parameters
    ----------
        I: np.ndarray
            Image
        beta: float
            Beta parameter
        alpha: float
            Alpha parameter

    Returns
    -------
        np.ndarray: Stain matrix
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    OD = OD[(OD > beta).any(axis=1), :]
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
    V = V[:, [2, 1]]
    if V[0, 0] < 0:
        V[:, 0] *= -1
    if V[0, 1] < 0:
        V[:, 1] *= -1
    That = np.dot(OD, V)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    return normalize_rows(HE)


###


class Normalizer(object):
    """
    Stain normalization object
    """

    def __init__(self):
        self.stain_matrix_target = None
        self.target_concentrations = None

    def fit(self, target: np.ndarray) -> None:
        target = standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)
        self.target_concentrations = get_concentrations(
            target, self.stain_matrix_target
        )

    def target_stains(self) -> np.ndarray:
        return OD_to_RGB(self.stain_matrix_target)

    def transform(self, I: np.ndarray) -> np.ndarray:
        I = standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape(
            (1, 2)
        )
        source_concentrations *= maxC_target / maxC_source
        return (
            255
            * np.exp(
                -1
                * np.dot(source_concentrations, self.stain_matrix_target).reshape(
                    I.shape
                )
            )
        ).astype(np.uint8)

    def hematoxylin(self, I: np.ndarray) -> np.ndarray:
        I = standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)
        return H
