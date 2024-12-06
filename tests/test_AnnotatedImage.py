import numpy as np
import astropy.units as u

import pytest

import logging

logger = logging.getLogger(__name__)

from L4_guiding_util import AnnotatedImage


def test_basic():

    a = AnnotatedImage(np.arange(10), unit=u.ph / u.s, pix_sca=0.01 * u.arcsec / u.pix)
    logger.debug(str(a))

    assert (a.quantity() == (np.arange(10) * u.ph / u.s)).all()
