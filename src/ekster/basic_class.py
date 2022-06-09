#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic class for a code wrapper. Not meant to be called on its own.
"""


class BasicCode(object):
    """Basic code class"""
    def __init__(
            self,
            code=None
    ):
        if code is not None:
            self.code = code()

    @property
    def model_time(self):
        """Return the current time of the code"""
        return self.code.model_time

    @property
    def particles(self):
        """Return the particleset in the code"""
        return self.code.particles
