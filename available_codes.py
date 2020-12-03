try:
    from amuse.community.fi.interface import Fi
except ImportError:
    Fi = None
try:
    from amuse.community.bhtree.interface import BHTree
except ImportError:
    BHTree = None
try:
    from amuse.community.fastkick.interface import FastKick
except ImportError:
    FastKick = None
try:
    from amuse.community.hermite.interface import Hermite
except ImportError:
    Hermite = None
try:
    from amuse.community.ph4.interface import ph4
except ImportError:
    ph4 = None
try:
    from amuse.community.petar.interface import Petar
except ImportError:
    Petar = None
try:
    from amuse.community.phantom.interface import Phantom
except ImportError:
    Phantom = None
try:
    from amuse.community.seba.interface import SeBa
except ImportError:
    SeBa = None
