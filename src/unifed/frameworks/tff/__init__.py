import sys

from unifed.frameworks.tff import protocol
from unifed.frameworks.tff.workload_sim import *
from unifed.frameworks.tff.models import *
from unifed.frameworks.tff.evaluate import *
from unifed.frameworks.tff.data_loader import *


def run_protocol():
    print('Running protocol...')
    protocol.pop.run()  # FIXME: require extra testing here

