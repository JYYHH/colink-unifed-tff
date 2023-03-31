import sys

from unifed.frameworks.tff import protocol
from unifed.frameworks.tff.workload_sim import *


def run_protocol():
    print('Running protocol...')
    protocol.pop.run()  # FIXME: require extra testing here

