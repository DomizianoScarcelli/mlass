import unittest
import torch
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from typing import Dict, Sequence
from diba.diba.interfaces import Likelihood, SeparationPrior
from tqdm import tqdm
import torch.nn.functional as F
