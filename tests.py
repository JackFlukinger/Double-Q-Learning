from environments.climbing_environment import ClimbingEnvironment
from environments.coordination_environment import CoordinationEnvironment
from environments.stochastic_coordination_environment import StochasticCoordinationEnvironment
from environments.fully_stochastic_climbing_environment import FullyStochasticClimbingEnvironment
from environments.partially_stochastic_climbing_environment import PartiallyStochasticClimbingEnvironment
from environments.penalty_environment import PenaltyEnvironment


def test_climbing():
    environment = ClimbingEnvironment()
    environment.run_tests("1/T", "1/T", 5000, 0.997, 0.95, 3000, 100)


def test_penalty(k):
    environment = PenaltyEnvironment(k)
    environment.run_tests("1/T", "1/T", 5000, 0.997, 0.95, 3000, 100)


def test_partially_stochastic_climbing():
    environment = PartiallyStochasticClimbingEnvironment()
    environment.run_tests("1/T", "1/T", 5000, 0.997, 0.95, 4000, 100)


def test_fully_stochastic_climbing():
    environment = FullyStochasticClimbingEnvironment()
    environment.run_tests("1/T", "1/T", 5000, 0.997, 0.95, 4000, 100)


def test_coordination(k):
    environment = CoordinationEnvironment(k)
    environment.run_tests("1/T", "1/T", 5000, 0.997, 0.95, 2000, 100)


def test_stochastic_coordination(k):
    environment = StochasticCoordinationEnvironment(k)
    environment.run_tests("1/T", "1/T", 5000, 0.997, 0.95, 2000, 100)


test_stochastic_coordination(-100)
