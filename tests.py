from environments.climbing_environment import ClimbingEnvironment
from environments.coordination_environment import CoordinationEnvironment
from environments.custom_environment import CustomEnvironment
from environments.fully_stochastic_climbing_environment import FullyStochasticClimbingEnvironment
from environments.partially_stochastic_climbing_environment import PartiallyStochasticClimbingEnvironment
from environments.penalty_environment import PenaltyEnvironment


def test_climbing():
    environment = ClimbingEnvironment()
    environment.run_tests(0.1, "1/T", 0.97, 0.95, 200, 1000)


def test_penalty():
    environment = PenaltyEnvironment(100)
    environment.run_tests(0.1, "1/T", 0.97, 0.95, 200, 1000)


def test_partially_stochastic_climbing():
    environment = PartiallyStochasticClimbingEnvironment()
    environment.run_tests(0.1, "1/T", 0.97, 0.95, 200, 1000)


def test_fully_stochastic_climbing():
    environment = FullyStochasticClimbingEnvironment()
    environment.run_tests(0.1, "1/T", 0.97, 0.95, 200, 1000)


def test_coordination():
    environment = CoordinationEnvironment(0)
    environment.run_tests(0.1, "1/T", 0.97, 0.95, 200, 1000)


def test_custom():
    environment = CustomEnvironment()
    environment.run_tests(0.1, "1/T", 0.97, 0.8, 200, 1000)


test_partially_stochastic_climbing()
