from double_q_learning import DoubleQLearning
from climbing_environment import ClimbingEnvironment

def test_climbing():
    environment = ClimbingEnvironment()
    environment.run_tests(0.1, 0.05, 0.1, 0.9, 200, 1000)


test_climbing()