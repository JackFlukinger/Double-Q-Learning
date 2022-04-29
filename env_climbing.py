a = "a"
b = "b"
c = "c"

actions = [a, b, c]

t_func = {
    (a, a): 11,
    (a, b): -30,
    (a, c): 0,

    (b, a): -30,
    (b, b): 7,
    (b, c): 6,

    (c, a): 0,
    (c, b): 0,
    (c, c): 5,
}


def get_reward(p1_action, p2_action):
    return t_func[(p1_action, p2_action)]
