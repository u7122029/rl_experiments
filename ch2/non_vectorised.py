import numpy as np
from icecream import ic

pS0 = {"hungry": 1/2,
       "full": 1/2}

states = ["hungry", "full"]
actions = ["ignore", "feed"]
dynamics = {("hungry", "ignore", -2, "hungry"): 1,
            ("hungry", "feed", -3, "hungry"): 1/3,
            ("hungry", "feed", 1, "full"): 2/3,
            ("full", "ignore", -2, "hungry"): 3/4,
            ("full", "ignore", 2, "full"): 1/4,
            ("full", "feed", 1, "full"): 1} # p(s',r|s,a)

rewards = set([x[2] for x in dynamics.keys()])

# Transitions s -a-> s'
s_a_sp = set([(x[0], x[1], x[3]) for x in dynamics.keys()])

# Transition probability p(s'|s,a)
p_sp_given_s_a = {(sp, s, a): sum([dynamics.get((s, a, r, sp), 0) for r in rewards]) for s, a, sp in s_a_sp}

# Expected reward given state-action pair r(s,a)
r_sa = {(s, a): sum([r * dynamics.get((s, a, r, sp), 0) for r in rewards for sp in states]) for s in states for a in actions}

# Expected reward given state, action, and next state r(s,a,s')
r_s_a_sp = {(sp,a,s): sum([r * dynamics.get((s, a, r, sp), 0) for r in rewards]) / p_sp_given_s_a.get((sp, s, a), 1)
            for s in states for sp in states for a in actions}

# Policies pi(a|s)
pi = {("ignore", "hungry"): 1/4,
      ("feed", "hungry"): 3/4,
      ("ignore", "full"): 5/6,
      ("feed", "full"): 1/6}

pi_deterministic = {"hungry": "feed",
                    "full": "feed"}

# Initial state-action distribution
p_S0_A0_pi = {(s, a): pS0[s] * pi[(a, s)] for s in states for a in actions}

# Transition probability from state to next state p_pi(s'|s)
p_pi = {(sp, s): sum(p_sp_given_s_a.get((sp, s, a), 0) * pi[(a, s)] for a in actions) for s in states for sp in states}

# Transition prob from state-action pair to next state-action pair.
p_pi_sp_ap_given_s_a = {(sp, ap, s, a): pi[(ap, sp)] * p_sp_given_s_a.get((sp, s, a), 0)
                        for sp in states for ap in actions for s in states for a in actions}
ic(p_pi_sp_ap_given_s_a)

# Expected state reward
r_pi_s = {s: sum(r_sa.get((s, a), 0) * pi[(a, s)] for a in actions) for s in states}

