from salp_mdp import SalpEnvironment, SalpAgent

Env = SalpEnvironment("grids/salp/illustration_eddy.txt", 0)
agent = SalpAgent(Env)

with open('icp_data1.txt', 'w') as file:
    file.write("State\tR1\tR2\tR3\tContext\n")  # Writing the header
    for s in agent.S:
        file.write("{} {} {} {} {}\n".format(s, Env.R1(s, 'Noop'), Env.R2(s, 'Noop'), Env.R3(s, 'Noop'), Env.state2context_map[s]))

    