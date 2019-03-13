import glob
import argparse
from argparse import RawTextHelpFormatter
import ModTox.cpptraj.analisis as an

def parse_args():
    
    parser = argparse.ArgumentParser(description='Specify trajectories and topology to be analised.\n  \
    i.e python -m ModTox.main traj.xtc --top topology.pdb', formatter_class=RawTextHelpFormatter)
    parser.add_argument('trajs', nargs="+", help='Trajectory to analise')
    parser.add_argument('--top', type=str, help='Topology of your trajectory')
    args = parser.parse_args()
    return args.trajs, args.top

def main(trajs, top):
    #In case of pdb
    top = top if top else trajs[0]
    for traj in trajs:
        an.analise(traj, top)

if __name__ == "__main__":
    trajs, top = parse_args()
    main(trajs, top)
