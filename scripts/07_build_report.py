import argparse
from pathlib import Path
from evaluation.report import report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);a=p.parse_args();print(report(a.comparison))
