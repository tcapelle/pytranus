import argparse
import os
from pytranus import Lcal, BinaryInterface, L1s, L1sParam, TranusConfig

description = "<<Interactive pyLcal progra>>"
help = "Help, I need HELP!!!!"
parser = argparse.ArgumentParser(prog='PROG', description=description, add_help=help)
parser.add_argument('cmd', choices=['create','delete','help','quit'])

print "Enter path to binary Tranus files:"
bin_path = raw_input('(C://ProgramFiles/Tranus/) >> ')
print "Enter path to Tranus Project:"
project_path = raw_input('(/ExampleC/) >> ')
print "Project ID:"
project_ID = raw_input('(EXC) >> ')
print "Enter scenario Name:"
scn = raw_input('(03A) >> ')
t = TranusConfig(tranusBinPath = bin_path, workingDirectory = project_path, 
                 projectId=project_ID, scenarioId=scn) 
while True:
    astr = raw_input(' OperaName >> ')
    # print astr
    try:
        args = parser.parse_args(astr.split())
    except SystemExit:
        # trap argparse error message
        print 'error'
        continue
    if args.cmd in ['create', 'delete']:
        print 'doing', args.cmd
    elif args.cmd == 'help':
        parser.print_help()
    else:
        print 'done'
        break