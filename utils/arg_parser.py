"""! @package argParser

Parse argument from user command line input.

"""

# import library
import argparse
import numpy as np
 
def argParser():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument( "--enable-plot", required=False, action='store_true',
		help="enable/disable plotting" )

	ap.add_argument( "-v", "--verbose", required=False, default=1, type=int,
		help="	0: silent run\
				1: print info\
			  " )

	ap.add_argument( "-ns", "--numsample", required=False, default='1e7',
		help="input maximum number of samples" )

	ap.add_argument( "-ne", "--numepoch", required=False, default='30',
		help="input maximum number of epochs" )

	ap.add_argument("--enable-log", action='store_true',
		help="1: log the data\n\
			  0: no data logging\
			  ")

	ap.add_argument("--obj", required=False, default='l2', type=str,
		help="Objective for SNE example only, l1, l2")

	ap.add_argument("--alg", required=False, nargs="*", type=int, default=[1],
		help="select estimator:\
				1: GN\
				2: SGN\
				3: SGN2\
				4: Nested-SPIDER\
				5: SCGD")

	ap.add_argument( "-bs", "--batchsize", required=False, nargs="*", type=int,
		help="batchsize to estimate function and jacobian" )
	
	ap.add_argument( "--data", required=False, default='w8a',
		help="input the name/prefix of the dataset. \
			Supported extension: \
			 - train data: tr,train\
			 - test data: t,test\
			Ex: data.tr & data.t => - d data	\
			Ex: ndata & ndata.t => - d ndata	\
				" )

	ap.add_argument( "--seed", required=False, default=42, type=int,
		help="fix random seed" )

	# read arguments
	args = ap.parse_args()

	# create a dictionary to stor program parameters
	prog_option = {}

	# check whether to plot
	prog_option["PlotOption"] = args.enable_plot

	# whether to log data during training
	prog_option["LogEnable"] = args.enable_log

	# maximum number of epochs to run
	prog_option["MaxNumEpoch"] = int(float( args.numepoch ))

	# get objective type
	prog_option["Objective"] = args.obj
		
	# verbosity level
	prog_option["Verbose"] = args.verbose

	# random seed
	prog_option["Seed"] = args.seed

	# get batchsize
	prog_option["BatchSize"] = args.batchsize

	# whether to log data during training
	available_algs = ['GN','SGN','SGN2','N-SPIDER','SCGD']
	prog_option["Alg"] = []
	for i in range(len(available_algs)):
		if (i+1) in args.alg:
			prog_option["Alg"].append(available_algs[i])

	# get dataset name
	prog_option['Dataset'] = args.data

	return prog_option
