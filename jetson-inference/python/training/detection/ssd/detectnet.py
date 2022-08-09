# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import jetson.inference
import jetson.utils

import argparse

import rospy
from std_msgs.msg import Float32


# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video output object 
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
	
# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

if __name__ == "__main__":

	rospy.init_node("detect_human_leg_and_pub")
	rate = rospy.Rate(10.0)
	pub = rospy.Publisher("keyop/angle", Float32, queue_size = 1)
	msg = Float32()
	pre_pos = 0
	cur_pos = 0
	v_pixel = 0
	# process frames until the user exits
	while not rospy.is_shutdown():
		# capture the next image
		img = input.Capture()
		pos_sum = 0

		# detect objects in the image (with overlay)
		detections = net.Detect(img, overlay=opt.overlay)

		# print the detections
		print "detected {:d} objects in image".format(len(detections))

		count = 0
		for detection in detections:
			if detection.Confidence > 0.8:
        			pos_sum += detection.Right + detection.Left
        			count = count + 1    
        			print detection
            
		if count != 0:
			cur_pos = pos_sum / count
        
		if not (pre_pos == 0):
			v_pixel = cur_pos - pre_pos
			print v_pixel
			msg.data = (0 - v_pixel) * 0.001
			pub.publish(msg)
		pre_pos = cur_pos

		# render the image
		output.Render(img)

		# update the title bar
		output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

		# print out performance info
		net.PrintProfilerTimes()

		# exit on input/output EOS
		if not input.IsStreaming() or not output.IsStreaming():
			break
		rate.sleep()


