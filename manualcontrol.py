# pins working are: 0,1,3,8,11
# pin 0 close is 0/open is 180
# pin 1 close is 150/ open is 0
# pin 3 close is  70/open is 180
# pin 8 close is 60/ open is 180

import time
from adafruit_servokit import ServoKit

# Set channels to the number of servo channels on your kit.
# 8 for FeatherWing, 16 for Shield/HAT/Bonnet.
# pin 0 is pinky,pin 1 is ring finger, pin 3 is middle finger, pin 8 is index, pin 11 is thumb
kit = ServoKit(channels=16)

kit.servo[0].angle = 180
kit.continuous_servo[1].throttle = 1
time.sleep(1)
kit.continuous_servo[1].throttle = -1
time.sleep(1)
kit.servo[0].angle = 0
kit.continuous_servo[1].throttle = 0

kit.servo[8].angle = 180
kit.continuous_servo[3].throttle = 1
time.sleep(1)
kit.continuous_servo[3].throttle = -1
time.sleep(1)
kit.servo[8].angle = 0
kit.continuous_servo[1].throttle = 0

kit.servo[11].angle = 180
time.sleep(1)
kit.servo[11].angle = 0

