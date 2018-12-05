#####POST IT INSTRUMENT BY SOHAN VICHARE
#####TO PLAY: POINT CAMERA AT A WHITE WALL AND USE THREE POST ITS (ORANGE, BLUE, AND GREEN) MOVE THEM UP AND DOWN
import pygame, pygame.sndarray
import scipy.signal
import numpy
import cv2
pygame.mixer.pre_init(channels=1)
pygame.init()

def play_for(sample_wave, ms=1):
    """Play the given NumPy array, as a sound, for ms milliseconds."""
    sound = pygame.sndarray.make_sound(sample_wave)
    sound.play(-1)
    pygame.time.delay(ms)
    sound.stop()


sample_rate = 44100

def sine_wave(hz, peak, n_samples=sample_rate):
    """Compute N samples of a sine wave with given frequency and peak amplitude.
       Defaults to one second.
    """
    length = sample_rate / float(hz)
    omega = numpy.pi * 2 / length
    xvalues = numpy.arange(int(length)) * omega
    onecycle = peak * numpy.sin(xvalues)
    return numpy.resize(onecycle, (n_samples,)).astype(numpy.int16)

def square_wave(hz, peak, duty_cycle=.5, n_samples=sample_rate):
    """Compute N samples of a sine wave with given frequency and peak amplitude.
       Defaults to one second.
    """
    t = numpy.linspace(0, 1, 500 * 440/hz, endpoint=False)
    wave = scipy.signal.square(2 * numpy.pi * 5 * t, duty=duty_cycle)
    wave = numpy.resize(wave, (n_samples,))
    return (peak / 2 * wave.astype(numpy.int16))


def filter_green(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (30, 100, 90), (50, 200,255))
    imask = mask>0
    green = numpy.zeros_like(image, numpy.uint8)
    green[imask] = image[imask]
    return green

def filter_blue(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 150, 0), (150, 255, 255))
    imask = mask>0
    blue = numpy.zeros_like(image, numpy.uint8)
    blue[imask] = image[imask]
    return blue

def filter_orange(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (12, 170, 100), (30, 255, 255))
    imask = mask>0
    orange = numpy.zeros_like(image, numpy.uint8)
    orange[imask] = image[imask]
    return orange

def detect_blue(image):
    blue = filter_blue(image)
    imgray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 40, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 9000:
            x,y,w,h = cv2.boundingRect(contour)
            print("Blue found at " + str(y))
            return [True, y]
    return [False, None]

def detect_green(image):
    green = filter_green(image)
    imgray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 40, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 9000:
            x,y,w,h = cv2.boundingRect(contour)
            print("Green found at " + str(y))
            return [True, y]
    return [False, None]

def detect_orange(image):
    orange = filter_orange(image)
    imgray = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 40, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 9000:
            x,y,w,h = cv2.boundingRect(contour)
            print("Orange found at " + str(y))
            return [True, y]
    return [False, None]



def play_orange(y):
    if y > 0:
        return sine_wave(y, 4096)


def play_blue(y):
    if y > 0:
        return sine_wave(y, 4096)

def play_green(y):
    if y > 0:
        return sine_wave(y, 4096)

def play(frame):
    notes = []
    ##ORANGE
    orange = detect_orange(frame)
    if orange[0]:
        notes.append(play_orange(orange[1]))
    ##BLUE
    blue = detect_blue(frame)
    if blue[0]:
        notes.append(play_blue(blue[1]))
    ##GREEN
    green = detect_green(frame)
    if green[0]:
        notes.append(play_green(green[1]))

    if (orange[0] or blue[0]) or green[0]:
        #print(sum(notes))
        try:
            play_for(sum(notes), 500)
        except:
            print("cannot add noneType")

cap = cv2.VideoCapture(0)
while(True):

    ret, frame = cap.read()
    play(frame)

    imO = cv2.resize(filter_orange(frame), (300, 300))
    imB = cv2.resize(filter_blue(frame), (300, 300))
    imG = cv2.resize(filter_green(frame), (300, 300))
    frame = cv2.resize(frame, (300, 300))
    cv2.imshow("orange", imO )
    cv2.imshow("blue", imB)
    cv2.imshow("green", imG)
    cv2.imshow("original", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
